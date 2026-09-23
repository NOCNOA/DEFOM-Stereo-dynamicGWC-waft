"""CorGA-fast-binmono with explicit correlation statistics in refine motion."""

import math

import torch
import torch.nn as nn

from core.defom_cor_ga_fast_binmono import DEFOMStereo as CorGAFastBinMonoStereo
from core.update_pru import MotionEncoder


class CorrelationStatisticsMotionEncoder(MotionEncoder):
    """Add direction and uncertainty statistics through a zero-init residual."""

    def __init__(self, hidden_dim, corplane, warp_feat_dim, corr_levels,
                 corr_radius, adapter_hidden=32, temperature=1.0):
        super().__init__(hidden_dim, corplane, warp_feat_dim)
        self.corr_levels = int(corr_levels)
        self.corr_radius = int(corr_radius)
        self.corr_window = 2 * self.corr_radius + 1
        self.corr_stats_temperature = float(temperature)
        if self.corr_levels < 1 or self.corr_radius < 1:
            raise ValueError("corr_levels and corr_radius must be positive")
        if self.corr_stats_temperature <= 0.0:
            raise ValueError("corr_stats_temperature must be positive")
        if corplane != self.corr_levels * self.corr_window:
            raise ValueError(
                f"Expected {self.corr_levels * self.corr_window} FullCorr "
                f"channels, got {corplane}"
            )

        adapter_hidden = int(adapter_hidden)
        if adapter_hidden < 1:
            raise ValueError("corr_stats_hidden must be positive")
        self.corr_stats_adapter = nn.Sequential(
            nn.Conv2d(self.corr_levels * 5, adapter_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_hidden, hidden_dim - 1, 1),
        )
        nn.init.zeros_(self.corr_stats_adapter[-1].weight)
        nn.init.zeros_(self.corr_stats_adapter[-1].bias)
        offsets = torch.arange(
            -self.corr_radius, self.corr_radius + 1, dtype=torch.float32
        )
        self.register_buffer(
            "corr_stats_offsets",
            offsets.view(1, 1, self.corr_window, 1, 1),
        )

    def correlation_statistics(self, corr):
        batch, channels, height, width = corr.shape
        expected = self.corr_levels * self.corr_window
        if channels != expected:
            raise ValueError(f"Expected {expected} correlation channels, got {channels}")
        grouped = corr.float().reshape(
            batch, self.corr_levels, self.corr_window, height, width
        )
        local_mean = grouped.mean(dim=2, keepdim=True)
        local_std = grouped.std(
            dim=2, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        scores = (grouped - local_mean) / local_std
        probability = torch.softmax(
            scores / self.corr_stats_temperature, dim=2
        )
        offsets = self.corr_stats_offsets.to(device=corr.device)
        expected_offset = (probability * offsets).sum(dim=2)
        # CorrBlock samples right-image x, hence disparity delta has the
        # opposite sign. Level identity is retained by the channel layout.
        suggested_delta = -expected_offset / float(self.corr_radius)
        variance = (
            probability * (offsets - expected_offset.unsqueeze(2)).square()
        ).sum(dim=2)
        offset_std = variance.clamp_min(0.0).sqrt() / float(self.corr_radius)
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=2) / math.log(float(self.corr_window))
        peak = probability.amax(dim=2)
        top2 = grouped.topk(k=2, dim=2).values
        margin = top2[:, :, 0] - top2[:, :, 1]
        statistics = torch.stack(
            (suggested_delta, offset_std, entropy, peak, margin), dim=2
        )
        return statistics.flatten(1, 2)

    def forward(self, corr, disp, warp_feat):
        # Call the original implementation verbatim so zero residual means
        # exact behavioral compatibility rather than a duplicated code path.
        base_motion = super().forward(corr, disp, warp_feat)
        base_out, raw_disp = base_motion[:, :-1], base_motion[:, -1:]
        stats = self.correlation_statistics(corr)
        residual = self.corr_stats_adapter(stats.to(base_out.dtype))
        # Do not alter the final raw-disparity channel of the motion feature.
        return torch.cat((base_out + residual, raw_disp), dim=1)


class DEFOMStereo(CorGAFastBinMonoStereo):
    """Use correlation statistics only in the refine MotionEncoder."""

    def __init__(self, args):
        super().__init__(args)
        hidden_dim = int(args.hidden_dims[0])
        corr_levels = int(args.corr_levels)
        corr_radius = int(args.corr_radius)
        old_encoder = self.refine_update_block.motion_encoder
        new_encoder = CorrelationStatisticsMotionEncoder(
            hidden_dim=hidden_dim,
            corplane=corr_levels * (2 * corr_radius + 1),
            warp_feat_dim=32,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            adapter_hidden=int(getattr(args, "corr_stats_hidden", 32)),
            temperature=float(getattr(args, "corr_stats_temperature", 1.0)),
        )
        result = new_encoder.load_state_dict(old_encoder.state_dict(), strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected MotionEncoder keys: {result.unexpected_keys}")
        self.refine_update_block.motion_encoder = new_encoder
