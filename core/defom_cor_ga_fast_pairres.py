"""CorGA-fast with a separately trainable per-candidate matching residual.

The original CorGA-fast path is kept intact. During refinement only, every
sampled right candidate is paired with the left feature before scale reduction:

    pair(s, k) = ReLU(P_left(left) + P_right(right(s, k)))

All inherited parameters are frozen so this experiment trains only the new
residual branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defom_cor_ga_fast import DEFOMStereo as CorGAFast
from core.submodules import lookup_right_features
from core.warp import sample_right_feature_pyramid


class DEFOMStereo(CorGAFast):
    """Train only a lightweight left/right candidate residual on CorGA-fast."""

    def __init__(self, args):
        super().__init__(args)

        feature_dim = 32
        pair_dim = int(getattr(args, "pair_feature_dim", 8))
        if pair_dim <= 0:
            raise ValueError(f"pair_feature_dim must be positive, got {pair_dim}")

        window_size = 2 * self.corr_radius + 1
        self.pair_feature_dim = pair_dim
        self.pair_left_proj = nn.Conv2d(
            feature_dim, pair_dim, kernel_size=1, bias=True
        )
        self.pair_right_proj = nn.Conv2d(
            feature_dim, pair_dim, kernel_size=1, bias=False
        )
        self.pair_residual_head = nn.Sequential(
            nn.Conv2d(pair_dim * window_size, feature_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
        )

        # Small non-zero output weights let all new layers receive gradients on
        # the first batch while perturbing the pretrained path only minimally.
        init_std = float(getattr(args, "pair_residual_init_std", 1e-4))
        if init_std <= 0:
            raise ValueError(
                f"pair_residual_init_std must be positive, got {init_std}"
            )
        nn.init.normal_(
            self.pair_residual_head[-1].weight, mean=0.0, std=init_std
        )

        # fetch_optimizer() selects parameters by requires_grad. Frozen modules
        # still propagate input gradients from the final loss to this branch.
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.pair_left_proj,
            self.pair_right_proj,
            self.pair_residual_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _lookup_right_features(
        self,
        fmap1_low,
        right_feature_pyramid,
        disp,
        warp_offsets,
        feature_fuse,
        rope_galerkin=None,
        scale_weights=None,
        compression_ratios=(1, 2, 4),
        compact_sampling=False,
    ):
        # Coarse lookup has one scale and no scale weights; leave it unchanged.
        if scale_weights is None:
            return lookup_right_features(
                fmap1_low,
                right_feature_pyramid,
                disp,
                warp_offsets,
                feature_fuse,
                rope_galerkin=rope_galerkin,
                scale_weights=None,
                compression_ratios=compression_ratios,
                compact_sampling=compact_sampling,
            )

        sampled_right = sample_right_feature_pyramid(
            right_feature_pyramid,
            disp,
            offsets=warp_offsets.flatten(),
            compression_ratios=compression_ratios,
            padding_mode="zeros",
            align_corners=True,
            compact=compact_sampling,
        )

        batch, channels, height, width = fmap1_low.shape
        num_scales = len(compression_ratios)
        total_samples = sampled_right.shape[1]
        if total_samples % num_scales != 0:
            raise RuntimeError(
                f"Cannot split {total_samples} samples into {num_scales} scales"
            )
        samples_per_scale = total_samples // num_scales
        sampled_right = sampled_right.reshape(
            batch, num_scales, samples_per_scale, channels, height, width
        )

        expected_weights = (batch, num_scales, height, width)
        if tuple(scale_weights.shape) != expected_weights:
            raise ValueError(
                f"scale_weights must be {expected_weights}, "
                f"got {tuple(scale_weights.shape)}"
            )
        weights = scale_weights.to(sampled_right.dtype)[:, :, None, None]

        # Original pretrained path.
        selected_right = (sampled_right * weights).sum(dim=1)
        original_input = torch.cat(
            [fmap1_low.to(sampled_right.dtype), selected_right.flatten(1, 2)],
            dim=1,
        )
        original_feature = feature_fuse(original_input)

        # Candidate-specific nonlinear interaction before scale/candidate fusion.
        left_token = self.pair_left_proj(fmap1_low)
        right_token = self.pair_right_proj(
            sampled_right.reshape(
                batch * num_scales * samples_per_scale,
                channels,
                height,
                width,
            )
        ).reshape(
            batch,
            num_scales,
            samples_per_scale,
            self.pair_feature_dim,
            height,
            width,
        )
        pair_tokens = F.relu(left_token[:, None, None] + right_token)
        selected_pairs = (pair_tokens * weights).sum(dim=1)
        pair_feature = self.pair_residual_head(selected_pairs.flatten(1, 2))

        fused_feature = original_feature + pair_feature
        if rope_galerkin is not None:
            fused_feature = rope_galerkin(fused_feature)
        return fused_feature
