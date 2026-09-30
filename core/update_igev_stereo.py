"""IGEV-style recurrent update blocks for the pure-stereo CorGA model."""

import torch
import torch.nn as nn

from core.update_pru import DispHead, MotionEncoder, interp, pool2x


class ConvGRU(nn.Module):
    """Convolutional GRU with additive context biases for all three gates."""

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super().__init__()
        channels = hidden_dim + input_dim
        padding = kernel_size // 2
        self.convz = nn.Conv2d(channels, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(channels, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(channels, hidden_dim, kernel_size, padding=padding)

    def forward(self, hidden, context_z, context_r, context_q, *inputs):
        features = torch.cat(inputs, dim=1)
        hidden_features = torch.cat((hidden, features), dim=1)
        update_gate = torch.sigmoid(self.convz(hidden_features) + context_z)
        reset_gate = torch.sigmoid(self.convr(hidden_features) + context_r)
        candidate = torch.tanh(
            self.convq(torch.cat((reset_gate * hidden, features), dim=1))
            + context_q
        )
        return (1.0 - update_gate) * hidden + update_gate * candidate


class StereoIGEVUpdateBlock(nn.Module):
    """Multi-scale ConvGRU update without a monocular/anchor disparity prompt.

    This follows the IGEV family update pattern: coarse recurrent states are
    updated first, the finest state consumes stereo motion features, and static
    context supplies learned biases to each GRU gate. CorGA's existing FullCorr
    samples and right-warp features replace IGEV's geometry-volume encoder.
    """

    def __init__(self, args, hidden_dim, feat_dim, volume_dim,
                 warp_feat_dim, cor_planes=None, num_layers=None,
                 use_warp_features=True):
        super().__init__()
        del feat_dim
        self.num_layers = (
            args.n_gru_layers if num_layers is None else int(num_layers)
        )
        if not 1 <= self.num_layers <= 3:
            raise ValueError(
                f"StereoIGEVUpdateBlock supports 1-3 levels, got {self.num_layers}"
            )

        if cor_planes is None:
            cor_planes = (
                args.corr_levels * (2 * args.corr_radius + 1) * volume_dim
            )
        self.motion_encoder = MotionEncoder(
            hidden_dim, cor_planes, warp_feat_dim,
            use_warp_features=use_warp_features,
        )

        # The current CorGA context encoder exposes a finest-scale context
        # tensor. Pool it once per level and learn IGEV-style z/r/q biases.
        self.context_encoders = nn.ModuleList([
            nn.Conv2d(hidden_dim, 3 * hidden_dim, 3, padding=1)
            for _ in range(self.num_layers)
        ])

        gru_input_dims = []
        for level in range(self.num_layers):
            input_dim = hidden_dim if level == 0 else 0  # stereo motion
            if level > 0:
                input_dim += hidden_dim                  # finer state
            if level + 1 < self.num_layers:
                input_dim += hidden_dim                  # coarser state
            gru_input_dims.append(input_dim)

        self.gru_layers = nn.ModuleList([
            ConvGRU(hidden_dim, input_dim, kernel_size=3)
            for input_dim in gru_input_dims
        ])
        self.disp_head = DispHead(hidden_dim)
        upsample_factor = 2 ** args.n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, (upsample_factor ** 2) * 9, 1),
        )

    def prepare_static(self, ctx, anchor_disp=None):
        """Cache multi-scale context gate biases for one image pair."""
        del anchor_disp
        context = ctx.float()
        context_gates = []
        for level, encoder in enumerate(self.context_encoders):
            if level:
                context = pool2x(context)
            context_gates.append(encoder(context).chunk(3, dim=1))
        return context_gates

    def forward(self, net, corr, disp, ctx, warp_feat, anchor_disp=None,
                static_features=None, need_mask=True):
        del anchor_disp
        if len(net) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} recurrent states, got {len(net)}"
            )

        context_gates = (
            self.prepare_static(ctx) if static_features is None
            else static_features
        )
        if len(context_gates) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} context levels, "
                f"got {len(context_gates)}"
            )

        motion = self.motion_encoder(corr, disp, warp_feat)

        # Match IGEV's high-to-low schedule. Each newly updated coarse state is
        # immediately available to the next finer level in this iteration.
        for level in reversed(range(self.num_layers)):
            inputs = []
            if level == 0:
                inputs.append(motion)
            if level > 0:
                inputs.append(pool2x(net[level - 1]))
            if level + 1 < self.num_layers:
                inputs.append(interp(net[level + 1], net[level]))
            net[level] = self.gru_layers[level](
                net[level], *context_gates[level], *inputs
            )

        delta_disp = self.disp_head(net[0])
        mask = 0.25 * self.mask(net[0]) if need_mask else None
        return net, delta_disp, mask
