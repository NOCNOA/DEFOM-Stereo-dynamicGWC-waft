"""Multi-Range Epipolar Feature Augmentation for the bins4 model.

MEFA augments unary left/right descriptors once before recurrent disparity
sampling. It does not construct a cost volume or perform left-right attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.warp import sample_right_feature_pyramid


class LocalEnhance(nn.Module):
    """Depthwise 3x3 local structure enhancement followed by pointwise mixing."""

    def __init__(self, channels=32):
        super().__init__()
        self.channels = int(channels)
        self.depthwise = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, padding=1,
            groups=self.channels
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode='fan_out', nonlinearity='relu'
        )
        nn.init.zeros_(self.depthwise.bias)
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, feature):
        return self.pointwise(self.activation(self.depthwise(feature)))


class HorizontalConvEnhance(nn.Module):
    """Lightweight epipolar context using depthwise horizontal convolution."""

    def __init__(self, channels=32, kernel_size=9):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(1, self.kernel_size),
            padding=(0, self.kernel_size // 2),
            groups=self.channels,
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode='fan_out', nonlinearity='relu'
        )
        nn.init.zeros_(self.depthwise.bias)
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, feature):
        return self.pointwise(self.activation(self.depthwise(feature)))


class HorizontalSelfAttention(nn.Module):
    """One standard pre-norm MHSA block over each horizontal image row."""

    def __init__(self, channels=32, num_heads=4, ffn_expansion=2,
                 attention_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        hidden_channels = int(self.channels * ffn_expansion)
        self.attention_norm = nn.LayerNorm(self.channels)
        self.attention = nn.MultiheadAttention(
            self.channels,
            self.num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_channels, self.channels),
            nn.Dropout(ffn_dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        if self.attention.in_proj_bias is not None:
            nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _sinusoidal_position(length, channels, device, dtype):
        """Return dynamic horizontal PE [1,W,C], computed stably in FP32."""
        position = torch.arange(length, device=device, dtype=torch.float32)
        position = position[:, None]
        frequencies = torch.arange(
            0, channels, 2, device=device, dtype=torch.float32
        )
        frequencies = torch.exp(-math.log(10000.0) * frequencies / channels)
        angles = position * frequencies[None]
        encoding = torch.zeros(
            length, channels, device=device, dtype=torch.float32
        )
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
        return encoding[None].to(dtype=dtype)

    def forward(self, feature):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"feature must be [B,{self.channels},H,W], "
                f"got {tuple(feature.shape)}"
            )
        batch, channels, height, width = feature.shape
        # [B,C,H,W] -> [B,H,W,C] -> [B*H,W,C]. W is the sequence axis.
        sequence = feature.permute(0, 2, 3, 1).reshape(
            batch * height, width, channels
        )
        sequence = sequence + self._sinusoidal_position(
            width, channels, sequence.device, sequence.dtype
        )
        normalized = self.attention_norm(sequence)
        attention_update, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        sequence = sequence + attention_update
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        return sequence.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2
        )


class SelectiveFusion(nn.Module):
    """Spatially select local detail or epipolar/global context."""

    def __init__(self, channels=32):
        super().__init__()
        self.channels = int(channels)
        self.gate = nn.Conv2d(2 * self.channels, 1, kernel_size=3, padding=1)
        # Begin with an unbiased 50/50 mixture.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, local_feature, context_feature):
        if local_feature.shape != context_feature.shape:
            raise ValueError(
                "local and context features must have identical shapes, got "
                f"{tuple(local_feature.shape)} and {tuple(context_feature.shape)}"
            )
        gate = torch.sigmoid(
            self.gate(torch.cat([local_feature, context_feature], dim=1))
        )
        return gate * local_feature + (1.0 - gate) * context_feature


class CrossRangeFusion(nn.Module):
    """Propagate a coarser horizontal range into the adjacent finer range."""

    def __init__(self, channels=32):
        super().__init__()
        self.channels = int(channels)
        self.projection = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.gate = nn.Conv2d(2 * self.channels, 1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, fine_feature, coarse_feature):
        if fine_feature.ndim != 4 or coarse_feature.ndim != 4:
            raise ValueError("fine_feature and coarse_feature must be 4D tensors")
        if fine_feature.shape[:3] != coarse_feature.shape[:3]:
            raise ValueError(
                "cross-range inputs must match in batch, channels and height, "
                f"got {tuple(fine_feature.shape)} and {tuple(coarse_feature.shape)}"
            )
        upsampled = F.interpolate(
            coarse_feature,
            size=fine_feature.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        upsampled = self.projection(upsampled)
        gate = torch.sigmoid(
            self.gate(torch.cat([fine_feature, upsampled], dim=1))
        )
        return fine_feature + gate * upsampled


class MultiRangeEpipolarFeatureAugmentation(nn.Module):
    """One-shot MEFA over left and width-compressed right unary features.

    Left/right features at the same range are concatenated along the batch axis
    before enhancement. This both enforces parameter sharing and avoids any
    cross-view interaction: every batch item is still processed independently.
    """

    def __init__(self, channels=32, num_heads=4, horizontal_kernel=9,
                 layerscale_init=1e-2):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if layerscale_init < 0:
            raise ValueError(
                f"layerscale_init must be non-negative, got {layerscale_init}"
            )
        self.channels = int(channels)

        # Auxiliary left pyramid only; output sizes are aligned to the actual
        # right branch sizes in forward, which also handles odd input widths.
        self.left_down_2 = nn.Conv2d(
            self.channels, self.channels, kernel_size=(1, 3),
            stride=(1, 2), padding=(0, 1)
        )
        self.left_down_4 = nn.Conv2d(
            self.channels, self.channels, kernel_size=(1, 3),
            stride=(1, 2), padding=(0, 1)
        )

        self.local_branches = nn.ModuleList([
            LocalEnhance(self.channels) for _ in range(3)
        ])
        self.fine_context = HorizontalConvEnhance(
            self.channels, horizontal_kernel
        )
        self.medium_context = HorizontalConvEnhance(
            self.channels, horizontal_kernel
        )
        self.coarse_context = HorizontalSelfAttention(
            self.channels,
            num_heads=num_heads,
            ffn_expansion=2,
            attention_dropout=0.0,
            ffn_dropout=0.0,
        )
        self.selective_fusions = nn.ModuleList([
            SelectiveFusion(self.channels) for _ in range(3)
        ])
        self.residual_projections = nn.ModuleList([
            nn.Conv2d(self.channels, self.channels, kernel_size=1)
            for _ in range(3)
        ])
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.full((1, self.channels, 1, 1), layerscale_init))
            for _ in range(3)
        ])
        self.coarse_to_medium = CrossRangeFusion(self.channels)
        self.medium_to_fine = CrossRangeFusion(self.channels)
        self.reset_parameters()

    def reset_parameters(self):
        for downsample in (self.left_down_2, self.left_down_4):
            nn.init.kaiming_normal_(
                downsample.weight, mode='fan_out', nonlinearity='relu'
            )
            nn.init.zeros_(downsample.bias)
        for projection in self.residual_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _match_target_size(feature, target):
        if feature.shape[-2:] == target.shape[-2:]:
            return feature
        return F.interpolate(
            feature,
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    def _enhance_scale(self, paired_feature, scale_index):
        local_feature = self.local_branches[scale_index](paired_feature)
        if scale_index == 0:
            context_feature = self.fine_context(paired_feature)
        elif scale_index == 1:
            context_feature = self.medium_context(paired_feature)
        else:
            context_feature = self.coarse_context(paired_feature)
        mixed_feature = self.selective_fusions[scale_index](
            local_feature, context_feature
        )
        residual_update = self.residual_projections[scale_index](mixed_feature)
        return paired_feature + self.layer_scales[scale_index] * residual_update

    def forward(self, left, right_1, right_2, right_4):
        features = (left, right_1, right_2, right_4)
        for name, feature in zip(
                ('left', 'right_1', 'right_2', 'right_4'), features):
            if feature.ndim != 4:
                raise ValueError(
                    f"{name} must be [B,C,H,W_s], got {tuple(feature.shape)}"
                )
            if feature.shape[1] != self.channels:
                raise ValueError(
                    f"{name} must have {self.channels} channels, "
                    f"got {feature.shape[1]}"
                )
        batch, _, height, width = left.shape
        for name, feature in zip(
                ('right_1', 'right_2', 'right_4'), features[1:]):
            if feature.shape[0] != batch or feature.shape[2] != height:
                raise ValueError(
                    f"{name} must match left batch and height, got "
                    f"{tuple(feature.shape)} versus {tuple(left.shape)}"
                )
        if right_1.shape[-1] != width:
            raise ValueError(
                f"right_1 width must match left width, got {right_1.shape[-1]} "
                f"and {width}"
            )

        left_2 = self._match_target_size(self.left_down_2(left), right_2)
        left_4 = self._match_target_size(self.left_down_4(left_2), right_4)

        # Shared left/right processing without correspondence interaction.
        pair_1 = torch.cat([left, right_1], dim=0)
        pair_2 = torch.cat([left_2, right_2], dim=0)
        pair_4 = torch.cat([left_4, right_4], dim=0)
        pair_1 = self._enhance_scale(pair_1, 0)
        pair_2 = self._enhance_scale(pair_2, 1)
        pair_4 = self._enhance_scale(pair_4, 2)

        pair_2 = self.coarse_to_medium(pair_2, pair_4)
        pair_1 = self.medium_to_fine(pair_1, pair_2)

        left_aug, right_1_aug = pair_1.split(batch, dim=0)
        _, right_2_aug = pair_2.split(batch, dim=0)
        _, right_4_aug = pair_4.split(batch, dim=0)
        return left_aug, right_1_aug, right_2_aug, right_4_aug


class MultiRangeSparseMatchingLookup(nn.Module):
    """Bins4-only joint multi-scale lookup over cached MEFA descriptors.

    This module performs the disparity-dependent work that cannot be cached:
    right-feature sampling and left-right matching token construction. It does
    not mix or enhance hypotheses and contains no recurrent-time attention.
    All scale/hypothesis tokens are concatenated before one shared fusion MLP;
    no per-scale reduction or scale-weighted sum is applied.
    """

    def __init__(self, compression_ratios=(1, 2, 4), group=8, channels=32,
                 samples_per_scale=9, projected_token=32, output_channels=128):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by group ({group})"
            )
        self.compression_ratios = tuple(compression_ratios)
        self.group = int(group)
        self.channels = int(channels)
        self.samples_per_scale = int(samples_per_scale)
        self.projected_token = int(projected_token)
        self.output_channels = int(output_channels)
        self.token_projection = nn.Sequential(
            nn.Linear(self.channels + self.group, self.projected_token),
            nn.GELU(),
        )
        self.match_fuse = nn.Sequential(
            nn.Linear(
                len(self.compression_ratios)
                * self.samples_per_scale
                * self.projected_token,
                self.output_channels,
            ),
            nn.GELU(),
            nn.Linear(self.output_channels, self.output_channels),
        )

    def _match_single_scale(self, left, right, disp, warp_offsets, ratio):
        batch, channels, height, width = left.shape
        sampled_right = sample_right_feature_pyramid(
            (right,),
            disp,
            offsets=warp_offsets.flatten(),
            compression_ratios=(ratio,),
            padding_mode='zeros',
            align_corners=True,
        )
        hypothesis_count = sampled_right.shape[1]
        if hypothesis_count != self.samples_per_scale:
            raise ValueError(
                f"expected {self.samples_per_scale} hypotheses, "
                f"got {hypothesis_count}"
            )

        feature_difference = left[:, None] - sampled_right
        channels_per_group = channels // self.group
        left_grouped = left.reshape(
            batch, self.group, channels_per_group, height, width
        )[:, None]
        right_grouped = sampled_right.reshape(
            batch, hypothesis_count, self.group,
            channels_per_group, height, width
        )
        groupwise_corr = (
            left_grouped.float() * right_grouped.float()
        ).mean(dim=3).to(sampled_right.dtype)

        match_tokens = torch.cat(
            [groupwise_corr, feature_difference], dim=2
        )
        match_tokens = match_tokens.permute(0, 3, 4, 1, 2).reshape(
            -1, hypothesis_count, channels + self.group
        )
        return self.token_projection(match_tokens)

    def forward(self, left, right_feature_pyramid, disp, warp_offsets,
                scale_weights):
        if left.ndim != 4 or left.shape[1] != self.channels:
            raise ValueError(
                f"left must be [B,{self.channels},H,W], got {tuple(left.shape)}"
            )
        batch, _, height, width = left.shape
        num_scales = len(self.compression_ratios)
        if len(right_feature_pyramid) != num_scales:
            raise ValueError(
                f"expected {num_scales} right scales, "
                f"got {len(right_feature_pyramid)}"
            )
        expected_weights = (batch, num_scales, height, width)
        if tuple(scale_weights.shape) != expected_weights:
            raise ValueError(
                f"scale_weights must be {expected_weights}, "
                f"got {tuple(scale_weights.shape)}"
            )

        all_scale_tokens = []
        for right, ratio in zip(
                right_feature_pyramid, self.compression_ratios):
            scale_tokens = self._match_single_scale(
                left, right, disp, warp_offsets, ratio
            )
            all_scale_tokens.append(scale_tokens)

        # Scale-major order: [s1:k1..kK, s2:k1..kK, s4:k1..kK].
        joint_tokens = torch.cat(all_scale_tokens, dim=1)
        fused = self.match_fuse(joint_tokens.flatten(start_dim=1))
        return fused.reshape(
            batch, height, width, self.output_channels
        ).permute(0, 3, 1, 2)


class MultiRangeFeatureFusionLookup(nn.Module):
    """Fuse sampled MEFA descriptors without constructing matching tokens.

    Each scale samples the same ordered local disparity window. Per-pixel
    ``scale_weights`` reduce the scale axis before the K sampled right
    descriptors are concatenated with the left descriptor and fused by a
    small convolutional network. The module intentionally contains no GWC,
    feature-difference token, hypothesis mixer, attention, or token MLP.
    """

    def __init__(self, compression_ratios=(1, 2, 4), channels=32,
                 samples_per_scale=9, hidden_channels=64,
                 output_channels=128):
        super().__init__()
        if not compression_ratios:
            raise ValueError("compression_ratios must not be empty")
        if channels <= 0 or samples_per_scale <= 0:
            raise ValueError(
                "channels and samples_per_scale must both be positive"
            )
        if hidden_channels <= 0 or output_channels <= 0:
            raise ValueError(
                "hidden_channels and output_channels must both be positive"
            )

        self.compression_ratios = tuple(compression_ratios)
        self.channels = int(channels)
        self.samples_per_scale = int(samples_per_scale)
        self.output_channels = int(output_channels)
        fusion_channels = self.channels * (self.samples_per_scale + 1)
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(
                fusion_channels, hidden_channels, kernel_size=3,
                padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, self.output_channels, kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, left, right_feature_pyramid, disp, warp_offsets,
                scale_weights):
        if left.ndim != 4 or left.shape[1] != self.channels:
            raise ValueError(
                f"left must be [B,{self.channels},H,W], got {tuple(left.shape)}"
            )
        batch, channels, height, width = left.shape
        num_scales = len(self.compression_ratios)
        if len(right_feature_pyramid) != num_scales:
            raise ValueError(
                f"expected {num_scales} right scales, "
                f"got {len(right_feature_pyramid)}"
            )
        expected_weights = (batch, num_scales, height, width)
        if tuple(scale_weights.shape) != expected_weights:
            raise ValueError(
                f"scale_weights must be {expected_weights}, "
                f"got {tuple(scale_weights.shape)}"
            )

        # Reduce scales immediately instead of retaining S*K projected tokens
        # until a joint MLP. Processing one branch at a time also avoids an
        # additional concatenated [B,S*K,C,H,W] sampling tensor.
        selected_right = None
        for scale_index, (right, ratio) in enumerate(zip(
                right_feature_pyramid, self.compression_ratios)):
            sampled_right = sample_right_feature_pyramid(
                (right,),
                disp,
                offsets=warp_offsets.flatten(),
                compression_ratios=(ratio,),
                padding_mode='zeros',
                align_corners=True,
            )
            hypothesis_count = sampled_right.shape[1]
            if hypothesis_count != self.samples_per_scale:
                raise ValueError(
                    f"expected {self.samples_per_scale} hypotheses, "
                    f"got {hypothesis_count} at scale {ratio}"
                )
            if sampled_right.shape[2:] != (channels, height, width):
                raise ValueError(
                    "sampled right feature must be "
                    f"[B,K,{channels},{height},{width}], got "
                    f"{tuple(sampled_right.shape)}"
                )

            scale_weight = scale_weights[
                :, scale_index:scale_index + 1, None
            ].to(sampled_right.dtype)
            weighted_right = sampled_right * scale_weight
            selected_right = (
                weighted_right if selected_right is None
                else selected_right + weighted_right
            )

        fused_input = torch.cat([
            left.to(selected_right.dtype),
            selected_right.flatten(1, 2),
        ], dim=1)
        return self.feature_fuse(fused_input)
