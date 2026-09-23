"""Horizontal-only sampling primitives for CorGA-fast2."""
from __future__ import annotations

import torch

from core.corr import CorrBlock1D1


class HorizontalCoordinateCache:
    """Cache immutable horizontal pixel-coordinate rows."""

    def __init__(self):
        self._x = {}

    def x(self, width, device, dtype=torch.float32):
        key = (int(width), device.type, device.index, dtype)
        value = self._x.get(key)
        if value is None:
            value = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
            self._x[key] = value
        return value


def horizontal_linear_sample(source, x):
    """Sample [B,C,H,Ws] at horizontal pixel coordinates [B,K,H,Wq]."""
    if source.ndim != 4 or x.ndim != 4:
        raise ValueError(
            f"Expected source [B,C,H,W] and x [B,K,H,W], got {source.shape} and {x.shape}"
        )
    batch, channels, height, source_width = source.shape
    if x.shape[0] != batch or x.shape[2] != height:
        raise ValueError(
            f"Source and coordinate batch/height differ: {source.shape} and {x.shape}"
        )

    # Explicit masks reproduce grid_sample(..., padding_mode='zeros') within
    # the one-pixel interpolation band outside the image.
    x_fp32 = x.float()
    left_unclamped = torch.floor(x_fp32)
    right_unclamped = left_unclamped + 1.0
    right_weight = x_fp32 - left_unclamped
    left_weight = 1.0 - right_weight
    left_valid = (left_unclamped >= 0) & (left_unclamped < source_width)
    right_valid = (right_unclamped >= 0) & (right_unclamped < source_width)
    left_index = left_unclamped.clamp(0, source_width - 1).long()
    right_index = right_unclamped.clamp(0, source_width - 1).long()

    hypotheses, query_width = x.shape[1], x.shape[3]
    expanded_source = source.unsqueeze(1).expand(
        batch, hypotheses, channels, height, source_width
    )
    left_index = left_index.unsqueeze(2).expand(
        batch, hypotheses, channels, height, query_width
    )
    right_index = right_index.unsqueeze(2).expand_as(left_index)
    left = torch.gather(expanded_source, -1, left_index)
    right = torch.gather(expanded_source, -1, right_index)
    left_weight = (left_weight * left_valid).to(source.dtype).unsqueeze(2)
    right_weight = (right_weight * right_valid).to(source.dtype).unsqueeze(2)
    return left * left_weight + right * right_weight


def sample_right_feature_pyramid_horizontal(
    right_features, disp, offsets, coordinate_cache, compression_ratios=(1, 2, 4)
):
    """Horizontal equivalent of sample_right_feature_pyramid."""
    if disp.ndim != 4 or disp.shape[1] != 1:
        raise ValueError(f"disp must be [B,1,H,W], got {tuple(disp.shape)}")
    if len(right_features) != len(compression_ratios):
        raise ValueError("right_features and compression_ratios must have equal length")

    batch, _, query_height, query_width = disp.shape
    disp_fp32 = disp.float()
    offsets = torch.as_tensor(
        offsets, device=disp.device, dtype=torch.float32
    ).flatten()
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one value")
    x_left = coordinate_cache.x(query_width, disp.device, torch.float32)
    sampled_scales = []
    for right_feature, ratio in zip(right_features, compression_ratios):
        if right_feature.shape[0] != batch or right_feature.shape[2] != query_height:
            raise ValueError(
                f"Right feature batch/height must match disp, got "
                f"{right_feature.shape} and {disp.shape}"
            )
        center_offset = (float(ratio) - 1.0) / 2.0
        x_right = (x_left - disp_fp32 - center_offset) / float(ratio)
        x_right = x_right + offsets.view(1, -1, 1, 1)
        with torch.cuda.amp.autocast(enabled=False):
            sampled = horizontal_linear_sample(right_feature.float(), x_right)
        sampled_scales.append(sampled.to(right_feature.dtype))
    return torch.cat(sampled_scales, dim=1)


def lookup_right_features_horizontal(
    fmap1_low,
    right_feature_pyramid,
    disp,
    warp_offsets,
    feature_fuse,
    coordinate_cache,
    rope_galerkin=None,
    scale_weights=None,
    compression_ratios=(1, 2, 4),
    **_ignored,
):
    sampled_right = sample_right_feature_pyramid_horizontal(
        right_feature_pyramid,
        disp,
        warp_offsets.flatten(),
        coordinate_cache,
        compression_ratios=compression_ratios,
    )
    batch, channels, height, width = fmap1_low.shape
    num_scales = len(compression_ratios)
    samples_per_scale = sampled_right.shape[1] // num_scales
    sampled_right = sampled_right.reshape(
        batch, num_scales, samples_per_scale, channels, height, width
    )
    if scale_weights is None:
        if num_scales != 1:
            raise ValueError("scale_weights is required for multiple right-feature scales")
        selected_right = sampled_right[:, 0]
    else:
        expected = (batch, num_scales, height, width)
        if tuple(scale_weights.shape) != expected:
            raise ValueError(
                f"scale_weights must be {expected}, got {tuple(scale_weights.shape)}"
            )
        selected_right = (
            sampled_right
            * scale_weights.to(sampled_right.dtype)[:, :, None, None]
        ).sum(dim=1)
    fused_input = torch.cat(
        [fmap1_low.to(sampled_right.dtype), selected_right.flatten(1, 2)], dim=1
    )
    fused_feature = feature_fuse(fused_input)
    if rope_galerkin is not None:
        fused_feature = rope_galerkin(fused_feature)
    return fused_feature


class HorizontalCorrBlock1D(CorrBlock1D1):
    """FullCorr pyramid with horizontal-only lookup and unchanged layout."""

    def __call__(self, disp, scaling=False):
        if scaling:
            return super().__call__(disp, scaling=True)
        disparity = disp.float().reshape(
            self.batch * self.h1 * self.w1, 1, 1, 1
        )
        offsets = self.dx.reshape(1, 1, 1, -1).float()
        outputs = []
        for level, volume in enumerate(self.volume_pyramid):
            x = (self.coords - disparity) / (2 ** level) + offsets
            sampled = horizontal_linear_sample(volume, x).squeeze(1)
            outputs.append(
                sampled.reshape(self.batch, self.h1, self.w1, -1)
            )
        output = torch.cat(outputs, dim=-1)
        return output.permute(0, 3, 1, 2).contiguous().float()

    def sample_many(self, disparities):
        disparities = tuple(disparities)
        if not disparities:
            raise ValueError("sample_many requires at least one disparity field")
        if self.coordinate_mode != "right":
            raise ValueError(
                "sample_many currently supports right-coordinate volumes only"
            )
        branch_count = len(disparities)
        pixel_count = self.batch * self.h1 * self.w1
        disparity = torch.stack([value.float() for value in disparities])
        disparity = disparity.reshape(
            branch_count, pixel_count, 1, 1
        ).permute(1, 0, 2, 3)
        coords = self.coords.reshape(pixel_count, 1, 1, 1)
        offsets = self.dx.reshape(1, 1, 1, -1).float()
        outputs = []
        for level, volume in enumerate(self.volume_pyramid):
            x = (coords - disparity) / (2 ** level) + offsets
            sampled = horizontal_linear_sample(volume, x)
            channels = sampled.shape[2]
            sampled = sampled.squeeze(3).permute(1, 0, 2, 3)
            sampled = sampled.reshape(
                branch_count,
                self.batch,
                self.h1,
                self.w1,
                channels * offsets.shape[-1],
            ).permute(0, 1, 4, 2, 3)
            outputs.append(sampled)
        return torch.cat(outputs, dim=2).contiguous().float()
