"""Local correspondence tokens for disparity-conditioned right-feature lookup.

Sampled right candidates are weighted and summed across scales before matching.
The fused window keeps its ordered candidate axis through token projection and
residual 1D aggregation before reduction to a spatial feature map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.warp import sample_right_feature_pyramid
from core.horizontal_sampler import HorizontalCoordinateCache, sample_right_feature_pyramid_horizontal


class CandidateResidualBlock(nn.Module):
    """Compare neighboring candidates independently at each image location."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, tokens, valid):
        # Mask intermediate activations too: an invalid candidate must not
        # relay information between otherwise separated valid candidates.
        update = F.gelu(self.conv1(tokens)) * valid
        update = self.conv2(update) * valid
        return (tokens + update) * valid


class MatchingTokenEncoder(nn.Module):
    """Encode sampled correspondence evidence as a low-channel warp feature.

    Evidence: group-wise cosine similarity, normalized signed L-R difference,
    and optional scale identity. Geometric validity only masks invalid tokens;
    neither the physical disparity offset nor validity is an input channel.
    Right candidates are weighted and summed before normalization and matching.
    Projection, candidate aggregation, and window reduction run once on the
    fused window. Its map is concatenated with the left feature for spatial fusion.
    """

    def __init__(self, channels=32, groups=4, token_dim=32, radius=4,
                 compression_ratios=(1, 2, 4), aggregation_layers=1,
                 scale_encoding=False, checkpoint_scales=True, output_channels=None):
        super().__init__()
        if channels < 1 or groups < 1 or channels % groups:
            raise ValueError('channels must be positive and divisible by groups')
        if token_dim < 1 or radius < 0 or aggregation_layers < 0:
            raise ValueError('token_dim must be positive; radius/layers must be nonnegative')
        if not compression_ratios or any(r < 1 for r in compression_ratios):
            raise ValueError('compression_ratios must contain positive ratios')
        output_channels = channels if output_channels is None else int(output_channels)
        if output_channels < 1:
            raise ValueError('output_channels must be positive')
        self.channels = channels
        self.output_channels = output_channels
        self.groups = groups
        self.token_dim = token_dim
        self.compression_ratios = tuple(compression_ratios)
        self.scale_encoding = scale_encoding
        self.checkpoint_scales = checkpoint_scales
        self.samples_per_scale = 2 * radius + 1
        self.register_buffer('offsets', torch.arange(-radius, radius + 1).float(),
                             persistent=False)
        evidence_dim = groups + channels + int(scale_encoding)
        # Keep raw evidence when its width already matches the candidate tokens.
        # The candidate convolutions learn how to combine cosine/difference channels.
        self.token_projection = (nn.Identity() if evidence_dim == token_dim else nn.Sequential(
            nn.Conv2d(evidence_dim, token_dim, 1), nn.GELU(),
            nn.Conv2d(token_dim, token_dim, 1),
        ))
        self.candidate_blocks = nn.ModuleList([
            CandidateResidualBlock(token_dim) for _ in range(aggregation_layers)
        ])
        self.window_projection = nn.Sequential(
            nn.Conv2d(self.samples_per_scale * token_dim, token_dim, 1, bias=False),
            nn.GELU(),
        )
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(channels + token_dim, output_channels, 3,
                      padding=1, bias=False), nn.ReLU(inplace=True),
        )

    def _candidate_validity(self, disp, valid, ratio):
        disparity_offsets = -float(ratio) * self.offsets.float()
        candidate_disp = disp.float()[:, None] + disparity_offsets.view(
            1, self.samples_per_scale, 1, 1, 1)
        return valid & (candidate_disp >= 0)

    def _build_evidence(self, left, right, disp, valid, ratio, scale_feature=None):
        """Keep normalization and matching statistics in FP32 under AMP."""
        batch, samples, channels, height, width = right.shape
        with torch.autocast(device_type=left.device.type, enabled=False):
            left_fp32, right_fp32 = left.float(), right.float()
            if ratio is not None:
                valid = self._candidate_validity(disp, valid, ratio)
            mask = valid.float()
            left_grouped = left_fp32.reshape(
                batch, self.groups, channels // self.groups, height, width)[:, None]
            right_grouped = right_fp32.reshape(
                batch, samples, self.groups, channels // self.groups, height, width)
            corr = (F.normalize(left_grouped, dim=3, eps=1e-6)
                    * F.normalize(right_grouped, dim=3, eps=1e-6)).sum(dim=3)
            difference = (F.normalize(left_fp32, dim=1, eps=1e-6)[:, None]
                          - F.normalize(right_fp32, dim=2, eps=1e-6))
            evidence = [corr * mask, difference * mask]
            if self.scale_encoding:
                if scale_feature is None:
                    scale = left_fp32.new_tensor(float(ratio)).log2() / 2.0
                    scale_feature = scale.expand(batch, samples, 1, height, width)
                evidence.append(scale_feature.float() * mask)
            evidence = torch.cat(evidence, dim=2)
        return evidence.to(left.dtype), valid

    def _sample_right_features(self, right, disp, ratio):
        # Process scales sequentially and sample without replicating the entire
        # right feature K times. This bounds the FP32 sampling working set.
        return sample_right_feature_pyramid(
            (right,), disp, self.offsets, compression_ratios=(ratio,),
            padding_mode='zeros', align_corners=True, compact=True,
            return_valid=True,
        )

    def _encode_scale(self, left, right, disp, ratio):
        sampled, valid = self._sample_right_features(right, disp, ratio)
        return self._encode_candidates(left, sampled, disp, valid, ratio)

    def _encode_candidates(self, left, sampled, disp, valid, ratio=None, scale_feature=None):
        evidence, valid = self._build_evidence(
            left, sampled, disp, valid, ratio, scale_feature)
        batch, samples, _, height, width = evidence.shape
        tokens = self.token_projection(evidence.flatten(0, 1)).reshape(
            batch, samples, self.token_dim, height, width)
        tokens = tokens * valid.to(tokens.dtype)
        if self.candidate_blocks:
            # [B,K,C,H,W] -> [B*H*W,C,K], never mix different image locations.
            sequence = tokens.permute(0, 3, 4, 2, 1).reshape(
                batch * height * width, self.token_dim, samples)
            sequence_mask = valid.permute(0, 3, 4, 2, 1).reshape(
                batch * height * width, 1, samples).to(sequence.dtype)
            for block in self.candidate_blocks:
                sequence = block(sequence, sequence_mask)
            tokens = sequence.reshape(batch, height, width, self.token_dim, samples)
            tokens = tokens.permute(0, 4, 3, 1, 2).contiguous()
        return self.window_projection(tokens.flatten(1, 2))

    def _encode_pyramid(self, left, disp, scale_weights, *right_features):
        mixed_right, mixed_valid, mixed_scale = None, None, None
        for index, (right, ratio) in enumerate(zip(right_features, self.compression_ratios)):
            sampled, valid = self._sample_right_features(right, disp, ratio)
            valid = self._candidate_validity(disp, valid, ratio)
            weight = (1.0 if scale_weights is None else
                      scale_weights[:, index:index + 1, None].float())
            # Keep the original scale weights; missing candidates contribute
            # zero, without renormalizing weights over the remaining scales.
            contribution = sampled.float() * valid * weight
            active = valid if scale_weights is None else valid & (weight != 0)
            mixed_right = contribution if mixed_right is None else mixed_right + contribution
            mixed_valid = active if mixed_valid is None else mixed_valid | active
            if self.scale_encoding:
                scale = left.new_tensor(float(ratio), dtype=torch.float32).log2() / 2.0
                contribution_scale = valid.float() * weight * scale
                mixed_scale = (contribution_scale if mixed_scale is None else
                               mixed_scale + contribution_scale)
        return self._encode_candidates(
            left, mixed_right, disp, mixed_valid, scale_feature=mixed_scale)

    def forward(self, left, right_feature_pyramid, disp, scale_weights=None):
        if left.ndim != 4 or left.shape[1] != self.channels:
            raise ValueError(f'left must be [B,{self.channels},H,W]')
        batch, _, height, width = left.shape
        if tuple(disp.shape) != (batch, 1, height, width):
            raise ValueError('disp must match the left feature spatial dimensions')
        if len(right_feature_pyramid) != len(self.compression_ratios):
            raise ValueError('right pyramid must match compression_ratios')
        num_scales = len(self.compression_ratios)
        if scale_weights is None and num_scales != 1:
            raise ValueError('scale_weights is required for multiple scales')
        if scale_weights is not None and tuple(scale_weights.shape) != (batch, num_scales, height, width):
            raise ValueError('scale_weights must be [B,num_scales,H,W]')
        for right in right_feature_pyramid:
            if right.ndim != 4 or right.shape[:3] != (batch, self.channels, height):
                raise ValueError('right features must have matching batch, channels and height')
        # Recompute sampling, weighted fusion and the single matching encoding
        # together, so per-iteration candidate activations need not be retained.
        if self.checkpoint_scales and self.training and torch.is_grad_enabled():
            feature = checkpoint(self._encode_pyramid, left, disp, scale_weights,
                                 *right_feature_pyramid, use_reentrant=False)
        else:
            feature = self._encode_pyramid(left, disp, scale_weights, *right_feature_pyramid)
        fused_input = torch.cat([left.to(feature.dtype), feature], dim=1)
        return self.spatial_fuse(fused_input)


class HorizontalMatchingTokenEncoder(MatchingTokenEncoder):
    """Matching tokens with fast2 sampling and identical parameter names."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coordinate_cache = HorizontalCoordinateCache()

    def _sample_right_features(self, right, disp, ratio):
        return sample_right_feature_pyramid_horizontal(
            (right,), disp, self.offsets, self._coordinate_cache,
            compression_ratios=(ratio,), return_valid=True,
        )
