"""Compressed bin evidence queried at the actual multiscale warp positions."""

from typing import NamedTuple

import torch
from torch import nn


class WaveletBinEvidence(NamedTuple):
    coarse: torch.Tensor
    detail_values: torch.Tensor
    detail_indices: torch.Tensor
    peak_values: torch.Tensor
    peak_indices: torch.Tensor


class WaveletBinGuidance(nn.Module):
    """Keep low-frequency bin mass and sparse peaks, then condition local samples.

    The 1D Haar transform is along each pixel's disparity-bin axis, never
    along image width. Three 48-bin branches retain 6 coarse coefficients,
    four signed detail coefficients and two exact peaks per branch. Indices
    are stored as uint8 and converted to long only while querying.
    """

    def __init__(self, num_bins=48, bin_width=1.0, ratios=(1, 2, 4),
                 num_details=4, num_peaks=2, feature_dim=32):
        super().__init__()
        if num_bins < 8 or num_bins % 8:
            raise ValueError("wavelet bin guidance requires num_bins divisible by 8")
        if num_bins > 256:
            raise ValueError("uint8 bin indices require num_bins <= 256")
        if not 0 < num_details <= num_bins - num_bins // 8:
            raise ValueError("num_details is outside the Haar detail count")
        if not 0 < num_peaks <= num_bins:
            raise ValueError("num_peaks must be between 1 and num_bins")
        self.num_bins = int(num_bins)
        self.bin_width = float(bin_width)
        self.ratios = tuple(ratios)
        self.num_details = int(num_details)
        self.num_peaks = int(num_peaks)
        self.feature_dim = int(feature_dim)
        self.register_buffer('ratio_tensor', torch.tensor(ratios, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('haar_scale', torch.tensor(2.0 ** -0.5), persistent=False)
        self.register_buffer('coarse_scale', torch.tensor(8.0 ** -0.5),
                             persistent=False)
        self.register_buffer('max_disparity', torch.tensor(
            max(ratios) * (num_bins - 1) * bin_width, dtype=torch.float32),
            persistent=False)
        self.conditioner = None  # built once from the model's sampling radius

    def configure(self, num_offsets):
        """Create the trainable fusion head after the warp radius is known."""
        scales = len(self.ratios)
        summary_channels = (self.num_bins // 8) * scales
        summary_channels += 2 * self.num_peaks * scales + 2 * scales
        sample_channels = scales * int(num_offsets)
        self.conditioner = nn.Sequential(
            nn.Conv2d(2 * sample_channels + summary_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, sample_channels + self.feature_dim, 1),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        return self

    def compress(self, logits, valid):
        """Return compressed evidence for [B,scale,bin,H,W] initializer logits."""
        if logits.ndim != 5 or logits.shape[1:3] != (len(self.ratios), self.num_bins):
            raise ValueError(f"Expected logits [B,{len(self.ratios)},{self.num_bins},H,W]")
        if valid.shape != logits.shape:
            raise ValueError("valid mask must have the same shape as logits")
        probability = torch.softmax(logits.float(), dim=2) * valid.float()
        probability = probability / probability.sum(dim=2, keepdim=True).clamp_min(1e-6)
        approximation = probability
        detail_levels = []
        for _ in range(3):
            even, odd = approximation[:, :, 0::2], approximation[:, :, 1::2]
            detail_levels.append((even - odd) * self.haar_scale)
            approximation = (even + odd) * self.haar_scale
        details = torch.cat(detail_levels, dim=2)
        detail_indices = details.abs().topk(self.num_details, dim=2).indices
        detail_values = details.gather(2, detail_indices)
        peak_values, peak_indices = probability.topk(self.num_peaks, dim=2)
        return WaveletBinEvidence(
            approximation, detail_values, detail_indices.to(torch.uint8),
            peak_values, peak_indices.to(torch.uint8))

    @staticmethod
    def _sparse_detail(evidence, coefficient_index):
        matching = evidence.detail_indices.long().unsqueeze(3) == coefficient_index.unsqueeze(2)
        return (evidence.detail_values.unsqueeze(3) * matching).sum(dim=2)

    def _reconstruct_bins(self, evidence, index):
        """Reconstruct only queried integer bin positions, [B,S,K,H,W]."""
        coarse = evidence.coarse.gather(2, index // 8)
        first = self._sparse_detail(evidence, index // 2)
        second = self._sparse_detail(evidence, self.num_bins // 2 + index // 4)
        third = self._sparse_detail(evidence, 3 * self.num_bins // 4 + index // 8)
        first_sign = 1 - 2 * (index % 2)
        second_sign = 1 - 2 * ((index % 4) // 2)
        third_sign = 1 - 2 * ((index % 8) // 4)
        result = (coarse + third_sign * third) * self.coarse_scale
        result = result + second_sign * second * 0.5
        result = result + first_sign * first * self.haar_scale
        # Preserve narrow modes even if their high-frequency coefficients were
        # omitted by the sparse transform.
        peak_match = evidence.peak_indices.long().unsqueeze(3) == index.unsqueeze(2)
        exact_peak = (evidence.peak_values.unsqueeze(3) * peak_match).sum(dim=2)
        return torch.where(peak_match.any(dim=2), exact_peak, result.clamp_min(0))

    def query(self, evidence, disp, offsets, right_widths):
        """Query the prior at each scale's actual right-feature sample position."""
        if disp.ndim != 4 or disp.shape[1] != 1:
            raise ValueError("disp must be [B,1,H,W]")
        if len(right_widths) != len(self.ratios):
            raise ValueError("right_widths must match compression ratios")
        batch, _, height, width = disp.shape
        ratios = self.ratio_tensor.view(1, -1, 1, 1, 1)
        offsets = offsets.float().flatten().view(1, 1, -1, 1, 1)
        candidate_disp = disp.float().unsqueeze(1) - ratios * offsets
        bin_position = candidate_disp / (ratios * self.bin_width)
        clipped = bin_position.clamp(0, self.num_bins - 1)
        lower = clipped.floor().long().expand(batch, -1, -1, height, width)
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        fractional = clipped - lower.float()
        lower_value = self._reconstruct_bins(evidence, lower)
        upper_value = self._reconstruct_bins(evidence, upper)
        prior = (1 - fractional) * lower_value + fractional * upper_value
        x_left = torch.arange(width, device=disp.device, dtype=torch.float32).view(1, 1, 1, 1, width)
        right_x = (x_left - candidate_disp - (ratios - 1) * 0.5) / ratios
        source_max = torch.as_tensor(right_widths, device=disp.device,
                                     dtype=torch.float32).view(1, -1, 1, 1, 1) - 1
        valid = ((bin_position >= 0) & (bin_position <= self.num_bins - 1)
                 & (right_x >= 0) & (right_x <= source_max))
        return prior.clamp(0, 1) * valid.float()

    def summary(self, evidence, disp, scale_weights):
        """Expose all coarse ranges and distant peak directions to the update."""
        batch, scales, _, height, width = evidence.coarse.shape
        ratios = self.ratio_tensor.view(1, scales, 1, 1, 1)
        peak_disparities = evidence.peak_indices.float() * ratios * self.bin_width
        relative_peaks = ((peak_disparities - disp.float().unsqueeze(1))
                          / self.max_disparity.clamp_min(1)).clamp(-1, 1)
        relative_peaks = relative_peaks * (evidence.peak_values > 0).float()
        coarse_mass = evidence.coarse * (8.0 ** 0.5)
        occupied = coarse_mass.sum(dim=2)
        return torch.cat([
            coarse_mass.reshape(batch, -1, height, width),
            relative_peaks.reshape(batch, -1, height, width),
            evidence.peak_values.reshape(batch, -1, height, width),
            occupied,
            scale_weights.float(),
        ], dim=1)

    def condition_samples(self, evidence, left, sampled, disp, offsets,
                          scale_weights, right_widths):
        """Fuse global bin evidence with local left/right match similarity."""
        if self.conditioner is None:
            raise RuntimeError("configure(num_offsets) must be called before use")
        batch, scales, samples, channels, height, width = sampled.shape
        if scales != len(self.ratios) or channels != left.shape[1]:
            raise ValueError("sampled feature scales/channels do not match guidance")
        if scale_weights is None:
            raise ValueError("scale_weights are required for multiscale guidance")
        sampled_float = sampled.float()
        local_dot = torch.einsum('bchw,bskchw->bskhw', left.float(), sampled_float)
        left_norm = torch.linalg.vector_norm(left.float(), dim=1).clamp_min(1e-6)
        right_norm = torch.linalg.vector_norm(sampled_float, dim=3).clamp_min(1e-6)
        local_similarity = local_dot / (left_norm[:, None, None] * right_norm)
        prior = self.query(evidence, disp, offsets, right_widths)
        global_summary = self.summary(evidence, disp, scale_weights)
        inputs = torch.cat([
            local_similarity.reshape(batch, scales * samples, height, width),
            torch.log1p(self.num_bins * prior).reshape(batch, scales * samples, height, width),
            global_summary,
        ], dim=1)
        conditioned = self.conditioner(inputs)
        gains = 1 + 0.25 * torch.tanh(conditioned[:, :scales * samples])
        gains = gains.reshape(batch, scales, samples, 1, height, width)
        residual = 0.25 * torch.tanh(conditioned[:, scales * samples:])
        return sampled * gains.to(sampled.dtype), residual
