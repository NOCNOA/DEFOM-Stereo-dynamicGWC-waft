"""Per-pixel coarse-bin evidence, read by local warp features each iteration."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .peak_selection import distinct_peak_indices


class CoarseTop8Guidance(nn.Module):
    def __init__(self, feature_dim=32, token_dim=32, topk=8,
                 min_separation_px=8., pixel_factor=4., max_disp=192.):
        super().__init__()
        self.topk = topk
        self.min_separation_px = min_separation_px
        self.pixel_factor = pixel_factor
        self.max_disp = max_disp  # All internal disparities are on the 1/4 grid.
        self.token_encoder = nn.Sequential(
            nn.Conv2d(3 * feature_dim + 5, token_dim, 1), nn.ReLU(),
            nn.Conv2d(token_dim, token_dim, 1))
        self.query = nn.Conv2d(2 * feature_dim + 1, token_dim, 1)
        self.relative_position = nn.Conv2d(3, token_dim, 1)
        self.key = nn.Conv2d(token_dim, token_dim, 1)
        self.value = nn.Conv2d(token_dim, token_dim, 1)
        self.gate = nn.Conv2d(feature_dim + token_dim, feature_dim, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(feature_dim + token_dim, feature_dim, 1), nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def position(value):
        return torch.cat((value, torch.sin(math.pi * value),
                          torch.cos(math.pi * value)), dim=1)

    def build(self, left, right, logits, valid, bins, ratio=4):
        """Cache tokens from one branch. Bins/coordinates use 1/4 pixel units.

        Discrete peak identities are detached. Selected scores and image
        features retain gradients. No dense token tensor over all bins is built.
        """
        b, c, h, w = left.shape
        indices, mask = distinct_peak_indices(
            logits, valid, bins[None].expand(b, -1) * self.pixel_factor,
            max_peaks=self.topk, min_separation_px=self.min_separation_px)
        disparities = bins[indices].detach()
        x = torch.arange(w, device=left.device).float()[None, None, None]
        y = torch.arange(h, device=left.device).float()[None, None, :, None]
        # Match the initializer's average-pooling center convention exactly.
        xr = (x - disparities - (ratio - 1.) / 2.) / ratio
        mask = mask & (xr >= 0) & (xr <= right.shape[-1] - 1)
        grid = torch.stack((2 * xr / max(right.shape[-1] - 1, 1) - 1,
                            (2 * y / max(h - 1, 1) - 1).expand_as(xr)), -1)
        sampled = F.grid_sample(right.float(), grid.reshape(b, self.topk * h, w, 2),
                                align_corners=True, padding_mode='zeros')
        sampled = sampled.reshape(b, c, self.topk, h, w).permute(0, 2, 1, 3, 4)
        left_pair = left.float()[:, None].expand_as(sampled)
        cosine = F.cosine_similarity(left_pair, sampled, dim=2, eps=1e-6)[:, :, None]
        usable = valid & torch.isfinite(logits)
        logprob = F.log_softmax(logits.float().masked_fill(~usable, -1e4), dim=1)
        scores = logprob.gather(1, indices).masked_fill(~mask, 0)[:, :, None]
        pos = self.position(disparities.reshape(b * self.topk, 1, h, w) / self.max_disp)
        pair = torch.cat((left_pair, sampled, left_pair-sampled, cosine, scores), dim=2)
        pair = torch.cat((pair.reshape(b * self.topk, 3*c + 2, h, w), pos), dim=1)
        tokens = self.token_encoder(pair).reshape(b, self.topk, -1, h, w)
        tokens = tokens.masked_fill(~mask[:, :, None], 0)
        return dict(tokens=tokens, disparities=disparities, valid=mask)

    def forward(self, evidence, left, local, disp):
        b, k, d, h, w = evidence['tokens'].shape
        relative = (evidence['disparities'] - disp.float()) / self.max_disp
        position = self.relative_position(self.position(relative.reshape(b*k, 1, h, w)))
        tokens = evidence['tokens'].reshape(b*k, d, h, w) + position
        keys = self.key(tokens).reshape(b, k, d, h, w)
        values = self.value(tokens).reshape(b, k, d, h, w)
        query = self.query(torch.cat((left.float(), local.float(), disp.float()/self.max_disp), 1))
        scores = (query[:, None] * keys).sum(2) / math.sqrt(d)
        mask = evidence['valid']
        attention = torch.softmax(scores.masked_fill(~mask, -1e4), dim=1) * mask
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
        global_feature = (attention[:, :, None] * values).sum(1)
        fused = torch.cat((local.float(), global_feature), 1)
        residual = torch.sigmoid(self.gate(fused)) * self.residual(fused)
        residual = residual * mask.any(1, keepdim=True)
        return local.float() + residual
