"""Read cached global candidates separately for every local matching sample."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .top8_guidance import CoarseTop8Guidance


class SampleTop8Guidance(CoarseTop8Guidance):
    def __init__(self, feature_dim=32, token_dim=32, query_dim=8, samples_per_scale=9,
                 **kwargs):
        super().__init__(feature_dim=feature_dim, token_dim=token_dim, **kwargs)
        # Only evidence construction is shared with the previous head.
        del self.query, self.gate, self.residual
        self.query_dim = query_dim
        self.key = nn.Conv2d(token_dim, query_dim, 1)
        self.value = nn.Conv2d(token_dim, query_dim, 1)
        self.sample_query = nn.Sequential(
            nn.Conv2d(3 * feature_dim + 4, query_dim, 1), nn.ReLU(),
            nn.Conv2d(query_dim, query_dim, 1))
        self.sample_ffn = nn.Sequential(
            nn.Conv2d(2 * query_dim, query_dim, 1), nn.ReLU(),
            nn.Conv2d(query_dim, query_dim, 1))
        self.sample_fuse = nn.Sequential(
            nn.Conv2d(3 * samples_per_scale * query_dim, feature_dim, 1),
            nn.ReLU(), nn.Conv2d(feature_dim, feature_dim, 3, padding=1))

    def condition_samples(self, evidence, left, sampled, disp, offsets,
                          right_widths, ratios=(1, 2, 4)):
        """Return an independent enhanced feature; leave original samples intact.

        Attention has shape [B,scale*offset,Top8,H,W]. No tensor with
        both candidate axes AND a feature-channel axis is materialized.
        All coordinates and disparities use the uncompressed feature grid.
        """
        b, scales, count, c, h, w = sampled.shape
        k, token_dim = evidence['tokens'].shape[1:3]
        d = self.query_dim
        s = scales * count
        ratios_t = disp.new_tensor(ratios).view(1, scales, 1, 1, 1)
        offsets_t = offsets.to(disp).view(1, 1, count, 1, 1)
        # Positive right-coordinate offsets correspond to smaller disparities.
        local_disps = disp[:, None] - ratios_t * offsets_t
        local_disps = local_disps.expand(b, scales, count, h, w).reshape(b*s, 1, h, w)
        position = self.position(local_disps / self.max_disp)
        right = sampled.reshape(b*s, c, h, w)
        left_pair = left[:, None].expand(b, s, c, h, w).reshape(b*s, c, h, w)
        cosine = F.cosine_similarity(left_pair.float(), right.float(), dim=1)[:, None]
        pair = torch.cat((left_pair, right, left_pair-right,
                          cosine.to(right), position.to(right)), dim=1)
        query = self.sample_query(pair).reshape(b, s, d, h, w)
        relative = (evidence['disparities'] - disp.float()) / self.max_disp
        tokens = evidence['tokens'].reshape(b*k, token_dim, h, w)
        tokens = tokens + self.relative_position(self.position(relative.reshape(b*k, 1, h, w)))
        keys = self.key(tokens).reshape(b, k, d, h, w)
        values = self.value(tokens).reshape(b, k, d, h, w)
        with torch.cuda.amp.autocast(enabled=False):
            scores = torch.einsum('bsdhw,bkdhw->bskhw', query.float(), keys.float()) / math.sqrt(d)
            mask = evidence['valid'][:, None]
            attention = torch.softmax(scores.masked_fill(~mask, -1e4), dim=2) * mask
            attention = attention / attention.sum(2, keepdim=True).clamp_min(1e-6)
            readout = torch.einsum('bskhw,bkdhw->bsdhw', attention, values.float())
        enhanced = self.sample_ffn(torch.cat((query, readout.to(query)), dim=2).reshape(b*s, 2*d, h, w))
        x = torch.arange(w, device=disp.device)[None, None, None, None]
        xr = (x - disp[:, None] - (ratios_t-1)/2) / ratios_t + offsets_t
        widths = disp.new_tensor(right_widths).view(1, scales, 1, 1, 1)
        valid_local = ((xr >= 0) & (xr <= widths-1)).expand(b, scales, count, h, w)
        valid_local = valid_local.reshape(b, s, 1, h, w)
        enhanced = enhanced.reshape(b, s, d, h, w) * valid_local
        feature = self.sample_fuse(enhanced.flatten(1, 2))
        return feature * evidence['valid'].any(1, keepdim=True) * valid_local.any(1)
