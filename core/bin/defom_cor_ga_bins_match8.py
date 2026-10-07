"""Bin initialization with compact, explicitly encoded local match samples."""

import torch
import torch.nn as nn

from core.warp import sample_right_feature_pyramid
from .defom_cor_ga_bins import DEFOMStereo as BinStereo


def encode_sampled_matches(left, right, num_groups=4, eps=1e-4):
    """Encode one scale as [right-left, group cosine], [B,K,C+G,H,W].

    Match the encoding used by the user's PIVNO gated model. Clamp the
    squared denominator before sqrt so zero/out-of-bounds features have
    finite gradients. Work one scale at a time to bound temporary buffers.
    """
    batch, samples, channels, height, width = right.shape
    if tuple(left.shape) != (batch, channels, height, width):
        raise ValueError('Left and sampled-right feature shapes do not match')
    if channels % num_groups:
        raise ValueError('Feature channels must be divisible by num_groups')
    residual = right - left.to(right.dtype)[:, None]
    with torch.cuda.amp.autocast(enabled=False):
        left_group = left.float().reshape(batch, num_groups, channels // num_groups, height, width)
        right_group = right.float().reshape(batch, samples, num_groups, channels // num_groups, height, width)
        numerator = (left_group[:, None] * right_group).sum(dim=3)
        squared_denominator = (left_group.square().sum(dim=2)[:, None]
                               * right_group.square().sum(dim=3))
        valid = squared_denominator > eps * eps
        cosine = numerator / squared_denominator.clamp_min(eps * eps).sqrt()
        cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    return torch.cat((residual, cosine.to(residual.dtype)), dim=2)


class DEFOMStereo(BinStereo):
    """Preserve scale identity before local-match fusion; use static bin gates.

    This first ablation changes only local-match encoding/fusion. It retains
    the initializer, static w_s, FullCorr, RoPE, FP32 recurrent update, no
    delta clamp, and existing execution optimizations of the base bin model.
    """

    MODEL_VARIANT = 'cor_ga_bins_match8'
    MATCH_NUM_GROUPS = 4
    MATCH_ENCODED_CHANNELS = 8

    def __init__(self, args):
        if getattr(args, 'enable_top8_guidance', False):
            raise ValueError('match8 is an isolated encoding ablation; disable Top8 guidance')
        super().__init__(args)
        # Keep the baseline's post-construction RNG for matched data ordering.
        rng_state = torch.get_rng_state()
        try:
            self.sample_match_encoder = nn.Sequential(
                nn.Conv2d(32 + self.MATCH_NUM_GROUPS, self.MATCH_ENCODED_CHANNELS, 1, bias=False),
                nn.GroupNorm(4, self.MATCH_ENCODED_CHANNELS),
                nn.GELU(),
            )
            fusion_channels = 32 + 3 * self.warp_offsets.numel() * self.MATCH_ENCODED_CHANNELS
            # Preserve the baseline's 64->32 layer and its checkpoint name.
            self.refine_right_fuse[0] = nn.Conv2d(fusion_channels, 64, 3, padding=1, bias=False)
        finally:
            torch.set_rng_state(rng_state)

    def initialize_from_bin(self, state_dict):
        """Load a base-bin checkpoint, replacing only the changed fusion input.

        Do not silently accept unrelated missing/extra parameters. Converted
        match8 checkpoints subsequently load with normal strict=True.
        """
        expected = set(self.state_dict()) - {
            'refine_right_fuse.0.weight', *self.sample_match_encoder.state_dict(prefix='sample_match_encoder.').keys()}
        actual = set(state_dict) - {'refine_right_fuse.0.weight'}
        if actual != expected:
            raise ValueError(f'Not a compatible base-bin checkpoint: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
        old_shape = (64, 32 * (self.warp_offsets.numel() + 1), 3, 3)
        if tuple(state_dict['refine_right_fuse.0.weight'].shape) != old_shape:
            raise ValueError('Base-bin fusion weight has an unexpected shape')
        filtered = {name: value for name, value in state_dict.items()
                    if name != 'refine_right_fuse.0.weight'}
        result = self.load_state_dict(filtered, strict=False)
        return dict(reinitialized=list(result.missing_keys), unexpected=list(result.unexpected_keys))

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp, scale_weights,
                             global_evidence=None, sample_evidence=None, enhanced=None,
                             rope_positions=None):
        if global_evidence is not None or sample_evidence is not None:
            raise ValueError('match8 does not combine global conditioning with this ablation')
        batch, channels, height, width = fmap1_low.shape
        if len(right_feature_pyramid) != 3 or tuple(scale_weights.shape) != (batch, 3, height, width):
            raise ValueError('match8 requires three right scales and three static scale weights')
        encoded_scales = []
        for scale, (feature, ratio) in enumerate(zip(right_feature_pyramid, (1, 2, 4))):
            sampled = sample_right_feature_pyramid(
                (feature,), disp, self.warp_offsets, compression_ratios=(ratio,),
                padding_mode='zeros', align_corners=True, compact=True)
            samples = sampled.shape[1]
            matches = encode_sampled_matches(fmap1_low, sampled, self.MATCH_NUM_GROUPS)
            encoded = self.sample_match_encoder(matches.reshape(batch * samples, channels + self.MATCH_NUM_GROUPS, height, width))
            encoded = encoded.reshape(batch, samples, self.MATCH_ENCODED_CHANNELS, height, width)
            # Same amplitude convention as PIVNO: uniform gates have factor 1.
            weight = (3. * scale_weights[:, scale]).to(encoded.dtype)
            encoded_scales.append((encoded * weight[:, None, None]).flatten(1, 2))
        fused_input = torch.cat((fmap1_low.to(encoded_scales[0].dtype), *encoded_scales), dim=1)
        fused = self.refine_right_fuse(fused_input)
        return self.refine_rope(fused, freqs_cis=rope_positions)
