"""Match8 on spatial pyramids, with tokens built at native or dense grids."""
import torch
import torch.nn.functional as F

from core.warp import sample_right_feature_pyramid
from .defom_cor_ga_bins_match8 import DEFOMStereo as Match8, encode_sampled_matches


class DEFOMStereo(Match8):
    """Change refinement evidence resolution, preserving all parameter shapes.

    The bin initializer retains its width-only representation. In native
    mode, each branch pools left/right features, samples and encodes at its
    own grid, then upsamples only 9x8 token channels. Dense mode provides
    the control: spatial right sources sampled/encoded on the quarter grid.
    Both retain the original full-grid static w_s and 248->64->32 fuser.
    """
    MODEL_VARIANT = 'cor_ga_bins_match8_spatial'

    def __init__(self, args):
        if getattr(args, 'refine_right_compression', 'avg') != 'avg':
            raise ValueError('Spatial match8 currently uses average-pool refinement pyramids')
        super().__init__(args)
        self.match_token_grid = getattr(args, 'match_token_grid', 'native')
        if self.match_token_grid not in ('native', 'dense'):
            raise ValueError('match_token_grid must be native or dense')

    def _build_right_feature_pyramids(self, right):
        initial, _ = super()._build_right_feature_pyramids(right)
        refine = (right, F.avg_pool2d(right, 2, 2), F.avg_pool2d(right, 4, 4))
        return initial, refine

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp, scale_weights,
                             global_evidence=None, sample_evidence=None, enhanced=None,
                             rope_positions=None):
        if global_evidence is not None or sample_evidence is not None:
            raise ValueError('Spatial match8 is an isolated token-grid ablation')
        batch, channels, height, width = fmap1_low.shape
        if len(right_feature_pyramid) != 3 or tuple(scale_weights.shape) != (batch, 3, height, width):
            raise ValueError('Spatial match8 requires three source scales and gates')
        encoded_scales = []
        for scale, (feature, ratio) in enumerate(zip(right_feature_pyramid, (1, 2, 4))):
            if self.match_token_grid == 'native':
                left = fmap1_low if ratio == 1 else F.avg_pool2d(fmap1_low, ratio, ratio)
                # d is in quarter-grid units. Native-grid disparity is d/r.
                # Pooling left and right with matching centers removes the
                # center offset from their native coordinate difference.
                native_disp = disp if ratio == 1 else F.avg_pool2d(disp, ratio, ratio) / float(ratio)
                sampled = sample_right_feature_pyramid(
                    (feature,), native_disp, self.warp_offsets,
                    compression_ratios=(1,), compact=True)
            else:
                left = fmap1_low
                sampled = sample_right_feature_pyramid(
                    (feature,), disp, self.warp_offsets, compression_ratios=(ratio,),
                    height_compression_ratios=(ratio,), compact=True)
            sample_count, native_h, native_w = sampled.shape[1], sampled.shape[-2], sampled.shape[-1]
            matches = encode_sampled_matches(left, sampled, self.MATCH_NUM_GROUPS)
            encoded = self.sample_match_encoder(matches.reshape(
                batch * sample_count, channels + self.MATCH_NUM_GROUPS, native_h, native_w))
            encoded = encoded.reshape(batch, sample_count * self.MATCH_ENCODED_CHANNELS, native_h, native_w)
            if (native_h, native_w) != (height, width):
                # align_corners=False follows pooling's half-cell centers.
                encoded = F.interpolate(encoded, size=(height, width), mode='bilinear', align_corners=False)
            # Apply gates after upsampling: the same quarter-grid w_s as match8.
            weight = (3. * scale_weights[:, scale:scale + 1]).to(encoded.dtype)
            encoded_scales.append(encoded * weight)
        fused_input = torch.cat((fmap1_low.to(encoded_scales[0].dtype), *encoded_scales), dim=1)
        fused = self.refine_right_fuse(fused_input)
        return self.refine_rope(fused, freqs_cis=rope_positions)
