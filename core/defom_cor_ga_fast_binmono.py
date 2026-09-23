"""CorGA-fast with PromptStereo-style bin/monocular initialization fusion."""

import math

import torch
import torch.nn as nn

from core.defom_cor_ga import autocast
from core.defom_cor_ga_fast_bininit import DEFOMStereo as CorGAFastBinInitStereo
from core.submodules import normalize_disparity
from core.warp import sample_right_feature_pyramid


class DEFOMStereo(CorGAFastBinInitStereo):
    """Fuse each stereo-bin estimate with scale-aligned monocular disparity.

    The three branches share one confidence detector. The detector sees the
    left feature and right feature sampled at the raw bin disparity. Its output
    is the per-pixel stereo weight:

        fused = confidence * bin_disp + (1 - confidence) * aligned_mono
    """

    def __init__(self, args):
        super().__init__(args)
        feature_dim = 32
        hidden_dim = int(getattr(args, 'binmono_conf_hidden', 32))
        initial_weight = float(
            getattr(args, 'binmono_init_stereo_weight', 0.05)
        )
        if not 0.0 < initial_weight < 1.0:
            raise ValueError(
                "binmono_init_stereo_weight must be in (0, 1), got "
                f"{initial_weight}"
            )

        self.binmono_confidence = nn.Sequential(
            nn.Conv2d(2 * feature_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Start close to aligned mono. Tiny non-zero output weights keep the
        # initial confidence nearly constant and let both layers train at step 1.
        nn.init.normal_(self.binmono_confidence[-1].weight, std=1e-4)
        nn.init.constant_(
            self.binmono_confidence[-1].bias,
            math.log(initial_weight / (1.0 - initial_weight)),
        )

        if getattr(args, 'freeze_binmono_base', False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.binmono_confidence.parameters():
                parameter.requires_grad = True

    def _initialize_coarse_disparities(
        self, fmap1_low, right_feature_pyramid, mono_idepth
    ):
        bin_disps, initialization_aux = super()._initialize_coarse_disparities(
            fmap1_low, right_feature_pyramid, mono_idepth
        )

        # Mono supplies spatial structure; each stereo branch supplies robust
        # global scale and shift, as in PromptStereo's alignment-and-blend path.
        with autocast(enabled=False):
            normalized_mono, _, _ = normalize_disparity(mono_idepth.float())

        fused_disps = []
        aligned_mono_disps = []
        stereo_confidences = []
        zero_offset = fmap1_low.new_zeros(1)

        for bin_disp, right_feature, ratio in zip(
            bin_disps, right_feature_pyramid, (1, 2, 4)
        ):
            with autocast(enabled=False):
                _, stereo_scale, stereo_shift = normalize_disparity(
                    bin_disp.float()
                )
                aligned_mono = (
                    normalized_mono * stereo_scale[..., None, None]
                    + stereo_shift[..., None, None]
                )
                sampled_right = sample_right_feature_pyramid(
                    (right_feature,),
                    bin_disp.float(),
                    zero_offset.float(),
                    compression_ratios=(ratio,),
                    compact=True,
                )[:, 0]

            with autocast(
                enabled=self.args.mixed_precision, dtype=self.amp_dtype
            ):
                confidence_logit = self.binmono_confidence(torch.cat(
                    (fmap1_low.to(sampled_right.dtype), sampled_right), dim=1
                ))
            stereo_confidence = torch.sigmoid(confidence_logit.float())
            fused_disp = self._bound_disparity(
                stereo_confidence * bin_disp.float()
                + (1.0 - stereo_confidence) * aligned_mono
            )

            fused_disps.append(fused_disp)
            aligned_mono_disps.append(aligned_mono)
            stereo_confidences.append(stereo_confidence)

        initialization_aux.update({
            'raw_bin_disps': torch.cat(bin_disps, dim=1),
            'aligned_mono_disps': torch.cat(aligned_mono_disps, dim=1),
            'binmono_fused_disps': torch.cat(fused_disps, dim=1),
            'binmono_confidence': torch.cat(stereo_confidences, dim=1),
        })
        return fused_disps, initialization_aux
