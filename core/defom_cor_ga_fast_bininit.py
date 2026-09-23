"""CorGA-fast with WAFT/bin coarse-disparity initialization only."""

import torch

from core.defom_cor_ga import autocast
from core.defom_cor_ga_fast import DEFOMStereo as CorGAFastStereo
from core.defom_cor_ga_bins2 import MultiRangeBinInitializer
from core.submodules import ParallelRightWidthCompressor


class DEFOMStereo(CorGAFastStereo):
    """Replace only fixed monocular scales with three bin estimates."""

    def __init__(self, args):
        super().__init__(args)
        low_feature_dim = 32
        compression = getattr(args, 'init_right_compression', 'avg')
        if compression not in ('avg', 'conv'):
            raise ValueError(
                "init_right_compression must be 'avg' or 'conv', got "
                f"{compression}"
            )
        self.bin_right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim, low_feature_dim, mode=compression
        )
        self.bin_initializer = MultiRangeBinInitializer(
            feature_dim=low_feature_dim,
            num_bins=getattr(args, 'num_init_bins', 48),
            max_offset=getattr(args, 'init_max_offset', 48.0),
            corr_scale=getattr(args, 'init_corr_scale', 10.0),
        )
        if getattr(args, 'freeze_bininit_base', False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (self.bin_right_width_compressor, self.bin_initializer):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def _initialize_coarse_disparities(
        self, fmap1_low, right_feature_pyramid, mono_idepth
    ):
        # Keep an independent compressor so the inherited PRU pyramid is exact.
        fmap2_low = right_feature_pyramid[0]
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            bin_half, bin_quarter = self.bin_right_width_compressor(fmap2_low)
        with autocast(enabled=False):
            init_disps, init_logits, init_confidence, init_valid = (
                self.bin_initializer(
                    fmap1_low,
                    (fmap2_low, bin_half, bin_quarter),
                    mono_idepth,
                )
            )
        return init_disps, {
            'waft_init_disps': torch.cat(init_disps, dim=1),
            'init_logits': init_logits,
            'init_confidence': init_confidence,
            'init_valid': init_valid,
            'init_bins': self.bin_initializer.bin_disparities,
        }

    def _augment_initialization_aux(
        self, aux, initialization_aux, coarse_disps, hypothesis_logits
    ):
        aux.update(initialization_aux)
        # The inherited gate fuses post-coarse hypotheses, not raw bin means.
        aux['init_disps'] = torch.cat(coarse_disps, dim=1)
        aux['w_h_logits'] = hypothesis_logits
