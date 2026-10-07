"""Bin-initialized stereo refinement with compressed global matching evidence."""

import torch

from .defom_cor_ga_bins import DEFOMStereo as BinStereo
from .wavelet_guidance import WaveletBinGuidance


class DEFOMStereo(BinStereo):
    """Keep the base bin model and condition its local warp features per step."""

    def __init__(self, args):
        super().__init__(args)
        # The extra head must not shift the DataLoader's RNG stream relative
        # to the baseline when both runs start from the same seed.
        cpu_rng_state = torch.get_rng_state()
        try:
            self.wavelet_guidance = WaveletBinGuidance(
                num_bins=self.bin_initializer.num_bins,
                bin_width=self.bin_initializer.bin_width,
                ratios=self.bin_initializer.compression_ratios,
                feature_dim=32,
            ).configure(self.warp_offsets.numel())
        finally:
            torch.set_rng_state(cpu_rng_state)
