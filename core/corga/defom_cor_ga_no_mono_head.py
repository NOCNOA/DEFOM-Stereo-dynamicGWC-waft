"""CorGA-fast3 without the monocular inverse-depth extraction head."""

import torch

from core.corga.defom_cor_ga import autocast
from core.bin.defom_cor_ga_bins2 import MultiRangeBinInitializer
from core.corga.defom_cor_ga_fast3 import DEFOMStereo as CorGAFast3
from core.submodules import ParallelRightWidthCompressor


class DEFOMStereo(CorGAFast3):
    """Use three stereo bin-matching hypotheses instead of monocular depth.

    The DINOv2 backbone and DPT stereo feature decoder are retained. The DPT
    monocular depth head is not constructed, so it consumes neither parameters
    nor forward compute. Initial disparities are estimated independently over
    the 1x/2x/4x right-feature ranges without a monocular refiner input.
    """

    use_mono_head = False
    _legacy_mono_prefix = "defomencoder.depth_anything.depth_head."
    _bin_prefixes = ("bin_right_width_compressor.", "bin_initializer.")

    def __init__(self, args):
        super().__init__(args)
        low_feature_dim = 32
        compression = getattr(args, "init_right_compression", "avg")
        if compression not in ("avg", "conv"):
            raise ValueError(
                "init_right_compression must be 'avg' or 'conv', got "
                f"{compression}"
            )
        self.bin_right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim, low_feature_dim, mode=compression
        )
        self.bin_initializer = MultiRangeBinInitializer(
            feature_dim=low_feature_dim,
            num_bins=getattr(args, "num_init_bins", 48),
            max_offset=getattr(args, "init_max_offset", 48.0),
            corr_scale=getattr(args, "init_corr_scale", 10.0),
            use_mono=False,
        )

        if getattr(args, "freeze_bininit_base", False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (
                self.bin_right_width_compressor,
                self.bin_initializer,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def _initialize_coarse_disparities(
        self, fmap1_low, right_feature_pyramid, mono_idepth
    ):
        if mono_idepth is not None:
            raise RuntimeError(
                "cor_ga_no_mono_head must not receive monocular inverse depth"
            )

        fmap2_low = right_feature_pyramid[0]
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            bin_half, bin_quarter = self.bin_right_width_compressor(fmap2_low)
        with autocast(enabled=False):
            init_disps, init_logits, init_confidence, init_valid = (
                self.bin_initializer(
                    fmap1_low,
                    (fmap2_low, bin_half, bin_quarter),
                    mono_idepth=None,
                )
            )
        init_disps = [self._bound_disparity(disp) for disp in init_disps]
        return init_disps, {
            "waft_init_disps": torch.cat(init_disps, dim=1),
            "init_logits": init_logits,
            "init_confidence": init_confidence,
            "init_valid": init_valid,
            "init_bins": self.bin_initializer.bin_disparities,
        }

    def _augment_initialization_aux(
        self, aux, initialization_aux, coarse_disps, hypothesis_logits
    ):
        aux.update(initialization_aux)
        aux["init_disps"] = torch.cat(coarse_disps, dim=1)
        aux["w_h_logits"] = hypothesis_logits

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept legacy Fast3 and mono-refined bin-initializer weights."""
        compatible_state = state_dict.copy()
        for key in tuple(compatible_state):
            if key.startswith(self._legacy_mono_prefix):
                compatible_state.pop(key)

        current_state = self.state_dict()
        has_bin_initializer = any(
            key.startswith("bin_initializer.") for key in compatible_state
        )
        if not has_bin_initializer:
            # A legacy Fast3 checkpoint has no bin initializer. Preserve strict
            # checking for the shared model while retaining the new layers'
            # deterministic initialization.
            for key, value in current_state.items():
                if key.startswith(self._bin_prefixes):
                    compatible_state[key] = value
        else:
            # A cor_ga_fast_bininit checkpoint has one extra input channel in
            # each first refiner convolution for mono_idepth. Drop that channel
            # while retaining its trained stereo-logit weights.
            for key, value in tuple(compatible_state.items()):
                target = current_state.get(key)
                if (
                    target is not None
                    and value.ndim == 4
                    and target.ndim == 4
                    and value.shape[0] == target.shape[0]
                    and value.shape[1] == target.shape[1] + 1
                    and value.shape[2:] == target.shape[2:]
                    and key.startswith("bin_initializer.logit_refiners.")
                ):
                    compatible_state[key] = value[:, :target.shape[1]].contiguous()
        return super().load_state_dict(
            compatible_state, strict=strict, assign=assign
        )
