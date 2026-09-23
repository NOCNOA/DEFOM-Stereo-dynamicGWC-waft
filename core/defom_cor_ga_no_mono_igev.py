"""Pure-stereo CorGA using an IGEV-style multi-scale ConvGRU update."""

from core.defom_cor_ga_no_mono_head import DEFOMStereo as NoMonoStereo
from core.parallel_igev import parallel_igev_coarse_update
from core.update_igev_stereo import StereoIGEVUpdateBlock


class DEFOMStereo(NoMonoStereo):
    """Replace prompt residual units with prompt-free stereo ConvGRUs."""

    update_block_class = StereoIGEVUpdateBlock

    def _update_coarse_branches(
        self, active, coarse_nets, correlations, disps, inp, warp_features,
        initial_disps, static_features, need_mask,
    ):
        blocks = [self.coarse_update_blocks[index] for index in active]
        nets = [
            [state.float() for state in coarse_nets[index]]
            for index in active
        ]
        return parallel_igev_coarse_update(
            blocks=blocks,
            nets=nets,
            correlations=[value.float() for value in correlations],
            disps=[value.float() for value in disps],
            contexts=[inp.float() for _ in active],
            warp_features=[value.float() for value in warp_features],
            anchor_disps=[initial_disps[index].float() for index in active],
            static_features=[static_features[index] for index in active],
            need_mask=need_mask,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Transfer shared weights from a PRU checkpoint when requested."""
        compatible_state = state_dict.copy()
        legacy_markers = (".stereo_pru.", ".structure_encoder.", ".update.")
        has_legacy_update = any(
            key.startswith(("coarse_update_blocks.", "refine_update_block."))
            and any(marker in key for marker in legacy_markers)
            for key in compatible_state
        )
        if has_legacy_update:
            current_state = self.state_dict()
            for key in tuple(compatible_state):
                if (
                    key.startswith(
                        ("coarse_update_blocks.", "refine_update_block.")
                    )
                    and key not in current_state
                ):
                    compatible_state.pop(key)
            for key, value in current_state.items():
                if (
                    ".context_encoders." in key or ".gru_layers." in key
                ):
                    compatible_state.setdefault(key, value)
        return super().load_state_dict(
            compatible_state, strict=strict, assign=assign
        )
