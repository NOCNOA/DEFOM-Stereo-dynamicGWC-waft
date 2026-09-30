"""CorGA-fast with three independent coarse PRUs executed in grouped convs."""
from core.corga.defom_cor_ga_fast import DEFOMStereo as CorGAFast
from core.parallel_pru import parallel_coarse_update


class DEFOMStereo(CorGAFast):
    """Preserve all parameters while vectorizing active coarse branches."""

    def _update_coarse_branches(
        self,
        active,
        coarse_nets,
        correlations,
        disps,
        inp,
        warp_features,
        initial_disps,
        static_features,
        need_mask,
    ):
        blocks = [self.coarse_update_blocks[index] for index in active]
        nets = [
            [state.float() for state in coarse_nets[index]]
            for index in active
        ]
        return parallel_coarse_update(
            blocks=blocks,
            nets=nets,
            correlations=[value.float() for value in correlations],
            disps=[value.float() for value in disps],
            contexts=[inp.float() for _ in active],
            warp_features=[value.float() for value in warp_features],
            mono_disps=[initial_disps[index].float() for index in active],
            static_features=[static_features[index] for index in active],
            need_mask=need_mask,
        )
