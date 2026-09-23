"""CorGA with one recurrent layer in each coarse disparity branch."""

from core.defom_cor_ga import DEFOMStereo as CorGAStereo


class DEFOMStereo(CorGAStereo):
    """Keep three-layer refinement while making each coarse branch one-layer."""

    def __init__(self, args):
        super().__init__(args, coarse_gru_layers=1)
