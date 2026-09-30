"""Lightweight CorGA with shallow, short coarse refinement branches."""

from core.corga.defom_cor_ga import DEFOMStereo as CorGAStereo


class DEFOMStereo(CorGAStereo):
    """Keep all three coarse hypotheses but use one recurrent level per branch.

    The intended configuration uses two coarse iterations per hypothesis. The
    iteration count remains a forward argument so training and evaluation can
    use the same model class without changing the original CorGA behavior.
    Refinement still uses ``args.n_gru_layers`` (normally three levels).
    """

    def __init__(self, args):
        super().__init__(args, coarse_gru_layers=1)

    def forward(self, image1, image2, iters=12, scale_iters=2,
                test_mode=False, return_aux=False, coarse_ablation=None):
        return super().forward(
            image1,
            image2,
            iters=iters,
            scale_iters=scale_iters,
            test_mode=test_mode,
            return_aux=return_aux,
            coarse_ablation=coarse_ablation,
        )
