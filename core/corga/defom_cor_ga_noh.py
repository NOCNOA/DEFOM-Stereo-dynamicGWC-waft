"""CorGA ablation that excludes coarse hidden-state fusion from refinement."""

from core.corga.defom_cor_ga import DEFOMStereo as CorGAStereo


class DEFOMStereo(CorGAStereo):
    """Keep CorGA's coarse disparities and gates, but reset refinement hidden state.

    This subclass deliberately defines no parameters. Its state-dict structure is
    therefore identical to :class:`core.corga.defom_cor_ga.DEFOMStereo`, so an existing
    CorGA checkpoint can be loaded strictly and other architectures are unaffected.
    """

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False,
                return_aux=False):
        return super().forward(
            image1,
            image2,
            iters=iters,
            scale_iters=scale_iters,
            test_mode=test_mode,
            return_aux=return_aux,
            coarse_ablation={'use_context_hidden': True},
        )
