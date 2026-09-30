"""CorGA-fast with horizontal-only correlation/right-feature sampling."""
from core.corr import CorrBlock1D1
from core.corga.defom_cor_ga_fast import DEFOMStereo as CorGAFast
from core.horizontal_sampler import (
    HorizontalCoordinateCache,
    HorizontalCorrBlock1D,
    lookup_right_features_horizontal,
)
from core.utils.utils import coords_grid


class DEFOMStereo(CorGAFast):
    """Execution-only optimization; state_dict is identical to CorGA-fast."""

    def __init__(self, args):
        super().__init__(args)
        self._horizontal_coordinate_cache = HorizontalCoordinateCache()

    def _lookup_right_features(self, *args, **kwargs):
        return lookup_right_features_horizontal(
            *args,
            coordinate_cache=self._horizontal_coordinate_cache,
            **kwargs,
        )

    def _build_full_corr(self, fmap1, fmap2):
        batch, _, height, width = fmap1.shape
        coords = coords_grid(batch, height, width)[:, :1].to(fmap1.device)
        volume = CorrBlock1D1.corr(fmap1.float(), fmap2.float())
        return HorizontalCorrBlock1D(
            volume,
            coords,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
            coordinate_mode="right",
            volume_layout="full_corr",
        )
