"""IGEV++ CorGA variant using only FullCorr samples in coarse GRU motion encoding."""

from core.corga.defom_cor_ga_stereo_igevpp_encoder import DEFOMStereo as BaseStereo


class DEFOMStereo(BaseStereo):
    def __init__(self, args):
        super().__init__(args, coarse_corr_only=True)
