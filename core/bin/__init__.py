"""Bin-series stereo model and compatibility for CorGA's shared initializer."""

import sys
from types import ModuleType

from .defom_cor_ga_bins import MultiRangeBinInitializer


# Some CorGA models import this shared component from the old bins2 module.
# Expose only the initializer there; the bins2 model and source file are gone.
_compat = ModuleType(__name__ + '.defom_cor_ga_bins2')
_compat.MultiRangeBinInitializer = MultiRangeBinInitializer
sys.modules[_compat.__name__] = _compat
