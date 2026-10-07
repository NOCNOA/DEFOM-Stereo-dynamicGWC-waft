"""Shared physical-distance peak selection for bin evidence."""
import math
import torch

def _prepare(logits, valid, bins_px):
    if logits.ndim != 4 or logits.shape != valid.shape:
        raise ValueError("logits and valid must both have shape [R,K,H,W]")
    if not logits.is_floating_point() or valid.dtype != torch.bool:
        raise ValueError("logits must be floating point and valid must be bool")
    if valid.device != logits.device:
        raise ValueError("logits and valid must be on the same device")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("at least one branch and one bin are required")
    if bins_px.shape != logits.shape[:2]:
        raise ValueError("bins_px must have shape [R,K]")
    bins = bins_px.to(device=logits.device, dtype=torch.float32)
    if not torch.isfinite(bins).all() or not (bins[:, 1:] > bins[:, :-1]).all():
        raise ValueError("bin disparities must be finite and strictly increasing")
    usable = valid & torch.isfinite(logits)
    return bins, usable


def _finite_error(prediction, target):
    error = (prediction.float() - target.float()).abs()
    return error.masked_fill(~torch.isfinite(error), float("inf"))


@torch.no_grad()
def distinct_peak_indices(logits, valid, bins_px, max_peaks=4,
                          min_separation_px=8.0):
    """Select genuine local maxima independently for every branch and pixel.

    Flat local maxima use their leftmost bin. Endpoints and the boundaries of
    valid runs can be peaks. Highest scores are selected greedily, suppressing
    other peaks at distance strictly less than ``min_separation_px``; equality
    is allowed. Equal scores prefer the lower bin index. A missing peak has
    index zero and a false mask, rather than being filled by a nonmaximum bin.

    Returns indices and validity masks of shape [R,max_peaks,H,W]. Working
    tensors never add another candidate axis to [R,K,H,W].
    """
    if not isinstance(max_peaks, int) or isinstance(max_peaks, bool) or max_peaks < 1:
        raise ValueError("max_peaks must be a positive integer")
    if not math.isfinite(min_separation_px) or min_separation_px < 0:
        raise ValueError("min_separation_px must be finite and nonnegative")
    bins, usable = _prepare(logits, valid, bins_px)
    scores = logits.masked_fill(~usable, float("-inf"))
    maxima = torch.zeros_like(usable)

    # Propagate the comparison at the right edge through each equal-score run.
    # Testing only immediate neighbors incorrectly labels a rising plateau as
    # a peak. This scan avoids a second K-sized candidate dimension.
    right_lower = torch.ones_like(usable[:, 0])
    for k in range(logits.shape[1] - 1, -1, -1):
        if k + 1 < logits.shape[1]:
            right_lower = ((scores[:, k] > scores[:, k + 1]) |
                           ((scores[:, k] == scores[:, k + 1]) & right_lower))
        left_lower = True if k == 0 else scores[:, k] > scores[:, k - 1]
        maxima[:, k] = usable[:, k] & left_lower & right_lower

    remaining = scores.masked_fill(~maxima, float("-inf"))
    shape = (logits.shape[0], max_peaks, *logits.shape[2:])
    indices = torch.zeros(shape, dtype=torch.long, device=logits.device)
    selected_valid = torch.zeros(shape, dtype=torch.bool, device=logits.device)
    bin_indices = torch.arange(logits.shape[1], device=logits.device)[None, :, None, None]
    for p in range(max_peaks):
        best, index = remaining.max(dim=1)
        exists = torch.isfinite(best)
        indices[:, p] = torch.where(exists, index, 0)
        selected_valid[:, p] = exists
        selected_disp = bins.gather(1, index.flatten(1)).view_as(index)
        too_close = (bins[:, :, None, None] - selected_disp[:, None]).abs() < min_separation_px
        # Explicitly suppress the selected bin even when the separation is zero.
        suppress = (too_close | (bin_indices == index[:, None])) & exists[:, None]
        remaining.masked_fill_(suppress, float("-inf"))
    return indices, selected_valid


