"""Per-pixel diagnostics for disparity-bin predictions (no model dependencies).

All disparities and separation thresholds use original-image pixel units. The
bin axis must be ordered by increasing disparity. Nonfinite logits are treated
as invalid evidence. Missing candidates produce infinite errors/ranks, so an
oracle cannot silently count an invalid bin as a successful match.
"""

import math

import torch


from core.bin.peak_selection import _prepare, _finite_error, distinct_peak_indices


@torch.no_grad()
def candidate_errors(logits, valid, bins_px, branch_disps_px, fused_disp_px,
                     gt_px, peak_counts=(1, 2, 4), min_separation_px=8.0):
    """Return pixelwise errors for the model and several GT-only oracles.

    ``topN_oracle`` searches the union of at most N distinct peaks from EACH
    branch. ``branch_oracle`` searches branch disparity predictions, excluding
    branches with no valid evidence. ``all_bin_oracle`` searches every valid
    bin regardless of score and measures the range/quantization floor.

    The result has keys ``errors``, ``peak_indices``, and ``peak_valid``.
    The fused error is computed directly from the supplied model prediction;
    validity filtering of GT pixels is left to the evaluator.
    """
    bins, usable = _prepare(logits, valid, bins_px)
    expected_branch_shape = (logits.shape[0], *logits.shape[2:])
    if branch_disps_px.shape != expected_branch_shape:
        raise ValueError("branch_disps_px must have shape [R,H,W]")
    if fused_disp_px.shape != logits.shape[2:] or gt_px.shape != logits.shape[2:]:
        raise ValueError("fused_disp_px and gt_px must have shape [H,W]")
    if not peak_counts or any(not isinstance(n, int) or isinstance(n, bool) or n < 1
                              for n in peak_counts):
        raise ValueError("peak_counts must contain positive integers")
    indices, peak_valid = distinct_peak_indices(
        logits, valid, bins, max_peaks=max(peak_counts),
        min_separation_px=min_separation_px)
    peak_disps = bins[:, :, None, None].expand_as(logits).gather(1, indices)
    peak_errors = _finite_error(peak_disps, gt_px).masked_fill(~peak_valid, float("inf"))
    branch_errors = _finite_error(branch_disps_px, gt_px)
    branch_errors.masked_fill_(~usable.any(dim=1), float("inf"))
    all_errors = _finite_error(bins[:, :, None, None], gt_px)
    all_errors.masked_fill_(~usable, float("inf"))
    errors = {
        "fused": _finite_error(fused_disp_px, gt_px),
        "branch_oracle": branch_errors.amin(dim=0),
        "all_bin_oracle": all_errors.amin(dim=(0, 1)),
    }
    for n in peak_counts:
        errors[f"top{n}_oracle"] = peak_errors[:, :n].amin(dim=(0, 1))
    return {"errors": errors, "peak_indices": indices, "peak_valid": peak_valid}


@torch.no_grad()
def nearest_gt_bin_rank(logits, valid, bins_px, gt_px):
    """Rank the physically nearest valid GT bin within each branch.

    Rank is 1 + the number of valid bins with STRICTLY higher scores, so score
    ties share a lower-bound rank. Equal-distance nearest-bin ties choose the
    lower bin index. Returns float [R,H,W], with inf for no valid bin or a
    nonfinite ground truth. This ranks bins, not the NMS-selected peaks.
    """
    bins, usable = _prepare(logits, valid, bins_px)
    if gt_px.shape != logits.shape[2:]:
        raise ValueError("gt_px must have shape [H,W]")
    distance = _finite_error(bins[:, :, None, None], gt_px)
    distance.masked_fill_(~usable, float("inf"))
    nearest_distance, nearest = distance.min(dim=1)
    nearest_score = logits.gather(1, nearest[:, None]).squeeze(1)
    rank = 1 + ((logits > nearest_score[:, None]) & usable).sum(dim=1)
    return rank.float().masked_fill(~torch.isfinite(nearest_distance), float("inf"))
