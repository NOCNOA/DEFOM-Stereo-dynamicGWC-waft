import logging
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random

def normalize_disparity(disp):
    dtype = disp.dtype
    scale, shift = compute_scale_and_shift(disp)
    norm_disp = (disp - shift[..., None, None]) / scale[..., None, None]

    return norm_disp.to(dtype), scale.to(dtype), shift.to(dtype)

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def sequence_loss_d0L1(flow_preds, flow_gt, valid, loss_gamma=0.9,
                       max_flow=700, smooth_l1_beta=1.0,
                       debug_finite=False):
    """ 
    Loss function defined over sequence of flow predictions.

    flow_preds[0] is treated as initial disparity d0 and supervised by SmoothL1.
    flow_preds[1:] keep the original RAFT-style L1 sequence loss.
    """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    if debug_finite:
        assert not torch.isinf(flow_gt[valid.bool()]).any()
    valid_float = valid.float()
    valid_denominator = valid_float.sum().clamp_min(1.0)

    # same adjusted gamma logic as original RAFT-Stereo
    adjusted_loss_gamma = loss_gamma ** (15 / n_predictions)

    for i in range(n_predictions):
        pred = flow_preds[i]

        if debug_finite:
            assert torch.isfinite(pred).all()

        i_weight = adjusted_loss_gamma ** (n_predictions - i)

        if i == 0:
            # d0: SmoothL1 loss
            i_loss = F.smooth_l1_loss(
                pred,
                flow_gt,
                reduction='none',
                beta=smooth_l1_beta
            )
        else:
            # other disparity predictions: original L1 loss
            i_loss = (pred - flow_gt).abs()

        assert i_loss.shape == valid.shape, [
            i_loss.shape, valid.shape, flow_gt.shape, pred.shape
        ]

        flow_loss += (
            i_weight * (i_loss * valid_float).sum() / valid_denominator
        )

    # metrics keep unchanged, only evaluate final prediction
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().detach(),
        '1px': (epe < 1).float().mean().detach(),
        '3px': (epe < 3).float().mean().detach(),
        '5px': (epe < 5).float().mean().detach(),
    }

    return flow_loss, metrics


def sequence_loss_multi_init_waft(
    init_disps,
    init_logits,
    hypothesis_logits,
    flow_preds,
    flow_gt,
    valid,
    init_bins,
    init_valid,
    loss_gamma=0.9,
    max_flow=700,
    smooth_l1_beta=1.0,
    downsample=4,
    init_target_temperature=1.0,
    gate_loss_weight=0.1,
    gate_target_temperature=4.0,
    d0_loss_weight=1.0,
    use_gru_edge_loss=False,
    edge_mode="topk",
    edge_topk=0.10,
    edge_threshold=1.0,
    edge_dilation=5,
    gru_edge_lambda=1.0,
    use_gru_disp_grad_loss=False,
    gru_grad_lambda=0.02,
    init_branch_loss_weight=None,
    debug_finite=False,
):
    """WAFT initialization losses plus optional edge supervision on GRU outputs only.

    The bin classification, hypothesis gate, and fused d0 losses are deliberately
    left unchanged. Edge weights are applied only to ``flow_preds[1:]``; the
    optional disparity-gradient term is computed from the final GRU prediction.
    """
    if not isinstance(flow_preds, (list, tuple)):
        flow_preds = [flow_preds]
    if not flow_preds:
        raise ValueError("flow_preds must contain at least the fused d0 prediction")
    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}")
    if smooth_l1_beta < 0:
        raise ValueError(f"smooth_l1_beta must be non-negative, got {smooth_l1_beta}")
    if init_target_temperature <= 0 or gate_target_temperature <= 0:
        raise ValueError(f"target temperatures must be positive, got {init_target_temperature} and {gate_target_temperature}")
    if gate_loss_weight < 0:
        raise ValueError(f"gate_loss_weight must be non-negative, got {gate_loss_weight}")
    if d0_loss_weight < 0:
        raise ValueError(f"d0_loss_weight must be non-negative, got {d0_loss_weight}")
    if gru_edge_lambda < 0 or gru_grad_lambda < 0:
        raise ValueError(
            f"gru_edge_lambda and gru_grad_lambda must be non-negative, got "
            f"{gru_edge_lambda} and {gru_grad_lambda}"
        )
    flow_gt = _as_disp_4d(flow_gt).float()
    valid = _as_disp_4d(valid).float()
    if init_disps.ndim != 4:
        raise ValueError(f"init_disps must be [B,R,H,W], got {tuple(init_disps.shape)}")
    batch, num_ranges, height, width = init_disps.shape
    if init_logits.ndim != 5 or init_logits.shape[:2] != (batch, num_ranges) or init_logits.shape[-2:] != (height, width):
        raise ValueError(f"init_logits must be [B,R,K,H,W] and match init_disps, got {tuple(init_logits.shape)} and {tuple(init_disps.shape)}")
    if tuple(init_valid.shape) != tuple(init_logits.shape):
        raise ValueError(f"init_valid must match init_logits, got {tuple(init_valid.shape)} and {tuple(init_logits.shape)}")
    if tuple(hypothesis_logits.shape) != tuple(init_disps.shape):
        raise ValueError(f"hypothesis_logits must match init_disps, got {tuple(hypothesis_logits.shape)} and {tuple(init_disps.shape)}")
    if debug_finite:
        assert torch.isfinite(init_disps).all()
        assert torch.isfinite(init_logits).all()
        assert torch.isfinite(hypothesis_logits).all()
    num_bins = init_logits.shape[2]
    if tuple(init_bins.shape) != (num_ranges, num_bins):
        raise ValueError(f"init_bins must be {(num_ranges, num_bins)}, got {tuple(init_bins.shape)}")
    gt_full_low = F.interpolate(flow_gt, size=(height, width), mode='nearest')
    gt_low = gt_full_low / float(downsample)
    gt_for_target = torch.nan_to_num(gt_full_low, nan=0.0, posinf=0.0, neginf=0.0)
    valid_low = F.interpolate(valid, size=(height, width), mode='nearest') >= 0.5
    valid_low = valid_low & torch.isfinite(gt_low) & (gt_low >= 0.0) & (gt_low * float(downsample) < float(max_flow))
    init_classification_losses = []
    branch_eligible_masks = []
    for branch_index in range(num_ranges):
        bins_low = init_bins[branch_index].float().view(1, num_bins, 1, 1)
        bins_full = bins_low * float(downsample)
        candidate_valid = init_valid[:, branch_index].bool()
        valid_bins_low = bins_low.expand_as(candidate_valid)
        candidate_exists = candidate_valid.any(dim=1, keepdim=True)
        min_valid_bin = valid_bins_low.masked_fill(~candidate_valid, float('inf')).amin(dim=1, keepdim=True)
        max_valid_bin = valid_bins_low.masked_fill(~candidate_valid, float('-inf')).amax(dim=1, keepdim=True)
        branch_valid = valid_low & candidate_exists & (gt_low >= min_valid_bin) & (gt_low <= max_valid_bin)
        target_logits = -(bins_full - gt_for_target).abs() / float(init_target_temperature)
        target_probability = torch.softmax(target_logits.masked_fill(~candidate_valid, -1e4), dim=1) * candidate_valid.float()
        target_probability = (target_probability / target_probability.sum(dim=1, keepdim=True).clamp_min(1e-6)).detach()
        prediction_logits = init_logits[:, branch_index].float().masked_fill(~candidate_valid, -1e4)
        classification_map = -(target_probability * F.log_softmax(prediction_logits, dim=1)).sum(dim=1, keepdim=True)
        branch_valid_float = branch_valid.float()
        init_classification_losses.append(
            (classification_map * branch_valid_float).sum()
            / branch_valid_float.sum().clamp_min(1.0)
        )
        branch_eligible_masks.append(branch_valid)
    init_classification_loss = torch.stack(init_classification_losses).sum()
    branch_eligible = torch.cat(branch_eligible_masks, dim=1)
    gate_error = (init_disps.detach().float() * float(downsample) - gt_for_target).abs()
    gate_target_logits = (-gate_error / float(gate_target_temperature)).masked_fill(~branch_eligible, -1e4)
    gate_target_probability = torch.softmax(gate_target_logits, dim=1) * branch_eligible.float()
    gate_target_probability = (gate_target_probability / gate_target_probability.sum(dim=1, keepdim=True).clamp_min(1e-6)).detach()
    gate_log_probability = F.log_softmax(hypothesis_logits.float(), dim=1)
    gate_probability = gate_log_probability.exp()
    gate_loss_map = -(gate_target_probability * gate_log_probability).sum(dim=1)
    gate_valid = branch_eligible.any(dim=1)
    gate_valid_float = gate_valid.float()
    gate_denominator = gate_valid_float.sum().clamp_min(1.0)
    gate_loss = (gate_loss_map * gate_valid_float).sum() / gate_denominator
    gate_correct = (
        gate_probability.argmax(dim=1)
        == gate_target_probability.argmax(dim=1)
    ).float()
    gate_accuracy = (
        (gate_correct * gate_valid_float).sum() / gate_denominator
    ).detach()
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid_full = (valid >= 0.5) & torch.isfinite(flow_gt) & (flow_gt >= 0.0) & (mag < float(max_flow))
    valid_float = valid_full.float()
    valid_den = valid_float.sum().clamp_min(1.0)
    flow_gt_safe = torch.where(valid_full, flow_gt, torch.zeros_like(flow_gt))
    d0_prediction = _as_disp_4d(flow_preds[0]).float()
    if debug_finite:
        assert torch.isfinite(d0_prediction).all()
    d0_loss_map = F.smooth_l1_loss(d0_prediction, flow_gt_safe, reduction='none', beta=float(smooth_l1_beta))
    d0_loss = (d0_loss_map * valid_float).sum() / valid_den

    if use_gru_edge_loss:
        edge_mask, edge_ratio, _ = disparity_edge_mask(
            flow_gt_safe,
            valid_full,
            edge_mode=edge_mode,
            edge_topk=edge_topk,
            edge_threshold=edge_threshold,
            edge_dilation=edge_dilation,
        )
        recurrent_valid_weight = valid_float * (1.0 + float(gru_edge_lambda) * edge_mask)
        recurrent_den = recurrent_valid_weight.sum().clamp_min(1.0)
        edge_weighted_mass_ratio = (recurrent_valid_weight * edge_mask).sum() / recurrent_den
    else:
        edge_mask = torch.zeros_like(valid_float)
        edge_ratio = valid_float.new_tensor(0.0)
        edge_weighted_mass_ratio = valid_float.new_tensor(0.0)
        recurrent_valid_weight = valid_float
        recurrent_den = valid_den

    recurrent_predictions = flow_preds[1:]
    num_recurrent_predictions = len(recurrent_predictions)
    adjusted_loss_gamma = float(loss_gamma) ** (15.0 / max(num_recurrent_predictions, 1))
    recurrent_weights = []
    recurrent_loss = flow_gt.new_tensor(0.0)
    recurrent_loss_raw = flow_gt.new_tensor(0.0)
    for prediction_index, prediction in enumerate(recurrent_predictions):
        prediction = _as_disp_4d(prediction).float()
        if debug_finite:
            assert torch.isfinite(prediction).all()
        prediction_loss = (prediction - flow_gt_safe).abs()
        prediction_weight = adjusted_loss_gamma ** (num_recurrent_predictions - 1 - prediction_index)
        recurrent_weights.append(prediction_weight)
        recurrent_loss_raw = recurrent_loss_raw + prediction_weight * (prediction_loss * valid_float).sum() / valid_den
        recurrent_loss = recurrent_loss + prediction_weight * (prediction_loss * recurrent_valid_weight).sum() / recurrent_den

    final_prediction = _as_disp_4d(flow_preds[-1]).float()
    if use_gru_disp_grad_loss and num_recurrent_predictions > 0:
        gru_disp_grad_loss = disparity_gradient_loss(final_prediction, flow_gt_safe, valid_full)
    else:
        gru_disp_grad_loss = flow_gt.new_tensor(0.0)
    weighted_gru_disp_grad_loss = float(gru_grad_lambda) * gru_disp_grad_loss

    total_loss = (
        init_classification_loss
        + float(gate_loss_weight) * gate_loss
        + float(d0_loss_weight) * d0_loss
        + recurrent_loss
        + weighted_gru_disp_grad_loss
    )
    epe = torch.sum((final_prediction - flow_gt_safe) ** 2, dim=1).sqrt().view(-1)[valid_full.view(-1)]
    if epe.numel() > 0:
        epe_mean = epe.mean().detach()
        px1 = (epe < 1).float().mean().detach()
        px3 = (epe < 3).float().mean().detach()
        px5 = (epe < 5).float().mean().detach()
    else:
        epe_mean = px1 = px3 = px5 = flow_gt.new_tensor(0.0)
    metrics = {
        'epe': epe_mean,
        '1px': px1,
        '3px': px3,
        '5px': px5,
        'init_soft_ce': init_classification_loss.detach(),
        'gate_soft_ce': gate_loss.detach(),
        'gate_weighted_soft_ce': (float(gate_loss_weight) * gate_loss).detach(),
        'gate_accuracy': gate_accuracy,
        'd0_smooth_l1': d0_loss.detach(),
        'd0_weighted_smooth_l1': (float(d0_loss_weight) * d0_loss).detach(),
        'recurrent_l1': recurrent_loss.detach(),
        'recurrent_l1_raw': recurrent_loss_raw.detach(),
        'edge_pixel_ratio': edge_ratio.detach(),
        'edge_weighted_mass_ratio': edge_weighted_mass_ratio.detach(),
        'gru_disp_grad_loss': gru_disp_grad_loss.detach(),
        'gru_weighted_disp_grad_loss': weighted_gru_disp_grad_loss.detach(),
        'recurrent_weight_first': recurrent_weights[0] if recurrent_weights else 0.0,
        'recurrent_weight_last': recurrent_weights[-1] if recurrent_weights else 0.0,
        'total_loss': total_loss.detach(),
    }
    for branch_index, branch_loss in enumerate(init_classification_losses):
        metrics[f'init_soft_ce_{branch_index}'] = branch_loss.detach()
    return total_loss, metrics



def sequence_loss_multi_init_smooth_l1_edge(init_disps, init_logits, hypothesis_logits, flow_preds, flow_gt, valid, init_bins, init_valid,
                                             loss_gamma=0.9, max_flow=700, smooth_l1_beta=1.0, downsample=4,
                                             init_branch_loss_weight=0.1, d0_loss_weight=1.0,
                                             edge_mode='topk', edge_topk=0.10, edge_threshold=1.0, edge_dilation=5,
                                             gru_edge_lambda=1.0, use_gru_disp_grad_loss=False, gru_grad_lambda=0.02, **unused):
    """bins3: branch/d0 Smooth-L1 and compulsory edge-weighted GRU supervision."""
    if not isinstance(flow_preds, (list, tuple)):
        flow_preds = [flow_preds]
    if not flow_preds or init_disps.ndim != 4:
        raise ValueError('flow_preds must be non-empty and init_disps must be [B,R,H,W]')
    flow_gt, valid = _as_disp_4d(flow_gt).float(), _as_disp_4d(valid).float()
    batch, ranges, height, width = init_disps.shape
    if init_valid.ndim != 5 or init_valid.shape[:2] != (batch, ranges) or init_valid.shape[-2:] != (height, width):
        raise ValueError('init_valid must be [B,R,K,H,W] and match init_disps')
    bins = init_valid.shape[2]
    if tuple(init_bins.shape) != (ranges, bins):
        raise ValueError(f'init_bins must be {(ranges, bins)}, got {tuple(init_bins.shape)}')
    gt_low_full = F.interpolate(flow_gt, size=(height, width), mode='nearest')
    gt_low = gt_low_full / float(downsample)
    valid_low = (F.interpolate(valid, size=(height, width), mode='nearest') >= .5) & torch.isfinite(gt_low) & (gt_low >= 0) & (gt_low_full < max_flow)
    branch_losses = []
    for r in range(ranges):
        candidate_valid = init_valid[:, r].bool()
        branch_bins = init_bins[r].float().view(1, bins, 1, 1).expand_as(candidate_valid)
        present = candidate_valid.any(dim=1, keepdim=True)
        lo = branch_bins.masked_fill(~candidate_valid, float('inf')).amin(dim=1, keepdim=True)
        hi = branch_bins.masked_fill(~candidate_valid, float('-inf')).amax(dim=1, keepdim=True)
        mask = valid_low & present & (gt_low >= lo) & (gt_low <= hi)
        prediction = init_disps[:, r:r + 1].float() * float(downsample)
        loss_map = F.smooth_l1_loss(prediction, gt_low_full, reduction='none', beta=smooth_l1_beta)
        branch_losses.append(loss_map[mask].mean() if mask.any() else prediction.sum() * 0.)
    branch_loss = torch.stack(branch_losses).sum()
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid_full = (valid >= .5) & torch.isfinite(flow_gt) & (flow_gt >= 0) & (mag < max_flow)
    valid_float = valid_full.float()
    valid_den = valid_float.sum().clamp_min(1.)
    target = torch.where(valid_full, flow_gt, torch.zeros_like(flow_gt))
    d0 = _as_disp_4d(flow_preds[0]).float()
    d0_loss = (F.smooth_l1_loss(d0, target, reduction='none', beta=smooth_l1_beta) * valid_float).sum() / valid_den
    edge_mask, edge_ratio, _ = disparity_edge_mask(target, valid_full, edge_mode, edge_topk, edge_threshold, edge_dilation)
    weighted_valid = valid_float * (1. + gru_edge_lambda * edge_mask)
    weighted_den = weighted_valid.sum().clamp_min(1.)
    weighted_mass = (weighted_valid * edge_mask).sum() / weighted_den
    recurrent, raw, weights = target.new_tensor(0.), target.new_tensor(0.), []
    predictions = flow_preds[1:]
    gamma = loss_gamma ** (15. / max(len(predictions), 1))
    for i, prediction in enumerate(predictions):
        weight = gamma ** (len(predictions) - 1 - i)
        error = (_as_disp_4d(prediction).float() - target).abs()
        weights.append(weight)
        raw = raw + weight * (error * valid_float).sum() / valid_den
        recurrent = recurrent + weight * (error * weighted_valid).sum() / weighted_den
    final = _as_disp_4d(flow_preds[-1]).float()
    grad = disparity_gradient_loss(final, target, valid_full) if use_gru_disp_grad_loss and predictions else target.new_tensor(0.)
    total = init_branch_loss_weight * branch_loss + d0_loss_weight * d0_loss + recurrent + gru_grad_lambda * grad
    epe = torch.sum((final - target) ** 2, dim=1).sqrt().view(-1)[valid_full.view(-1)]
    metrics = {'epe': epe.mean().item() if epe.numel() else 0., '1px': (epe < 1).float().mean().item() if epe.numel() else 0.,
               '3px': (epe < 3).float().mean().item() if epe.numel() else 0., '5px': (epe < 5).float().mean().item() if epe.numel() else 0.,
               'init_branch_smooth_l1': branch_loss.detach().item(), 'init_branch_weighted_smooth_l1': (init_branch_loss_weight * branch_loss).detach().item(),
               'd0_smooth_l1': d0_loss.detach().item(), 'd0_weighted_smooth_l1': (d0_loss_weight * d0_loss).detach().item(),
               'recurrent_l1': recurrent.detach().item(), 'recurrent_l1_raw': raw.detach().item(), 'edge_pixel_ratio': edge_ratio.detach().item(),
               'edge_weighted_mass_ratio': weighted_mass.detach().item(), 'gru_disp_grad_loss': grad.detach().item(),
               'gru_weighted_disp_grad_loss': (gru_grad_lambda * grad).detach().item(), 'recurrent_weight_first': weights[0] if weights else 0.,
               'recurrent_weight_last': weights[-1] if weights else 0., 'total_loss': total.detach().item()}
    for r, loss in enumerate(branch_losses):
        metrics[f'init_branch_smooth_l1_{r}'] = loss.detach().item()
    return total, metrics

def _as_disp_4d(x):
    """Convert disparity-like tensors to [B, 1, H, W]."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected disparity tensor with 3 or 4 dims, got shape {tuple(x.shape)}")


def disparity_edge_mask(
    disp_gt,
    valid,
    edge_mode="topk",
    edge_topk=0.10,
    edge_threshold=1.0,
    edge_dilation=5,
):
    """Build a GT-disparity edge mask from disparity gradients.

    Returns:
        edge_mask: float tensor in [0, 1], shape [B, 1, H, W]
        edge_ratio: scalar tensor, edge pixels over valid pixels after dilation
        grad: raw gradient strength, shape [B, 1, H, W]
    """
    disp_gt = _as_disp_4d(disp_gt).float()
    valid = _as_disp_4d(valid).bool()

    b, _, h, w = disp_gt.shape
    grad = disp_gt.new_zeros((b, 1, h, w))
    grad_valid = torch.zeros((b, 1, h, w), device=disp_gt.device, dtype=torch.bool)

    if w > 1:
        grad_x = (disp_gt[:, :, :, 1:] - disp_gt[:, :, :, :-1]).abs()
        valid_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        grad[:, :, :, :-1] += grad_x * valid_x.float()
        grad_valid[:, :, :, :-1] |= valid_x

    if h > 1:
        grad_y = (disp_gt[:, :, 1:, :] - disp_gt[:, :, :-1, :]).abs()
        valid_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        grad[:, :, :-1, :] += grad_y * valid_y.float()
        grad_valid[:, :, :-1, :] |= valid_y

    if edge_mode == "threshold":
        edge_mask = (grad > edge_threshold) & grad_valid
    elif edge_mode == "topk":
        edge_mask = torch.zeros_like(grad_valid)
        edge_topk = max(0.0, min(float(edge_topk), 1.0))

        for batch_idx in range(b):
            image_valid = grad_valid[batch_idx, 0]
            values = grad[batch_idx, 0][image_valid]
            if values.numel() == 0 or edge_topk <= 0.0:
                continue

            k = max(1, int(torch.ceil(values.new_tensor(values.numel() * edge_topk)).item()))
            k = min(k, values.numel())
            topk_indices = torch.topk(values, k, largest=True).indices
            valid_indices = image_valid.nonzero(as_tuple=False)
            selected = valid_indices[topk_indices]
            edge_mask[batch_idx, 0, selected[:, 0], selected[:, 1]] = True
    else:
        raise ValueError(f"Unsupported edge_mode '{edge_mode}', expected 'topk' or 'threshold'")

    edge_mask = edge_mask.float()

    if edge_dilation and edge_dilation > 1:
        kernel_size = int(edge_dilation)
        padding = kernel_size // 2
        edge_mask = F.max_pool2d(edge_mask, kernel_size=kernel_size, stride=1, padding=padding)
        if kernel_size % 2 == 0:
            edge_mask = edge_mask[:, :, :h, :w]

    edge_mask = edge_mask * valid.float()
    valid_count = valid.float().sum().clamp_min(1.0)
    edge_ratio = edge_mask.sum() / valid_count

    return edge_mask, edge_ratio, grad


def disparity_gradient_loss(pred, disp_gt, valid):
    """Gradient consistency loss on adjacent valid GT pixels."""
    pred = _as_disp_4d(pred)
    disp_gt = _as_disp_4d(disp_gt).to(dtype=pred.dtype)
    valid = _as_disp_4d(valid).bool()

    total = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)

    if pred.shape[-1] > 1:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = disp_gt[:, :, :, 1:] - disp_gt[:, :, :, :-1]
        valid_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        loss_x = (pred_dx - gt_dx).abs() * valid_x.float()
        total = total + loss_x.sum()
        count = count + valid_x.float().sum()

    if pred.shape[-2] > 1:
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = disp_gt[:, :, 1:, :] - disp_gt[:, :, :-1, :]
        valid_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        loss_y = (pred_dy - gt_dy).abs() * valid_y.float()
        total = total + loss_y.sum()
        count = count + valid_y.float().sum()

    return total / count.clamp_min(1.0)


def sequence_loss_d0L1_edge(
    flow_preds,
    flow_gt,
    valid,
    loss_gamma=0.9,
    max_flow=700,
    smooth_l1_beta=1.0,
    use_edge_weight_loss=True,
    edge_mode="topk",
    edge_topk=0.10,
    edge_threshold=1.0,
    edge_dilation=5,
    lambda_edge=2.0,
    use_disp_grad_loss=True,
    lambda_grad=0.05,
):
    """D0 SmoothL1 + recurrent L1 sequence loss with GT edge-aware weighting."""
    if not isinstance(flow_preds, (list, tuple)):
        flow_preds = [flow_preds]

    flow_gt = _as_disp_4d(flow_gt)
    valid = _as_disp_4d(valid)
    n_predictions = len(flow_preds)
    assert n_predictions >= 1

    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]

    if valid.any():
        assert not torch.isinf(flow_gt[valid.bool()]).any()

    edge_mask, edge_ratio, _ = disparity_edge_mask(
        flow_gt,
        valid,
        edge_mode=edge_mode,
        edge_topk=edge_topk,
        edge_threshold=edge_threshold,
        edge_dilation=edge_dilation,
    )

    if use_edge_weight_loss:
        loss_weight = 1.0 + float(lambda_edge) * edge_mask
    else:
        loss_weight = torch.ones_like(edge_mask)
        edge_mask = torch.zeros_like(edge_mask)
        edge_ratio = edge_ratio.detach() * 0.0

    valid_float = valid.float()
    raw_sequence_loss = flow_gt.new_tensor(0.0)
    weighted_sequence_loss = flow_gt.new_tensor(0.0)
    adjusted_loss_gamma = loss_gamma ** (15 / n_predictions)

    for i, pred in enumerate(flow_preds):
        pred = _as_disp_4d(pred)
        assert not torch.isnan(pred).any() and not torch.isinf(pred).any()

        i_weight = adjusted_loss_gamma ** (n_predictions - i)

        if i == 0:
            i_loss = F.smooth_l1_loss(pred, flow_gt, reduction='none', beta=smooth_l1_beta)
        else:
            i_loss = (pred - flow_gt).abs()

        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, pred.shape]

        raw_den = valid_float.sum().clamp_min(1.0)
        raw_sequence_loss = raw_sequence_loss + i_weight * (i_loss * valid_float).sum() / raw_den

        weighted_valid = valid_float * loss_weight.to(dtype=i_loss.dtype)
        weighted_den = weighted_valid.sum().clamp_min(1.0)
        weighted_sequence_loss = weighted_sequence_loss + i_weight * (i_loss * weighted_valid).sum() / weighted_den

    final_pred = _as_disp_4d(flow_preds[-1])
    if use_disp_grad_loss:
        grad_loss = disparity_gradient_loss(final_pred, flow_gt, valid)
    else:
        grad_loss = final_pred.sum() * 0.0

    total_loss = weighted_sequence_loss + float(lambda_grad) * grad_loss

    epe = torch.sum((final_pred - flow_gt) ** 2, dim=1).sqrt()
    valid_flat = valid.view(-1)
    epe = epe.view(-1)[valid_flat]

    if epe.numel() > 0:
        epe_mean = epe.mean().item()
        px1 = (epe < 1).float().mean().item()
        px3 = (epe < 3).float().mean().item()
        px5 = (epe < 5).float().mean().item()
    else:
        epe_mean = px1 = px3 = px5 = 0.0

    metrics = {
        'epe': epe_mean,
        '1px': px1,
        '3px': px3,
        '5px': px5,
        'sequence_loss_raw': raw_sequence_loss.detach().item(),
        'sequence_loss_edge_weighted': weighted_sequence_loss.detach().item(),
        'disp_grad_loss': grad_loss.detach().item(),
        'edge_pixel_ratio': edge_ratio.detach().item(),
        'total_loss': total_loss.detach().item(),
    }

    return total_loss, metrics

def sequence_loss_withconf(flow_preds, conf_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    B, _, H, W = flow_gt.shape
    #print("asd", flow_gt.shape)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        
        
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # with torch.no_grad():
        #     conf_gt = (i_loss <= 2).float()

        disp_loss = i_loss 
        #loss_conf_map = F.binary_cross_entropy_with_logits(logit_preds[i], conf_gt, reduction='none')
        w = (1 - conf_preds[i]).detach()
        w = w / (w.mean(dim=[2,3], keepdim=True) + 1e-6)
        w = w.clamp(0.25, 4.0)

        final_loss = disp_loss * w
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += (i_weight * final_loss)[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def gradient_x(img):
    return img[..., :, 1:] - img[..., :, :-1]

def gradient_y(img):
    return img[..., 1:, :] - img[..., :-1, :]

def edge_aware_smoothness(disp, image, alpha=10.0):
    """
    disp:  [B,1,H,W]
    image: [B,3,H,W] or [B,1,H,W]  (用于边缘权重)
    """
    # image 转灰度也行，这里简单用均值
    if image.shape[1] > 1:
        img = image.mean(dim=1, keepdim=True)
    else:
        img = image

    dx_disp = gradient_x(disp)
    dy_disp = gradient_y(disp)
    dx_img  = gradient_x(img).abs()
    dy_img  = gradient_y(img).abs()

    wx = torch.exp(-alpha * dx_img)
    wy = torch.exp(-alpha * dy_img)

    # charbonnier 比 L1 更平滑
    smooth = (wx * charbonnier(dx_disp)).mean() + (wy * charbonnier(dy_disp)).mean()
    return smooth

def sequence_loss_smoothness(
    flow_preds, flow_gt, valid,
    loss_gamma=0.9, max_flow=700,

    # ---- 新增：两类正则的权重 ----
    lambda_step=0.01,      # 建议 0.01 起
    lambda_osc=0.05,       # 建议 0.05 起
    lambda_smooth=0.01,    # 建议 0.001~0.01（需要 image 才能用）
    image=None,           # 传入左图 [B,3,H,W] 才启用 edge-aware smoothness
):
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # gamma 调整保持你原逻辑
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))

    prev_delta = None
    for i in range(n_predictions):
        pred = flow_preds[i]
        assert not torch.isnan(pred).any() and not torch.isinf(pred).any()

        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (pred - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, pred.shape]
        flow_loss = flow_loss + i_weight * i_loss[valid.bool()].mean()

        # --- (1) 步长正则：抑制每步更新过大（不需要额外输出） ---
        if lambda_step > 0.0:
            if i == 0:
                delta = pred - pred.detach()  # 0（第一步没有上一步；你也可以跳过 i==0）
            else:
                delta = pred - flow_preds[i-1].detach()  # 用 detach 防止“互相拉扯”
            # 只在 valid 上算，避免无效区影响
            flow_loss = flow_loss + lambda_step * delta.abs()[valid.bool()].mean()

            # --- (2) 振荡正则：抑制来回震荡 Δt-Δt-1 ---
            if lambda_osc > 0.0 and prev_delta is not None:
                osc = (delta - prev_delta).abs()
                flow_loss = flow_loss + lambda_osc * osc[valid.bool()].mean()
            prev_delta = delta

        # --- (3) 边缘感知平滑：需要 image ---
        if lambda_smooth > 0.0 and image is not None:
            # pred 可能是 flow [B,2,H,W]，你若是 disparity 就是 [B,1,H,W]
            # stereo disparity 情况：只取 x 分量或直接 pred
            if pred.shape[1] == 2:
                disp = pred[:, :1]  # 只用 x 分量当作 disparity
            else:
                disp = pred
            flow_loss = flow_loss + (lambda_smooth * i_weight) * edge_aware_smoothness(disp, image)

    # metrics（保持不动）
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def fetch_optimizer(args, model, last_epoch=-1, checkpoint=None):
    """ Create the optimizer and learning rate scheduler """
    trainable_params = [parameter for parameter in model.parameters()
                        if parameter.requires_grad]
    optimizer = optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.01,
                                              cycle_momentum=False, anneal_strategy='linear', last_epoch=last_epoch)

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = 'runs/' + name
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar("train/" + k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar("valid/" + key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
