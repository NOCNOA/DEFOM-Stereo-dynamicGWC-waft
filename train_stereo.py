from __future__ import print_function, division
import os
import sys
import gc
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from utils.utils import *
from core.defom_cor_ga_bins2 import DEFOMStereo as CorGABinsNoRoPEStereo
from core.defom_cor_ga_bins3 import DEFOMStereo as CorGABinsSmoothInitEdgeStereo
from core.defom_cor_ga_bins3_gwc import DEFOMStereo as CorGABins3GWCStereo
from core.defom_cor_ga_bins4 import DEFOMStereo as CorGABinsAttentionStereo
from core.defom_cor_ga_bins5 import DEFOMStereo as CorGABins5Stereo
from core.defom_cor_ga_bins5_shared_parallel import DEFOMStereo as CorGABins5SharedParallelStereo
from core.defom_cor_ga_bins6 import DEFOMStereo as CorGABins6Stereo
from core.defom_cor_ga_bins_residual import DEFOMStereo as CorGABinsResidualStereo
from core.defom_cor_ga import DEFOMStereo as CorGAStereo
from core.defom_cor_ga_coarse1 import DEFOMStereo as CorGACoarse1Stereo
from core.defom_cor_ga_lite import DEFOMStereo as CorGALiteStereo
from core.defom_cor_ga_fast import DEFOMStereo as CorGAFastStereo
from core.defom_cor_ga_fast2 import DEFOMStereo as CorGAFast2Stereo
from core.defom_cor_ga_fast3 import DEFOMStereo as CorGAFast3Stereo
from core.defom_cor_ga_no_mono_head import DEFOMStereo as CorGANoMonoHeadStereo
from core.defom_cor_ga_no_mono_igev import DEFOMStereo as CorGANoMonoIGEVStereo
from core.defom_cor_ga_stereo_igev import DEFOMStereo as CorGAStereoIGEVStereo
from core.defom_cor_ga_fast_pairres import DEFOMStereo as CorGAFastPairResidualStereo
from core.defom_cor_ga_fast_bininit import DEFOMStereo as CorGAFastBinInitStereo
from core.defom_cor_ga_fast_binmono import DEFOMStereo as CorGAFastBinMonoStereo
from core.defom_cor_ga_fast_binmono_corrstats import DEFOMStereo as CorGAFastBinMonoCorrStatsStereo
from core.defom_cor_ga_noh import DEFOMStereo as CorGANoHiddenStereo
from core.defom_cor_ga_bins import DEFOMStereo as CorGABinsStereo
#from core.defom_GRU_2multi import DEFOMStereo
#from core.defom_onlygwc import DEFOMStereo
#from core.defom_pru import DEFOMStereo
from evaluate_stereo import validate_things, count_parameters
import core.stereo_datasets as datasets

try:
    from torch.amp import GradScaler
    GRAD_SCALER_HAS_DEVICE_ARG = True
except ImportError:
    from torch.cuda.amp import GradScaler
    GRAD_SCALER_HAS_DEVICE_ARG = False


@torch.no_grad()
def clip_grad_norm_stable(parameters, max_norm, chunk_elements=1048576):
    """Clip an L2 gradient norm with bounded temporary memory and FP64 reduction."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    if chunk_elements < 1:
        raise ValueError(f"chunk_elements must be positive, got {chunk_elements}")

    chunk_norms = []
    for parameter in parameters:
        flat_grad = parameter.grad.detach().reshape(-1)
        for start in range(0, flat_grad.numel(), chunk_elements):
            chunk = flat_grad[start:start + chunk_elements]
            max_abs = chunk.abs().amax()
            safe_scale = torch.where(
                torch.isfinite(max_abs) & (max_abs > 0),
                max_abs,
                torch.ones_like(max_abs),
            )
            chunk_norms.append(
                torch.linalg.vector_norm(chunk / safe_scale).double()
                * safe_scale.double()
            )

    total_norm = torch.linalg.vector_norm(torch.stack(chunk_norms))
    if torch.isfinite(total_norm):
        clip_coefficient = min(
            1.0,
            float(max_norm) / (total_norm.item() + 1e-6),
        )
        if clip_coefficient < 1.0:
            for parameter in parameters:
                parameter.grad.mul_(clip_coefficient)

    return total_norm


@torch.no_grad()
def coarse_branch_valid_masks(
    init_bins,
    init_valid,
    disp_gt,
    valid,
    downsample,
    max_flow,
):
    """Return full-resolution masks within each coarse branch's valid bin range."""
    if init_valid.ndim != 5:
        raise ValueError(f"init_valid must be [B,R,K,H,W], got {tuple(init_valid.shape)}")
    _, branches, bins_count, height, width = init_valid.shape
    if tuple(init_bins.shape) != (branches, bins_count):
        raise ValueError(
            f"init_bins must be {(branches, bins_count)}, got {tuple(init_bins.shape)}"
        )
    if disp_gt.ndim != 4 or valid.shape != disp_gt.shape:
        raise ValueError(
            f"disp_gt/valid must have matching 4D shapes, got "
            f"{tuple(disp_gt.shape)}/{tuple(valid.shape)}"
        )

    candidate_valid = init_valid.bool()
    bins = init_bins.float().view(1, branches, bins_count, 1, 1)
    expanded_bins = bins.expand_as(candidate_valid)
    candidate_exists = candidate_valid.any(dim=2, keepdim=True)
    minimum = expanded_bins.masked_fill(
        ~candidate_valid, float("inf")
    ).amin(dim=2, keepdim=True)
    maximum = expanded_bins.masked_fill(
        ~candidate_valid, float("-inf")
    ).amax(dim=2, keepdim=True)

    low_size = (height, width)
    gt_low = F.interpolate(disp_gt.float(), size=low_size, mode="nearest")
    gt_low = gt_low / float(downsample)
    valid_low = F.interpolate(valid.float(), size=low_size, mode="nearest") >= 0.5
    valid_low &= torch.isfinite(gt_low) & (gt_low >= 0.0)
    valid_low &= gt_low * float(downsample) < float(max_flow)
    gt_low = gt_low.unsqueeze(1)
    eligible_low = candidate_exists & valid_low.unsqueeze(1)
    eligible_low &= gt_low >= minimum
    eligible_low &= gt_low <= maximum

    eligible_full = F.interpolate(
        eligible_low.squeeze(2).float(),
        size=disp_gt.shape[-2:],
        mode="nearest",
    ) >= 0.5
    full_valid = (valid >= 0.5) & torch.isfinite(disp_gt)
    full_valid &= (disp_gt >= 0.0) & (disp_gt < float(max_flow))
    return [
        full_valid & eligible_full[:, branch:branch + 1]
        for branch in range(branches)
    ]


def train(args):

    seed_everything(args.seed)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if args.train_max_disp <= 0 or args.eval_max_disp <= 0:
    #     raise ValueError(f"train_max_disp and eval_max_disp must be positive, got {args.train_max_disp} and {args.eval_max_disp}")
    # if args.d0_loss_weight < 0 or args.gate_loss_weight < 0:
    #     raise ValueError(f"d0_loss_weight and gate_loss_weight must be non-negative, got {args.d0_loss_weight} and {args.gate_loss_weight}")
    # if args.init_target_temperature <= 0 or args.gate_target_temperature <= 0 or args.max_delta_disp <= 0 or args.amp_init_scale <= 0:
    #     raise ValueError(f"temperatures, max_delta_disp, and amp_init_scale must be positive, got {args.init_target_temperature}, {args.gate_target_temperature}, {args.max_delta_disp}, and {args.amp_init_scale}")
    # if args.gru_edge_lambda < 0 or args.gru_grad_lambda < 0:
    #     raise ValueError(
    #         f"gru_edge_lambda and gru_grad_lambda must be non-negative, got "
    #         f"{args.gru_edge_lambda} and {args.gru_grad_lambda}"
    #     )
    # if not 0.0 <= args.edge_topk <= 1.0:
    #     raise ValueError(f"edge_topk must be in [0, 1], got {args.edge_topk}")
    # if args.edge_threshold < 0 or args.edge_dilation < 1:
    #     raise ValueError(f"edge_threshold must be non-negative and edge_dilation positive, got {args.edge_threshold} and {args.edge_dilation}")

    # if args.launcher == 'none':
    #     args.distributed = False
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # else:
    #     args.distributed = True

    #     # adjust batch size for each gpu
    #     assert args.batch_size % torch.cuda.device_count() == 0
    #     args.batch_size = args.batch_size // torch.cuda.device_count()

    #     dist_params = dict(backend='nccl')
    #     init_dist(args.launcher, **dist_params)
    #     # re-set gpu_ids with distributed training mode
    #     _, world_size = get_dist_info()
    #     args.gpu_ids = range(world_size)
    #     device = torch.device('cuda:{}'.format(args.local_rank))

    #     setup_for_distributed(args.local_rank == 0)

    model_classes = {
        'cor_ga_coarse1': CorGACoarse1Stereo,
        'cor_ga_lite': CorGALiteStereo,
        'cor_ga': CorGAStereo,
        'cor_ga_fast': CorGAFastStereo,
        'cor_ga_fast2': CorGAFast2Stereo,
        'cor_ga_fast3': CorGAFast3Stereo,
        'cor_ga_no_mono_head': CorGANoMonoHeadStereo,
        'cor_ga_no_mono_igev': CorGANoMonoIGEVStereo,
        'cor_ga_stereo_igev': CorGAStereoIGEVStereo,
        'cor_ga_fast_pairres': CorGAFastPairResidualStereo,
        'cor_ga_fast_bininit': CorGAFastBinInitStereo,
        'cor_ga_fast_binmono': CorGAFastBinMonoStereo,
        'cor_ga_fast_binmono_corrstats': CorGAFastBinMonoCorrStatsStereo,
        'cor_ga_noh': CorGANoHiddenStereo,
        'cor_ga_bins': CorGABinsStereo,
        'cor_ga_bins2': CorGABinsNoRoPEStereo,
        'cor_ga_bins3': CorGABinsSmoothInitEdgeStereo,
        'cor_ga_bins3_gwc': CorGABins3GWCStereo,
        'cor_ga_bins4': CorGABinsAttentionStereo,
        'cor_ga_bins5': CorGABins5Stereo,
        'cor_ga_bins5_shared_parallel': CorGABins5SharedParallelStereo,
        'cor_ga_bins6': CorGABins6Stereo,
        'cor_ga_bins_residual': CorGABinsResidualStereo,
    }
    model_class = model_classes[args.model_arch]
    model = model_class(args).to(device)
    print("Parameter Count: %d, Trainable: %d" % count_parameters(model))

    if args.distributed:
        process_group = torch.distributed.new_group(list(range(len(args.gpu_ids))))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model_without_ddp = model.module
        else:
            model_without_ddp = model

    start_epoch = 0
    start_step = 0
    optimizer, scheduler = fetch_optimizer(args, model)
    if args.mixed_precision and args.amp_dtype == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 mixed precision is not supported by this CUDA device')
    scaler_enabled = args.mixed_precision and args.amp_dtype == 'fp16' and device.type == 'cuda'
    scaler = (GradScaler('cuda', enabled=scaler_enabled, init_scale=args.amp_init_scale)
              if GRAD_SCALER_HAS_DEVICE_ARG
              else GradScaler(enabled=scaler_enabled, init_scale=args.amp_init_scale))

    if args.resume_ckpt:
        assert args.resume_ckpt.endswith((".pth", ".pth.gz"))
        logging.info("Loading checkpoint: %s" % args.resume_ckpt)
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu', weights_only=False, mmap=True)
        if 'model' in checkpoint:
            load_result = model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        else:
            load_result = model_without_ddp.load_state_dict(checkpoint, strict=args.strict_resume)
        if not args.strict_resume and (load_result.missing_keys or load_result.unexpected_keys):
            logging.info("Checkpoint load: %d missing and %d unexpected keys", len(load_result.missing_keys), len(load_result.unexpected_keys))

        can_resume_training = all(key in checkpoint for key in ('optimizer', 'step', 'epoch')) and not args.no_resume_optimizer
        if can_resume_training:
            logging.info('Restoring optimizer and training state')
            start_step = int(checkpoint['step'])
            start_epoch = int(checkpoint['epoch'])
            del optimizer, scheduler
            legacy_scheduler_epoch = -1 if 'scheduler' in checkpoint else start_step - 1
            optimizer, scheduler = fetch_optimizer(args, model, last_epoch=legacy_scheduler_epoch, checkpoint=checkpoint)
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                restored_lrs = scheduler.get_last_lr()
                if len(restored_lrs) != len(optimizer.param_groups):
                    raise ValueError(f"Scheduler has {len(restored_lrs)} learning rates for {len(optimizer.param_groups)} optimizer groups")
                for param_group, restored_lr in zip(optimizer.param_groups, restored_lrs):
                    param_group['lr'] = restored_lr
            else:
                logging.warning("Checkpoint has no scheduler state; reconstructing it from step %d", start_step)
            if scaler.is_enabled() and checkpoint.get('scaler'):
                scaler.load_state_dict(checkpoint['scaler'])
            elif scaler.is_enabled():
                logging.warning("Checkpoint has no usable GradScaler state; starting with a fresh AMP scale")
        elif not args.no_resume_optimizer:
            logging.warning("Checkpoint has no complete optimizer/step/epoch state; loading model weights only")
        del checkpoint
        gc.collect()

    train_data = datasets.fetch_dataset(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
    else:
        train_sampler = None
    
    loader_worker_options = (
        {'persistent_workers': True, 'prefetch_factor': 2}
        if args.num_workers > 0 else {}
    )
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler, **loader_worker_options)

    total_steps = start_step
    epoch = start_epoch
    logger = Logger(model, scheduler, args.name)
    logger.total_steps = total_steps

    model.train()
    if not args.distributed:
        model_without_ddp.freeze_bn()
    should_keep_training = total_steps < args.num_steps
    reported_grad_stride_mismatch = False
    amp_overflow_count = 0
    nonfinite_gradient_count = 0
    large_gradient_count = 0
    accumulated_microbatches = 0

    while should_keep_training:

        # mannually change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            if accumulated_microbatches == 0:
                optimizer.zero_grad(set_to_none=True)
            image1 = data_blob["img1"].cuda(non_blocking=True)
            image2 = data_blob["img2"].cuda(non_blocking=True)
            disp_gt = data_blob["disp"].cuda(non_blocking=True)
            valid = data_blob["valid"].cuda(non_blocking=True)

            disp_predictions, aux = [], {}
            try:
                assert model.training
                disp_predictions, aux = model(image1, image2, iters=args.train_iters, scale_iters=args.scale_iters, return_aux=True)
                assert model.training
                if 'init_disps' in aux:
                    init_bins = model_without_ddp.bin_initializer.bin_disparities
                    main_loss, metrics = (sequence_loss_multi_init_smooth_l1_edge if args.model_arch in ('cor_ga_bins3', 'cor_ga_bins3_gwc') else sequence_loss_multi_init_waft)(
                        aux['init_disps'],
                        aux['init_logits'],
                        aux['w_h_logits'],
                        disp_predictions,
                        disp_gt,
                        valid,
                        init_bins,
                        aux['init_valid'],
                        loss_gamma=args.loss_gamma,
                        max_flow=args.train_max_disp,
                        smooth_l1_beta=args.smooth_l1_beta,
                        downsample=2 ** args.n_downsample,
                        init_target_temperature=args.init_target_temperature,
                        gate_loss_weight=args.gate_loss_weight,
                        gate_target_temperature=args.gate_target_temperature,
                        d0_loss_weight=args.d0_loss_weight,
                        use_gru_edge_loss=args.use_gru_edge_loss,
                        edge_mode=args.edge_mode,
                        edge_topk=args.edge_topk,
                        edge_threshold=args.edge_threshold,
                        edge_dilation=args.edge_dilation,
                        gru_edge_lambda=args.gru_edge_lambda,
                        use_gru_disp_grad_loss=args.use_gru_disp_grad_loss,
                        gru_grad_lambda=args.gru_grad_lambda,
                        init_branch_loss_weight=args.init_branch_loss_weight,
                        debug_finite=args.debug_nan,
                    )
                    if args.model_arch in ('cor_ga_bins5', 'cor_ga_bins5_shared_parallel', 'cor_ga_fast_bininit', 'cor_ga_no_mono_head', 'cor_ga_no_mono_igev', 'cor_ga_stereo_igev', 'cor_ga_fast_binmono', 'cor_ga_fast_binmono_corrstats'):
                        coarse_valid_masks = coarse_branch_valid_masks(
                            init_bins,
                            aux['init_valid'],
                            disp_gt,
                            valid,
                            downsample=2 ** args.n_downsample,
                            max_flow=args.train_max_disp,
                        )
                        coarse_losses = []
                        for branch_index, (branch_predictions, branch_valid) in enumerate(
                            zip(aux['coarse_predictions'], coarse_valid_masks)
                        ):
                            metrics[f'coarse_valid_ratio_{branch_index}'] = (
                                branch_valid.float().mean().detach()
                            )
                            coarse_losses.append(
                                sequence_loss_d0L1(
                                    branch_predictions,
                                    disp_gt,
                                    branch_valid,
                                    loss_gamma=args.loss_gamma,
                                    max_flow=args.train_max_disp,
                                    smooth_l1_beta=args.smooth_l1_beta,
                                    debug_finite=args.debug_nan,
                                )[0]
                            )
                        coarse_loss = (
                            torch.stack(coarse_losses).mean()
                            if coarse_losses
                            else main_loss.new_tensor(0.0)
                        )
                        loss = main_loss + args.coarse_loss_weight * coarse_loss
                    else:
                        coarse_loss = main_loss.new_tensor(0.0)
                        loss = main_loss
                else:
                    main_loss, metrics = sequence_loss_d0L1(disp_predictions, disp_gt, valid, loss_gamma=args.loss_gamma, max_flow=args.train_max_disp, smooth_l1_beta=args.smooth_l1_beta, debug_finite=args.debug_nan)
                    coarse_losses = [sequence_loss_d0L1(branch_predictions, disp_gt, valid, loss_gamma=args.loss_gamma, max_flow=args.train_max_disp, smooth_l1_beta=args.smooth_l1_beta, debug_finite=args.debug_nan)[0] for branch_predictions in aux.get('coarse_predictions', [])]
                    coarse_loss = torch.stack(coarse_losses).mean() if coarse_losses else main_loss.new_tensor(0.0)
                    loss = main_loss + args.coarse_loss_weight * coarse_loss
            except (AssertionError, FloatingPointError) as error:
                if args.local_rank == 0 and args.debug_nan:
                    nonfinite_outputs = []
                    for prediction_index, prediction in enumerate(disp_predictions):
                        nan_count = torch.isnan(prediction).sum().item()
                        inf_count = torch.isinf(prediction).sum().item()
                        if nan_count or inf_count:
                            nonfinite_outputs.append('refine[%d]:nan=%d,inf=%d' % (prediction_index, nan_count, inf_count))
                    for branch_index, branch_predictions in enumerate(aux.get('coarse_predictions', [])):
                        for prediction_index, prediction in enumerate(branch_predictions):
                            nan_count = torch.isnan(prediction).sum().item()
                            inf_count = torch.isinf(prediction).sum().item()
                            if nan_count or inf_count:
                                nonfinite_outputs.append('coarse[%d][%d]:nan=%d,inf=%d' % (branch_index, prediction_index, nan_count, inf_count))
                    for aux_name in ('init_disps', 'init_logits', 'w_h', 'w_h_logits', 'w_s'):
                        tensor = aux.get(aux_name)
                        if tensor is not None:
                            nan_count = torch.isnan(tensor).sum().item()
                            inf_count = torch.isinf(tensor).sum().item()
                            if nan_count or inf_count:
                                nonfinite_outputs.append('%s:nan=%d,inf=%d' % (aux_name, nan_count, inf_count))
                    failure_batch_path = Path('checkpoints/%s/nan_batch_snapshot_step_%d.pth' % (args.name, total_steps))
                    failure_model_path = Path('checkpoints/%s/nan_model_snapshot_step_%d.pth' % (args.name, total_steps))
                    torch.save({'step': total_steps, 'image1': image1.detach().cpu(), 'image2': image2.detach().cpu(), 'disp_gt': disp_gt.detach().cpu(), 'valid': valid.detach().cpu()}, failure_batch_path)
                    torch.save({'step': total_steps, 'model': model_without_ddp.state_dict()}, failure_model_path)
                    logging.exception("Non-finite failure (%s) in %s; batch saved to %s and model saved to %s", error, ', '.join(nonfinite_outputs) or 'stage reported by exception', failure_batch_path.absolute(), failure_model_path.absolute())
                raise
            metrics['main_loss'] = main_loss.detach()
            metrics['coarse_loss'] = coarse_loss.detach()
            init_metric = metrics.get('init_soft_ce', 0.0)
            metrics['init_loss'] = (
                init_metric.detach()
                if torch.is_tensor(init_metric) else float(init_metric)
            )
            metrics['total_loss'] = loss.detach()

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            accumulated_microbatches += 1
            if accumulated_microbatches < args.gradient_accumulation_steps:
                continue

            if (not reported_grad_stride_mismatch) and args.distributed and args.local_rank == 0:
                mismatch_lines = []
                for name, param in model_without_ddp.named_parameters():
                    if param.grad is None:
                        continue
                    if tuple(param.grad.stride()) != tuple(param.stride()):
                        mismatch_lines.append(
                            f"{name}: grad_shape={tuple(param.grad.shape)}, "
                            f"grad_stride={tuple(param.grad.stride())}, "
                            f"param_stride={tuple(param.stride())}"
                        )
                if mismatch_lines:
                    logging.warning("DDP grad/param stride mismatch detected:\n%s", "\n".join(mismatch_lines[:20]))
                    reported_grad_stride_mismatch = True

            scaler.unscale_(optimizer)
            stable_clip_architectures = (
                'cor_ga',
                'cor_ga_fast',
                'cor_ga_fast2',
                'cor_ga_fast3',
                'cor_ga_no_mono_head',
                'cor_ga_no_mono_igev',
                'cor_ga_stereo_igev',
                'cor_ga_fast_pairres',
                'cor_ga_fast_bininit',
                'cor_ga_fast_binmono',
                'cor_ga_fast_binmono_corrstats',
                'cor_ga_bins5',
                'cor_ga_bins5_shared_parallel',
                'cor_ga_noh',
                'cor_ga_coarse1',
                'cor_ga_lite',
            )
            grad_norm = (
                clip_grad_norm_stable(model.parameters(), 1.0)
                if args.model_arch in stable_clip_architectures
                else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            )
            grad_norm_value = grad_norm.detach().item()
            metrics['grad_norm'] = grad_norm_value
            grad_norm_is_finite = np.isfinite(grad_norm_value)
            if (
                grad_norm_is_finite
                and args.large_grad_warn_threshold > 0
                and grad_norm_value > args.large_grad_warn_threshold
            ):
                large_gradient_count += 1
                if args.local_rank == 0 and (
                    large_gradient_count == 1 or large_gradient_count % 100 == 0
                ):
                    logging.warning(
                        "Large finite gradient #%d at optimizer step %d "
                        "(stable pre-clip norm=%.6e); gradients clipped to 1.0",
                        large_gradient_count,
                        total_steps,
                        grad_norm_value,
                    )
            if not grad_norm_is_finite:
                if scaler.is_enabled():
                    scale_before = scaler.get_scale()
                    # unscale_() has already recorded the non-finite gradients.
                    # GradScaler.step() therefore skips optimizer.step(), while
                    # update() lowers the scale so the next batch can recover.
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    amp_overflow_count += 1
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_microbatches = 0
                    if args.local_rank == 0 and (amp_overflow_count == 1 or amp_overflow_count % 10 == 0):
                        logging.warning(
                            "AMP gradient overflow #%d at optimizer step %d "
                            "(grad_norm=%s, scale %.1f -> %.1f); update skipped and scheduler held",
                            amp_overflow_count,
                            total_steps,
                            grad_norm.detach().item(),
                            scale_before,
                            scale_after,
                        )
                    continue

                nonfinite_gradient_count += 1
                save_failure_snapshot = (
                    args.local_rank == 0
                    and (not args.skip_nonfinite_gradients or nonfinite_gradient_count == 1)
                )
                if save_failure_snapshot:
                    failure_batch_path = Path('checkpoints/%s/nonfinite_gradient_batch_step_%d.pth' % (args.name, total_steps))
                    failure_model_path = Path('checkpoints/%s/nonfinite_gradient_model_step_%d.pth' % (args.name, total_steps))
                    torch.save({'step': total_steps, 'grad_norm': grad_norm.detach().cpu(), 'image1': image1.detach().cpu(), 'image2': image2.detach().cpu(), 'disp_gt': disp_gt.detach().cpu(), 'valid': valid.detach().cpu()}, failure_batch_path)
                    torch.save({'step': total_steps, 'model': model_without_ddp.state_dict()}, failure_model_path)
                if args.skip_nonfinite_gradients:
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_microbatches = 0
                    if args.local_rank == 0:
                        logging.warning(
                            "Non-finite gradient #%d at optimizer step %d; batch skipped, "
                            "optimizer and scheduler held%s",
                            nonfinite_gradient_count,
                            total_steps,
                            (
                                f"; batch saved to {failure_batch_path.absolute()} and "
                                f"model saved to {failure_model_path.absolute()}"
                                if save_failure_snapshot else ""
                            ),
                        )
                    continue
                if args.local_rank == 0:
                    logging.error("Non-finite gradient norm %s at optimizer step %d; stopping before optimizer update. Batch saved to %s and model saved to %s", grad_norm.detach().item(), total_steps, failure_batch_path.absolute(), failure_model_path.absolute())
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {total_steps}: {grad_norm.detach().item()}")

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
            if not optimizer_stepped:
                amp_overflow_count += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_microbatches = 0
                if args.local_rank == 0 and (amp_overflow_count == 1 or amp_overflow_count % 10 == 0):
                    logging.warning("AMP gradient overflow #%d at optimizer step %d; update skipped and scheduler held", amp_overflow_count, total_steps)
                continue
            scheduler.step()
            accumulated_microbatches = 0

            total_steps += 1

            if args.local_rank == 0:
                if total_steps % 10 == 0:
                    logger.writer.add_scalar(
                        "train/live_loss", loss.detach(), total_steps
                    )
                logger.writer.add_scalar(f'train/learning_rate', optimizer.param_groups[0]['lr'], total_steps)
                logger.push(metrics)

                if total_steps % args.save_latest_ckpt_freq == 0:
                    save_path = Path('checkpoints/%s/checkpoint_latest.pth' % (args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    temporary_save_path = save_path.with_suffix(save_path.suffix + '.tmp')
                    save_dict = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'step': total_steps, 'epoch': epoch}
                    torch.save(save_dict, temporary_save_path)
                    os.replace(temporary_save_path, save_path)

                if total_steps % args.save_ckpt_freq == 0:
                    save_path = Path('checkpoints/%s/%s_%6d.pth' % (args.name, args.name, total_steps))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(model_without_ddp.state_dict(), save_path)

                if total_steps % args.val_freq == 0:

                    # visualizing training results with tensorboard
                    disp = disp_predictions[-1]

                    for j in range(min(4, args.batch_size)):  # write a maxmimum of four images
                        logger.writer.add_image("image1/{}".format(j), image1[j].data.type(torch.uint8), total_steps)
                        logger.writer.add_image("image2/{}".format(j), image2[j].data.type(torch.uint8), total_steps)
                        logger.writer.add_image("disp/{}".format(j),
                                                (disp[j]).data.type(torch.uint8), total_steps)
                        logger.writer.add_image("gt_disp/{}".format(j),
                                                (disp_gt[j]).data.type(torch.uint8), total_steps)

                    results = validate_things(
                        model_without_ddp, args.valid_iters, args.scale_iters,
                        max_disp=args.eval_max_disp,
                        binmono_metrics=args.model_arch in ('cor_ga_fast_binmono', 'cor_ga_fast_binmono_corrstats'))
                    logger.write_dict(results)
                    model.train()
                    if not args.distributed: model_without_ddp.freeze_bn()

            if total_steps >= args.num_steps:
                should_keep_training = False
                break

        if accumulated_microbatches:
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
        epoch += 1

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%s/%d_epoch_%s.pth.gz' % (args.name, total_steps, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model_without_ddp.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model_without_ddp.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='defom-stereo', help="name your experiment")
    parser.add_argument('--model_arch', default='cor_ga_bins2', choices=['cor_ga_fast_binmono_corrstats', 'cor_ga_fast_binmono', 'cor_ga_fast_bininit', 'cor_ga_fast_pairres', 'cor_ga_no_mono_head', 'cor_ga_no_mono_igev', 'cor_ga_stereo_igev', 'cor_ga_fast3', 'cor_ga_fast2', 'cor_ga_fast', 'cor_ga', 'cor_ga_coarse1', 'cor_ga_lite', 'cor_ga_noh', 'cor_ga_bins', 'cor_ga_bins2', 'cor_ga_bins3', 'cor_ga_bins3_gwc', 'cor_ga_bins4', 'cor_ga_bins5', 'cor_ga_bins5_shared_parallel', 'cor_ga_bins6', 'cor_ga_bins_residual'], help='model architecture')

    # resume pretrained model or resume training
    parser.add_argument('--resume_ckpt', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='microbatches accumulated per optimizer update')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--train_folds', type=int, nargs='+', default=[1], help="training datasets' folds.")
    parser.add_argument('--lr', type=float, default=0.00002, help="max learning rate.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 768], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=18, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--scale_iters', type=int, default=8, help="number of coarse aggregation updates for cor_ga; ignored by cor_ga_bins")
    parser.add_argument('--coarse_loss_weight', type=float, default=0.2, help="weight of coarse cor_ga losses; ignored by cor_ga_bins")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default='fp16', help='autocast dtype used with --mixed_precision')
    parser.add_argument('--amp_init_scale', type=float, default=64.0, help='initial FP16 GradScaler scale; ignored without mixed precision and restored from full checkpoints')
    parser.add_argument('--debug_nan', action='store_true', help='enable targeted non-finite checks for NaN reproduction')
    parser.add_argument('--skip_nonfinite_gradients', action='store_true',
                        help='skip batches with non-finite gradients without stepping optimizer or scheduler')
    parser.add_argument('--large_grad_warn_threshold', type=float, default=1e6,
                        help='warn when the finite pre-clip gradient norm exceeds this value; set <= 0 to disable')
    parser.add_argument('--seed', default=1234565, type=int)

    # log
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--save_ckpt_freq', default=10000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--val_freq', default=10000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--max_disp', type=int, default=48, help="maximum disparity")
    parser.add_argument('--train_max_disp', type=float, default=768.0, help="maximum full-resolution GT disparity supervised by training losses")
    parser.add_argument('--eval_max_disp', type=float, default=768.0, help="maximum full-resolution GT disparity included in SceneFlow validation")
    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disparity field updates during validation forward pass')

    # Raft Architecure choices
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hypothesis_scales', type=float, nargs=3, default=[24.0, 48.0, 96.0],
                        help='three quarter-resolution scales applied to normalized monocular inverse depth; ignored by bin-initialized models')
    parser.add_argument('--num_init_bins', type=int, default=48, help='equal bin count used by all three disparity ranges')
    parser.add_argument('--init_max_offset', type=float, default=48.0, help='exclusive upper bound of bin offsets in each compressed right-feature coordinate')
    parser.add_argument('--init_corr_scale', type=float, default=10.0, help='scale applied to cosine matching logits')
    parser.add_argument('--init_gwc_groups', type=int, default=8, help='channel groups used by cor_ga_bins3_gwc initialization')
    parser.add_argument('--freeze_bininit_base', action='store_true', help='train only the new bin initializer/compressor in cor_ga_fast_bininit or cor_ga_no_mono_head')
    parser.add_argument('--freeze_binmono_base', action='store_true', help='train only the shared bin/mono confidence head in cor_ga_fast_binmono')
    parser.add_argument('--binmono_conf_hidden', type=int, default=32, help='hidden channels in the shared bin/mono confidence head')
    parser.add_argument('--binmono_init_stereo_weight', type=float, default=0.05, help='initial stereo-bin weight used by cor_ga_fast_binmono')
    parser.add_argument('--corr_stats_hidden', type=int, default=32, help='hidden channels in the refine correlation-statistics residual adapter')
    parser.add_argument('--corr_stats_temperature', type=float, default=1.0, help='softmax temperature after local correlation standardization')
    parser.add_argument('--init_gwc_hidden', type=int, default=32, help='hidden width of the shared per-bin GWC detector')
    parser.add_argument('--init_right_compression', choices=['avg', 'conv'], default='avg',
                        help='right-feature width compression used by cor_ga_bins bin matching')
    parser.add_argument('--refine_right_compression', choices=['avg', 'conv'], default='avg',
                        help='right-feature width compression used by cor_ga_bins GRU warp lookup')
    parser.add_argument('--loss_gamma', type=float, default=0.9, help='decay factor for successive recurrent L1 losses')
    parser.add_argument('--init_target_temperature', type=float, default=1.0, help='WAFT target temperature in full-resolution disparity pixels')
    parser.add_argument('--gate_loss_weight', type=float, default=0.1, help='weight of direct hypothesis-gate soft cross-entropy')
    parser.add_argument('--gate_target_temperature', type=float, default=4.0, help='hypothesis-gate target temperature in full-resolution disparity pixels')
    parser.add_argument('--smooth_l1_beta', type=float, default=1.0, help='SmoothL1 beta for fused d0 in full-resolution disparity pixels')
    parser.add_argument('--d0_loss_weight', type=float, default=1.0, help='weight of the directly supervised fused d0 SmoothL1 loss')
    parser.add_argument('--use_gru_edge_loss', action='store_true', help='apply GT-disparity edge weighting only to recurrent GRU predictions')
    parser.add_argument('--edge_mode', choices=['topk', 'threshold'], default='topk', help='method used to select GT disparity edges')
    parser.add_argument('--edge_topk', type=float, default=0.10, help='per-image fraction of strongest valid GT gradients selected in topk mode')
    parser.add_argument('--init_branch_loss_weight', type=float, default=0.1, help='bins3 weight for summed per-branch initial-disparity SmoothL1 losses')
    parser.add_argument('--edge_threshold', type=float, default=1.0, help='GT disparity-gradient threshold in pixels for threshold mode')
    parser.add_argument('--edge_dilation', type=int, default=5, help='edge-mask dilation kernel size in full-resolution pixels')
    parser.add_argument('--gru_edge_lambda', type=float, default=1.0, help='extra recurrent L1 weight assigned to selected edge pixels')
    parser.add_argument('--use_gru_disp_grad_loss', action='store_true', help='add final-GRU disparity-gradient consistency loss')
    parser.add_argument('--gru_grad_lambda', type=float, default=0.02, help='weight of the optional final-GRU disparity-gradient loss')
    parser.add_argument('--max_delta_disp', type=float, default=8.0, help='smooth per-iteration disparity-update limit in quarter-resolution pixels')
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the shared right-coordinate FullCorr pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="shared local radius for FullCorr and right-feature window sampling")
    parser.add_argument('--pair_feature_dim', type=int, default=8, help='candidate-pair token width used by cor_ga_fast_pairres')
    parser.add_argument('--pair_residual_init_std', type=float, default=1e-4, help='small non-zero initialization std for the pair residual output projection')

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2], help="fixed quarter-resolution disparity field")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, choices=[1, 2, 3], help="number of recurrent levels (1/4, 1/8, 1/16)")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128], help="quarter-resolution hidden state and context dimension")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0.0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default='v', choices=['v', 'None'], help='flip the images vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    Path("checkpoints/"+args.name).mkdir(exist_ok=True, parents=True)

    train(args)
