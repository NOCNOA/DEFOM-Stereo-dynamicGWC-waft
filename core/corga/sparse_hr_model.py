"""Frozen CorGA plus an independently trainable sparse HR recurrent refiner."""

import pickle
from types import SimpleNamespace

import torch
import torch.nn as nn

from core.corga.defom_cor_ga_fast import DEFOMStereo
from core.corga.sparse_hr_refiner import (
    SparseHRRefiner, dense_query_coordinates, dilate, disparity_boundary,
    scatter_queries,
)
from core.utils.utils import InputPadder


def default_corga_args(**overrides):
    """Match the existing ViT-L two-stage 200k evaluation configuration."""
    values = dict(dinov2_encoder="vitl", corr_levels=3, corr_radius=4,
                  hidden_dims=[128], n_gru_layers=3, n_downsample=2,
                  context_norm="batch", hypothesis_scales=[24., 48., 96.],
                  mixed_precision=True, amp_dtype="bf16", max_delta_disp=8.,
                  max_coarse_delta_disp=None, disable_delta_clamp=True,
                  disable_coarse_delta_clamp=False, train_max_disp=768.)
    values.update(overrides)
    return SimpleNamespace(**values)


def load_corga_checkpoint(model, path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)


class SparseHRCorGA(nn.Module):
    def __init__(self, checkpoint, coarse_iters=6, refine_iters=32,
                 hr_iters=4, feature_channels=16, hidden_channels=64,
                 context_channels=64,
                 delta_radius=2., disable_delta_clamp=True,
                 gate_init_bias=-2., direction_classes=False,
                 candidate_guided=False, context_scoring=False):
        super().__init__()
        self.coarse = DEFOMStereo(default_corga_args(
            disable_delta_clamp=disable_delta_clamp))
        load_corga_checkpoint(self.coarse, checkpoint)
        self.coarse.requires_grad_(False)
        self.coarse.eval()
        self.coarse_iters = int(coarse_iters)
        self.refine_iters = int(refine_iters)
        self.refiner = SparseHRRefiner(
            coarse_hidden_channels=128, coarse_context_channels=128,
            feature_channels=feature_channels, hidden_channels=hidden_channels,
            context_channels=context_channels,
            iterations=hr_iters, delta_radius=delta_radius,
            gate_init_bias=gate_init_bias,
            direction_classes=direction_classes,
            candidate_guided=candidate_guided, context_scoring=context_scoring)

    def train(self, mode=True):
        super().train(mode)
        self.coarse.eval()
        return self

    @torch.no_grad()
    def _coarse_state(self, left, right):
        padder = InputPadder(left.shape, divis_by=32)
        left_pad, right_pad = padder.pad(left, right)
        full_pad, state = self.coarse(
            left_pad, right_pad, iters=self.refine_iters,
            scale_iters=self.coarse_iters, test_mode=True, return_hr_state=True)
        return padder, left_pad, right_pad, full_pad, state

    def forward_queries(self, left, right, query_xy,
                        return_direction_logits=False,
                        return_candidate_info=False):
        """Train/evaluate arbitrary integer queries; return dense coarse and HR sequence."""
        padder, left_pad, right_pad, full_pad, state = self._coarse_state(left, right)
        pad_left, _, pad_top, _ = padder._pad
        xy_pad = query_xy + query_xy.new_tensor((pad_left, pad_top))
        predictions = self.refiner(
            left_pad, right_pad, full_pad, state["disp_lr_px"],
            state["hidden_lr"], state["context_lr"], xy_pad,
            return_direction_logits=return_direction_logits,
            return_candidate_info=return_candidate_info)
        return padder.unpad(full_pad), predictions

    @torch.no_grad()
    def infer(self, left, right, mode="predicted_boundary", gt=None,
              valid=None, band_radius=4, boundary_threshold=2., chunk_size=4096):
        """Return dense HR predictions for steps 0..T and the selected query mask."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        padder, left_pad, right_pad, full_pad, state = self._coarse_state(left, right)
        coarse_full = padder.unpad(full_pad)
        if mode == "all":
            query_mask = torch.ones_like(coarse_full, dtype=torch.bool)
        elif mode == "oracle_boundary":
            if gt is None or valid is None:
                raise ValueError("oracle_boundary requires GT and valid mask")
            query_mask = dilate(disparity_boundary(gt, valid.bool(), boundary_threshold),
                                band_radius) & valid.bool()
        elif mode == "oracle_correctable":
            if gt is None or valid is None:
                raise ValueError("oracle_correctable requires GT and valid mask")
            edge_band = dilate(disparity_boundary(gt, valid.bool(), boundary_threshold),
                               band_radius) & valid.bool()
            initial_error = (coarse_full - gt).abs()
            query_mask = edge_band & (initial_error >= 1.) & (initial_error <= 8.)
        elif mode == "predicted_boundary":
            finite = torch.ones_like(coarse_full, dtype=torch.bool)
            query_mask = dilate(disparity_boundary(coarse_full, finite,
                                                  boundary_threshold), band_radius)
        else:
            raise ValueError(f"Unknown query mode: {mode}")
        left_feature = self.refiner.feature(left_pad)
        right_feature = self.refiner.feature(right_pad)
        pad_left, _, pad_top, _ = padder._pad
        outputs = []
        for batch_index, xy_batch in enumerate(dense_query_coordinates(query_mask)):
            xy = xy_batch.to(left.device)
            batch_coarse = coarse_full[batch_index:batch_index + 1]
            predictions = [batch_coarse.clone() for _ in range(self.refiner.iterations + 1)]
            for start in range(0, xy.shape[1], chunk_size):
                query = xy[:, start:start + chunk_size]
                xy_pad = query + query.new_tensor((pad_left, pad_top))
                sparse = self.refiner(
                    left_pad[batch_index:batch_index + 1],
                    right_pad[batch_index:batch_index + 1],
                    full_pad[batch_index:batch_index + 1],
                    state["disp_lr_px"][batch_index:batch_index + 1],
                    state["hidden_lr"][batch_index:batch_index + 1],
                    state["context_lr"][batch_index:batch_index + 1],
                    xy_pad, feature_left=left_feature[batch_index:batch_index + 1],
                    feature_right=right_feature[batch_index:batch_index + 1])
                for iteration in range(1, len(sparse)):
                    predictions[iteration] = scatter_queries(predictions[iteration],
                                                              query, sparse[iteration])
            outputs.append(predictions)
        return [torch.cat([outputs[batch][iteration] for batch in range(len(outputs))])
                for iteration in range(self.refiner.iterations + 1)], query_mask
