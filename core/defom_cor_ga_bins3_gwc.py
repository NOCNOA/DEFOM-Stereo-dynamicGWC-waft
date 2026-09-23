"""Bins3 with a shared group-wise-correlation bin detector.

Only the multi-range disparity initializer is changed.  The three initial
disparities are still fused by bins3's original w_h gate, and the refinement
FullCorr/GRU path is kept unchanged.
"""

import torch
import torch.nn as nn

from core.defom_cor_ga_bins3 import DEFOMStereo as Bins3Stereo, autocast
from core.submodules import ParallelRightWidthCompressor
from core.utils.utils import upflow, get_danv2_io_size


class SharedScaleGWCInitializer(nn.Module):
    """Predict per-scale bin logits from shared group-wise correlations.

    Conceptual group volume shape is [B, S, G, D, H, W].  Scales are processed
    sequentially to avoid materializing that complete volume, while the same
    detector weights are applied to every scale.
    """

    def __init__(self, feature_dim=64, num_groups=8, detector_dim=32,
                 num_bins=48, max_offset=48.0, corr_scale=10.0):
        super().__init__()
        if feature_dim % num_groups != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_groups ({num_groups})"
            )
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        if max_offset <= 0:
            raise ValueError(f"max_offset must be positive, got {max_offset}")
        if detector_dim < 1:
            raise ValueError(f"detector_dim must be positive, got {detector_dim}")

        self.feature_dim = int(feature_dim)
        self.num_groups = int(num_groups)
        self.channels_per_group = self.feature_dim // self.num_groups
        self.num_bins = int(num_bins)
        self.corr_scale = float(corr_scale)
        self.compression_ratios = (1, 2, 4)

        bin_width = float(max_offset) / self.num_bins
        offsets = torch.arange(self.num_bins, dtype=torch.float32) * bin_width
        ratios = torch.tensor(self.compression_ratios, dtype=torch.float32)
        self.bin_width = bin_width
        self.register_buffer("bin_offsets", offsets, persistent=False)
        self.register_buffer(
            "bin_disparities", ratios[:, None] * offsets[None, :], persistent=False
        )

        # Conv3d with kernel 1 is an MLP applied independently at every
        # (disparity-bin, y, x) location.  The module is shared by all scales.
        self.bin_detector = nn.Sequential(
            nn.Conv3d(self.num_groups, detector_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(detector_dim, 1, kernel_size=1),
        )

    @staticmethod
    def _require_finite(name, tensor, branch_index=None):
        if torch.isfinite(tensor).all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        branch = "" if branch_index is None else f" in branch {branch_index}"
        raise FloatingPointError(
            f"Non-finite {name}{branch}: nan={nan_count}, inf={inf_count}"
        )

    def _groupwise_bin_correlation(self, left_feature, right_feature, ratio):
        batch, channels, height, width = left_feature.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} input channels, got {channels}"
            )
        if right_feature.shape[:3] != (batch, channels, height):
            raise ValueError(
                f"Incompatible left/right feature shapes: "
                f"{tuple(left_feature.shape)} and {tuple(right_feature.shape)}"
            )

        right_width = right_feature.shape[-1]
        left = left_feature.float().reshape(
            batch, self.num_groups, self.channels_per_group, height, width
        )
        right = right_feature.float().reshape(
            batch, self.num_groups, self.channels_per_group, height, right_width
        )

        # [B, G, H, W_left, W_right], equivalent to a GWC volume before
        # selecting the discrete disparity bins.
        all_pairs = torch.einsum("bgchw,bgchv->bghwv", left, right)
        all_pairs = all_pairs / float(self.channels_per_group)

        x_left = torch.arange(
            width, device=left_feature.device, dtype=torch.float32
        ).view(1, 1, width, 1)
        offsets = self.bin_offsets.view(1, 1, 1, self.num_bins)
        center_offset = (float(ratio) - 1.0) / 2.0
        sample_x = (x_left - center_offset) / float(ratio) - offsets
        valid = (sample_x >= 0.0) & (sample_x <= float(right_width - 1))
        sample_x = sample_x.clamp(0.0, float(max(right_width - 1, 0)))

        index0 = sample_x.floor().long()
        index1 = (index0 + 1).clamp(max=max(right_width - 1, 0))
        alpha = sample_x - index0.float()
        gather_shape = (batch, self.num_groups, height, width, self.num_bins)
        index0 = index0.view(1, 1, 1, width, self.num_bins).expand(gather_shape)
        index1 = index1.view(1, 1, 1, width, self.num_bins).expand(gather_shape)
        alpha = alpha.view(1, 1, 1, width, self.num_bins)
        value0 = torch.gather(all_pairs, dim=-1, index=index0)
        value1 = torch.gather(all_pairs, dim=-1, index=index1)
        group_corr = ((1.0 - alpha) * value0 + alpha * value1)
        group_corr = group_corr.permute(0, 1, 4, 2, 3).contiguous()

        valid = valid.permute(0, 3, 1, 2).expand(
            batch, self.num_bins, height, width
        )
        return group_corr, valid

    def forward(self, left_feature, right_feature_pyramid, mono_idepth):
        if len(right_feature_pyramid) != len(self.compression_ratios):
            raise ValueError(
                f"Expected {len(self.compression_ratios)} right features, "
                f"got {len(right_feature_pyramid)}"
            )
        self._require_finite("left_feature", left_feature)
        self._require_finite("mono_idepth", mono_idepth)

        branch_logits = []
        branch_confidences = []
        branch_disps = []
        branch_valid = []
        # GWC, the shared detector, softmax and expectation stay in FP32.
        with autocast(enabled=False):
            for branch_index, (right_feature, ratio) in enumerate(
                zip(right_feature_pyramid, self.compression_ratios)
            ):
                self._require_finite("right_feature", right_feature, branch_index)
                group_corr, valid = self._groupwise_bin_correlation(
                    left_feature, right_feature, ratio
                )
                self._require_finite("group_corr", group_corr, branch_index)
                logits = self.bin_detector(self.corr_scale * group_corr).squeeze(1)
                logits = logits.masked_fill(~valid, -1e4)
                self._require_finite("masked_logits", logits, branch_index)

                probability = torch.softmax(logits, dim=1) * valid.float()
                probability = probability / probability.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
                bins = self.bin_disparities[branch_index].view(
                    1, self.num_bins, 1, 1
                )
                branch_disp = torch.sum(probability * bins, dim=1, keepdim=True)
                self._require_finite("branch_disp", branch_disp, branch_index)

                branch_logits.append(logits)
                branch_confidences.append(probability.amax(dim=1))
                branch_disps.append(branch_disp)
                branch_valid.append(valid)

        return (
            branch_disps,
            torch.stack(branch_logits, dim=1),
            torch.stack(branch_confidences, dim=1),
            torch.stack(branch_valid, dim=1),
        )


class DEFOMStereo(Bins3Stereo):
    """Bins3 whose three bin logits come from one shared 64-channel GWC detector."""

    def __init__(self, args):
        super().__init__(args)
        self.bin_initializer = SharedScaleGWCInitializer(
            feature_dim=64,
            num_groups=getattr(args, "init_gwc_groups", 8),
            detector_dim=getattr(args, "init_gwc_hidden", 32),
            num_bins=getattr(args, "num_init_bins", 48),
            max_offset=getattr(args, "init_max_offset", 48.0),
            corr_scale=getattr(args, "init_corr_scale", 10.0),
        )
        self.gwc_right_width_compressor = ParallelRightWidthCompressor(
            64, 64, mode=self.init_right_compression
        )

    def forward(self, image1, image2, iters=12, scale_iters=3,
                test_mode=False, return_aux=False):
        _, _, image_height, image_width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        danv2_io_sizes = get_danv2_io_size(
            image_height, image_width, self.args.n_downsample
        )
        dfeat1_4, dfeat2_4, mono_idepth1_4 = self.defomencoder(
            image1, image2, danv2_io_sizes
        )
        fmap1_4, fmap2_4 = self.fnet(
            [image1, image2], [dfeat1_4, dfeat2_4]
        )

        with autocast(enabled=False):
            context_pyramid = self.cnet(
                image1.float(), dfeat1_4.float(), num_layers=self.args.n_gru_layers
            )
            net_4, inp_4 = torch.chunk(context_pyramid[0], chunks=2, dim=1)
            initial_nets = [torch.tanh(net_4)]
            initial_nets.extend(torch.tanh(context) for context in context_pyramid[1:])
            inp_4 = torch.relu(inp_4)

        with autocast(enabled=self.args.mixed_precision):
            fmap1_low = self.low_channel(fmap1_4)
            fmap2_low = self.low_channel(fmap2_4)
            _, refine_right_feature_pyramid = self._build_right_feature_pyramids(
                fmap2_low
            )
            gwc_half, gwc_quarter = self.gwc_right_width_compressor(fmap2_4)
            gwc_right_feature_pyramid = (fmap2_4, gwc_half, gwc_quarter)

        with autocast(enabled=False):
            full_corr = self._build_full_corr(fmap1_4, fmap2_4)
            init_disps, init_logits, init_confidence, init_valid = self.bin_initializer(
                fmap1_4, gwc_right_feature_pyramid, mono_idepth1_4
            )
            init_disps = [self._bound_disparity(disp) for disp in init_disps]
            aggregation_input = torch.cat(
                [inp_4, fmap1_low.float(), mono_idepth1_4.float()]
                + init_disps + [init_confidence], dim=1
            )
            weight_logits = self.initial_weight_head(aggregation_input.float())
            hypothesis_logits = weight_logits[:, :self.num_init_ranges]
            hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
            scale_weights = torch.softmax(
                weight_logits[:, self.num_init_ranges:], dim=1
            )
            refine_nets = [state.float() for state in initial_nets]
            inp_4 = inp_4.float()

        self._debug_require_finite("weight_logits", weight_logits, -1)
        self._debug_require_finite("hypothesis_weights", hypothesis_weights, -1)
        self._debug_require_finite("scale_weights", scale_weights, -1)
        disp = sum(
            hypothesis_weights[:, index:index + 1] * init_disp.float()
            for index, init_disp in enumerate(init_disps)
        )
        disp = self._bound_disparity(disp)
        self._debug_require_finite("initial_disp", disp, -1)

        del (
            image1, image2, dfeat1_4, dfeat2_4, fmap1_4, fmap2_4,
            context_pyramid, net_4, aggregation_input, weight_logits,
            gwc_right_feature_pyramid, gwc_half, gwc_quarter,
        )

        d0 = disp
        disp_predictions = [upflow(d0, factor=2 ** self.args.n_downsample)]
        disp_up = disp_predictions[0]
        for iteration in range(iters):
            disp = disp.detach()
            with autocast(enabled=self.args.mixed_precision):
                corr = full_corr(disp)
                warp_feature = self._refine_warp_feature(
                    fmap1_low, refine_right_feature_pyramid, disp, scale_weights
                )
            self._debug_require_finite("corr", corr, iteration)
            self._debug_require_finite("warp_feature", warp_feature, iteration)
            with autocast(enabled=False):
                refine_nets, delta_disp, up_mask = self.refine_update_block(
                    refine_nets, corr.float(), disp.float(), inp_4,
                    warp_feature.float(), d0.float()
                )
            delta_disp = self._limit_delta_disparity(delta_disp)
            disp = self._bound_disparity(disp + delta_disp)
            self._debug_require_finite("disp", disp, iteration)
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self._upsample_prediction(disp, up_mask)
            self._debug_require_finite("disp_up", disp_up, iteration)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up
        if return_aux:
            return disp_predictions, {
                "w_h": hypothesis_weights,
                "w_h_logits": hypothesis_logits,
                "w_s": scale_weights,
                "init_disps": torch.cat(init_disps, dim=1),
                "init_logits": init_logits,
                "init_bins": self.bin_initializer.bin_disparities,
                "init_valid": init_valid,
            }
        return disp_predictions
