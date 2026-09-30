import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update_pru import MultiPromptUpdateBlock
from core.extractor import BasicEncoder, BasicEncoder2, DefomEncoder
from core.submodules import ParallelRightWidthCompressor
from core.mefa import (
    MultiRangeEpipolarFeatureAugmentation,
    MultiRangeFeatureFusionLookup,
)
from core.utils.utils import upflow, get_danv2_io_size


try:
    autocast = torch.cuda.amp.autocast
except Exception:
    class autocast:
        def __init__(self, enabled, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class MultiRangeBinInitializer(nn.Module):
    """Estimate three initial disparities with equal-width bins over 1/2/4x ranges."""

    def __init__(self, feature_dim=32, num_bins=48, max_offset=48.0, corr_scale=10.0):
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        if max_offset <= 0:
            raise ValueError(f"max_offset must be positive, got {max_offset}")
        self.num_bins = int(num_bins)
        self.corr_scale = float(corr_scale)
        self.compression_ratios = (1, 2, 4)
        bin_width = float(max_offset) / self.num_bins
        offsets = torch.arange(self.num_bins, dtype=torch.float32) * bin_width
        ratios = torch.tensor(self.compression_ratios, dtype=torch.float32)
        self.bin_width = bin_width
        self.register_buffer('bin_offsets', offsets, persistent=False)
        self.register_buffer('bin_disparities', ratios[:, None] * offsets[None, :], persistent=False)
        self.logit_refiners = nn.ModuleList([nn.Sequential(nn.Conv2d(self.num_bins + 1, feature_dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(feature_dim, self.num_bins, 3, padding=1)) for _ in self.compression_ratios])
        for refiner in self.logit_refiners:
            nn.init.zeros_(refiner[-1].weight)
            nn.init.zeros_(refiner[-1].bias)

    @staticmethod
    def _require_finite(name, tensor, branch_index=None):
        if torch.isfinite(tensor).all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        branch = "" if branch_index is None else f" in branch {branch_index}"
        raise FloatingPointError(f"Non-finite {name}{branch}: nan={nan_count}, inf={inf_count}")

    def _match_logits(self, left_feature, right_feature, ratio):
        batch, channels, height, width = left_feature.shape
        if right_feature.shape[:3] != (batch, channels, height):
            raise ValueError(f"Incompatible left/right feature shapes: {tuple(left_feature.shape)} and {tuple(right_feature.shape)}")
        right_width = right_feature.shape[-1]
        with autocast(enabled=False):
            left = F.normalize(left_feature.float(), p=2, dim=1, eps=1e-6)
            right = F.normalize(right_feature.float(), p=2, dim=1, eps=1e-6)
            all_pairs = torch.einsum('bchw,bchv->bhwv', left, right)
            x_left = torch.arange(width, device=left.device, dtype=torch.float32).view(1, 1, width, 1)
            offsets = self.bin_offsets.view(1, 1, 1, self.num_bins)
            center_offset = (float(ratio) - 1.0) / 2.0
            sample_x = (x_left - center_offset) / float(ratio) - offsets
            valid = (sample_x >= 0.0) & (sample_x <= float(right_width - 1))
            sample_x = sample_x.clamp(0.0, float(max(right_width - 1, 0)))
            index0 = sample_x.floor().long()
            index1 = (index0 + 1).clamp(max=max(right_width - 1, 0))
            alpha = sample_x - index0.float()
            gather_shape = (batch, height, width, self.num_bins)
            value0 = torch.gather(all_pairs, dim=-1, index=index0.expand(gather_shape))
            value1 = torch.gather(all_pairs, dim=-1, index=index1.expand(gather_shape))
            logits = ((1.0 - alpha) * value0 + alpha * value1).permute(0, 3, 1, 2).contiguous()
        return logits, valid.permute(0, 3, 1, 2).expand(batch, self.num_bins, height, width)

    def forward(self, left_feature, right_feature_pyramid, mono_idepth):
        if len(right_feature_pyramid) != len(self.compression_ratios):
            raise ValueError(f"Expected {len(self.compression_ratios)} right features, got {len(right_feature_pyramid)}")
        self._require_finite("left_feature", left_feature)
        self._require_finite("mono_idepth", mono_idepth)
        branch_logits = []
        branch_confidences = []
        branch_disps = []
        branch_valid = []
        # Keep correlation refinement, softmax, and expectation in FP32. These
        # operations are inexpensive at quarter resolution and sensitive to
        # FP16 overflow because one non-finite logit poisons the whole bin PDF.
        with autocast(enabled=False):
            for branch_index, (right_feature, ratio) in enumerate(zip(right_feature_pyramid, self.compression_ratios)):
                self._require_finite("right_feature", right_feature, branch_index)
                raw_logits, valid = self._match_logits(left_feature, right_feature, ratio)
                self._require_finite("raw_logits", raw_logits, branch_index)
                refiner_input = torch.cat([raw_logits.masked_fill(~valid, -1.0), mono_idepth.float()], dim=1)
                refiner_delta = self.logit_refiners[branch_index](refiner_input.float())
                self._require_finite("refiner_delta", refiner_delta, branch_index)
                logits = self.corr_scale * raw_logits + refiner_delta
                logits = logits.float().masked_fill(~valid, -1e4)
                self._require_finite("masked_logits", logits, branch_index)
                probability = torch.softmax(logits, dim=1) * valid.float()
                probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-6)
                self._require_finite("probability", probability, branch_index)
                bins = self.bin_disparities[branch_index].view(1, self.num_bins, 1, 1)
                branch_disp = torch.sum(probability * bins, dim=1, keepdim=True)
                self._require_finite("branch_disp", branch_disp, branch_index)
                branch_logits.append(logits)
                branch_confidences.append(probability.amax(dim=1))
                branch_disps.append(branch_disp)
                branch_valid.append(valid)
        return branch_disps, torch.stack(branch_logits, dim=1), torch.stack(branch_confidences, dim=1), torch.stack(branch_valid, dim=1)


class DEFOMStereo(nn.Module):
    """CorGA with multi-range WAFT-style discrete disparity initialization.
    1、特征升维度
    2、左右图像特征增强
    #"""
    def __init__(self, args):
        super().__init__()
        self.init_right_compression = getattr(args, 'init_right_compression', 'avg')
        self.refine_right_compression = getattr(args, 'refine_right_compression', 'avg')
        valid_compression_modes = ('avg', 'conv')
        if self.init_right_compression not in valid_compression_modes:
            raise ValueError(
                f"init_right_compression must be one of {valid_compression_modes}, "
                f"got {self.init_right_compression}"
            )
        if self.refine_right_compression not in valid_compression_modes:
            raise ValueError(
                f"refine_right_compression must be one of {valid_compression_modes}, "
                f"got {self.refine_right_compression}"
            )

        # Keep this historical module name so the bin-initialization compressor
        # retains stable parameter keys. The MEFA variant has its own checkpoint
        # namespace because its refinement modules intentionally differ from
        # earlier bins4 experiments.
        self.args = args
        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)
        self.corr_radius = int(args.corr_radius)
        self.register_buffer('warp_offsets', torch.arange(-self.corr_radius, self.corr_radius + 1, dtype=torch.float32))
        low_feature_dim = 32
        hidden_dim = args.hidden_dims[0]
        window_size = 2 * self.corr_radius + 1
        self.defomencoder = DefomEncoder(args.dinov2_encoder)
        self.low_channel = nn.Conv2d(64, low_feature_dim, kernel_size=1)
        self.right_width_compressor = ParallelRightWidthCompressor(low_feature_dim, low_feature_dim, mode=self.init_right_compression)
        self.refine_feature_augmentation = MultiRangeEpipolarFeatureAugmentation(
            channels=low_feature_dim,
            num_heads=4,
            horizontal_kernel=9,
            layerscale_init=1e-2,
        )
        self.refine_right_lookup = MultiRangeFeatureFusionLookup(
            compression_ratios=(1, 2, 4),
            channels=low_feature_dim,
            samples_per_scale=window_size,
            hidden_channels=64,
            output_channels=128,
        )
        self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=64, norm_fn='instance', downsample=2)
        self.cnet = BasicEncoder2(self.defomencoder.out_dim, output_dim=[2 * hidden_dim, hidden_dim, hidden_dim], norm_fn=args.context_norm, downsample=2)
        update_kwargs = dict(args=args, hidden_dim=hidden_dim, feat_dim=low_feature_dim, volume_dim=1, warp_feat_dim=128, cor_planes=0)
        self.refine_update_block = MultiPromptUpdateBlock(**update_kwargs)
        if self.refine_right_compression == self.init_right_compression:
            self.refine_right_width_compressor = None
        else:
            self.refine_right_width_compressor = ParallelRightWidthCompressor(
                low_feature_dim,
                low_feature_dim,
                mode=self.refine_right_compression,
            )
        self.bin_initializer = MultiRangeBinInitializer(feature_dim=32, num_bins=getattr(args, 'num_init_bins', 48), max_offset=getattr(args, 'init_max_offset', 48.0), corr_scale=getattr(args, 'init_corr_scale', 10.0))
        self.num_init_ranges = len(self.bin_initializer.compression_ratios)
        self.max_delta_disp = float(getattr(args, 'max_delta_disp', 8.0))
        hidden_dim = args.hidden_dims[0]
        aggregation_channels = hidden_dim + 32 + 1 + 2 * self.num_init_ranges
        self.initial_weight_head = nn.Sequential(nn.Conv2d(aggregation_channels, hidden_dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, 2 * self.num_init_ranges, 1))
        nn.init.zeros_(self.initial_weight_head[-1].weight)
        nn.init.zeros_(self.initial_weight_head[-1].bias)
        self._freeze_structurally_unused_parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def upsample_flow(self, flow, mask):
        """Upsample a quarter-resolution disparity with a convex mask."""
        batch, channels, height, width = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(batch, 1, 9, factor, factor, height, width)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(batch, channels, 9, 1, 1, height, width)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(batch, channels, factor * height, factor * width)

    def _upsample_prediction(self, disp, mask):
        if mask is None:
            return upflow(disp, factor=2 ** self.args.n_downsample)
        return self.upsample_flow(disp, mask)

    def _bound_disparity(self, disp):
        max_disp = float(getattr(self.args, 'train_max_disp', 768.0)) / (2 ** self.args.n_downsample)
        return disp.float().clamp(min=0.0, max=max_disp)

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp, scale_weights):
        return self.refine_right_lookup(
            fmap1_low,
            right_feature_pyramid,
            disp,
            warp_offsets=self.warp_offsets,
            scale_weights=scale_weights,
        )

    def _freeze_structurally_unused_parameters(self):
        for norm_name in ('norm2', 'norm3'):
            norm = getattr(self.cnet.convd, norm_name, None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad = False
        unused_depth_residual = self.defomencoder.depth_anything.depth_feat.scratch.refinenet4.resConfUnit1
        for parameter in unused_depth_residual.parameters():
            parameter.requires_grad = False

    def _limit_delta_disparity(self, delta_disp):
        return self.max_delta_disp * torch.tanh(delta_disp.float() / self.max_delta_disp)

    def _debug_require_finite(self, name, tensor, iteration):
        if getattr(self.args, 'debug_nan', False):
            self.bin_initializer._require_finite(f"{name} at refinement iteration {iteration}", tensor)

    def _build_right_feature_pyramids(self, fmap2_low):
        """Build independently selectable pyramids for initialization/refinement."""
        init_half, init_quarter = self.right_width_compressor(fmap2_low)
        init_pyramid = (fmap2_low, init_half, init_quarter)

        if self.refine_right_width_compressor is None:
            refine_pyramid = init_pyramid
        else:
            refine_half, refine_quarter = self.refine_right_width_compressor(fmap2_low)
            refine_pyramid = (fmap2_low, refine_half, refine_quarter)

        return init_pyramid, refine_pyramid

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False, return_aux=False):
        _, _, image_height, image_width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        danv2_io_sizes = get_danv2_io_size(image_height, image_width, self.args.n_downsample)
        dfeat1_4, dfeat2_4, mono_idepth1_4 = self.defomencoder(image1, image2, danv2_io_sizes)
        fmap1_4, fmap2_4 = self.fnet([image1, image2], [dfeat1_4, dfeat2_4])
        with autocast(enabled=False):
            context_pyramid = self.cnet(image1.float(), dfeat1_4.float(), num_layers=self.args.n_gru_layers)
            net_4, inp_4 = torch.chunk(context_pyramid[0], chunks=2, dim=1)
            initial_nets = [torch.tanh(net_4)]
            initial_nets.extend(torch.tanh(context) for context in context_pyramid[1:])
            inp_4 = torch.relu(inp_4)
        for state_index, state in enumerate(initial_nets):
            self._debug_require_finite(
                f"initial_hidden[{state_index}]", state, -1
            )
        self._debug_require_finite("inp_4", inp_4, -1)
        with autocast(enabled=self.args.mixed_precision):
            fmap1_low = self.low_channel(fmap1_4)
            fmap2_low = self.low_channel(fmap2_4)
            init_right_feature_pyramid, refine_right_feature_pyramid = self._build_right_feature_pyramids(fmap2_low)
        with autocast(enabled=self.args.mixed_precision):
            init_disps, init_logits, init_confidence, init_valid = self.bin_initializer(
                fmap1_low,
                init_right_feature_pyramid,
                mono_idepth1_4,
            )
            init_disps = [self._bound_disparity(disp) for disp in init_disps]
            aggregation_input = torch.cat([inp_4, fmap1_low, mono_idepth1_4.float()] + init_disps + [init_confidence], dim=1)
            self._debug_require_finite(
                "initial_weight_head input", aggregation_input, -1
            )
            # The gate head is small, while a single FP16 overflow here
            # invalidates both the hypothesis and scale softmaxes and then all
            # recurrent predictions. Keep this numerically sensitive head in
            # FP32 even when the feature encoders use mixed precision.
            with autocast(enabled=False):
                weight_logits = self.initial_weight_head(
                    aggregation_input.float()
                )
            self._debug_require_finite(
                "initial_weight_head output", weight_logits, -1
            )
            hypothesis_logits = weight_logits[:, :self.num_init_ranges].float()
            hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
            scale_weights = torch.softmax(weight_logits[:, self.num_init_ranges:].float(), dim=1)
            # Hidden states are fed back for all refinement iterations. Keeping
            # them in FP32 prevents one transient FP16 overflow from poisoning
            # every later prediction; the feature encoders remain under AMP.
            refine_nets = [state.float() for state in initial_nets]
            inp_4 = inp_4.float()
        disp = sum(hypothesis_weights[:, index:index + 1] * init_disp.float() for index, init_disp in enumerate(init_disps))
        disp = self._bound_disparity(disp)
        del (
            image1,
            image2,
            dfeat1_4,
            dfeat2_4,
            mono_idepth1_4,
            fmap1_4,
            fmap2_4,
            fmap2_low,
            init_right_feature_pyramid,
            init_confidence,
            context_pyramid,
            net_4,
            initial_nets,
            aggregation_input,
            weight_logits,
            danv2_io_sizes,
        )

        # MEFA is unary feature augmentation: execute it once and cache its
        # outputs. The recurrent loop below only performs disparity-dependent
        # sampling and left-right matching on these cached descriptors.
        with autocast(enabled=self.args.mixed_precision):
            fmap1_low, right_1_aug, right_2_aug, right_4_aug = (
                self.refine_feature_augmentation(
                    fmap1_low,
                    refine_right_feature_pyramid[0],
                    refine_right_feature_pyramid[1],
                    refine_right_feature_pyramid[2],
                )
            )
            refine_right_feature_pyramid = (
                right_1_aug,
                right_2_aug,
                right_4_aug,
            )
        del right_1_aug, right_2_aug, right_4_aug

        d0 = disp
        disp_predictions = [upflow(d0, factor=2 ** self.args.n_downsample)]
        disp_up = disp_predictions[0]
        for iteration in range(iters):
            disp = disp.detach()
            with autocast(enabled=self.args.mixed_precision):
                warp_feature = self._refine_warp_feature(
                    fmap1_low,
                    refine_right_feature_pyramid,
                    disp,
                    scale_weights,
                )
            # self._debug_require_finite("warp_feature", warp_feature, iteration)
            with autocast(enabled=False):
                refine_nets, delta_disp, up_mask = self.refine_update_block(
                    refine_nets, None, disp.float(), inp_4,
                    warp_feature.float(), d0.float())
            delta_disp = self._limit_delta_disparity(delta_disp)
            disp = self._bound_disparity(disp + delta_disp)
            del delta_disp
            #self._debug_require_finite("disp", disp, iteration)
            if test_mode and iteration < iters - 1:
                del up_mask
                continue
            disp_up = self._upsample_prediction(disp, up_mask)
            del up_mask
            #self._debug_require_finite("disp_up", disp_up, iteration)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up
        if return_aux:
            return disp_predictions, {'w_h': hypothesis_weights, 'w_h_logits': hypothesis_logits, 'w_s': scale_weights, 'init_disps': torch.cat(init_disps, dim=1), 'init_logits': init_logits, 'init_bins': self.bin_initializer.bin_disparities, 'init_valid': init_valid}
        return disp_predictions
