"""Standalone CorGA-fast2 with horizontal-only correlation and feature sampling.

Parameter names and execution behavior remain compatible with CorGA-fast.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr import CorrBlock1D1
from core.update_pru import MultiPromptUpdateBlock
from core.extractor import BasicEncoder, BasicEncoder2, DefomEncoder
from core.galerkin import simple_attn_rope_2d
from core.utils.utils import coords_grid, upflow, get_danv2_io_size
from core.submodules import ParallelRightWidthCompressor

autocast = torch.cuda.amp.autocast


from core.horizontal_sampler import (
    HorizontalCoordinateCache,
    HorizontalCorrBlock1D,
    lookup_right_features_horizontal,
)


class DEFOMStereo(nn.Module):
    """Independent fast2 implementation with three coarse PRUs and refinement."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        amp_dtype_name = getattr(args, 'amp_dtype', 'fp16')
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.defomencoder = DefomEncoder(
            args.dinov2_encoder, use_mono_head=True
        )
        self.corr_levels = int(args.corr_levels)
        self.corr_radius = int(args.corr_radius)

        hypothesis_scales = tuple(float(x) for x in args.hypothesis_scales)
        self.register_buffer('hypothesis_scales', torch.tensor(hypothesis_scales, dtype=torch.float32).view(3, 1, 1, 1))
        self.register_buffer('warp_offsets', torch.arange(-self.corr_radius, self.corr_radius + 1, dtype=torch.float32))

        low_feature_dim = 32
        hidden_dim = args.hidden_dims[0]
        window_size = 2 * self.corr_radius + 1
        fusion_in_channels = low_feature_dim * (window_size + 1)
        corr_channels = self.corr_levels * window_size

        self.low_channel = nn.Conv2d(64, low_feature_dim, kernel_size=1)
        self.right_width_compressor = ParallelRightWidthCompressor(low_feature_dim, low_feature_dim, mode="conv")
        self.coarse_right_fuse = self._make_right_fuser(fusion_in_channels)
        self.coarse_rope = simple_attn_rope_2d(low_feature_dim, 4)
        self.refine_right_fuse = self._make_right_fuser(fusion_in_channels)
        self.refine_rope = simple_attn_rope_2d(low_feature_dim, 4)

        self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=64, norm_fn='instance', downsample=2)
        self.cnet = BasicEncoder2(self.defomencoder.out_dim, output_dim=[2 * hidden_dim, hidden_dim, hidden_dim], norm_fn=args.context_norm, downsample=2)

        update_kwargs = dict(args=self.args, hidden_dim=hidden_dim, feat_dim=low_feature_dim, volume_dim=1, warp_feat_dim=low_feature_dim, cor_planes=corr_channels)
        self.coarse_gru_layers = args.n_gru_layers
        self.coarse_update_blocks = nn.ModuleList([
            MultiPromptUpdateBlock(
                **update_kwargs, num_layers=self.coarse_gru_layers
            ) for _ in range(3)
        ])
        self.refine_update_block = MultiPromptUpdateBlock(**update_kwargs)

        self.hypothesis_weight_head = nn.Sequential(nn.Conv2d(3 * hidden_dim + 3, hidden_dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, 6, 1))
        self.refine_state_projections = nn.ModuleList([
            nn.Conv2d(3 * hidden_dim, hidden_dim, 1)
            for _ in range(self.coarse_gru_layers)
        ])
        self._horizontal_coordinate_cache = HorizontalCoordinateCache()

    @staticmethod
    def _make_right_fuser(in_channels):
        return nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(64, 32, 1, bias=False), nn.ReLU(inplace=True))

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

    def _bound_delta_disparity(self, delta_disp, *, coarse=False):
        """Smoothly bound a coarse or refinement update in quarter pixels."""
        if (getattr(self.args, 'disable_delta_clamp', False)
                or (coarse and getattr(self.args, 'disable_coarse_delta_clamp', False))):
            return delta_disp.float()
        coarse_limit = getattr(self.args, 'max_coarse_delta_disp', None)
        max_delta = float(coarse_limit if coarse and coarse_limit is not None
                          else self.args.max_delta_disp)
        if max_delta <= 0:
            raise ValueError(f"disparity update limit must be positive, got {max_delta}")
        delta_disp = delta_disp.float()
        return max_delta * torch.tanh(delta_disp / max_delta)

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

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False,
                return_aux=False):
        # Cover the frozen ViT and small tensor operations that sit outside the
        # narrower autocast regions below. Numerically sensitive FullCorr and
        # disparity accumulation explicitly opt back into FP32.
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            return self._forward_impl(
                image1, image2, iters=iters, scale_iters=scale_iters,
                test_mode=test_mode, return_aux=return_aux)

    def _lookup_right_features(self, *args, **kwargs):
        return lookup_right_features_horizontal(
            *args,
            coordinate_cache=self._horizontal_coordinate_cache,
            **kwargs,
        )

    def _update_coarse_branches(
        self, active, coarse_nets, correlations, disps, inp, warp_features,
        initial_disps, static_features, need_mask,
    ):
        """Run the three independent coarse PRUs."""
        outputs = []
        for branch, corr, disp, feature in zip(
            active, correlations, disps, warp_features
        ):
            outputs.append(self.coarse_update_blocks[branch](
                [state.float() for state in coarse_nets[branch]],
                corr.float(),
                disp.float(),
                inp.float(),
                feature.float(),
                initial_disps[branch].float(),
                static_features=static_features[branch],
                need_mask=need_mask,
            ))
        return outputs

    def _forward_impl(self, image1, image2, iters=12, scale_iters=3,
                      test_mode=False, return_aux=False):
        _, _, image_height, image_width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()

        danv2_io_sizes = get_danv2_io_size(image_height, image_width, self.args.n_downsample)
        # Unlike the original path, include the frozen DINO backbone and both
        # stereo feature heads in AMP. FullCorr is still explicitly FP32 below.
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            dfeat1_4, dfeat2_4, mono_idepth1_4 = self.defomencoder(
                image1, image2, danv2_io_sizes)
            fmap1_4, fmap2_4 = self.fnet(
                [image1, image2], [dfeat1_4, dfeat2_4])
        # Use one explicit AMP dtype throughout the recurrent path. BF16 keeps
        # half-precision memory while providing a much larger exponent range.
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            context_pyramid = self.cnet(image1.float(), dfeat1_4.float(), num_layers=self.args.n_gru_layers)
            net_4, inp_4 = torch.chunk(context_pyramid[0], chunks=2, dim=1)
            initial_nets = [torch.tanh(net_4)]
            initial_nets.extend(torch.tanh(context) for context in context_pyramid[1:])
            inp_4 = torch.relu(inp_4)
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            fmap1_low = self.low_channel(fmap1_4)
            fmap2_low = self.low_channel(fmap2_4)
            fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
        with autocast(enabled=False):
            full_corr = self._build_full_corr(fmap1_4, fmap2_4)

        right_feature_pyramid = (fmap2_low, fmap2_half, fmap2_quarter)
        del image1, image2, dfeat1_4, dfeat2_4, fmap1_4, fmap2_4, context_pyramid, net_4

        coarse_nets = [
            [state.clone() for state in initial_nets[:self.coarse_gru_layers]]
            for _ in range(3)
        ]
        coarse_disps = [
            mono_idepth1_4 * self.hypothesis_scales[index]
            for index in range(3)
        ]
        initial_coarse_disps = list(coarse_disps)
        coarse_predictions = None if test_mode else [
            [upflow(disp, factor=2 ** self.args.n_downsample)]
            for disp in coarse_disps
        ]

        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            coarse_positions = self.coarse_rope.prepare_positions(fmap1_low)
            refine_positions = self.refine_rope.prepare_positions(fmap1_low)
            static_coarse = [block.prepare_static(inp_4.float(), prompt.float())
                             for block, prompt in zip(self.coarse_update_blocks, initial_coarse_disps)]
        active = range(3)
        for iteration in range(scale_iters):
            disps = [coarse_disps[i].detach() for i in active]
            correlations = full_corr.sample_many(disps)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp = self._lookup_right_features(
                    torch.cat([fmap1_low] * len(active)),
                    (torch.cat([fmap2_quarter] * len(active)),), torch.cat(disps),
                    self.warp_offsets, self.coarse_right_fuse,
                    rope_galerkin=lambda x: self.coarse_rope(x, freqs_cis=coarse_positions),
                    compression_ratios=(4,), compact_sampling=True)
            branch_inputs = tuple(zip(
                active, disps, correlations.unbind(0), warp.chunk(len(active))
            ))
            with autocast(
                enabled=self.args.mixed_precision, dtype=self.amp_dtype
            ):
                updates = self._update_coarse_branches(
                    active, coarse_nets, correlations.unbind(0), disps,
                    inp_4, warp.chunk(len(active)), initial_coarse_disps,
                    static_coarse, need_mask=not test_mode,
                )
            for (i, disp, _, _), (net, delta, mask) in zip(
                branch_inputs, updates
            ):
                coarse_nets[i] = net
                coarse_disps[i] = self._bound_disparity(
                    disp + self._bound_delta_disparity(delta, coarse=True)
                )
                if not test_mode:
                    coarse_predictions[i].append(
                        self._upsample_prediction(coarse_disps[i], mask)
                    )

        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            aggregation_input = torch.cat(
                [coarse_nets[index][0].float() for index in range(3)]
                + [coarse_disp.float() for coarse_disp in coarse_disps], dim=1)
            weight_logits = self.hypothesis_weight_head(aggregation_input)
            hypothesis_weights = torch.softmax(weight_logits[:, :3], dim=1)
            scale_weights = torch.softmax(weight_logits[:, 3:], dim=1)
            refine_nets = [
                projection(torch.cat(
                    [coarse_nets[index][level].float() for index in range(3)],
                    dim=1,
                ))
                for level, projection in enumerate(self.refine_state_projections)
            ]

        disp = sum(
            hypothesis_weights[:, index:index + 1].float() * coarse_disps[index].float()
            for index in range(3)
        )
        disp = self._bound_disparity(disp)
        del aggregation_input, weight_logits

        d0 = disp
        if test_mode:
            disp_up = None
            disp_predictions = None
        else:
            disp_up = upflow(d0, factor=2 ** self.args.n_downsample)
            disp_predictions = [disp_up]
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            static_refine = self.refine_update_block.prepare_static(inp_4.float(), d0.float())
        for iteration in range(iters):
            disp = disp.detach()
            corr = full_corr(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp_feature = self._lookup_right_features(fmap1_low, right_feature_pyramid, disp, self.warp_offsets,
                    self.refine_right_fuse, rope_galerkin=lambda x: self.refine_rope(x, freqs_cis=refine_positions),
                    scale_weights=scale_weights, compression_ratios=(1, 2, 4),
                    compact_sampling=True)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                refine_nets, delta_disp, up_mask = self.refine_update_block(
                    [state.float() for state in refine_nets], corr.float(), disp.float(),
                    inp_4.float(), warp_feature.float(), d0.float(), static_features=static_refine,
                    need_mask=not test_mode or iteration == iters - 1)
            delta_disp = self._bound_delta_disparity(delta_disp)
            disp = self._bound_disparity(disp + delta_disp)

            if test_mode and iteration < iters - 1:
                continue
            disp_up = self._upsample_prediction(disp, up_mask)
            if not test_mode:
                disp_predictions.append(disp_up)

        if test_mode:
            if disp_up is None:
                disp_up = upflow(d0, factor=2 ** self.args.n_downsample)
            return disp_up

        if return_aux:
            aux = {
                'coarse_predictions': coarse_predictions,
                'w_h': hypothesis_weights,
                'w_s': scale_weights,
            }
            return disp_predictions, aux
        return disp_predictions
