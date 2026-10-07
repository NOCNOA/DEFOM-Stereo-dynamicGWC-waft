"""Standalone CorGA architecture with local matching-token evidence.

The mono-scale coarse initialization, full correlation, hypothesis fusion and
PRU refinement follow CorGA's design. This model inherits only nn.Module;
its coarse and refinement lookup paths encode correspondence before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

from core.corr import CorrBlock1D1
from core.extractor import BasicEncoder, BasicEncoder2, DefomEncoder
from core.galerkin import simple_attn_rope_2d
from core.matching import MatchingTokenEncoder
from core.submodules import ParallelRightWidthCompressor
from core.update_pru import MultiPromptUpdateBlock
from core.utils.utils import coords_grid, get_danv2_io_size, upflow


class DEFOMStereo(nn.Module):
    """CorGA with grouped cosine/difference tokens and candidate aggregation."""

    matching_encoder_class = MatchingTokenEncoder
    default_match_feature_dim = 32
    default_match_token_dim = 32

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.checkpoint_updates = not getattr(args, 'disable_activation_checkpoint', False)
        if args.n_downsample != 2:
            raise ValueError('cor_ga_match uses quarter-resolution disparity (n_downsample=2)')
        self.corr_levels = int(args.corr_levels)
        self.corr_radius = int(args.corr_radius)
        if self.corr_levels < 1 or self.corr_radius < 0:
            raise ValueError('corr_levels must be positive and corr_radius nonnegative')
        scales = tuple(float(x) for x in args.hypothesis_scales)
        if len(scales) != 3 or any(x <= 0 for x in scales):
            raise ValueError('hypothesis_scales must contain three positive scales')
        self.amp_dtype = (torch.bfloat16 if getattr(args, 'amp_dtype', 'fp16') == 'bf16'
                          else torch.float16)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None] * 255)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None] * 255)
        self.register_buffer('hypothesis_scales', torch.tensor(scales).view(3, 1, 1, 1))
        self.register_buffer('warp_offsets', torch.arange(-self.corr_radius, self.corr_radius + 1).float())

        feature_dim = getattr(args, 'match_feature_dim', None)
        feature_dim = self.default_match_feature_dim if feature_dim is None else int(feature_dim)
        if feature_dim < 1:
            raise ValueError('match_feature_dim must be positive')
        hidden_dim = args.hidden_dims[0]
        # Image and candidate widths are independent of the fixed 32-channel
        # matching output and PRU input, including four-head RoPE attention.
        warp_feature_dim = 32
        token_dim = getattr(args, 'match_token_dim', None)
        token_dim = self.default_match_token_dim if token_dim is None else int(token_dim)
        self.defomencoder = DefomEncoder(args.dinov2_encoder)
        self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=64,
                                 norm_fn='instance', downsample=2)
        self.cnet = BasicEncoder2(self.defomencoder.out_dim,
                                  output_dim=[2 * hidden_dim, hidden_dim, hidden_dim],
                                  norm_fn=args.context_norm, downsample=2)
        self.low_channel = nn.Conv2d(64, feature_dim, 1)
        self.right_width_compressor = ParallelRightWidthCompressor(
            feature_dim, feature_dim, mode=getattr(args, 'match_right_compression', 'conv'))
        token_kwargs = dict(
            channels=feature_dim, output_channels=warp_feature_dim,
            groups=getattr(args, 'match_groups', 4),
            token_dim=token_dim, radius=self.corr_radius,
            aggregation_layers=getattr(args, 'match_aggregation_layers', 1),
            scale_encoding=getattr(args, 'match_scale_encoding', False),
            checkpoint_scales=self.checkpoint_updates,
        )
        self.coarse_match_encoder = self.matching_encoder_class(
            **token_kwargs, compression_ratios=(4,))
        self.refine_match_encoder = self.matching_encoder_class(
            **token_kwargs, compression_ratios=(1, 2, 4))
        self.coarse_rope = simple_attn_rope_2d(warp_feature_dim, 4)
        self.refine_rope = simple_attn_rope_2d(warp_feature_dim, 4)
        update_kwargs = dict(
            args=args, hidden_dim=hidden_dim, feat_dim=warp_feature_dim, volume_dim=1,
            warp_feat_dim=warp_feature_dim,
            cor_planes=self.corr_levels * (2 * self.corr_radius + 1),
        )
        self.coarse_update_blocks = nn.ModuleList([
            MultiPromptUpdateBlock(**update_kwargs) for _ in range(3)
        ])
        self.refine_update_block = MultiPromptUpdateBlock(**update_kwargs)
        self.hypothesis_weight_head = nn.Sequential(
            nn.Conv2d(3 * hidden_dim + 3, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, 6, 1),
        )
        self.refine_state_projections = nn.ModuleList([
            nn.Conv2d(3 * hidden_dim, hidden_dim, 1) for _ in range(args.n_gru_layers)
        ])
        # These parameters are structurally unused by the shared encoders.
        for name in ('norm2', 'norm3'):
            norm = getattr(self.cnet.convd, name, None)
            if norm is not None:
                norm.requires_grad_(False)
        self.defomencoder.depth_anything.depth_feat.scratch.refinenet4.resConfUnit1.requires_grad_(False)

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def upsample_flow(self, flow, mask):
        batch, channels, height, width = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = torch.softmax(mask.view(batch, 1, 9, factor, factor, height, width), dim=2)
        neighborhoods = F.unfold(factor * flow, (3, 3), padding=1)
        neighborhoods = neighborhoods.view(batch, channels, 9, 1, 1, height, width)
        upsampled = (mask * neighborhoods).sum(dim=2)
        return upsampled.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, channels, factor * height, factor * width)

    def _bound_disparity(self, disp):
        maximum = float(getattr(self.args, 'train_max_disp', 768.0)) / (2 ** self.args.n_downsample)
        return disp.float().clamp(min=0.0, max=maximum)

    def _bound_delta_disparity(self, delta, coarse=False):
        if (getattr(self.args, 'disable_delta_clamp', False)
                or (coarse and getattr(self.args, 'disable_coarse_delta_clamp', False))):
            return delta.float()
        coarse_limit = getattr(self.args, 'max_coarse_delta_disp', None)
        limit = float(coarse_limit if coarse and coarse_limit is not None
                      else getattr(self.args, 'max_delta_disp', 8.0))
        return limit * torch.tanh(delta.float() / limit)

    def _build_full_corr(self, left, right):
        batch, _, height, width = left.shape
        coords = coords_grid(batch, height, width)[:, :1].to(left.device)
        volume = CorrBlock1D1.corr(left.float(), right.float())
        return CorrBlock1D1(volume, coords, num_levels=self.corr_levels,
                           radius=self.corr_radius, coordinate_mode='right',
                           volume_layout='full_corr')

    def _coarse_warp_feature(self, left, right_quarter, disp):
        return self.coarse_rope(self.coarse_match_encoder(left, (right_quarter,), disp))

    def _refine_warp_feature(self, left, right_pyramid, disp, weights):
        return self.refine_rope(self.refine_match_encoder(left, right_pyramid, disp, weights))

    def _require_finite(self, name, *tensors):
        if getattr(self.args, 'debug_nan', False):
            for tensor in tensors:
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(f'Non-finite {name}')

    def _run_update(self, block, nets, corr, disp, ctx, warp, prompt):
        if not (self.checkpoint_updates and self.training and torch.is_grad_enabled()):
            return block(nets, corr, disp, ctx, warp, prompt)
        num_states = len(nets)

        def update(*inputs):
            # PRU updates its state list in place. Build a fresh list on both
            # the original call and recomputation so saved inputs stay intact.
            states, delta, mask = block(list(inputs[:num_states]), *inputs[num_states:])
            return tuple(states) + (delta, mask)

        outputs = checkpoint(update, *nets, corr, disp, ctx, warp, prompt,
                             use_reentrant=False)
        return list(outputs[:num_states]), outputs[-2], outputs[-1]

    def forward(self, image1, image2, iters=12, scale_iters=3,
                test_mode=False, return_aux=False):
        if image1.shape != image2.shape or iters < 0 or scale_iters < 0:
            raise ValueError('stereo image shapes must match and iteration counts must be nonnegative')
        _, _, height, width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        sizes = get_danv2_io_size(height, width, self.args.n_downsample)
        dfeat1, dfeat2, mono = self.defomencoder(image1, image2, sizes)
        fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
        # Recurrent states and gate fusion stay FP32; matching projections and
        # attention use the selected AMP dtype. Full correlation is also FP32.
        with autocast(enabled=False):
            contexts = self.cnet(image1.float(), dfeat1.float(), num_layers=self.args.n_gru_layers)
            net, ctx = contexts[0].chunk(2, dim=1)
            initial_nets = [torch.tanh(net)] + [torch.tanh(x) for x in contexts[1:]]
            ctx = F.relu(ctx)
            full_corr = self._build_full_corr(fmap1, fmap2)
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            left = self.low_channel(fmap1)
            right = self.low_channel(fmap2)
            right_half, right_quarter = self.right_width_compressor(right)
        right_pyramid = (right, right_half, right_quarter)
        del image1, image2, dfeat1, dfeat2, fmap1, fmap2, contexts, net

        coarse_nets = [[state.clone() for state in initial_nets] for _ in range(3)]
        coarse_disps = [mono.float() * self.hypothesis_scales[i] for i in range(3)]
        coarse_predictions = [[upflow(d, factor=4)] for d in coarse_disps]
        for branch, update_block in enumerate(self.coarse_update_blocks):
            disp, nets = coarse_disps[branch], coarse_nets[branch]
            mono_prompt = disp
            for iteration in range(scale_iters):
                disp = disp.detach()
                corr = full_corr(disp)
                with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                    warp = self._coarse_warp_feature(left, right_quarter, disp)
                self._require_finite(f'coarse warp {branch}/{iteration}', warp)
                with autocast(enabled=False):
                    nets, delta, mask = self._run_update(
                        update_block, nets, corr.float(), disp.float(),
                        ctx.float(), warp.float(), mono_prompt.float())
                self._require_finite(f'coarse update {branch}/{iteration}', *nets, delta, mask)
                disp = self._bound_disparity(disp + self._bound_delta_disparity(delta, coarse=True))
                coarse_predictions[branch].append(self.upsample_flow(disp, mask))
            coarse_disps[branch], coarse_nets[branch] = disp, nets

        with autocast(enabled=False):
            fusion_input = torch.cat([nets[0].float() for nets in coarse_nets]
                                     + [d.float() for d in coarse_disps], dim=1)
            logits = self.hypothesis_weight_head(fusion_input)
            hypothesis_weights = logits[:, :3].softmax(dim=1)
            scale_weights = logits[:, 3:].softmax(dim=1)
            refine_nets = [projection(torch.cat([nets[level].float() for nets in coarse_nets], dim=1))
                           for level, projection in enumerate(self.refine_state_projections)]
            disp = self._bound_disparity(sum(hypothesis_weights[:, i:i + 1] * d
                                             for i, d in enumerate(coarse_disps)))
        self._require_finite('hypothesis fusion', disp, scale_weights, *refine_nets)
        d0 = disp
        disp_up = upflow(d0, factor=4)
        predictions = [disp_up]
        for iteration in range(iters):
            disp = disp.detach()
            corr = full_corr(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp = self._refine_warp_feature(left, right_pyramid, disp, scale_weights)
            self._require_finite(f'refine warp {iteration}', warp)
            with autocast(enabled=False):
                refine_nets, delta, mask = self._run_update(
                    self.refine_update_block, refine_nets, corr.float(), disp.float(),
                    ctx.float(), warp.float(), d0.float())
            self._require_finite(f'refine update {iteration}', *refine_nets, delta, mask)
            disp = self._bound_disparity(disp + self._bound_delta_disparity(delta))
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self.upsample_flow(disp, mask)
            self._require_finite(f'prediction {iteration}', disp_up)
            predictions.append(disp_up)
        if test_mode:
            return disp_up
        if return_aux:
            return predictions, {'coarse_predictions': coarse_predictions,
                                 'w_h': hypothesis_weights, 'w_s': scale_weights}
        return predictions
