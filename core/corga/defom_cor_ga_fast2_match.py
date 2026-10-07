"""CorGA matching tokens with fast2 horizontal lookup and execution caches.

State-dict keys and tensor shapes match cor_ga_match for the same settings.
The feature backbone and PRUs use AMP; context and hypothesis fusion stay FP32.
"""

import torch
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

from core.corga.defom_cor_ga_match import DEFOMStereo as CorGAMatchStereo
from core.corr import CorrBlock1D1
from core.horizontal_sampler import HorizontalCorrBlock1D
from core.matching import HorizontalMatchingTokenEncoder
from core.utils.utils import coords_grid, get_danv2_io_size, upflow


class DEFOMStereo(CorGAMatchStereo):
    """Match fused right candidates, batch coarse lookups and cache prompts."""

    matching_encoder_class = HorizontalMatchingTokenEncoder
    default_match_feature_dim = 24
    default_match_token_dim = 28

    def _build_full_corr(self, left, right):
        batch, _, height, width = left.shape
        coords = coords_grid(batch, height, width)[:, :1].to(left.device)
        volume = CorrBlock1D1.corr(left.float(), right.float())
        return HorizontalCorrBlock1D(
            volume, coords, num_levels=self.corr_levels, radius=self.corr_radius,
            coordinate_mode='right', volume_layout='full_corr',
        )

    def _run_cached_update(self, block, nets, corr, disp, ctx, warp, prompt,
                           static_features, need_mask=True):
        if not (self.checkpoint_updates and self.training and torch.is_grad_enabled()):
            return block(list(nets), corr, disp, ctx, warp, prompt,
                         static_features=static_features, need_mask=need_mask)
        num_states = len(nets)

        def update(*inputs):
            # Rebuild the mutable PRU state list on the initial and replay calls.
            states, delta, mask = block(
                list(inputs[:num_states]), *inputs[num_states:-2],
                static_features=inputs[-2:], need_mask=need_mask,
            )
            return tuple(states) + (delta, mask)

        outputs = checkpoint(
            update, *nets, corr, disp, ctx, warp, prompt, *static_features,
            use_reentrant=False,
        )
        return list(outputs[:num_states]), outputs[-2], outputs[-1]

    def _bound_delta_disparity(self, delta, coarse=False):
        if (getattr(self.args, 'disable_delta_clamp', False)
                or (coarse and getattr(self.args, 'disable_coarse_delta_clamp', False))):
            return delta.float()
        coarse_limit = getattr(self.args, 'max_coarse_delta_disp', None)
        limit = float(coarse_limit if coarse and coarse_limit is not None
                      else getattr(self.args, 'max_delta_disp', 8.0))
        if limit <= 0:
            raise ValueError(f'disparity update limit must be positive, got {limit}')
        return limit * torch.tanh(delta.float() / limit)

    def forward(self, image1, image2, iters=12, scale_iters=3,
                test_mode=False, return_aux=False):
        if image1.shape != image2.shape or iters < 0 or scale_iters < 0:
            raise ValueError('stereo image shapes must match and iteration counts must be nonnegative')
        _, _, height, width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        sizes = get_danv2_io_size(height, width, self.args.n_downsample)
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            dfeat1, dfeat2, mono = self.defomencoder(image1, image2, sizes)
            fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
        with autocast(enabled=False):
            contexts = self.cnet(image1.float(), dfeat1.float(), num_layers=self.args.n_gru_layers)
            net, ctx = contexts[0].chunk(2, dim=1)
            initial_nets = [torch.tanh(net)] + [torch.tanh(x) for x in contexts[1:]]
            ctx = torch.relu(ctx)
            full_corr = self._build_full_corr(fmap1, fmap2)
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            left, right = self.low_channel(fmap1), self.low_channel(fmap2)
            right_half, right_quarter = self.right_width_compressor(right)
            coarse_positions = self.coarse_rope.prepare_positions(left)
            refine_positions = self.refine_rope.prepare_positions(left)
        right_pyramid = (right, right_half, right_quarter)
        del image1, image2, dfeat1, dfeat2, fmap1, fmap2, contexts, net

        coarse_nets = [[state.clone() for state in initial_nets] for _ in range(3)]
        coarse_disps = [mono.float() * self.hypothesis_scales[i] for i in range(3)]
        prompts = list(coarse_disps)
        coarse_predictions = None if test_mode else [[upflow(d, factor=4)] for d in coarse_disps]
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            static_coarse = [block.prepare_static(ctx.float(), prompt.float())
                             for block, prompt in zip(self.coarse_update_blocks, prompts)]
        # Batch independent matching lookups; each branch still owns its PRU.
        if scale_iters:
            coarse_left = torch.cat([left] * 3)
            coarse_right = (torch.cat([right_quarter] * 3),)
        for iteration in range(scale_iters):
            disps = [d.detach() for d in coarse_disps]
            correlations = full_corr.sample_many(disps)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp = self.coarse_rope(
                    self.coarse_match_encoder(coarse_left, coarse_right, torch.cat(disps)),
                    freqs_cis=coarse_positions,
                )
            self._require_finite(f'coarse warp {iteration}', warp)
            for branch, (disp, corr, feature) in enumerate(zip(disps, correlations.unbind(0), warp.chunk(3))):
                with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                    nets, delta, mask = self._run_cached_update(
                        self.coarse_update_blocks[branch], [state.float() for state in coarse_nets[branch]],
                        corr.float(), disp.float(), ctx.float(), feature.float(),
                        prompts[branch].float(), static_coarse[branch], need_mask=not test_mode,
                    )
                self._require_finite(f'coarse update {branch}/{iteration}', *nets, delta)
                coarse_nets[branch] = nets
                coarse_disps[branch] = self._bound_disparity(
                    disp + self._bound_delta_disparity(delta, coarse=True))
                if not test_mode:
                    coarse_predictions[branch].append(self.upsample_flow(coarse_disps[branch], mask))
        if scale_iters:
            del coarse_left, coarse_right

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
            d0 = disp
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            static_refine = self.refine_update_block.prepare_static(ctx.float(), d0.float())
        self._require_finite('hypothesis fusion', disp, scale_weights, *refine_nets)
        del fusion_input, logits
        disp_up = None if test_mode else upflow(d0, factor=4)
        predictions = None if test_mode else [disp_up]
        for iteration in range(iters):
            disp = disp.detach()
            corr = full_corr(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp = self.refine_rope(
                    self.refine_match_encoder(left, right_pyramid, disp, scale_weights),
                    freqs_cis=refine_positions,
                )
            self._require_finite(f'refine warp {iteration}', warp)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                refine_nets, delta, mask = self._run_cached_update(
                    self.refine_update_block, [state.float() for state in refine_nets], corr.float(), disp.float(),
                    ctx.float(), warp.float(), d0.float(), static_refine,
                    need_mask=not test_mode or iteration == iters - 1,
                )
            self._require_finite(f'refine update {iteration}', *refine_nets, delta)
            disp = self._bound_disparity(disp + self._bound_delta_disparity(delta))
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self.upsample_flow(disp, mask)
            self._require_finite(f'prediction {iteration}', disp_up)
            if not test_mode:
                predictions.append(disp_up)
        if test_mode:
            return upflow(d0, factor=4) if disp_up is None else disp_up
        if return_aux:
            return predictions, {'coarse_predictions': coarse_predictions,
                                 'w_h': hypothesis_weights, 'w_s': scale_weights}
        return predictions
