"""CorGA execution optimizations; identical parameter names and independent PRUs."""
import torch
from core.corga.defom_cor_ga import DEFOMStereo as CorGA, autocast
from core.utils.utils import get_danv2_io_size, upflow
from core.submodules import lookup_right_features


class DEFOMStereo(CorGA):
    def __init__(self, args):
        super().__init__(args)
        # 保留原始三个独立 PRU、gate、hidden fusion 和全部训练监督。
        # 缓存仅在单次 forward 内使用，保留 context/prompt/RoPE 的梯度。
        # 公共 coarse corr/warp 按 batch 并行，warp 不再显式复制右特征。
        # 推理不构造中间上采样结果，只为最后一次 refine 计算 mask。

    def _lookup_right_features(self, *args, **kwargs):
        """Hook for sampler variants; CorGA-fast retains grid_sample."""
        return lookup_right_features(*args, **kwargs)

    def _initialize_coarse_disparities(
        self, fmap1_low, right_feature_pyramid, mono_idepth
    ):
        """Return the three coarse starting points and optional diagnostics."""
        del fmap1_low, right_feature_pyramid
        return [
            mono_idepth * self.hypothesis_scales[index]
            for index in range(3)
        ], {}

    def _augment_initialization_aux(
        self, aux, initialization_aux, coarse_disps, hypothesis_logits
    ):
        """Hook for initialization variants that require auxiliary losses."""
        del coarse_disps, hypothesis_logits
        aux.update(initialization_aux)

    def _update_coarse_branches(
        self, active, coarse_nets, correlations, disps, inp, warp_features,
        initial_disps, static_features, need_mask,
    ):
        """Run independent coarse PRUs; subclasses may vectorize execution."""
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

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False,
                return_aux=False, coarse_ablation=None, return_hr_state=False):
        # Cover the frozen ViT and small tensor operations that sit outside the
        # narrower autocast regions below. Numerically sensitive FullCorr and
        # disparity accumulation explicitly opt back into FP32.
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            return self._forward_impl(
                image1, image2, iters=iters, scale_iters=scale_iters,
                test_mode=test_mode, return_aux=return_aux,
                coarse_ablation=coarse_ablation, return_hr_state=return_hr_state)

    def _forward_impl(self, image1, image2, iters=12, scale_iters=3,
                      test_mode=False, return_aux=False,
                      coarse_ablation=None, return_hr_state=False):
        """Run CorGA, optionally intervening on the coarse-to-refine interface.

        ``coarse_ablation`` supports diagnostics and controlled ablation models.
        The default ``None`` preserves the training and evaluation path exactly.
        Supported keys are:

        - ``active_branches``: iterable of coarse branch indices to update;
        - ``branch_scale_iters``: per-branch iteration-count overrides;
        - ``use_initial_disparities``: form d0 from pre-coarse disparities;
        - ``use_context_hidden``: initialize refinement from context states;
        - ``uniform_scale_weights``: replace w_s with uniform weights.
        """
        ablation = {} if coarse_ablation is None else dict(coarse_ablation)
        active_branches = ablation.get('active_branches')
        if active_branches is not None:
            active_branches = {int(index) for index in active_branches}
            invalid_branches = active_branches.difference(range(3))
            if invalid_branches:
                raise ValueError(f"Invalid active coarse branches: {sorted(invalid_branches)}")
        branch_scale_iters = {
            int(index): int(count)
            for index, count in ablation.get('branch_scale_iters', {}).items()
        }
        if any(index not in range(3) for index in branch_scale_iters):
            raise ValueError(f"Invalid branch_scale_iters keys: {sorted(branch_scale_iters)}")
        if any(count < 0 for count in branch_scale_iters.values()):
            raise ValueError("branch_scale_iters values must be non-negative")
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
        coarse_disps, initialization_aux = self._initialize_coarse_disparities(
            fmap1_low, right_feature_pyramid, mono_idepth1_4
        )
        if len(coarse_disps) != 3:
            raise ValueError(f"CorGA requires three coarse disparities, got {len(coarse_disps)}")
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
        counts = [branch_scale_iters.get(i, scale_iters)
                  if active_branches is None or i in active_branches else 0 for i in range(3)]
        for iteration in range(max(counts)):
            active = [i for i in range(3) if iteration < counts[i]]
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
            hypothesis_logits = weight_logits[:, :3]
            hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
            scale_weights = torch.softmax(weight_logits[:, 3:], dim=1)
            if ablation.get('use_context_hidden', False):
                refine_nets = [state.float() for state in initial_nets]
            else:
                refine_nets = []
                for level in range(self.args.n_gru_layers):
                    if level < self.coarse_gru_layers:
                        projection = self.refine_state_projections[level]
                        refine_nets.append(projection(torch.cat(
                            [coarse_nets[index][level].float() for index in range(3)],
                            dim=1,
                        )))
                    else:
                        refine_nets.append(initial_nets[level].float())

        if ablation.get('uniform_scale_weights', False):
            scale_weights = torch.full_like(scale_weights, 1.0 / scale_weights.shape[1])

        disparity_sources = (
            initial_coarse_disps
            if ablation.get('use_initial_disparities', False)
            else coarse_disps
        )
        disp = sum(
            hypothesis_weights[:, index:index + 1].float() * disparity_sources[index].float()
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
            if return_hr_state:
                scale = 2 ** self.args.n_downsample
                return disp_up, {
                    "disp_lr_px": disp.float() * scale,
                    "hidden_lr": refine_nets[0].float(),
                    "context_lr": inp_4.float(),
                }
            return disp_up

        if return_aux:
            aux = {
                'coarse_predictions': coarse_predictions,
                'w_h': hypothesis_weights,
                'w_s': scale_weights,
            }
            self._augment_initialization_aux(
                aux, initialization_aux, coarse_disps, hypothesis_logits
            )
            if coarse_ablation is not None:
                aux['coarse_disps'] = torch.cat(
                    [value.float() for value in coarse_disps], dim=1)
                aux['initial_coarse_disps'] = torch.cat(
                    [value.float() for value in initial_coarse_disps], dim=1)
            return disp_predictions, aux
        return disp_predictions
