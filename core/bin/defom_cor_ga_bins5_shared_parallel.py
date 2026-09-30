"""Bin5 with one shared coarse GRU and branch-parallel candidate updates.

This module intentionally lives beside ``defom_cor_ga_bins5`` so experiments with
the shared updater cannot change the behaviour or checkpoint layout of bin5.
"""

import torch

from core.bin.defom_cor_ga_bins5 import DEFOMStereo as Bin5Stereo, autocast
from core.utils.utils import get_danv2_io_size, upflow


class DEFOMStereo(Bin5Stereo):
    def __init__(self, args):
        super().__init__(args)

        # 相较于 bin5 的改动：
        # 1. 三个 coarse 候选共用下面这一套 GRU 参数，不再各自持有一套 GRU。
        # 2. 三个候选的视差、hidden、context 和 warp feature 沿 batch 维打包，
        #    每次迭代只调用一次共享 GRU；FullCorr volume 仍然只构建、保存一份。
        # 3. coarse 后的视差 gate、hidden fusion 和 refine 阶段完全沿用 bin5。
        # 该实现位于独立模型文件中，不修改 bin5，也不影响其旧 checkpoint。
        self.num_coarse_branches = len(self.coarse_update_blocks)
        shared_block = self.coarse_update_blocks[0]
        del self.coarse_update_blocks
        self.shared_coarse_update_block = shared_block

    @staticmethod
    def _repeat_branches(tensor, num_branches):
        """Repeat B-first data in branch-major order: [branch0(B), branch1(B), ...]."""
        return tensor.repeat(num_branches, *([1] * (tensor.ndim - 1)))

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

        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            context_pyramid = self.cnet(
                image1.float(), dfeat1_4.float(),
                num_layers=self.args.n_gru_layers,
            )
            initial_nets = []
            context_gates = []
            inp_4 = None
            for level, context in enumerate(context_pyramid):
                net, context_input = torch.chunk(context, chunks=2, dim=1)
                initial_nets.append(torch.tanh(net))
                context_input = torch.relu(context_input)
                if level == 0:
                    inp_4 = context_input
                zqr = self.context_zqr_convs[level](context_input)
                context_gates.append(torch.chunk(zqr, chunks=3, dim=1))

            fmap1_low = self.low_channel(fmap1_4)
            fmap2_low = self.low_channel(fmap2_4)
            fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
            waft_half, waft_quarter = self.waft_right_width_compressor(fmap2_low)

        with autocast(enabled=False):
            full_corr = self._build_full_corr(fmap1_4, fmap2_4)
            waft_init_disps, init_logits, init_confidence, init_valid = (
                self.bin_initializer(
                    fmap1_low,
                    (fmap2_low, waft_half, waft_quarter),
                    mono_idepth1_4,
                )
            )
            waft_init_disps = [
                self._bound_disparity(disp) for disp in waft_init_disps
            ]

        right_feature_pyramid = (fmap2_low, fmap2_half, fmap2_quarter)
        del (
            image1,
            image2,
            dfeat1_4,
            dfeat2_4,
            mono_idepth1_4,
            fmap1_4,
            fmap2_4,
            context_pyramid,
            waft_half,
            waft_quarter,
            danv2_io_sizes,
        )

        num_branches = len(waft_init_disps)
        if num_branches != self.num_coarse_branches:
            raise RuntimeError(
                f"Expected {self.num_coarse_branches} coarse candidates, got {num_branches}"
            )

        coarse_predictions = [
            [upflow(disp, factor=2 ** self.args.n_downsample)]
            for disp in waft_init_disps
        ]

        # The packed layout is branch-major. Splitting it on batch later exactly
        # restores the three original B-sized branches.
        parallel_disp = torch.cat(waft_init_disps, dim=0)
        parallel_prompt = torch.cat(waft_init_disps, dim=0)
        parallel_nets = [
            self._repeat_branches(state, num_branches)
            for state in initial_nets[:self.coarse_gru_layers]
        ]
        parallel_context_gates = [
            tuple(self._repeat_branches(gate, num_branches) for gate in gates)
            for gates in context_gates
        ]
        parallel_inp_4 = self._repeat_branches(inp_4, num_branches)
        parallel_fmap1_low = self._repeat_branches(fmap1_low, num_branches)
        parallel_fmap2_quarter = self._repeat_branches(
            fmap2_quarter, num_branches
        )

        for _ in range(scale_iters):
            parallel_disp = parallel_disp.detach()

            # FullCorr internally remembers the original batch B. Query the one
            # shared volume with each B-sized disparity, then pack only its lookup.
            corr = torch.cat(
                [full_corr(disp) for disp in parallel_disp.chunk(num_branches, dim=0)],
                dim=0,
            )
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp_feature = self._coarse_warp_feature(
                    parallel_fmap1_low,
                    parallel_fmap2_quarter,
                    parallel_disp,
                )
                parallel_nets, delta_disp, up_mask = self.shared_coarse_update_block(
                    [state.float() for state in parallel_nets],
                    parallel_context_gates,
                    corr.float(),
                    parallel_disp.float(),
                    parallel_inp_4.float(),
                    warp_feature.float(),
                    parallel_prompt.float(),
                )
            delta_disp = self._bound_delta_disparity(delta_disp)
            parallel_disp = self._bound_disparity(parallel_disp + delta_disp)

            # Coarse predictions are needed only by the training auxiliary loss.
            if not test_mode:
                prediction = self._upsample_prediction(parallel_disp, up_mask)
                for branch_index, branch_prediction in enumerate(
                    prediction.chunk(num_branches, dim=0)
                ):
                    coarse_predictions[branch_index].append(branch_prediction)

        coarse_disps = list(parallel_disp.chunk(num_branches, dim=0))
        split_states = [state.chunk(num_branches, dim=0) for state in parallel_nets]
        coarse_nets = [
            [split_states[level][branch] for level in range(self.coarse_gru_layers)]
            for branch in range(num_branches)
        ]

        # From here onward this is deliberately identical to bin5.
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            aggregation_input = torch.cat(
                [coarse_nets[index][0].float() for index in range(3)]
                + [coarse_disp.float() for coarse_disp in coarse_disps],
                dim=1,
            )
            weight_logits = self.hypothesis_weight_head(aggregation_input)
            hypothesis_logits = weight_logits[:, :3].float()
            hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
            scale_weights = torch.softmax(weight_logits[:, 3:].float(), dim=1)
            refine_nets = [
                projection(torch.cat(
                    [coarse_nets[index][level].float() for index in range(3)],
                    dim=1,
                ))
                for level, projection in enumerate(self.refine_state_projections)
            ]

        disp = sum(
            hypothesis_weights[:, index:index + 1] * coarse_disps[index].float()
            for index in range(3)
        )
        disp = self._bound_disparity(disp)
        del (
            aggregation_input,
            weight_logits,
            initial_nets,
            parallel_fmap1_low,
            parallel_fmap2_quarter,
            parallel_inp_4,
            parallel_context_gates,
        )

        d0 = disp
        disp_up = upflow(d0, factor=2 ** self.args.n_downsample)
        disp_predictions = [disp_up]
        for iteration in range(iters):
            disp = disp.detach()
            corr = full_corr(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp_feature = self._refine_warp_feature(
                    fmap1_low,
                    right_feature_pyramid,
                    disp,
                    scale_weights,
                )
                refine_nets, delta_disp, up_mask = self.refine_update_block(
                    [state.float() for state in refine_nets],
                    context_gates,
                    corr.float(),
                    disp.float(),
                    inp_4.float(),
                    warp_feature.float(),
                    d0.float(),
                )
            delta_disp = self._bound_delta_disparity(delta_disp)
            disp = self._bound_disparity(disp + delta_disp)

            if test_mode and iteration < iters - 1:
                continue
            disp_up = self._upsample_prediction(disp, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up

        if return_aux:
            return disp_predictions, {
                'coarse_predictions': coarse_predictions,
                'w_h': hypothesis_weights,
                'w_h_logits': hypothesis_logits,
                'w_s': scale_weights,
                'init_disps': torch.cat(coarse_disps, dim=1),
                'waft_init_disps': torch.cat(waft_init_disps, dim=1),
                'init_logits': init_logits,
                'init_confidence': init_confidence,
                'init_bins': self.bin_initializer.bin_disparities,
                'init_valid': init_valid,
            }
        return disp_predictions
