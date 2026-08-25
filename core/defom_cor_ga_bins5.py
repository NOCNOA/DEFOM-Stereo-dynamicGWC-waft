"""Bin5: WAFT initialization followed by bounded multi-scale ConvGRU stages."""

import torch
import torch.nn as nn

from core.defom_cor_ga import DEFOMStereo as CorGAStereo, autocast
from core.defom_cor_ga_bins2 import MultiRangeBinInitializer
from core.submodules import ParallelRightWidthCompressor
from core.update_pru import MultiPromptGRUUpdateBlock
from core.utils.utils import get_danv2_io_size, upflow


class DEFOMStereo(CorGAStereo):
    """Use bins2 WAFT hypotheses as the three CorGA coarse-GRU prompts.

    Coarse and refinement updates use bounded multi-scale ConvGRUs while
    retaining CorGA's prompt features and hidden-state fusion. Bin5 keeps its
    WAFT initializer and width compressor in an independent namespace.
    """

    update_block_class = MultiPromptGRUUpdateBlock

    def __init__(self, args):
        super().__init__(args)

        # 模型改动：
        # 1. 保留原 CorGA 的三尺度 coarse/refine 分支、FullCorr 查找与多假设融合。
        # 2. 使用 bins2 的 WAFT 三范围离散视差初始化，替代固定单目深度尺度初始化；
        #    三个初始视差同时作为各尺度递归起点和 monocular prompt。
        # 3. WAFT 右图宽度压缩器与 CorGA 原有的 learned compressor 相互独立，
        #    初始化阶段可通过 init_right_compression 选择 avg 或 conv。
        # 4. 将无界 PromptStereo PRU 替换为 IGEV 风格 1x1 ConvGRU；三个尺度均由
        #    独立的 context z/r/q 偏置调制，并保留结构/运动/跨尺度 prompt 特征。
        # 5. 每次递归预测的视差增量经 tanh 限幅，并将累计视差约束在合法范围内，
        #    避免长递归中 hidden state 与 disparity 持续发散。
        # 6. aux 输出保留 WAFT 初始化、coarse 结果及 bin 置信度，供辅助损失与诊断使用。
        # 7. 训练时仅在各 coarse 分支的有效 bin 视差范围内施加辅助损失，并将
        #    train/validation refine 迭代次数统一为 18。

        hidden_dim = args.hidden_dims[0]
        self.cnet.d_out = [2 * hidden_dim] * 3
        for output_name in ('out16', 'out32'):
            projection = nn.Conv2d(128, 2 * hidden_dim, kernel_size=1)
            nn.init.kaiming_normal_(projection.weight, mode='fan_out', nonlinearity='relu')
            setattr(self.cnet, output_name, projection)

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, 3 * hidden_dim, kernel_size=3, padding=1)
            for _ in range(args.n_gru_layers)
        ])

        init_compression = getattr(args, 'init_right_compression', 'avg')
        if init_compression not in ('avg', 'conv'):
            raise ValueError(
                "init_right_compression must be one of ('avg', 'conv'), "
                f"got {init_compression}"
            )

        low_feature_dim = 32
        self.waft_right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim,
            low_feature_dim,
            mode=init_compression,
        )
        self.bin_initializer = MultiRangeBinInitializer(
            feature_dim=low_feature_dim,
            num_bins=getattr(args, 'num_init_bins', 48),
            max_offset=getattr(args, 'init_max_offset', 48.0),
            corr_scale=getattr(args, 'init_corr_scale', 10.0),
        )
        if len(self.bin_initializer.compression_ratios) != len(self.coarse_update_blocks):
            raise ValueError(
                "Bin5 requires one CorGA coarse update block per WAFT range, "
                f"got {len(self.coarse_update_blocks)} blocks and "
                f"{len(self.bin_initializer.compression_ratios)} ranges"
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

        # Match CorGA's recurrent precision and context initialization.
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

            # CorGA's learned compressor remains exclusive to its coarse and
            # refinement feature lookup.  WAFT has an independent namespace.
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

        # The three WAFT expectations replace CorGA's fixed mono-depth scales.
        # The recurrent update, prompts, and per-branch hidden states are CorGA.
        coarse_nets = [
            [state.clone() for state in initial_nets]
            for _ in waft_init_disps
        ]
        coarse_disps = list(waft_init_disps)
        coarse_predictions = [
            [upflow(disp, factor=2 ** self.args.n_downsample)]
            for disp in coarse_disps
        ]

        for branch_index, update_block in enumerate(self.coarse_update_blocks):
            disp = coarse_disps[branch_index]
            waft_prompt = disp
            net = coarse_nets[branch_index]
            for _ in range(scale_iters):
                disp = disp.detach()
                corr = full_corr(disp)
                with autocast(
                    enabled=self.args.mixed_precision,
                    dtype=self.amp_dtype,
                ):
                    warp_feature = self._coarse_warp_feature(
                        fmap1_low, fmap2_quarter, disp
                    )
                    net, delta_disp, up_mask = update_block(
                        [state.float() for state in net],
                        context_gates,
                        corr.float(),
                        disp.float(),
                        inp_4.float(),
                        warp_feature.float(),
                        waft_prompt.float(),
                    )
                delta_disp = self._bound_delta_disparity(delta_disp)
                disp = self._bound_disparity(disp + delta_disp)
                coarse_predictions[branch_index].append(
                    self._upsample_prediction(disp, up_mask)
                )
            coarse_disps[branch_index] = disp
            coarse_nets[branch_index] = net

        # Keep CorGA's post-coarse disparity gate and hidden-state fusion.
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
        del aggregation_input, weight_logits, initial_nets

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
                # Gate supervision must compare the post-coarse hypotheses
                # that are actually fused, while bin CE still uses init_logits.
                'init_disps': torch.cat(coarse_disps, dim=1),
                'waft_init_disps': torch.cat(waft_init_disps, dim=1),
                'init_logits': init_logits,
                'init_confidence': init_confidence,
                'init_bins': self.bin_initializer.bin_disparities,
                'init_valid': init_valid,
            }
        return disp_predictions
