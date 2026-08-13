import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defom_cor_ga import DEFOMStereo as CorGAStereo, autocast
from core.utils.utils import upflow, get_danv2_io_size


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
        branch_logits = []
        branch_confidences = []
        branch_disps = []
        branch_valid = []
        with autocast(enabled=torch.is_autocast_enabled()):
            for branch_index, (right_feature, ratio) in enumerate(zip(right_feature_pyramid, self.compression_ratios)):
                raw_logits, valid = self._match_logits(left_feature, right_feature, ratio)
                refiner_input = torch.cat([raw_logits.masked_fill(~valid, -1.0), mono_idepth.float()], dim=1)
                logits = self.corr_scale * raw_logits + self.logit_refiners[branch_index](refiner_input)
                logits = logits.float().masked_fill(~valid, -1e4)
                probability = torch.softmax(logits, dim=1) * valid.float()
                probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-6)
                bins = self.bin_disparities[branch_index].view(1, self.num_bins, 1, 1)
                branch_logits.append(logits)
                branch_confidences.append(probability.amax(dim=1))
                branch_disps.append(torch.sum(probability * bins, dim=1, keepdim=True))
                branch_valid.append(valid)
        return branch_disps, torch.stack(branch_logits, dim=1), torch.stack(branch_confidences, dim=1), torch.stack(branch_valid, dim=1)


class DEFOMStereo(CorGAStereo):
    """CorGA with multi-range WAFT-style discrete disparity initialization.

    模型改动：
    1. 右特征宽度压缩由可学习卷积改为平均池化，压缩倍率保持为 1/2/4。
    2. 移除三个“单目固定倍率初始化 + coarse GRU”分支。
    3. 新增 MultiRangeBinInitializer，以 1/2/4 倍间隔构造三组离散视差候选。
    4. 使用同行全局余弦相关性为每组视差 bin 生成初始匹配 logits。
    5. 新增三个独立的 logit refiner，结合相关性和单目逆深度修正 bin logits。
    6. 对每组 bin 概率做加权求和，生成三张连续初始视差图及其置信度。
    7. 新增 initial_weight_head，根据上下文、左特征、单目深度、初始视差和置信度预测融合权重。
    8. 分别预测三组初始视差的融合权重 w_h，以及递归 warp 特征的尺度权重 w_s。
    9. 直接使用原始 context hidden states 初始化 refinement GRU，不再融合三个 coarse hidden states。
    10. 使用 tanh 将每次递归视差更新平滑限制在 max_delta_disp 范围内。
    11. 冻结因移除 coarse 分支而不再参与前向计算的结构性冗余参数。
    12. 额外返回 bin logits、有效掩码、分支初始视差和 gate logits，供 WAFT 风格监督使用。
    #"""
    def __init__(self, args):
        super().__init__(args, build_coarse_gru=False, right_width_compression_mode="avg")
        self.bin_initializer = MultiRangeBinInitializer(feature_dim=32, num_bins=getattr(args, 'num_init_bins', 48), max_offset=getattr(args, 'init_max_offset', 48.0), corr_scale=getattr(args, 'init_corr_scale', 10.0))
        self.num_init_ranges = len(self.bin_initializer.compression_ratios)
        self.max_delta_disp = float(getattr(args, 'max_delta_disp', 8.0))
        if self.max_delta_disp <= 0:
            raise ValueError(f"max_delta_disp must be positive, got {self.max_delta_disp}")
        hidden_dim = args.hidden_dims[0]
        aggregation_channels = hidden_dim + 32 + 1 + 2 * self.num_init_ranges
        self.initial_weight_head = nn.Sequential(nn.Conv2d(aggregation_channels, hidden_dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, 2 * self.num_init_ranges, 1))
        nn.init.zeros_(self.initial_weight_head[-1].weight)
        nn.init.zeros_(self.initial_weight_head[-1].bias)
        self._freeze_structurally_unused_parameters()

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

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False, return_aux=False):
        _, _, image_height, image_width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        danv2_io_sizes = get_danv2_io_size(image_height, image_width, self.args.n_downsample)
        dfeat1_4, dfeat2_4, mono_idepth1_4 = self.defomencoder(image1, image2, danv2_io_sizes)
        fmap1_4, fmap2_4 = self.fnet([image1, image2], [dfeat1_4, dfeat2_4])
        with autocast(enabled=self.args.mixed_precision):
            context_pyramid = self.cnet(image1, dfeat1_4, num_layers=self.args.n_gru_layers)
            net_4, inp_4 = torch.chunk(context_pyramid[0], chunks=2, dim=1)
            initial_nets = [torch.tanh(net_4)]
            initial_nets.extend(torch.tanh(context) for context in context_pyramid[1:])
            inp_4 = torch.relu(inp_4)
            fmap1_low = self.low_channel(fmap1_4)
            fmap2_low = self.low_channel(fmap2_4)
            fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
        with autocast(enabled=False):
            full_corr = self._build_full_corr(fmap1_4, fmap2_4)
        right_feature_pyramid = (fmap2_low, fmap2_half, fmap2_quarter)
        with autocast(enabled=self.args.mixed_precision):
            init_disps, init_logits, init_confidence, init_valid = self.bin_initializer(fmap1_low, right_feature_pyramid, mono_idepth1_4)
            init_disps = [self._bound_disparity(disp) for disp in init_disps]
            aggregation_input = torch.cat([inp_4, fmap1_low, mono_idepth1_4.float()] + init_disps + [init_confidence], dim=1)
            weight_logits = self.initial_weight_head(aggregation_input)
            hypothesis_logits = weight_logits[:, :self.num_init_ranges].float()
            hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
            scale_weights = torch.softmax(weight_logits[:, self.num_init_ranges:].float(), dim=1)
            refine_nets = initial_nets
        disp = sum(hypothesis_weights[:, index:index + 1] * init_disp.float() for index, init_disp in enumerate(init_disps))
        disp = self._bound_disparity(disp)
        del image1, image2, dfeat1_4, dfeat2_4, fmap1_4, fmap2_4, context_pyramid, net_4, aggregation_input, weight_logits

        d0 = disp
        disp_predictions = [upflow(d0, factor=2 ** self.args.n_downsample)]
        disp_up = disp_predictions[0]
        for iteration in range(iters):
            disp = disp.detach()
            with autocast(enabled=self.args.mixed_precision):
                corr = full_corr(disp)
                warp_feature = self._refine_warp_feature(fmap1_low, right_feature_pyramid, disp, scale_weights)
                refine_nets, delta_disp, up_mask = self.refine_update_block(refine_nets, corr, disp, inp_4, warp_feature, d0)
            delta_disp = self._limit_delta_disparity(delta_disp)
            disp = self._bound_disparity(disp + delta_disp)
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self._upsample_prediction(disp, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up
        if return_aux:
            return disp_predictions, {'w_h': hypothesis_weights, 'w_h_logits': hypothesis_logits, 'w_s': scale_weights, 'init_disps': torch.cat(init_disps, dim=1), 'init_logits': init_logits, 'init_bins': self.bin_initializer.bin_disparities, 'init_valid': init_valid}
        return disp_predictions
