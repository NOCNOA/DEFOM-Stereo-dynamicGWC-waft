import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr import CorrBlock1D1
from core.update_pru import MultiPromptUpdateBlock
from core.extractor import BasicEncoder, BasicEncoder2, DefomEncoder
from core.galerkin import simple_attn_rope_2d
from core.utils.utils import coords_grid, upflow, get_danv2_io_size
from core.submodules import ParallelRightWidthCompressor, lookup_right_features

try:
    autocast = torch.cuda.amp.autocast
except Exception:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class DEFOMStereo(nn.Module):
    """
    1\基于cor_ga1
    2\不直接采低分辨率了，使用一个很粗的volume 替代
    """

    def __init__(self, args):
        super(DEFOMStereo, self).__init__()
        self.args = args
        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.defomencoder = DefomEncoder(args.dinov2_encoder)
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
        self.right_width_compressor = ParallelRightWidthCompressor(low_feature_dim, low_feature_dim)
        self.coarse_right_fuse = self._make_right_fuser(fusion_in_channels)
        self.refine_right_fuse = self._make_right_fuser(fusion_in_channels)
        self.coarse_rope = simple_attn_rope_2d(low_feature_dim, 4)
        self.refine_rope = simple_attn_rope_2d(low_feature_dim, 4)

        self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=64, norm_fn='instance', downsample=2)
        self.cnet = BasicEncoder2(self.defomencoder.out_dim, output_dim=[2 * hidden_dim, hidden_dim, hidden_dim], norm_fn=args.context_norm, downsample=2)

        update_kwargs = dict(args=self.args, hidden_dim=hidden_dim, feat_dim=low_feature_dim, volume_dim=1, warp_feat_dim=low_feature_dim, cor_planes=corr_channels)
        self.coarse_update_blocks = nn.ModuleList([MultiPromptUpdateBlock(**update_kwargs) for _ in range(3)])
        self.refine_update_block = MultiPromptUpdateBlock(**update_kwargs)

        self.hypothesis_weight_head = nn.Sequential(nn.Conv2d(3 * hidden_dim + 3, hidden_dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, 6, 1))
        self.refine_state_projections = nn.ModuleList([nn.Conv2d(3 * hidden_dim, hidden_dim, 1) for _ in range(args.n_gru_layers)])

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

    def _build_full_corr(self, fmap1, fmap2):
        batch, _, height, width = fmap1.shape
        coords = coords_grid(batch, height, width)[:, :1].to(fmap1.device)
        volume = CorrBlock1D1.corr(fmap1.float(), fmap2.float())
        return CorrBlock1D1(volume, coords, num_levels=self.corr_levels, radius=self.corr_radius, coordinate_mode='right', volume_layout='full_corr')

    def _coarse_warp_feature(self, fmap1_low, fmap2_quarter, disp):
        return lookup_right_features(fmap1_low, (fmap2_quarter,), disp, warp_offsets=self.warp_offsets, feature_fuse=self.coarse_right_fuse, rope_galerkin=self.coarse_rope, compression_ratios=(4,))

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp, scale_weights):
        return lookup_right_features(fmap1_low, right_feature_pyramid, disp, warp_offsets=self.warp_offsets, feature_fuse=self.refine_right_fuse, rope_galerkin=self.refine_rope, scale_weights=scale_weights, compression_ratios=(1, 2, 4))

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
        del image1, image2, dfeat1_4, dfeat2_4, fmap1_4, fmap2_4, context_pyramid, net_4

        coarse_nets = [[state.clone() for state in initial_nets] for _ in range(3)]
        coarse_disps = [
            mono_idepth1_4 * self.hypothesis_scales[index]
            for index in range(3)
        ]
        coarse_predictions = [
            [upflow(disp, factor=2 ** self.args.n_downsample)]
            for disp in coarse_disps
        ]

        for branch_index, update_block in enumerate(self.coarse_update_blocks):
            disp = coarse_disps[branch_index]
            mono_prompt = disp
            net = coarse_nets[branch_index]
            for coarse_iteration in range(scale_iters):
                disp = disp.detach()
                corr = full_corr(disp)
                with autocast(enabled=self.args.mixed_precision):
                    warp_feature = self._coarse_warp_feature(fmap1_low, fmap2_quarter, disp)
                    net, delta_disp, up_mask = update_block(net, corr, disp, inp_4, warp_feature, mono_prompt)
                disp = self._bound_disparity(disp + delta_disp.float())
                prediction = self._upsample_prediction(disp, up_mask)
                coarse_predictions[branch_index].append(prediction)
            coarse_disps[branch_index] = disp
            coarse_nets[branch_index] = net

        with autocast(enabled=self.args.mixed_precision):
            aggregation_input = torch.cat([coarse_nets[index][0] for index in range(3)] + coarse_disps, dim=1)
            weight_logits = self.hypothesis_weight_head(aggregation_input)
            hypothesis_weights = torch.softmax(weight_logits[:, :3], dim=1)
            scale_weights = torch.softmax(weight_logits[:, 3:], dim=1)
            refine_nets = [projection(torch.cat([coarse_nets[index][level] for index in range(3)], dim=1)) for level, projection in enumerate(self.refine_state_projections)]

        disp = sum(hypothesis_weights[:, index:index + 1].float() * coarse_disps[index].float() for index in range(3))
        disp = self._bound_disparity(disp)
        del aggregation_input, weight_logits

        d0 = disp
        disp_up = upflow(d0, factor=2 ** self.args.n_downsample)
        disp_predictions = [disp_up]
        for iteration in range(iters):
            disp = disp.detach()
            corr = full_corr(disp)
            with autocast(enabled=self.args.mixed_precision):
                warp_feature = self._refine_warp_feature(fmap1_low, right_feature_pyramid, disp, scale_weights)
                refine_nets, delta_disp, up_mask = self.refine_update_block(refine_nets, corr, disp, inp_4, warp_feature, d0)
            disp = self._bound_disparity(disp + delta_disp.float())

            if test_mode and iteration < iters - 1:
                continue
            disp_up = self._upsample_prediction(disp, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up

        if return_aux:
            aux = {
                'coarse_predictions': coarse_predictions,
                'w_h': hypothesis_weights,
                'w_s': scale_weights,
            }
            return disp_predictions, aux
        return disp_predictions
