"""Pure-stereo CorGA with a MobileNetV2 feature encoder and IGEV updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import autocast

from core.corr import CorrBlock1D1
from core.bin.defom_cor_ga_bins2 import MultiRangeBinInitializer
from core.extractor import BasicEncoder, BasicEncoder2
from core.galerkin import simple_attn_rope_2d
from core.parallel_igev import parallel_igev_coarse_update
from core.submodules import ParallelRightWidthCompressor, lookup_right_features
from core.update_igev_stereo import StereoIGEVUpdateBlock
from core.utils.utils import coords_grid, upflow


class MobileNetStereoEncoder(nn.Module):
    """Encode both images with shared MobileNetV2 weights at 1/4 resolution."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilenetv2_100', pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        self.out_dim = 64
        self.lateral = nn.ModuleList([
            nn.Conv2d(channels, self.out_dim, 1)
            for channels in self.backbone.feature_info.channels()
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image1, image2):
        batch = image1.shape[0]
        features = self.backbone(torch.cat((image1, image2), dim=0))
        size = features[0].shape[-2:]
        fused = self.lateral[0](features[0])
        for projection, feature in zip(self.lateral[1:], features[1:]):
            fused = fused + F.interpolate(
                projection(feature), size=size,
                mode='bilinear', align_corners=False,
            )
        fused = self.fuse(fused)
        return fused[:batch], fused[batch:]


class DEFOMStereo(nn.Module):
    """MobileNetV2 features, stereo bin initialization and IGEV ConvGRUs."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.register_buffer(
            'mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255
        )
        self.register_buffer(
            'std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255
        )
        self.amp_dtype = (
            torch.bfloat16 if getattr(args, 'amp_dtype', 'fp16') == 'bf16'
            else torch.float16
        )
        self.stereo_encoder = MobileNetStereoEncoder()
        self.corr_levels = int(args.corr_levels)
        self.corr_radius = int(args.corr_radius)
        self.register_buffer(
            'warp_offsets',
            torch.arange(-self.corr_radius, self.corr_radius + 1, dtype=torch.float32),
        )

        low_feature_dim = 32
        hidden_dim = args.hidden_dims[0]
        window_size = 2 * self.corr_radius + 1
        fusion_in_channels = low_feature_dim * (window_size + 1)
        corr_channels = self.corr_levels * window_size

        self.low_channel = nn.Conv2d(64, low_feature_dim, kernel_size=1)
        self.right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim, low_feature_dim, mode='conv'
        )
        self.coarse_right_fuse = self._make_right_fuser(fusion_in_channels)
        self.coarse_rope = simple_attn_rope_2d(low_feature_dim, 4)
        self.refine_right_fuse = self._make_right_fuser(fusion_in_channels)
        self.refine_rope = simple_attn_rope_2d(low_feature_dim, 4)

        self.fnet = BasicEncoder(
            self.stereo_encoder.out_dim, output_dim=64,
            norm_fn='instance', downsample=2,
        )
        self.cnet = BasicEncoder2(
            self.stereo_encoder.out_dim,
            output_dim=[2 * hidden_dim, hidden_dim, hidden_dim],
            norm_fn=args.context_norm, downsample=2,
        )

        update_kwargs = dict(
            args=args, hidden_dim=hidden_dim, feat_dim=low_feature_dim,
            volume_dim=1, warp_feat_dim=low_feature_dim,
            cor_planes=corr_channels,
        )
        self.coarse_gru_layers = args.n_gru_layers
        self.coarse_update_blocks = nn.ModuleList([
            StereoIGEVUpdateBlock(
                **update_kwargs, num_layers=self.coarse_gru_layers
            ) for _ in range(3)
        ])
        self.refine_update_block = StereoIGEVUpdateBlock(**update_kwargs)
        self.hypothesis_weight_head = nn.Sequential(
            nn.Conv2d(3 * hidden_dim + 3, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 6, 1),
        )
        self.refine_state_projections = nn.ModuleList([
            nn.Conv2d(3 * hidden_dim, hidden_dim, 1)
            for _ in range(self.coarse_gru_layers)
        ])

        compression = getattr(args, 'init_right_compression', 'avg')
        self.bin_right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim, low_feature_dim, mode=compression
        )
        self.bin_initializer = MultiRangeBinInitializer(
            feature_dim=low_feature_dim,
            num_bins=getattr(args, 'num_init_bins', 48),
            max_offset=getattr(args, 'init_max_offset', 48.0),
            corr_scale=getattr(args, 'init_corr_scale', 10.0),
            use_mono=False,
        )

        if getattr(args, 'freeze_bininit_base', False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (self.bin_right_width_compressor, self.bin_initializer):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    @staticmethod
    def _make_right_fuser(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def freeze_bn(self):
        # Match IGEV++: retain ImageNet BN statistics during stereo training.
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def upsample_flow(self, disp, mask):
        batch, channels, height, width = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(batch, 1, 9, factor, factor, height, width)
        mask = torch.softmax(mask, dim=2)
        values = F.unfold(factor * disp, [3, 3], padding=1)
        values = values.view(batch, channels, 9, 1, 1, height, width)
        values = torch.sum(mask * values, dim=2)
        values = values.permute(0, 1, 4, 2, 5, 3)
        return values.reshape(batch, channels, factor * height, factor * width)

    def _upsample_prediction(self, disp, mask):
        if mask is None:
            return upflow(disp, factor=2 ** self.args.n_downsample)
        return self.upsample_flow(disp, mask)

    def _bound_disparity(self, disp):
        maximum = float(self.args.train_max_disp) / (2 ** self.args.n_downsample)
        return disp.float().clamp(min=0.0, max=maximum)

    def _bound_delta_disparity(self, delta):
        maximum = float(self.args.max_delta_disp)
        return maximum * torch.tanh(delta.float() / maximum)

    def _build_full_corr(self, left, right):
        batch, _, height, width = left.shape
        coords = coords_grid(batch, height, width)[:, :1].to(left.device)
        volume = CorrBlock1D1.corr(left.float(), right.float())
        return CorrBlock1D1(
            volume, coords, num_levels=self.corr_levels,
            radius=self.corr_radius, coordinate_mode='right',
            volume_layout='full_corr',
        )

    def _initialize_coarse_disparities(self, left, right):
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            right_half, right_quarter = self.bin_right_width_compressor(right)
        with autocast(enabled=False):
            disps, logits, confidence, valid = self.bin_initializer(
                left, (right, right_half, right_quarter),
            )
        disps = [self._bound_disparity(disp) for disp in disps]
        return disps, {
            'waft_init_disps': torch.cat(disps, dim=1),
            'init_logits': logits,
            'init_confidence': confidence,
            'init_valid': valid,
            'init_bins': self.bin_initializer.bin_disparities,
        }

    def forward(self, image1, image2, iters=12, scale_iters=3,
                test_mode=False, return_aux=False):
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()

        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            left_feat, right_feat = self.stereo_encoder(image1, image2)
            fmap1, fmap2 = self.fnet(
                [image1, image2], [left_feat, right_feat]
            )
            context = self.cnet(
                image1.float(), left_feat.float(),
                num_layers=self.args.n_gru_layers,
            )
            net_4, inp = torch.chunk(context[0], chunks=2, dim=1)
            initial_nets = [torch.tanh(net_4)]
            initial_nets.extend(torch.tanh(level) for level in context[1:])
            inp = torch.relu(inp)
            fmap1_low = self.low_channel(fmap1)
            fmap2_low = self.low_channel(fmap2)
            fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
        with autocast(enabled=False):
            full_corr = self._build_full_corr(fmap1, fmap2)

        right_pyramid = (fmap2_low, fmap2_half, fmap2_quarter)
        coarse_disps, init_aux = self._initialize_coarse_disparities(
            fmap1_low, fmap2_low
        )
        initial_disps = list(coarse_disps)
        coarse_nets = [
            [state.clone() for state in initial_nets[:self.coarse_gru_layers]]
            for _ in range(3)
        ]
        coarse_predictions = None if test_mode else [
            [upflow(disp, factor=2 ** self.args.n_downsample)]
            for disp in coarse_disps
        ]

        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            coarse_positions = self.coarse_rope.prepare_positions(fmap1_low)
            refine_positions = self.refine_rope.prepare_positions(fmap1_low)
            static_coarse = [
                block.prepare_static(inp.float())
                for block in self.coarse_update_blocks
            ]

        for _ in range(scale_iters):
            disps = [disp.detach() for disp in coarse_disps]
            correlations = full_corr.sample_many(disps).unbind(0)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp = lookup_right_features(
                    torch.cat([fmap1_low] * 3),
                    (torch.cat([fmap2_quarter] * 3),), torch.cat(disps),
                    self.warp_offsets, self.coarse_right_fuse,
                    rope_galerkin=lambda x: self.coarse_rope(
                        x, freqs_cis=coarse_positions
                    ),
                    compression_ratios=(4,), compact_sampling=True,
                ).chunk(3)
                updates = parallel_igev_coarse_update(
                    blocks=list(self.coarse_update_blocks),
                    nets=[[state.float() for state in net] for net in coarse_nets],
                    correlations=[corr.float() for corr in correlations],
                    disps=[disp.float() for disp in disps],
                    contexts=[inp.float()] * 3,
                    warp_features=[feature.float() for feature in warp],
                    anchor_disps=[disp.float() for disp in initial_disps],
                    static_features=static_coarse,
                    need_mask=not test_mode,
                )
            for index, (net, delta, mask) in enumerate(updates):
                coarse_nets[index] = net
                coarse_disps[index] = self._bound_disparity(
                    disps[index] + self._bound_delta_disparity(delta)
                )
                if not test_mode:
                    coarse_predictions[index].append(
                        self._upsample_prediction(coarse_disps[index], mask)
                    )

        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            aggregation = torch.cat(
                [net[0].float() for net in coarse_nets]
                + [disp.float() for disp in coarse_disps], dim=1,
            )
            weight_logits = self.hypothesis_weight_head(aggregation)
            hypothesis_logits = weight_logits[:, :3]
            hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
            scale_weights = torch.softmax(weight_logits[:, 3:], dim=1)
            refine_nets = [
                projection(torch.cat(
                    [net[level].float() for net in coarse_nets], dim=1,
                ))
                for level, projection in enumerate(self.refine_state_projections)
            ]

        disp = sum(
            hypothesis_weights[:, index:index + 1].float()
            * coarse_disps[index].float()
            for index in range(3)
        )
        d0 = self._bound_disparity(disp)
        predictions = None if test_mode else [
            upflow(d0, factor=2 ** self.args.n_downsample)
        ]
        upsampled = None
        with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
            static_refine = self.refine_update_block.prepare_static(inp.float())

        for iteration in range(iters):
            disp = disp.detach()
            corr = full_corr(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=self.amp_dtype):
                warp_feature = lookup_right_features(
                    fmap1_low, right_pyramid, disp, self.warp_offsets,
                    self.refine_right_fuse,
                    rope_galerkin=lambda x: self.refine_rope(
                        x, freqs_cis=refine_positions
                    ),
                    scale_weights=scale_weights,
                    compression_ratios=(1, 2, 4), compact_sampling=True,
                )
                refine_nets, delta, mask = self.refine_update_block(
                    [state.float() for state in refine_nets], corr.float(),
                    disp.float(), inp.float(), warp_feature.float(),
                    static_features=static_refine,
                    need_mask=not test_mode or iteration == iters - 1,
                )
            disp = self._bound_disparity(
                disp + self._bound_delta_disparity(delta)
            )
            if test_mode and iteration < iters - 1:
                continue
            upsampled = self._upsample_prediction(disp, mask)
            if not test_mode:
                predictions.append(upsampled)

        if test_mode:
            if upsampled is None:
                upsampled = upflow(d0, factor=2 ** self.args.n_downsample)
            return upsampled
        if not return_aux:
            return predictions
        aux = {
            'coarse_predictions': coarse_predictions,
            'w_h': hypothesis_weights,
            'w_s': scale_weights,
            'init_disps': torch.cat(coarse_disps, dim=1),
            'w_h_logits': hypothesis_logits,
        }
        aux.update(init_aux)
        return predictions, aux
