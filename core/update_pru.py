import os
import torch,pdb,sys
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodules import *
from core.extractor import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from core.submodules import normalize_disparity

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[-2:], **interp_args)

class DispHead(nn.Module):
    def __init__(self, hidden_dim):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)

        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)

        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim, feat_dim):
        super(StructureEncoder, self).__init__()
        self.convc1 = nn.Conv2d(hidden_dim, hidden_dim // 2, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(feat_dim, hidden_dim // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, ctx, warp_feat):

        c = F.relu(self.convc1(ctx.float()), True)
        c = F.relu(self.convc2(c), True)

        d = F.relu(self.convd1(warp_feat), True)
        d = F.relu(self.convd2(d), True)
        out = F.relu(self.conv(torch.cat([c,d], dim=1)), True)

        return out

class MotionEncoder(nn.Module):
    def __init__(self, hidden_dim, corplane, warp_feat_dim, use_warp_features=True):
        super(MotionEncoder, self).__init__()
        self.use_corr = corplane > 0
        self.use_warp_features = use_warp_features
        if self.use_corr:
            self.convc1 = nn.Conv2d(corplane, hidden_dim // 2, 1, padding=0)
            self.convc2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(1, hidden_dim // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        if self.use_warp_features:
            self.convg1 = nn.Conv2d(warp_feat_dim, hidden_dim // 2, 1, padding=0)
            self.convg2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        motion_channels = hidden_dim // 2 * (1 + self.use_corr + self.use_warp_features)
        self.conv = nn.Conv2d(motion_channels, hidden_dim - 1, 3, padding=1)

    def forward(self, corr, disp, warp_feat):
        motion_features = []
        if self.use_corr:
            cor = F.relu(self.convc1(corr), True)
            cor = F.relu(self.convc2(cor), True)
            motion_features.append(cor)
        dis = F.relu(self.convd1(disp), True)
        dis = F.relu(self.convd2(dis), True)
        motion_features.append(dis)
        if self.use_warp_features:
            war = F.relu(self.convg1(warp_feat), True)
            war = F.relu(self.convg2(war), True)
            motion_features.append(war)
        out = F.relu(self.conv(torch.cat(motion_features, dim=1)), True)

        return torch.cat([out, disp], dim=1)

class PromptStereoRecurrentUnit(nn.Module):
    def __init__(self, features, activation=nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True, motion=False, size=None, use_res_input=True):
        super(PromptStereoRecurrentUnit, self).__init__()
        self.deconv = deconv
        self.expand = expand
        self.align_corners = align_corners
        self.size = size
        self.groups = 1

        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn) if use_res_input else None
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        if motion:
            self.resConfUnitStructure = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
                )
            
            self.resConfUnitMotion = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
                )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, structure=None, motion=None, size=None):
        output = xs[0]

        if len(xs) == 2 and self.resConfUnit1 is not None:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if motion is not None:
            
            structure = self.resConfUnitStructure(structure)
            output = self.skip_add.add(output, structure)
            
            motion = self.resConfUnitMotion(motion)
            output = self.skip_add.add(output, motion)

        output = self.out_conv(output)

        return output


class MultiPromptUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim, feat_dim, volume_dim, warp_feat_dim, cor_planes=None, num_layers=None):
        """
        
        """
        super(MultiPromptUpdateBlock, self).__init__()
        self.num_layers = args.n_gru_layers if num_layers is None else int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        self.stereo_pru = nn.ModuleList([
            PromptStereoRecurrentUnit(
                features=hidden_dim,
                bn=False,
                motion=(i == 0),
                use_res_input=(i != self.num_layers - 1),
            )
            for i in range(self.num_layers)
        ])
        self.structure_encoder = StructureEncoder(hidden_dim, 1)
        if cor_planes is None:
            cor_planes = args.corr_levels * (2*args.corr_radius + 1) * volume_dim
        self.motion_encoder = MotionEncoder(hidden_dim, cor_planes, warp_feat_dim)
        self.disp_head = DispHead(hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, (2 ** 2 ** 2) * 9, 1, padding=0)
        )

        self.update = nn.ModuleList([
            nn.Sequential(
                BasicConv(hidden_dim * (2 + (i == 0)), hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()                
            ) for i in range(self.num_layers)
        ])

    def prepare_static(self, ctx, mono_disp):
        """Per-forward cache with autograd retained."""
        enc = self.structure_encoder
        c = F.relu(enc.convc1(ctx.float()), True)
        return F.relu(enc.convc2(c), True), normalize_disparity(mono_disp)[0]

    def forward(self, net, corr, disp, ctx, warp_feat, mono_disp,
                static_features=None, need_mask=True):
        norm_mono = normalize_disparity(mono_disp)[0] if static_features is None else static_features[1]
        norm_disp, _, _ = normalize_disparity(disp)
        norm_res = norm_mono - norm_disp
        if static_features is None:
            structure = self.structure_encoder(ctx, norm_res)
        else:
            enc = self.structure_encoder
            d = F.relu(enc.convd1(norm_res), True)
            d = F.relu(enc.convd2(d), True)
            structure = F.relu(enc.conv(torch.cat([static_features[0], d], dim=1)), True)
        motion = self.motion_encoder(corr, disp, warp_feat)

        if len(net) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} recurrent states, got {len(net)}")

        if self.num_layers == 1:
            z = self.update[0](torch.cat([net[0], structure, motion], dim=1))
            pru_out = self.stereo_pru[0](net[0], structure=structure, motion=motion)
            net[0] = (1 - z) * net[0] + z * pru_out
            delta_disp = self.disp_head(net[0])
            mask = .25 * self.mask(net[0]) if need_mask else None
            return net, delta_disp, mask

        for i in reversed(range(len(net))):
            if (i == len(net) - 1):
                pooled = pool2x(net[i - 1])
                z = self.update[i](torch.cat([net[i], pooled], dim=1))
                pru_out = self.stereo_pru[i](net[i])
                net[i] = (1 - z) * net[i] + z * pru_out
            elif (i == 0):
                interp_feat = interp(net[i + 1], net[i])
                z = self.update[i](torch.cat([net[i], structure, motion], dim=1))
                pru_out = self.stereo_pru[i](net[i], interp_feat, structure=structure, motion=motion)
                net[i] = (1 - z) * net[i] + z * pru_out
            else:
                pooled = pool2x(net[i - 1])
                interp_feat = interp(net[i + 1], net[i])
                z = self.update[i](torch.cat([net[i], pooled], dim=1))
                pru_out = self.stereo_pru[i](net[i], interp_feat)
                net[i] = (1 - z) * net[i] + z * pru_out

        delta_disp = self.disp_head(net[0])
        mask = .25 * self.mask(net[0]) if need_mask else None

        return net, delta_disp, mask



class PromptConvGRU(nn.Module):
    """IGEV-style ConvGRU whose external inputs carry stereo prompt features."""

    def __init__(self, hidden_dim, input_dim, kernel_size=1):
        super().__init__()
        gate_dim = hidden_dim + input_dim
        padding = kernel_size // 2
        self.convz = nn.Conv2d(gate_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(gate_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(gate_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, hidden, context_z, context_r, context_q, *inputs):
        features = torch.cat(inputs, dim=1)
        hidden_features = torch.cat([hidden, features], dim=1)
        update_gate = torch.sigmoid(self.convz(hidden_features) + context_z)
        reset_gate = torch.sigmoid(self.convr(hidden_features) + context_r)
        candidate = torch.tanh(
            self.convq(torch.cat([reset_gate * hidden, features], dim=1))
            + context_q
        )
        return (1.0 - update_gate) * hidden + update_gate * candidate


class MultiPromptGRUUpdateBlock(nn.Module):
    """Multi-scale prompt update implemented with bounded ConvGRU cells."""

    def __init__(self, args, hidden_dim, feat_dim, volume_dim,
                 warp_feat_dim, cor_planes=None, num_layers=None):
        super().__init__()
        self.num_layers = args.n_gru_layers if num_layers is None else int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        self.structure_encoder = StructureEncoder(hidden_dim, 1)
        if cor_planes is None:
            cor_planes = args.corr_levels * (2 * args.corr_radius + 1) * volume_dim
        self.motion_encoder = MotionEncoder(hidden_dim, cor_planes, warp_feat_dim)

        input_dims = []
        for level in range(self.num_layers):
            if self.num_layers == 1:
                input_dims.append(2 * hidden_dim)
            elif level == 0:
                input_dims.append(3 * hidden_dim)
            elif level == self.num_layers - 1:
                input_dims.append(hidden_dim)
            else:
                input_dims.append(2 * hidden_dim)
        self.gru_layers = nn.ModuleList([
            PromptConvGRU(hidden_dim, input_dim)
            for input_dim in input_dims
        ])

        self.disp_head = DispHead(hidden_dim)
        upsample_factor = 2 ** args.n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, (upsample_factor ** 2) * 9, 1, padding=0),
        )

    def forward(self, net, context_gates, corr, disp, ctx, warp_feat, mono_disp):
        norm_mono, _, _ = normalize_disparity(mono_disp)
        norm_disp, _, _ = normalize_disparity(disp)
        structure = self.structure_encoder(ctx, norm_mono - norm_disp)
        motion = self.motion_encoder(corr, disp, warp_feat)

        if len(net) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} recurrent states, got {len(net)}"
            )
        if len(context_gates) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} context levels, got {len(context_gates)}"
            )

        if self.num_layers == 1:
            net[0] = self.gru_layers[0](
                net[0], *context_gates[0], structure, motion
            )
        else:
            for level in reversed(range(self.num_layers)):
                if level == self.num_layers - 1:
                    net[level] = self.gru_layers[level](
                        net[level], *context_gates[level],
                        pool2x(net[level - 1]),
                    )
                elif level == 0:
                    net[level] = self.gru_layers[level](
                        net[level], *context_gates[level], structure, motion,
                        interp(net[level + 1], net[level]),
                    )
                else:
                    net[level] = self.gru_layers[level](
                        net[level], *context_gates[level],
                        pool2x(net[level - 1]),
                        interp(net[level + 1], net[level]),
                    )

        delta_disp = self.disp_head(net[0])
        mask = 0.25 * self.mask(net[0])
        return net, delta_disp, mask
