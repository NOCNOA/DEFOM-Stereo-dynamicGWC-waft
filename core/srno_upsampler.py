import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNOStyleDisparityUpsampler(nn.Module):
    def __init__(self, hidden_dim, context_dim=0, hr_dim=0, scale=4, k=3, proj_dim=8):
        super().__init__()
        assert k % 2 == 1, "SRNOStyleDisparityUpsampler expects an odd kernel size"
        self.scale = scale
        self.k = k
        self.num_neighbors = k * k
        self.hr_dim = hr_dim

        self.hidden_proj = nn.Conv2d(hidden_dim, proj_dim, kernel_size=1, padding=0)  #降维
        if context_dim > 0:
            self.context_proj = nn.Conv2d(context_dim, proj_dim, kernel_size=1, padding=0)  #降维
            context_proj_dim = proj_dim
        else:
            self.context_proj = None
            context_proj_dim = 0

        # Per LR location we predict scale*scale different 3x3 kernels. The
        # coordinate channels are SRNO-style relative offsets and bilinear areas
        # for every sub-pixel query and every low-resolution neighbor.
        feature_channels = self.num_neighbors * (1 + proj_dim + context_proj_dim)
        coord_channels = self.num_neighbors * 3 * scale * scale
        hr_channels = hr_dim * scale * scale
        in_channels = feature_channels + coord_channels + hr_channels

        self.upsample_head = nn.Sequential(  #根据领域、坐标、高分辨率左图特征来预测权重
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, self.num_neighbors * scale * scale, kernel_size=1, padding=0),
        )

    def _unfold_neighbors(self, x): #3*3邻域展开
        b, c, h, w = x.shape
        pad = self.k // 2
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.unfold(x, kernel_size=self.k, padding=0).view(
            b, c * self.num_neighbors, h, w
        )

    def _coordinate_features(self, disp_low):
        device, dtype = disp_low.device, disp_low.dtype
        s, k, pad = self.scale, self.k, self.k // 2

        y_offsets, x_offsets = torch.meshgrid(  #每个邻居相对于当前低分辨率中心点的位置。
            torch.arange(-pad, pad + 1, device=device, dtype=dtype),
            torch.arange(-pad, pad + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        x_offsets = x_offsets.reshape(self.num_neighbors, 1, 1)
        y_offsets = y_offsets.reshape(self.num_neighbors, 1, 1)

        sub_y, sub_x = torch.meshgrid( #高分辨率子像素在低分辨率 cell 内的位置
            torch.arange(s, device=device, dtype=dtype),
            torch.arange(s, device=device, dtype=dtype),
            indexing="ij",
        )
        # Map each HR pixel center to continuous LR coordinates. For sub-pixels
        # inside LR cell q, x_c - q lies in roughly [-0.5, 0.5].
        sub_x = (sub_x + 0.5) / s - 0.5
        sub_y = (sub_y + 0.5) / s - 0.5

        rel_x = sub_x.unsqueeze(0) - x_offsets
        rel_y = sub_y.unsqueeze(0) - y_offsets
        bilinear = (1.0 - rel_x.abs()).clamp_min(0.0) * (1.0 - rel_y.abs()).clamp_min(0.0)  #相当于相对位置的大小

        coord = torch.stack([rel_x, rel_y, bilinear], dim=1)
        coord = coord.reshape(1, self.num_neighbors * 3 * s * s, 1, 1)
        return coord.expand(disp_low.shape[0], -1, disp_low.shape[-2], disp_low.shape[-1])

    def forward(self, disp_low, hidden_state, context_feature=None, left_hr_feature=None):
        """
        disp_low:       [B, 1, H, W]
        hidden_state:   [B, C, H, W]
        context_feature:[B, Cc, H, W] or None
        left_hr_feature:[B, Ch, H*scale, W*scale] or None

        return:
            disp_up: [B, 1, H*scale, W*scale]
        """
        assert disp_low.dim() == 4 and disp_low.shape[1] == 1, tuple(disp_low.shape)
        b, _, h, w = disp_low.shape
        s = self.scale
        out_dtype = disp_low.dtype
        work_dtype = self.hidden_proj.weight.dtype
        disp_work = disp_low.to(dtype=work_dtype)
        assert hidden_state.shape[0] == b and hidden_state.shape[-2:] == (h, w), (
            hidden_state.shape, disp_low.shape
        )
        if context_feature is not None:
            assert context_feature.shape[0] == b and context_feature.shape[-2:] == (h, w), (
                context_feature.shape, disp_low.shape
            )

        hidden = self.hidden_proj(hidden_state.to(dtype=work_dtype))  #降维 256->8
        features = [
            self._unfold_neighbors(disp_work),
            self._unfold_neighbors(hidden),
        ]

        if self.context_proj is not None:
            if context_feature is None:
                context = hidden.new_zeros(b, self.context_proj.in_channels, h, w)
            else:
                context = context_feature.to(dtype=work_dtype)
            features.append(self._unfold_neighbors(self.context_proj(context)))

        features.append(self._coordinate_features(disp_work))

        if self.hr_dim > 0:
            if left_hr_feature is None:
                hr_feat = disp_work.new_zeros(b, self.hr_dim, h * s, w * s)
            else:
                assert left_hr_feature.shape[0] == b and left_hr_feature.shape[-2:] == (h * s, w * s), (
                    left_hr_feature.shape, disp_low.shape
                )
                hr_feat = left_hr_feature.to(dtype=work_dtype)
            features.append(F.pixel_unshuffle(hr_feat, s))

        head_input = torch.cat(features, dim=1)
        weight_logits = self.upsample_head(head_input)
        weights = weight_logits.view(b, self.num_neighbors, s, s, h, w)
        weights = torch.softmax(weights, dim=1)

        pad = self.k // 2
        disp_pad = F.pad(disp_work, (pad, pad, pad, pad), mode="replicate") if pad > 0 else disp_work
        disp_neighbors = F.unfold(disp_pad, kernel_size=self.k, padding=0)
        disp_neighbors = disp_neighbors.view(b, 1, self.num_neighbors, h, w)
        disp_neighbors = disp_neighbors.unsqueeze(3).unsqueeze(3)

        disp_up = (weights.unsqueeze(1) * (s * disp_neighbors)).sum(dim=2)
        disp_up = disp_up.permute(0, 1, 4, 2, 5, 3).reshape(b, 1, h * s, w * s)
        assert disp_up.shape[-2:] == (h * s, w * s), (disp_up.shape, disp_low.shape)
        return disp_up.to(dtype=out_dtype)
