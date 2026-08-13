"""
1\输入视差图转为三维查询点的形式
2\引入USRNO
3、查询costvolume
4、经过一个galerkinatt
5、卷积生成视差
"""
from turtle import forward
import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
from USRNO_PRIVETE_main.models.sronet import SRNO3D

import torch
import torch.nn.functional as F


def make_disp_range_coords(
    disp: torch.Tensor,
    volume_depth: int,
    radius: int,
    *,
    step: float = 1.0,
    offsets: torch.Tensor = None,
    align_corners: bool = False,
    clamp: bool = True,
    coord_order: str = "dhw",
) -> torch.Tensor:
    """
    从视差图 [B,1,H,W] 生成“范围搜索”用的 cost volume 查询坐标。

    Args:
        disp: [B,1,H,W]
            当前视差图，默认数值单位是 volume 的 disparity 索引坐标。
        volume_depth: int
            cost volume 在 disparity 轴上的长度 D。
        radius: int
            左右搜索半径；若 offsets=None，则生成 K=2*radius+1 个点。
        step: float
            搜索步长。比如 radius=2, step=0.5 -> offsets=[-1,-0.5,0,0.5,1]
        offsets: Tensor or None
            若给定，则直接使用自定义 offsets，形状如 [K]。
        align_corners: bool
            需与后续 F.grid_sample 保持一致。
        clamp: bool
            是否裁剪到 [-1+eps, 1-eps]。
        coord_order: str
            "dhw": 返回 [d,h,w]，适合你当前代码，后续再 flip(-1)
            "whd": 直接返回 [w,h,d]，可直接送入 5D grid_sample

    Returns:
        coord: [B, K, H, W, 3]
    """
    if disp.ndim != 4 or disp.shape[1] != 1:
        raise ValueError(f"disp must be [B,1,H,W], got {tuple(disp.shape)}")
    if volume_depth <= 0:
        raise ValueError(f"volume_depth must be positive, got {volume_depth}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    B, _, H, W = disp.shape
    device = disp.device
    dtype = disp.dtype

    if offsets is None:
        offsets = torch.arange(
            -radius, radius + 1, device=device, dtype=dtype
        ) * step
    else:
        offsets = offsets.to(device=device, dtype=dtype)

    K = offsets.numel()

    # 基础像素网格
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    xx = xx.view(1, 1, H, W).expand(B, K, H, W)
    yy = yy.view(1, 1, H, W).expand(B, K, H, W)

    # disparity 查询位置: [B,K,H,W]
    dd = disp[:, 0:1, :, :] + offsets.view(1, K, 1, 1)

    # 索引坐标 -> [-1,1] 归一化坐标
    if align_corners:
        if W > 1:
            w_norm = 2.0 * xx / (W - 1) - 1.0
        else:
            w_norm = torch.zeros_like(xx)

        if H > 1:
            h_norm = 2.0 * yy / (H - 1) - 1.0
        else:
            h_norm = torch.zeros_like(yy)

        if volume_depth > 1:
            d_norm = 2.0 * dd / (volume_depth - 1) - 1.0
        else:
            d_norm = torch.zeros_like(dd)
    else:
        w_norm = 2.0 * (xx + 0.5) / W - 1.0
        h_norm = 2.0 * (yy + 0.5) / H - 1.0
        d_norm = 2.0 * (dd + 0.5) / volume_depth - 1.0

    if clamp:
        eps = 1e-6
        w_norm = w_norm.clamp(-1.0 + eps, 1.0 - eps)
        h_norm = h_norm.clamp(-1.0 + eps, 1.0 - eps)
        d_norm = d_norm.clamp(-1.0 + eps, 1.0 - eps)

    if coord_order == "dhw":
        coord = torch.stack([d_norm, h_norm, w_norm], dim=-1)  # [B,K,H,W,3]
    elif coord_order == "whd":
        coord = torch.stack([w_norm, h_norm, d_norm], dim=-1)  # [B,K,H,W,3]
    else:
        raise ValueError(f"coord_order must be 'dhw' or 'whd', got {coord_order}")
    return coord


class volume2disp(nn.Module):
    def __init__(self, args):
        self.max_disp = args.max_disp
        pass
    def forward(self, costvolume, disp):
        disp_coord = make_disp_range_coords(disp, self.max_disp)
        pass

