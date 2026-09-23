import torch
import torch.nn.functional as F


def meshgrid(img):
    """
    img: [B, C, H, W]
    return: [B, 2, H, W]
    """
    b, _, h, w = img.shape
    device = img.device
    dtype = img.dtype

    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, w).expand(1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, h, 1).expand(1, h, w)

    grid = torch.cat((x, y), dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, 2, H, W]

    return grid


def normalize_coords(grid, align_corners=True):
    """
    grid: [B, 2, H, W]
    return: [B, H, W, 2]
    """
    b, _, h, w = grid.shape

    x = grid[:, 0]
    y = grid[:, 1]

    if align_corners:
        x = 2 * x / (w - 1) - 1
        y = 2 * y / (h - 1) - 1
    else:
        x = 2 * (x + 0.5) / w - 1
        y = 2 * (y + 0.5) / h - 1

    return torch.stack((x, y), dim=-1)


def interp(x, sample_grid, padding_mode='border', align_corners=True):
    original_dtype = x.dtype

    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.grid_sample(
            x.float(),
            sample_grid.float(),
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=align_corners
        )

    return output_fp32.to(original_dtype)


def disp_warp(img, disp, padding_mode='border', align_corners=True):
    """
    Warp right image/feature to left view using left disparity.

    Args:
        img:  [B, C, H, W], usually right image/feature
        disp: [B, 1, H, W], positive left disparity
    Returns:
        warped_img: [B, C, H, W]
    """
    grid = meshgrid(img)  # [B, 2, H, W]

    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = grid + offset  # [B, 2, H, W]

    sample_grid_norm = normalize_coords(sample_grid, align_corners=align_corners)

    warped_img = interp(
        img,
        sample_grid_norm,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    return warped_img


def disp_warp_multi(img, disp_samples, padding_mode='border', align_corners=True):
    """
    Warp right feature with multiple disparity hypotheses.

    Args:
        img: [B, C, H, W]
        disp_samples: [B, S, H, W] or [B, S, 1, H, W]
    Returns:
        warped: [B, S, C, H, W]
    """
    if disp_samples.ndim == 5:
        disp_samples = disp_samples.squeeze(2)
    if disp_samples.ndim != 4:
        raise ValueError(f"disp_samples must have shape [B, S, H, W] or [B, S, 1, H, W], got {disp_samples.shape}")

    b, s, h, w = disp_samples.shape
    c = img.shape[1]

    img_rep = img.unsqueeze(1).expand(b, s, c, h, w).reshape(b * s, c, h, w)
    disp_rep = disp_samples.reshape(b * s, 1, h, w)
    warped = disp_warp(img_rep, disp_rep, padding_mode=padding_mode, align_corners=align_corners)

    return warped.view(b, s, c, h, w)


def sample_right_feature_pyramid(
    right_features,
    disp,
    offsets,
    compression_ratios=(1, 2, 4),
    padding_mode='zeros',
    align_corners=True,
    compact=False,
):
    """Sample right features around the current left-view disparity.

    Args:
        right_features: sequence of [B,C,H,W_r] tensors. Each tensor must be
            compressed directly from the original right feature by the
            corresponding ratio.
        disp: current disparity [B,1,H,W] in the uncompressed 1/4-feature unit.
        offsets: one-dimensional offsets in each branch's compressed-cell unit.
        compression_ratios: width compression ratio for every right feature.

    Returns:
        Sampled right features [B, num_scales*num_offsets, C, H, W].

    For a width-stride convolution with kernel_size == stride == ratio,
    compressed coordinate j has its center at
    ratio*j + (ratio-1)/2 in the original feature coordinates. Therefore the
    right lookup coordinate is:

        j = (x_left - disp - (ratio-1)/2) / ratio + offset
    """
    if disp.ndim != 4 or disp.shape[1] != 1:
        raise ValueError(f"disp must be [B,1,H,W], got {tuple(disp.shape)}")
    if len(right_features) != len(compression_ratios):
        raise ValueError(f"Expected one compression ratio per feature, got {len(right_features)} features and {len(compression_ratios)} ratios")

    batch, _, query_height, query_width = disp.shape
    offsets = torch.as_tensor(offsets, device=disp.device, dtype=disp.dtype).flatten()
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one value")

    x_left = torch.arange(query_width, device=disp.device, dtype=disp.dtype).view(1, 1, 1, query_width)
    y = torch.arange(query_height, device=disp.device, dtype=disp.dtype).view(1, 1, query_height, 1)
    sampled_scales = []

    for right_feat, ratio in zip(right_features, compression_ratios):
        if right_feat.ndim != 4:
            raise ValueError(f"Every right feature must be [B,C,H,W], got {tuple(right_feat.shape)}")
        if right_feat.shape[0] != batch or right_feat.shape[2] != query_height:
            raise ValueError(f"Right feature batch/height must match disp, got {tuple(right_feat.shape)} and {tuple(disp.shape)}")
        if ratio < 1:
            raise ValueError(f"compression ratios must be positive, got {ratio}")

        source_width = right_feat.shape[-1]
        source_height = right_feat.shape[-2]
        center_offset = (float(ratio) - 1.0) / 2.0
        x_right = (x_left - disp - center_offset) / float(ratio)
        x_right = x_right + offsets.view(1, -1, 1, 1)
        y_right = y.expand(batch, offsets.numel(), query_height, query_width)

        if align_corners:
            x_norm = 2.0 * x_right / max(source_width - 1, 1) - 1.0
            y_norm = 2.0 * y_right / max(source_height - 1, 1) - 1.0
        else:
            x_norm = 2.0 * (x_right + 0.5) / source_width - 1.0
            y_norm = 2.0 * (y_right + 0.5) / source_height - 1.0

        sample_count = offsets.numel()
        channels = right_feat.shape[1]
        grid = torch.stack([x_norm, y_norm], dim=-1)
        with torch.cuda.amp.autocast(enabled=False):
            if compact:
                # Treat the offset dimension as additional output rows. This
                # samples the same coordinates without materializing K copies
                # of the complete right feature tensor.
                compact_grid = grid.reshape(
                    batch, sample_count * query_height, query_width, 2
                )
                sampled = F.grid_sample(
                    right_feat.float(),
                    compact_grid.float(),
                    mode='bilinear',
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                sampled = sampled.reshape(
                    batch, channels, sample_count, query_height, query_width
                ).permute(0, 2, 1, 3, 4)
            else:
                grid = grid.reshape(
                    batch * sample_count, query_height, query_width, 2
                )
                expanded_right = right_feat.unsqueeze(1).expand(
                    batch, sample_count, channels, source_height, source_width
                )
                expanded_right = expanded_right.reshape(
                    batch * sample_count, channels, source_height, source_width
                )
                sampled = F.grid_sample(
                    expanded_right.float(),
                    grid.float(),
                    mode='bilinear',
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                sampled = sampled.reshape(
                    batch, sample_count, channels, query_height, query_width
                )
        sampled_scales.append(sampled.to(right_feat.dtype))

    return torch.cat(sampled_scales, dim=1) #batch, sample_count*level, channels, query_height, query_width
