"""DINOv2/Depth Anything V2 stereo descriptors for the encoder-only A/B."""

from pathlib import Path

import torch
import torch.nn as nn

from core.extractor import DefomEncoder
from core.utils.utils import get_danv2_io_size


class DINOv2MatchEncoder(nn.Module):
    """Return 1/4-resolution, 96-channel descriptors like IGEVPPMatchEncoder."""

    out_dim = 96

    def __init__(self, variant="vitl"):
        super().__init__()
        checkpoint = Path("checkpoints") / f"depth_anything_v2_{variant}.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                f"DINOv2 A/B requires pretrained weights at {checkpoint}; "
                "refusing to run with a randomly initialized encoder"
            )
        self.encoder = DefomEncoder(
            variant, pretrained=True, freeze=True, use_mono_head=False
        )
        self.adapter = nn.Conv2d(self.encoder.out_dim, self.out_dim, 1)
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Stereo images must have identical shapes")
        height, width = image1.shape[-2:]
        # The surrounding model uses [-1, 1]; Depth Anything uses ImageNet RGB.
        left = ((image1.float() + 1.0) * 0.5 - self.rgb_mean) / self.rgb_std
        right = ((image2.float() + 1.0) * 0.5 - self.rgb_mean) / self.rgb_std
        sizes = get_danv2_io_size(height, width, nds=2)
        left_desc, right_desc, _ = self.encoder(left, right, sizes)
        return self.adapter(left_desc), self.adapter(right_desc)
