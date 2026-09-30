"""IGEV++-style ImageNet MobileNetV2/FPN stereo descriptor.

The image path follows IGEV++'s Feature, stem_2/stem_4, conv and desc
modules. It intentionally does not include any cost-volume or monocular head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class IGEVBasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, deconv=False,
                 kernel_size=3, stride=1, padding=1, use_in=True):
        super().__init__()
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size, stride,
                         padding, bias=False)
        self.IN = nn.InstanceNorm2d(out_channels)
        self.use_in = use_in

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        return F.leaky_relu(x)


class IGEVConv2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = IGEVBasicConv(
            in_channels, out_channels, deconv=True,
            kernel_size=4, stride=2, padding=1,
        )
        self.conv2 = IGEVBasicConv(
            out_channels * 2, out_channels * 2,
            kernel_size=3, stride=1, padding=1,
        )

    def forward(self, x, skip):
        x = self.conv1(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        return self.conv2(torch.cat((x, skip), dim=1))


class IGEVPPFeature(nn.Module):
    """IGEV++ Feature: MobileNetV2 at 1/2..1/32 and top-down FPN."""

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = timm.create_model(
            'mobilenetv2_100', pretrained=pretrained, features_only=True,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = getattr(backbone, 'act1', nn.Identity())
        self.block0 = nn.Sequential(*backbone.blocks[0:1])
        self.block1 = nn.Sequential(*backbone.blocks[1:2])
        self.block2 = nn.Sequential(*backbone.blocks[2:3])
        self.block3 = nn.Sequential(*backbone.blocks[3:5])
        self.block4 = nn.Sequential(*backbone.blocks[5:6])
        self.deconv32_16 = IGEVConv2x(160, 96)
        self.deconv16_8 = IGEVConv2x(192, 32)
        self.deconv8_4 = IGEVConv2x(64, 24)
        self.conv4 = IGEVBasicConv(48, 48)

    def forward(self, image):
        x = self.act1(self.bn1(self.conv_stem(image)))
        feat2 = self.block0(x)
        feat4 = self.block1(feat2)
        feat8 = self.block2(feat4)
        feat16 = self.block3(feat8)
        feat32 = self.block4(feat16)
        feat16 = self.deconv32_16(feat32, feat16)
        feat8 = self.deconv16_8(feat16, feat8)
        feat4 = self.deconv8_4(feat8, feat4)
        return self.conv4(feat4)


class IGEVPPMatchEncoder(nn.Module):
    """Shared stereo encoder producing IGEV++ 96-channel match descriptors."""

    out_dim = 96

    def __init__(self, pretrained=True):
        super().__init__()
        self.feature = IGEVPPFeature(pretrained=pretrained)
        self.stem_2 = nn.Sequential(
            IGEVBasicConv(3, 32, stride=2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            IGEVBasicConv(32, 48, stride=2),
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU(),
        )
        self.conv = IGEVBasicConv(96, 96)
        self.desc = nn.Conv2d(96, 96, 1)

    def forward(self, image1, image2):
        if image1.shape != image2.shape:
            raise ValueError('Stereo images must have identical shapes')
        batch = image1.shape[0]
        images = torch.cat((image1, image2), dim=0)
        features = self.feature(images)
        stem4 = self.stem_4(self.stem_2(images))
        descriptors = self.desc(self.conv(torch.cat((features, stem4), dim=1)))
        return descriptors[:batch], descriptors[batch:]


class IGEVPPResidualBlock(nn.Module):
    """RGB context residual block with IGEV++ batch normalization."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.norm3 = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                self.norm3,
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        return self.relu(y + identity)


class IGEVPPContextEncoder(nn.Module):
    """IGEV++ RGB-only MultiBasicEncoder at 1/4, 1/8 and 1/16 resolution."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            IGEVPPResidualBlock(64, 64), IGEVPPResidualBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            IGEVPPResidualBlock(64, 96, stride=2),
            IGEVPPResidualBlock(96, 96),
        )
        self.layer3 = nn.Sequential(
            IGEVPPResidualBlock(96, 128, stride=2),
            IGEVPPResidualBlock(128, 128),
        )
        self.layer4 = nn.Sequential(
            IGEVPPResidualBlock(128, 128, stride=2),
            IGEVPPResidualBlock(128, 128),
        )
        self.layer5 = nn.Sequential(
            IGEVPPResidualBlock(128, 128, stride=2),
            IGEVPPResidualBlock(128, 128),
        )
        self.outputs04 = nn.ModuleList([
            self._output_head(hidden_dim) for _ in range(2)
        ])
        self.outputs08 = nn.ModuleList([
            self._output_head(hidden_dim) for _ in range(2)
        ])
        self.outputs16 = nn.ModuleList([
            nn.Conv2d(128, hidden_dim, 3, padding=1)
            for _ in range(2)
        ])

    @staticmethod
    def _output_head(hidden_dim):
        return nn.Sequential(
            IGEVPPResidualBlock(128, 128),
            nn.Conv2d(128, hidden_dim, 3, padding=1),
        )

    def forward(self, image, num_layers=3):
        if num_layers not in (1, 2, 3):
            raise ValueError(f'num_layers must be 1, 2, or 3, got {num_layers}')
        x = self.relu1(self.norm1(self.conv1(image)))
        x = self.layer3(self.layer2(self.layer1(x)))
        outputs = [tuple(head(x) for head in self.outputs04)]
        if num_layers >= 2:
            x = self.layer4(x)
            outputs.append(tuple(head(x) for head in self.outputs08))
        if num_layers >= 3:
            x = self.layer5(x)
            outputs.append(tuple(head(x) for head in self.outputs16))
        return outputs
