"""Attention blocks used by the optional CBAM-enhanced U-Net variant."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Reweight feature channels using pooled global descriptors."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combine average-pooled and max-pooled channel summaries before the
        # sigmoid gate so the module reacts to both broad and sharp responses.
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Reweight spatial positions using pooled channel summaries."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise mean and max summarize where activations are spatially
        # strong before the attention map is projected to one channel.
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Applies channel attention followed by spatial attention on the input feature map.

    Args:
        channels:        Number of input (and output) channels.
        reduction:       Channel reduction ratio for the MLP in channel attention.
        spatial_kernel:  Kernel size for the spatial attention convolution (3 or 7).

    Example::
        cbam = CBAM(channels=256)
        out  = cbam(feature_map)   # same shape as input
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CBAM applies channel recalibration first, then refines the remaining
        # activations spatially while preserving the input tensor shape.
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x
