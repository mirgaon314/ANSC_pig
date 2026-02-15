from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_spatial_module(
    x: torch.Tensor,
    module: nn.Module,
    bn: nn.Module | None = None,
    activation: nn.Module | None = None,
) -> torch.Tensor:
    """Apply a 2D module independently to each frame."""

    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = module(x)
    if bn is not None:
        x = bn(x)
    if activation is not None:
        x = activation(x)
    c_new = x.shape[1]
    h_new = x.shape[-2]
    w_new = x.shape[-1]
    x = x.reshape(b, t, c_new, h_new, w_new).permute(0, 2, 1, 3, 4)
    return x


def _apply_temporal_module(
    x: torch.Tensor,
    module: nn.Module,
    bn: nn.Module | None = None,
    activation: nn.Module | None = None,
) -> torch.Tensor:
    """Apply a 1D module along the temporal axis for every spatial location."""

    b, c, t, h, w = x.shape
    x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
    x = module(x)
    if bn is not None:
        x = bn(x)
    if activation is not None:
        x = activation(x)
    t_new = x.shape[-1]
    x = x.reshape(b, h, w, c, t_new).permute(0, 3, 4, 1, 2)
    return x


def _apply_spatial_pool(x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = pool(x)
    h_new = x.shape[-2]
    w_new = x.shape[-1]
    x = x.reshape(b, t, c, h_new, w_new).permute(0, 2, 1, 3, 4)
    return x


class SpatioTemporalUnit(nn.Module):
    """2+1D convolution: spatial Conv2d + depthwise temporal Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: int = 3,
        temporal_kernel: int = 3,
        spatial_stride: int = 1,
    ) -> None:
        super().__init__()
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=spatial_kernel,
            stride=spatial_stride,
            padding=padding,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=out_channels,
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _apply_spatial_module(x, self.spatial_conv, self.spatial_bn, self.act)
        x = _apply_temporal_module(x, self.temporal_conv, self.temporal_bn, self.act)
        return x


class ResidualSpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.unit1 = SpatioTemporalUnit(
            in_channels,
            out_channels,
            spatial_stride=stride,
        )
        self.unit2 = SpatioTemporalUnit(out_channels, out_channels)
        if downsample or in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.unit1(x)
        out = self.unit2(out)
        if self.shortcut_conv is not None:
            identity = _apply_spatial_module(identity, self.shortcut_conv, self.shortcut_bn)
        out += identity
        return self.act(out)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, centre_index: int) -> None:
        super().__init__()
        self.centre_index = centre_index
        self.att_linear = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (B, C, T)
        scores = self.att_linear(x.permute(0, 2, 1))  # (B, T, 1)
        weights = F.softmax(scores.squeeze(-1), dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x.permute(0, 2, 1))  # (B, 1, C)
        context = context.squeeze(1)  # (B, C)
        centre_feature = x[:, :, self.centre_index]
        return context, centre_feature, weights


class InteractionClassifier(nn.Module):
    def __init__(self, in_channels: int = 5, temporal_length: int = 5) -> None:
        super().__init__()
        self.temporal_length = temporal_length
        self.centre_index = temporal_length // 2

        self.stem_conv = nn.Conv2d(
            in_channels,
            32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.GELU()
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage1 = nn.Sequential(
            ResidualSpatioTemporalBlock(32, 64),
            ResidualSpatioTemporalBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ResidualSpatioTemporalBlock(64, 128, downsample=True),
            ResidualSpatioTemporalBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResidualSpatioTemporalBlock(128, 256, downsample=True),
            ResidualSpatioTemporalBlock(256, 256),
        )

        self.temporal_pool = TemporalAttention(in_channels=256, centre_index=self.centre_index)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, C, T, H, W)
        x = _apply_spatial_module(x, self.stem_conv, self.stem_bn, self.stem_act)
        x = _apply_spatial_pool(x, self.stem_pool)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        pooled = x.mean(dim=(-2, -1))  # (B, C, T)
        context, centre_feature, weights = self.temporal_pool(pooled)
        features = torch.cat([context, centre_feature], dim=-1)
        logits = self.head(features)
        return logits.squeeze(-1), weights
