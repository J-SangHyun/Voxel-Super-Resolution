# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from pathlib import Path


# VUNet (Volumetric U-Net)
class VUNet(nn.Module):
    name = 'VUNet'
    need_upsampling = True
    path = Path(os.path.dirname(__file__))

    def __init__(self, low_grid, high_grid):
        super(VUNet, self).__init__()
        self.low_grid, self.high_grid = low_grid, high_grid

        self.input_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # downsampling layers
        self.down1 = DownSamplingLayer(16, 32)
        self.down2 = DownSamplingLayer(32, 64)
        self.down3 = DownSamplingLayer(64, 128)

        # upsampling layers
        self.up1 = UpSamplingLayer(128, 64)
        self.up2 = UpSamplingLayer(64, 32)
        self.up3 = UpSamplingLayer(32, 16)

        self.output_conv = nn.Conv3d(16, 1, kernel_size=(1, 1, 1))

    def forward(self, lr):
        x1 = self.input_conv(lr)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.up1(x3, x)
        x = self.up2(x2, x)
        x = self.up3(x1, x)
        x = torch.sigmoid(self.output_conv(x))
        return x


class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingLayer, self).__init__()
        self.max_pool = nn.MaxPool3d(2)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class UpSamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingLayer, self).__init__()
        self.upsampling = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip_x, down_x):
        down_x = self.upsampling(down_x)
        x = torch.cat([skip_x, down_x], dim=1)
        x = self.double_conv(x)
        return x
