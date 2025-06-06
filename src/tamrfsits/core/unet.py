# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Modified Unet from following article:

@article{sambandham2024deep,
title={Deep learning-based harmonization and super-resolution of Landsat-8 and
        Sentinel-2 images},
author={Sambandham, Venkatesh Thirugnana and Kirchheim, Konstantin and
        Ortmeier, Frank and Mukhopadhaya, Sayan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={212},
  pages={274--288},
  year={2024},
  publisher={Elsevier}
}

Original code:
https://github.com/venkatesh-thiru/Deep-Harmonization/blob/main/training/models/UNet.py

Original code modified for:
- typehints
- style
"""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

UpsamplingMode = Literal["upsample", "upconv"]


class UNetConvBlock(nn.Module):
    """
    Down block for Unet
    """

    def __init__(self, in_channel: int, out_channel: int):
        """
        Class initializer
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        """
        out = self.block(x)

        return out


class UNetUpConvBlock(nn.Module):
    """
    Up block for Unet
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        upmode: UpsamplingMode,
        up_factor: int = 2,
    ):
        """
        Class initializer
        """
        super().__init__()

        self.upsize: nn.Module
        if upmode == "upsample":
            self.upsize = nn.Sequential(
                nn.Upsample(
                    scale_factor=up_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            )
        elif upmode == "upconv":
            self.upsize = nn.ConvTranspose2d(
                in_channel, out_channel, kernel_size=up_factor, stride=up_factor
            )
        else:
            raise ValueError(upmode)

        self.conv = UNetConvBlock(in_channel, out_channel)

    def forward(self, x: torch.Tensor, residue: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        """
        x = self.upsize(x)
        x = F.interpolate(x, size=residue.shape[2:], mode="bilinear")
        out = torch.cat([x, residue], dim=1)
        out = self.conv(out)
        return out


class SRUNet(nn.Module):
    """
    Modified UNet from sambandham2024deep
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 6,
        depth: int = 3,
        growth_factor: int = 6,
        interp_mode: str = "bilinear",
        scale_factor: float | None = None,
        up_mode: UpsamplingMode = "upsample",
    ):
        """
        Class initializer

        Parameter SR_model replaced by optional scale_factor
        (scale_factor = 3 is equivalent to SR_model=True in original
        code)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.growth_factor = growth_factor
        self.interp_mode = interp_mode
        prev_channels = self.in_channels
        self.encoding_module = nn.ModuleList()
        self.upmode = up_mode

        for i in range(self.depth):
            self.encoding_module.append(
                UNetConvBlock(
                    in_channel=prev_channels, out_channel=2 ** (self.growth_factor + i)
                )
            )
            prev_channels = 2 ** (self.growth_factor + i)

        self.decoding_module = nn.ModuleList()

        for i in reversed(range(self.depth - 1)):
            self.decoding_module.append(
                UNetUpConvBlock(
                    prev_channels, 2 ** (self.growth_factor + i), upmode="upconv"
                )
            )
            prev_channels = 2 ** (self.growth_factor + i)

        if scale_factor is not None:
            self.final: nn.Module = nn.Sequential(
                nn.Upsample(
                    scale_factor=scale_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding="same"),
                nn.PReLU(),
                nn.Conv2d(prev_channels, out_channels, kernel_size=1),
            )
        else:
            self.final = nn.Conv2d(prev_channels, out_channels, 1, 1)

    def forward(
        self, ms: torch.Tensor, pan: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        forward method
        """
        if pan is None:
            x = ms
        else:
            x = torch.cat([ms, pan], dim=1)
        blocks = []
        nb_blocks = len(self.encoding_module) - 1
        for i, down in enumerate(self.encoding_module):
            x = down(x)
            if i != nb_blocks:
                blocks.append(x)
                # pylint: disable=not-callable, dotted-import-in-loop
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.decoding_module):
            x = up(x, blocks[-i - 1])
        x = self.final(x)
        return x
