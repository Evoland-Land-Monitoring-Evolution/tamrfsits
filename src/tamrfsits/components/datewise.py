#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains a generic structure that applies a given torch.nn.Module
to all dates in a sits
"""

import torch
from einops import parse_shape, rearrange

from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.core.utils import CompiledTorchModule


class DateWiseSITSModule(torch.nn.Module):
    """
    Generic module for applying a torch.nn.Module to all dates in a SITS
    """

    def __init__(
        self,
        model: torch.nn.Module | CompiledTorchModule | None,
        upsampling_factor: float | None = None,
        upsampling_mode: str = "bicubic",
        early_upsampling: bool = False,
        compiled: bool = True,
    ):
        """
        Class initializer
        """
        super().__init__()

        self.model: torch.nn.Module | CompiledTorchModule | None = model
        if compiled and self.model is not None:
            self.model.compile(dynamic=True)
        self.upsampling_factor = upsampling_factor
        self.upsampling_mode = upsampling_mode
        self.early_upsampling = early_upsampling

    def forward(self, sits: MonoModalSITS) -> MonoModalSITS:
        """
        Implementation of forward method
        """

        # Process data tensor
        data_shape = parse_shape(sits.data, "b t c w h")
        data = rearrange(sits.data, "b t c w h -> (b t) c w h")
        # Apply upsampling factor if required
        if self.upsampling_factor and self.early_upsampling:
            data = torch.nn.functional.interpolate(
                data,
                scale_factor=self.upsampling_factor,
                mode=self.upsampling_mode,
                align_corners=False,
            )
        if self.model is not None:
            data = self.model(data)
        # Apply upsampling factor if required
        if self.upsampling_factor and not self.early_upsampling:
            data = torch.nn.functional.interpolate(
                data,
                scale_factor=self.upsampling_factor,
                mode=self.upsampling_mode,
                align_corners=False,
            )

        data = rearrange(
            data, "(b t) c w h -> b t c w h", b=data_shape["b"], t=data_shape["t"]
        )
        data_shape = parse_shape(data, "b t c w h")

        # Process mask tensor
        mask: torch.Tensor | None = None
        if sits.mask is not None:
            mask_shape = parse_shape(sits.mask, "b t w h")

            # If spatial dimension were not modified, there is nothing to do
            if (
                mask_shape["w"] == data_shape["w"]
                and mask_shape["h"] == data_shape["h"]
            ):
                mask = sits.mask
            else:
                # Else, we need to resample mask
                upsampling_factor = data_shape["w"] / mask_shape["w"]
                assert data_shape["h"] == upsampling_factor * mask_shape["h"]

                mask = rearrange(sits.mask, "b t w h -> (b t) w h")
                mask = (
                    torch.nn.functional.interpolate(
                        mask[:, None, ...].to(dtype=torch.float32),
                        scale_factor=upsampling_factor,
                        mode="nearest",
                    )[:, 0, ...]
                    > 0
                )
                mask = rearrange(
                    mask, "(b t) w h -> b t w h", b=mask_shape["b"], t=mask_shape["t"]
                )

        return MonoModalSITS(data, sits.doy, mask)
