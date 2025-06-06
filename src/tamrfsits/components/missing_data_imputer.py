#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains code for missing data inputers in sits
"""
import torch
from einops import parse_shape, rearrange, repeat

from tamrfsits.core.time_series import MonoModalSITS


class MissingDataImputer(torch.nn.Module):
    """
    A simple missing data inputer
    """

    def __init__(
        self,
        nb_features: int,
        default_value: float = 0.0,
        learnable_token: bool = False,
    ):
        """
        If value is not provided, a learnable token is used
        """
        super().__init__()
        default_token = torch.full(((nb_features,)), default_value)

        if learnable_token:
            self.register_parameter(
                "default_token", torch.nn.parameter.Parameter(default_token)
            )
        else:
            self.register_buffer("default_token", default_token)

    def forward(
        self,
        sits: MonoModalSITS,
        strip_mask: bool = False,
    ) -> MonoModalSITS:
        """
        Forward call
        """
        if sits.mask is None:
            return sits

        data_shape = parse_shape(sits.data, "b t c w h")

        data = rearrange(sits.data, "b t c w h -> b t w h c")

        mask = repeat(sits.mask, "b t w h -> b t w h c", c=data.shape[-1])
        token = repeat(
            self.default_token.to(dtype=data.dtype),
            "c -> b t w h c",
            b=data.shape[0],
            t=data.shape[1],
            w=data.shape[2],
            h=data.shape[3],
        )
        data = data.where(~mask, token)

        data = rearrange(data, "b t w h c -> b t c w h", **data_shape)

        if strip_mask:
            return MonoModalSITS(data, sits.doy)

        return MonoModalSITS(data, sits.doy, sits.mask)
