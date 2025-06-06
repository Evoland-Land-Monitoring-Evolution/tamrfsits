#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the positional encoding used in the TaMmFSiTS achitecture
"""

from enum import Enum

import torch
from einops import parse_shape, rearrange, repeat

from tamrfsits.core.time_series import MonoModalSITS


class PositionalEncodingMode(Enum):
    """
    Describe the different positional encoding modes
    """

    YIELD = "YIELD"
    ADD = "ADD"
    CONCAT = "CONCAT"


class FixedPositionalEncoder(torch.nn.Module):
    """
    Standard positional encoder
    """

    def __init__(
        self,
        div: float = 1000,
        offset: int = 0,
        nb_features: int = 64,
        mode: PositionalEncodingMode | str = PositionalEncodingMode.ADD,
    ):
        """
        Constructor
        """
        super().__init__()
        self.div = div
        self.offset = offset
        if isinstance(mode, str):
            self.mode = PositionalEncodingMode(mode)
        else:
            self.mode = mode
        self.nb_features = nb_features

    def forward(
        self,
        sits: MonoModalSITS,
    ) -> MonoModalSITS:
        """
        Forward method
        """
        nb_features = self.nb_features

        pe = self.compute_pe(sits.doy, nb_features)

        data_shape = parse_shape(sits.data, "b t c w h")

        pe = repeat(
            pe,
            "b t c -> b t c w h",
            w=data_shape["w"],
            h=data_shape["h"],
            c=self.nb_features,
        )

        if self.mode is PositionalEncodingMode.ADD:
            return MonoModalSITS(sits.data + pe, sits.doy, sits.mask)
        if self.mode is PositionalEncodingMode.YIELD:
            return MonoModalSITS(pe, sits.doy, sits.mask)
        if self.mode is PositionalEncodingMode.CONCAT:
            return MonoModalSITS(torch.cat([sits.data, pe], dim=2), sits.doy, sits.mask)

        raise NotImplementedError

    def get_embedding_size(self, nb_input_features: int) -> int:
        """
        Service to compute the embedding size based on the number of features
        """
        if self.mode is PositionalEncodingMode.ADD:
            assert nb_input_features == self.nb_features
            return nb_input_features
        if self.mode is PositionalEncodingMode.YIELD:
            return self.nb_features
        if self.mode is PositionalEncodingMode.CONCAT:
            return self.nb_features + nb_input_features

        raise NotImplementedError

    def compute_pe(
        self,
        doy: torch.Tensor,
        nb_features: int,
    ) -> torch.Tensor:
        """
        Forward method
        """
        denom = torch.pow(
            self.div,
            2
            * (
                torch.arange(
                    self.offset, self.offset + nb_features, device=doy.device
                ).float()
                // 2
            )
            / nb_features,
        )

        pe = doy[:, :, None] / denom[None, None, :]
        pe[:, :, 0::2] = torch.sin(pe[:, :, 0::2])
        pe[:, :, 1::2] = torch.cos(pe[:, :, 1::2])

        return pe


class LearnablePositionalEncoder(torch.nn.Module):
    """
    A class for learnable positional encoder (inspired from mTAN)
    """

    def __init__(
        self,
        nb_features: int = 64,
        mode: PositionalEncodingMode | str = PositionalEncodingMode.ADD,
    ):
        """
        Initializer
        """
        super().__init__()

        if isinstance(mode, str):
            self.mode = PositionalEncodingMode(mode)
        else:
            self.mode = mode
        self.nb_features = nb_features

        self.periodic_time_layer = torch.nn.Linear(1, self.nb_features - 1)
        self.linear_time_layer = torch.nn.Linear(1, 1)

    def forward(self, sits: MonoModalSITS) -> MonoModalSITS:
        """
        Implementation of forward method
        """
        pe = self.compute_pe(sits.doy)

        data_shape = parse_shape(sits.data, "b t c w h")

        pe = repeat(
            pe,
            "b t c -> b t c w h",
            w=data_shape["w"],
            h=data_shape["h"],
            c=self.nb_features,
        )

        if self.mode is PositionalEncodingMode.ADD:
            return MonoModalSITS(sits.data + pe, sits.doy, sits.mask)
        if self.mode is PositionalEncodingMode.YIELD:
            return MonoModalSITS(pe, sits.doy, sits.mask)
        if self.mode is PositionalEncodingMode.CONCAT:
            return MonoModalSITS(torch.cat([sits.data, pe], dim=2), sits.doy, sits.mask)

        raise NotImplementedError

    def get_embedding_size(self, nb_input_features: int) -> int:
        """
        Service to compute the embedding size based on the number of features
        """
        if self.mode is PositionalEncodingMode.ADD:
            assert nb_input_features == self.nb_features
            return nb_input_features
        if self.mode is PositionalEncodingMode.YIELD:
            return self.nb_features
        if self.mode is PositionalEncodingMode.CONCAT:
            return self.nb_features + nb_input_features

        raise NotImplementedError

    def compute_pe(self, doy: torch.Tensor):
        """
        Compute positional embedding
        """
        doy_shape = parse_shape(doy, "b t")
        doy = rearrange(doy, "b t -> (b t) 1")
        doy = doy.to(torch.float)

        # Unsqueeze is required so that linear layer acts on individual times
        linear_embedding = self.linear_time_layer(doy)
        periodic_embedding = torch.sin(self.periodic_time_layer(doy))

        # Concatenate against last (features) dimension
        out = torch.cat([linear_embedding, periodic_embedding], -1)

        # Rearange to final shape
        out = rearrange(out, "(b t) c -> b t c", **doy_shape)

        return out
