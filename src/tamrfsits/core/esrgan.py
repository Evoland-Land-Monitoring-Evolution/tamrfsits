#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the plain CARN model implementation
"""
from functools import partial

import numpy as np
from esrgan.models import ESREncoder, ESRNetDecoder  # type: ignore
from esrgan.nn.modules import Conv2d, InterpolateConv, SubPixelConv  # type: ignore
from torch import Tensor, no_grad
from torch.nn import LeakyReLU, Module, Sequential
from torch.nn.functional import interpolate


class ESRGANGenerator(Module):
    """ "
    ESRGAN generagor
    """

    def __init__(
        self,
        nb_bands: int,
        upsampling_factor: float,
        upsampling_base: int = 2,
        out_nb_bands: int | None = None,
        latent_size: int = 64,
        num_basic_blocks: int = 16,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        """
        Constructor:
        """
        super().__init__()

        # Memorize parameters
        self.nb_bands = nb_bands
        if out_nb_bands is None:
            out_nb_bands = nb_bands

        self.out_nb_bands = out_nb_bands

        self.num_basic_blocks = num_basic_blocks

        # Handle rounding of upsampling factor and final downsampling
        self.upsampling_factor = upsampling_factor
        self.rounded_upsampling_factor = upsampling_base * int(
            np.ceil(upsampling_factor / upsampling_base)
        )
        if self.rounded_upsampling_factor == upsampling_factor:
            self.final_downsampling_factor = None
        else:
            self.final_downsampling_factor = (
                self.rounded_upsampling_factor / self.upsampling_factor
            )

        self.encoder = ESREncoder(
            in_channels=nb_bands,
            out_channels=latent_size,
            num_basic_blocks=num_basic_blocks,
            growth_channels=growth_channels,
            activation=partial(LeakyReLU, negative_slope=0.2, inplace=True),
            residual_scaling=residual_scaling,
        )
        self.decoder = ESRNetDecoder(
            in_channels=latent_size,
            out_channels=out_nb_bands,
            upsampling_base=upsampling_base,
            scale_factor=self.rounded_upsampling_factor,
            activation=partial(LeakyReLU, negative_slope=0.2, inplace=True),
        )

    def forward(self, data: dict[str, Tensor] | Tensor) -> Tensor:
        """
        Forward pass of CARN

        :param data: Input tensor or dictionary with tensors of shape
        [nb_samples,nb_features,width,height]

        :return: Output tensor of shape
        [nb_samples,nb_features,upsampling_factor*width,upsampling_factor*height]
        """
        if isinstance(data, dict):
            assert (
                len([s for s in data.keys() if "source_std" in s]) == 1
            ), "You should choose double module"

            data = [data[s] for s in data.keys() if "source_std" in s][0]

        latent = self.encoder(data)
        out = self.decoder(latent)

        # Final downsampling
        if self.final_downsampling_factor is not None:
            out = interpolate(
                out,
                scale_factor=1 / self.final_downsampling_factor,
                mode="bicubic",
                recompute_scale_factor=False,
                align_corners=False,
                antialias=False,
            )

        return out

    def predict(self, data: dict[str, Tensor] | Tensor) -> Tensor:
        """
        Same as forward method but with final unstardardization and no_grad

        :param data: Input tensor of shape [nb_samples,nb_features,width,height]

        :return: Output tensor of shape
        [nb_samples,nb_features,upsampling_factor*width,upsampling_factor*height]
        """
        with no_grad():
            return self.forward(data)

    def get_prediction_margin(self) -> int:
        """

        Compute margin required for stable prediction
        """
        return int((5 * self.num_basic_blocks + 2) * self.upsampling_factor + 2)

    def get_upsampling_factor(self) -> float:
        """
        Return the upsampling factor
        """
        return self.upsampling_factor


class ESRSpatialEncoder(Module):
    """
    Spatial encoder based on ESRGAN architecture
    """

    def __init__(
        self,
        in_nb_bands: int,
        upsampling_factor: float,
        upsampling_base: int = 2,
        latent_size: int = 64,
        num_basic_blocks: int = 3,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
        upsampling_mode: str = "pixel_shuffle",
    ):
        """
        Constructor
        """
        super().__init__()
        # Store configuration
        self.upsampling_factor = upsampling_factor
        # Build internal ESRGAN instance

        self.encoder = ESREncoder(
            in_channels=in_nb_bands,
            out_channels=latent_size,
            num_basic_blocks=num_basic_blocks,
            growth_channels=growth_channels,
            activation=partial(LeakyReLU, negative_slope=0.2, inplace=True),
            residual_scaling=residual_scaling,
        )

        self.upsampling = Sequential()

        # upsampling
        nb_layers = int(upsampling_factor // upsampling_base)

        if upsampling_mode == "pixel_shuffle":
            self.upsampling = Sequential(
                *(
                    SubPixelConv(
                        num_features=latent_size,
                        conv=Conv2d,
                        activation=partial(LeakyReLU, negative_slope=0.2, inplace=True),
                        scale_factor=upsampling_base,
                    )
                    for _ in range(nb_layers)
                )
            )

        else:
            self.upsampling = Sequential(
                *(
                    InterpolateConv(
                        num_features=latent_size,
                        conv=Conv2d,
                        activation=partial(LeakyReLU, negative_slope=0.2, inplace=True),
                        scale_factor=upsampling_base,
                        mode=upsampling_mode,
                    )
                    for _ in range(nb_layers)
                )
            )

    def forward(self, data: Tensor) -> Tensor:
        """
        Implementation of forward method
        """

        # Process data tensor
        return self.upsampling(self.encoder(data))


class ESRSpatialDecoder(Module):
    """
    Decoder part of ERSGAN
    """

    def __init__(
        self,
        latent_size: int,
        out_nb_bands: int,
    ):
        super().__init__()

        self.decoder = Sequential(
            Conv2d(latent_size, latent_size),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Conv2d(latent_size, out_nb_bands),
        )

    def forward(self, data: Tensor) -> Tensor:
        """
        Forward method
        """
        return self.decoder(data)
