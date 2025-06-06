#!/usr/bin/env python
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the plain CARN model implementation
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import Conv2d


class EResBlock(torch.nn.Module):
    """
    Base block for CARN network:

    Ahn, N., Kang, B., & Sohn, K. A. (2018). Fast, accurate, and lightweight
    super-resolution with cascading residual network. In Proceedings of the European
    Conference on Computer Vision (ECCV) (pp. 252-268).
    """

    def __init__(
        self,
        nb_features: int = 64,
        groups: int = 5,
        kernel_size: int = 3,
        leaky_relu_slope: float = 0.2,
    ):
        """
        Constructor
        : param nb_features: Number of features for each deep layer
        : param kernel_size: Kernel size for all layers
        """
        super().__init__()
        self.nb_features = nb_features
        self.groups = groups
        self.kernel_size = kernel_size
        self.leaky_relu_slope = leaky_relu_slope
        padding = int(np.floor(kernel_size / 2.0))
        # Chain 2 x Conv2d + ReLU
        self.net = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_1",
                        Conv2d(
                            nb_features,
                            nb_features,
                            kernel_size,
                            groups=groups,
                            padding=padding,
                            padding_mode="reflect",
                        ),
                    ),
                    (
                        "relu_1",
                        torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
                    ),
                    (
                        "conv_2",
                        Conv2d(
                            nb_features,
                            nb_features,
                            kernel_size,
                            groups=groups,
                            padding=padding,
                            padding_mode="reflect",
                        ),
                    ),
                    (
                        "relu_2",
                        torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
                    ),
                ]
            )
        )
        # Add Linear Layer (equivalent to conv with ks == 1)
        self.single_conv = Conv2d(nb_features, nb_features, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ERes block

        :param data: Input tensor of shape [nb_samples,nb_features,width,height]

        :return: Output tensor of shape [nb_samples,nb_features,width,height]
        """
        net_out = self.net(data)
        single_conv_out = self.single_conv(net_out)
        return torch.nn.functional.leaky_relu(
            data + single_conv_out, negative_slope=self.leaky_relu_slope
        )

    def nb_params(self) -> int:
        """
        Return estimated and real number of parameters
        """
        return int(
            2
            * (
                self.nb_features
                + (((self.kernel_size * self.nb_features) ** 2) / self.groups)
            )
        ) + self.nb_features * (self.nb_features + 1)


class CascadingBlock(torch.nn.Module):
    """
    Implement cascading block
    """

    def __init__(
        self,
        nb_eres_blocks: int,
        nb_features: int,
        groups: int,
        kernel_size: int,
        shared_weights: bool,
        leaky_relu_slope: float = 0.2,
    ):
        """
        Initializer
        """
        super().__init__()
        self.residual_blocks = torch.nn.ModuleList()
        self.residual_1d_conv = torch.nn.ModuleList()
        self.nb_features = nb_features
        self.nb_eres_blocks = nb_eres_blocks
        self.shared_weights = shared_weights
        # Build each block
        self.residual_blocks.append(
            EResBlock(
                nb_features, groups, kernel_size, leaky_relu_slope=leaky_relu_slope
            )
        )
        self.residual_1d_conv.append(Conv2d(nb_features, nb_features, (1, 1)))

        if shared_weights:
            shared_block = self.residual_blocks[0]
            self.residual_blocks.extend(
                [shared_block for _ in range(1, nb_eres_blocks)]
            )
        else:
            self.residual_blocks.extend(
                [
                    EResBlock(
                        nb_features,
                        groups,
                        kernel_size,
                        leaky_relu_slope=leaky_relu_slope,
                    )
                    for _ in range(1, nb_eres_blocks)
                ]
            )
            # Intermediate skip convolution
            self.residual_1d_conv.extend(
                [
                    Conv2d(nb_features, nb_features, (1, 1))
                    for _ in range(1, nb_eres_blocks)
                ]
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CARN

        :param data: Input tensor with tensor of shape
        [nb_samples,nb_features,width,height]

        :return: Output tensor of shape
        [nb_samples,nb_features,upsampling_factor*width,upsampling_factor*height]
        """

        # Blocks with skip connections
        current_data = data
        residual_connections = [data]
        for block, residual_conv in zip(self.residual_blocks, self.residual_1d_conv):
            block_output = block(current_data)
            current_data = residual_conv(block_output) + sum(residual_connections)
            residual_connections.append(block_output)

        return current_data

    def nb_params(self) -> int:
        """
        Return the number of parameters in carn model
        """
        if self.shared_weights:
            return self.residual_blocks[0].nb_params() + self.nb_eres_blocks * (
                self.nb_features * (self.nb_features + 1)
            )
        return self.nb_eres_blocks * (
            self.residual_blocks[0].nb_params()
            + self.nb_features * (self.nb_features + 1)
        )


@dataclass(frozen=True)
class CARNConfig:
    """
    Configuration for CARN model

    : param nb_bands: Number of bands in input/output image
    : param groups: Grouping for cascading blocks
    : param shared_weights: If True, cascading blocks will share weights and bias
    : param nb_features_per_factor: Number of features dedicated to each upsampled pixel
    : param upsampling_factor: Output image will be this time bigger
    : param nb_blocks: number of residual blocks
    : param kernel_size: Kernel size for all layers
    """

    nb_bands: int
    out_nb_bands: int | None = None
    groups: int = 2
    shared_weights: bool = True
    nb_features_per_factor: int = 2
    upsampling_factor: float = 2.0
    nb_cascading_blocks: int = 3
    nb_eres_blocks_per_cascading_block: int = 3
    kernel_size: int = 3
    leaky_relu_slope: float = 0.2


class CARN(torch.nn.Module):
    """
    CARN network:

     Ahn, N., Kang, B., & Sohn, K. A. (2018). Fast, accurate, and lightweight
     super-resolution with cascading residual network. In Proceedings of the European
     Conference on Computer Vision (ECCV) (pp. 252-268).

     CARN belongs to the late upsampling family of methods, which means that uspampling
     will be performed by the network itself.

    """

    def __init__(self, config: CARNConfig):
        """
        Constructor:
        """

        super().__init__()

        # Memorize parameters
        self.config = config

        # Handle rounding of upsampling factor and final downsampling
        self.upsampling_factor = config.upsampling_factor
        self.rounded_upsampling_factor = 2 * int(np.ceil(config.upsampling_factor / 2))
        if self.rounded_upsampling_factor == config.upsampling_factor:
            self.final_downsampling_factor = None
        else:
            self.final_downsampling_factor = (
                self.rounded_upsampling_factor / self.upsampling_factor
            )
        # Compute number of features
        nb_features = int(
            config.nb_features_per_factor
            * np.ceil(self.rounded_upsampling_factor)
            * np.ceil(self.rounded_upsampling_factor)
        )
        self.nb_features = nb_features

        padding = int(np.floor(config.kernel_size / 2.0))
        self.cascading_blocks = torch.nn.ModuleList()
        self.residual_1d_conv = torch.nn.ModuleList()

        # Build each block
        self.cascading_blocks.append(
            CascadingBlock(
                config.nb_eres_blocks_per_cascading_block,
                nb_features,
                config.groups,
                config.kernel_size,
                config.shared_weights,
                leaky_relu_slope=config.leaky_relu_slope,
            )
        )
        self.residual_1d_conv.append(Conv2d(nb_features, nb_features, (1, 1)))

        for _ in range(1, config.nb_cascading_blocks):
            # Build residual block
            # Implement weight sharing
            self.cascading_blocks.append(
                CascadingBlock(
                    config.nb_eres_blocks_per_cascading_block,
                    nb_features,
                    config.groups,
                    config.kernel_size,
                    config.shared_weights,
                )
            )
            # Intermediate skip convolution
            self.residual_1d_conv.append(Conv2d(nb_features, nb_features, (1, 1)))

        self.input_conv = Conv2d(
            config.nb_bands,
            nb_features,
            config.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )

        self.output_conv1 = Conv2d(
            nb_features,
            nb_features,
            config.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )

        self.output_shuffle = torch.nn.PixelShuffle(self.rounded_upsampling_factor)

        out_nb_bands = config.out_nb_bands
        if out_nb_bands is None:
            out_nb_bands = config.nb_bands

        self.output_conv2 = Conv2d(
            config.nb_features_per_factor,
            out_nb_bands,
            config.kernel_size,
            padding=padding,
            padding_mode="reflect",
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CARN

        :param data: Input tensor with tensor of shape
        [nb_samples,nb_features,width,height]

        :return: Output tensor of shape
        [nb_samples,nb_features,upsampling_factor*width,upsampling_factor*height]
        """
        # Input convolution layer
        input_conv_data = self.input_conv(data)

        # Blocks with skip connections
        current_data = input_conv_data
        residual_connections = [input_conv_data]

        for block, residual_conv in zip(self.cascading_blocks, self.residual_1d_conv):
            block_output = block(current_data)
            current_data = residual_conv(block_output) + sum(residual_connections)
            residual_connections.append(block_output)

        # Output block with pixel shuffle
        out = self.output_conv1(current_data)
        out = self.output_shuffle(out)
        out = self.output_conv2(out)

        # Final downsampling
        if self.final_downsampling_factor is not None:
            out = torch.nn.functional.interpolate(
                out,
                scale_factor=1 / self.final_downsampling_factor,
                mode="bicubic",
                recompute_scale_factor=False,
                align_corners=False,
                antialias=False,
            )

        return out

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Same as forward method but with final unstardardization and no_grad

        :param data: Input tensor of shape [nb_samples,nb_features,width,height]

        :return: Output tensor of shape
        [nb_samples,nb_features,upsampling_factor*width,upsampling_factor*height]
        """
        with torch.no_grad():
            return self.forward(data)

    def get_upsampling_factor(self) -> float:
        """
        Return the upsampling factor
        """
        return self.upsampling_factor

    def get_prediction_margin(self) -> int:
        """

        Compute margin required for stable prediction
        """
        unitary_margin = int(np.floor(self.config.kernel_size / 2))
        return int(
            (
                (
                    2
                    * self.config.nb_eres_blocks_per_cascading_block
                    * self.config.nb_cascading_blocks
                    + 2
                )
                * self.upsampling_factor
                + 1
            )
            * unitary_margin
        )

    def nb_params(self):
        """
        Compute number of parameters
        """
        nb_params_input_conv1 = self.nb_features + (
            (self.config.kernel_size**2) * self.config.nb_bands * self.nb_features
        )

        nb_params_cascading = self.config.nb_cascading_blocks * (
            self.cascading_blocks[0].nb_params()
            + self.nb_features * (self.nb_features + 1)
        )
        nb_params_output_conv1 = (
            self.config.kernel_size**2 * self.nb_features**2
        ) + self.nb_features

        out_nb_bands = self.config.out_nb_bands

        if out_nb_bands is None:
            out_nb_bands = self.config.nb_bands

        nb_params_output_conv2 = out_nb_bands + (
            (self.config.kernel_size**2)
            * self.config.nb_features_per_factor
            * out_nb_bands
        )
        return (
            nb_params_input_conv1
            + nb_params_cascading
            + nb_params_output_conv1
            + nb_params_output_conv2
        )
