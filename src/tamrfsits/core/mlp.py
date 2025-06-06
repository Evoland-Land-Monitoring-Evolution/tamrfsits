"""MLP Pytorch model"""

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn


@dataclass
class MLPConfig:
    """Configuration of a MLP"""

    in_channels: int
    out_channels: int
    use_batch_norm: bool = True
    hidden_layers: list[int] = field(default_factory=list)


def linear_block(
    in_size: int,
    out_size: int,
    activation: str = "ReLU",
    batch_norm: bool = True,
) -> nn.Sequential:
    """Return linear block with batch norm and activation"""

    # Look up activation module from activation string
    activation_module = getattr(nn, activation)()

    # Check we get a correct module
    assert isinstance(activation_module, nn.Module)
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, 1),
            nn.BatchNorm2d(out_size),
            activation_module,
        )
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, 1),
        activation_module,
    )


class MLP(nn.Module):
    """
    Implements simple MLP with linear block
    """

    def __init__(self, config: MLPConfig):
        """
        Constructor

        :param config: Parameters of the MLP
        """
        super().__init__()
        self.n_outputs: int = config.out_channels
        self.n_inputs: int = config.in_channels

        # Define type for layers variable
        layers: list[nn.Module]

        if not config.hidden_layers:
            # Only one layer - Linear Classification
            layers = [nn.Conv2d(self.n_inputs, self.n_outputs, 1)]
        else:
            # Add first layers
            layers = [
                linear_block(
                    self.n_inputs,
                    config.hidden_layers[0],
                    batch_norm=config.use_batch_norm,
                )
            ]

            # Add hidden layers
            layers += [
                linear_block(
                    in_size,
                    out_size,
                    batch_norm=config.use_batch_norm,
                )
                for in_size, out_size in zip(
                    config.hidden_layers[:-1], config.hidden_layers[1:]
                )
            ]
            # Add last layers
            layers.append(nn.Conv2d(config.hidden_layers[-1], config.out_channels, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP

        :param batch: Data, tensor of shape [n_batch, n_inputs]
                      or [n_batch, n_inputs, h, w]

        :return: tensor of shape [n_batch, n_outputs] or [n_batch, n_inputs, h, w]
        """
        # Add extra dimension to go through conv2d if required
        if batch.dim() == 2:
            return cast(torch.Tensor, self.layers(batch[:, :, None, None]))[:, :, 0, 0]
        return cast(torch.Tensor, self.layers(batch))
