# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the DecorrelationLoss
"""

import torch
from einops import rearrange

from tamrfsits.core.time_series import MonoModalSITS


def decorrelation_loss(
    sits: MonoModalSITS, min_nb_samples: int = 1000
) -> torch.Tensor | None:
    """
    Implementation of forward method
    """
    data = rearrange(sits.data, "b t c w h -> c (b t w h)")

    # Can not estimate if too few samples
    if data.shape[-1] < min_nb_samples:
        return None

    covariance = torch.cov(data)

    return torch.nn.functional.l1_loss(
        covariance, torch.eye(covariance.shape[0], device=covariance.device)
    )
