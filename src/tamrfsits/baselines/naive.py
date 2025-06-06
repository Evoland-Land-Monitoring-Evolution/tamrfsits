# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
This module implement a naive baseline solution for SITS fusion :
1) Linear interpolation in time using cloud masks
2) Bicubic zoom to target resolution
"""
import torch

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.core.linear_gapfilling import linear_gapfilling
from tamrfsits.core.time_series import MonoModalSITS


class NaiveSITSFusion(torch.nn.Module):
    """
    Naive model
    """

    def __init__(self, lr_upsampling_factor: float):
        """
        class initializer
        """
        super().__init__()
        self.lr_upsampling = DateWiseSITSModule(
            model=None, upsampling_factor=lr_upsampling_factor
        )

    def forward(
        self,
        lr_sits: MonoModalSITS,
        hr_sits: MonoModalSITS,
        target_doy: torch.Tensor | None = None,
    ) -> tuple[MonoModalSITS | None, MonoModalSITS | None]:
        """
        Forward method
        """
        if target_doy is None:
            target_doy = torch.unique(torch.cat((lr_sits.doy, hr_sits.doy), dim=1))

        # Temporal linear interpolation of both SITS
        lr_sits_interp = (
            linear_gapfilling(lr_sits, target_doy) if lr_sits.doy.shape[1] > 1 else None
        )
        hr_sits_interp = (
            linear_gapfilling(hr_sits, target_doy) if hr_sits.doy.shape[1] > 1 else None
        )

        # Spatial bicubic interpolation of lr_sits
        lr_sits_interp_up = (
            self.lr_upsampling(lr_sits_interp) if lr_sits_interp else None
        )

        return lr_sits_interp_up, hr_sits_interp
