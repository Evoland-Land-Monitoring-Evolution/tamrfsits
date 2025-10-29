# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
This module implement UTILISE temporal interpolation and gap-filling of Sentinel-2 sits
"""
import os

import torch
from lib.eval_tools import Imputation

from tamrfsits.core.time_series import MonoModalSITS, subset_doy_monomodal_sits


class UtiliseGapFilling(torch.nn.Module):
    """
    UtiliseGapFilling

    see: https://github.com/prs-eth/U-TILISE
    """

    def __init__(self):
        """
        class initializer
        """
        super().__init__()

        # Retrieve required files from environment variables
        utilise_config = os.environ["UTILISE_CONFIGURATION_FILE"]
        utilise_weights = os.environ["UTILISE_WEIGHTS_FILE"]

        self.model = Imputation(
            utilise_config, method="utilise", checkpoint=utilise_weights
        )

    def forward(
        self,
        hr_sits: MonoModalSITS,
        target_doys: torch.Tensor | None = None,
    ) -> MonoModalSITS:
        """
        Forward method
        """
        assert hr_sits.data.shape[2] == 4

        if target_doys is None:
            target_doys = hr_sits.doy

        doys = torch.unique(torch.cat((hr_sits.doy, target_doys[None, ...]), dim=1))

        hr_sits_full = subset_doy_monomodal_sits(hr_sits, doys, fill_value=1.0)

        assert hr_sits_full.mask is not None

        _, predicted = self.model.impute_sample(
            {
                "x": hr_sits_full.data,
                "position_days": hr_sits_full.doy,
                "y": None,
                "masks": hr_sits_full.mask[:, :, None, ...].to(dtype=torch.float32),
            }
        )
        predicted = predicted.to(device=hr_sits_full.data.device)
        hr_predicted_full = MonoModalSITS(predicted, doys[None, ...])
        hr_predicted = subset_doy_monomodal_sits(hr_predicted_full, target_doys)

        return hr_predicted
