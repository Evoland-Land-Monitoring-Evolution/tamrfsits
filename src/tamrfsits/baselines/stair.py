# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
This module implement STAIR solution for SITS fusion:
Yunan Luo, Kaiyu Guan, Jian Peng,

STAIR: A generic and fully-automated method to fuse multiple sources of optical
satellite data to generate a high-resolution, daily and cloud-/gap-free surface
reflectance product,

Remote Sensing of Environment, Volume 214, 2018, Pages 87-99, ISSN 0034-4257,
https://doi.org/10.1016/j.rse.2018.04.042.
(https://www.sciencedirect.com/science/article/pii/S0034425718301998)
"""
from contextlib import suppress

import torch
from torch import tensor  # pylint: disable = no-name-in-module

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.core.linear_gapfilling import linear_gapfilling
from tamrfsits.core.time_series import MonoModalSITS, subset_doy_monomodal_sits
from tamrfsits.core.utils import common_doys


class STAIRSITSFusion(torch.nn.Module):
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
    ) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Forward method
        """
        # STAIR assumes same bands in both SITS
        assert lr_sits.data.shape[2] == hr_sits.data.shape[2]

        if lr_sits.doy.shape[1] == 0:
            raise ValueError("Cannot apply STAIR if there are no lr dates")

        # up-sample LR SITS
        lr_sits = self.lr_upsampling(lr_sits)

        # Next, find conjunctions between lr and hr sits
        conjunctions_doy = common_doys(lr_sits.doy, hr_sits.doy)

        # Raise if we cannot apply
        if conjunctions_doy.shape[0] == 0:
            raise ValueError("Cannot apply STAIR if no common doys are found")

        predictions: list[MonoModalSITS] = []
        assert hr_sits.mask is not None
        for current_doy in conjunctions_doy:
            doy_mask = hr_sits.doy != current_doy
            current_hr_sits = MonoModalSITS(
                hr_sits.data[:, doy_mask[0, ...], ...],
                hr_sits.doy[:, doy_mask[0, ...]],
                hr_sits.mask[:, doy_mask[0, ...], ...],
            )
            current_conjunctions_doy = common_doys(lr_sits.doy, current_hr_sits.doy)

            with suppress(IndexError):
                hr_common = linear_gapfilling(current_hr_sits, current_conjunctions_doy)
                lr_common = subset_doy_monomodal_sits(lr_sits, current_conjunctions_doy)

                # Build conjunctions differences
                conjunctions_diff_sits = MonoModalSITS(
                    hr_common.data - lr_common.data, doy=hr_common.doy
                )

                # Now, interpolate diff at target date
                conjunctions_diff_sits_only_lr = linear_gapfilling(
                    conjunctions_diff_sits,
                    tensor([current_doy], device=lr_common.data.device),
                )
                lr_only_lr = subset_doy_monomodal_sits(
                    lr_sits, tensor([current_doy], device=lr_common.data.device)
                )

                # HR prediction on dates with only LR
                pred_data = lr_only_lr.data + conjunctions_diff_sits_only_lr.data
                invalid_pred_data = pred_data.isnan().sum(dim=2) > 0
                pred_data = torch.nan_to_num(pred_data)
                pred_mask = torch.logical_or(lr_only_lr.mask, invalid_pred_data)
                predictions.append(
                    MonoModalSITS(
                        pred_data,
                        lr_only_lr.doy,
                        pred_mask,
                    )
                )

        if not predictions:
            raise ValueError("STAIR could not make any valid predictions")
        # Pixels that are masked in input lr should be masked in output
        hr_final = MonoModalSITS(
            torch.cat([s.data for s in predictions], dim=1),
            torch.cat(
                [s.doy for s in predictions],
                dim=1,
            ),
        )
        return lr_sits, hr_final
