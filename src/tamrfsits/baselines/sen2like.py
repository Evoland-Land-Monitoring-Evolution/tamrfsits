# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
This module implement the sen2like solution to fusion of landsat and sentinel2:

Saunier, S., Pflug, B., Lobos, I. M., Franch, B., Louis, J., De Los Reyes, R.,
Debaecker, V., Cadau, E. G., Boccia, V., Gascon, F., & Kocaman,
S. (2022). Sen2Like: Paving the Way towards Harmonization and Fusion of Optical
Data. Remote Sensing, 14(16), 3855. https://doi.org/10.3390/rs14163855

"""
import torch
from einops import repeat

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.core.downsampling import generate_psf_kernel, sits_high_pass_filtering
from tamrfsits.core.time_series import MonoModalSITS

ABOVE_MAX_DOY_VALUE = 10000


class Sen2LikeFusion(torch.nn.Module):
    """
    Sen2Like model
    """

    def __init__(
        self,
        lr_upsampling_factor: float,
        max_masked_rate_for_fusion: float = 0.5,
        mtf_for_hpf: float = 0.4,
    ):
        """
        class initializer
        """
        super().__init__()
        self.lr_upsampling = DateWiseSITSModule(
            model=None,
            upsampling_factor=lr_upsampling_factor,
            upsampling_mode="bilinear",
        )
        self.max_masked_rate_for_fusion = max_masked_rate_for_fusion
        self.mtf_for_hpf = mtf_for_hpf

    def forward(
        self,
        lr_sits: MonoModalSITS,
        hr_sits: MonoModalSITS,
    ) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Forward method
        """
        # Assume same bands in both sits
        assert lr_sits.data.shape[2] == hr_sits.data.shape[2]
        assert lr_sits.data.shape[0] == hr_sits.data.shape[0] == 1

        # up-sample LR SITS
        lr_sits = self.lr_upsampling(lr_sits)

        # Make predictions based on 2 most recent clear s2 dates for each landsat date
        doy_diff_matrix = lr_sits.doy[0, None, :] - hr_sits.doy[0, :, None]

        # Set posterior dates to max constant
        doy_diff_matrix[doy_diff_matrix < 0] = ABOVE_MAX_DOY_VALUE

        # Set masked hr dates to a max constant

        if hr_sits.mask is not None:
            hr_mask_rate = hr_sits.mask.sum(dim=(-1, -2)) / (
                hr_sits.mask.shape[-1] * hr_sits.mask.shape[-2]
            )
            # print(hr_mask_rate.shape, hr_mask_rate)
            doy_diff_matrix[
                hr_mask_rate[0, ...] > self.max_masked_rate_for_fusion, :
            ] = ABOVE_MAX_DOY_VALUE

        # Get first min
        first_diff_min, first_min_idx = torch.min(doy_diff_matrix, dim=0)
        first_min_values = hr_sits.doy.gather(1, first_min_idx[None, ...])
        # print(first_min_values, first_min_idx)
        # Get second min
        above_max_doy_value = ABOVE_MAX_DOY_VALUE  # Fixes W8292
        for i in range(first_min_idx.shape[0]):
            doy_diff_matrix[first_min_idx[i], i] = above_max_doy_value

        second_diff_min, second_min_idx = torch.min(doy_diff_matrix, dim=0)
        second_min_values = hr_sits.doy.gather(1, second_min_idx[None, ...])
        # print(second_min_values, second_min_idx)

        # We can only make prediction if there are two valid hr date before the lr date
        valid = torch.logical_and(
            first_diff_min < ABOVE_MAX_DOY_VALUE,
            second_diff_min < ABOVE_MAX_DOY_VALUE,
        )

        assert hr_sits.mask is not None
        masked_first = hr_sits.mask.gather(
            1,
            repeat(
                first_min_idx,
                "t -> b t w h",
                b=hr_sits.data.shape[0],
                w=hr_sits.data.shape[3],
                h=hr_sits.data.shape[4],
            ),
        )
        masked_second = hr_sits.mask.gather(
            1,
            repeat(
                second_min_idx,
                "t -> b t w h",
                b=hr_sits.data.shape[0],
                w=hr_sits.data.shape[3],
                h=hr_sits.data.shape[4],
            ),
        )

        # Now, lets compute wheights See
        # https://github.com/senbox-org/sen2like/blob/
        # master/sen2like/sen2like/s2l_processes/S2L_Fusion.py#L332
        first_min_idx = repeat(
            first_min_idx,
            "t -> b t c w h",
            b=hr_sits.data.shape[0],
            c=hr_sits.data.shape[2],
            w=hr_sits.data.shape[3],
            h=hr_sits.data.shape[4],
        )
        second_min_idx = repeat(
            second_min_idx,
            "t -> b t c w h",
            b=hr_sits.data.shape[0],
            c=hr_sits.data.shape[2],
            w=hr_sits.data.shape[3],
            h=hr_sits.data.shape[4],
        )
        first_min_values = repeat(
            first_min_values,
            "b t -> b t c w h",
            c=hr_sits.data.shape[2],
            w=hr_sits.data.shape[3],
            h=hr_sits.data.shape[4],
        )
        second_min_values = repeat(
            second_min_values,
            "b t -> b t c w h",
            c=hr_sits.data.shape[2],
            w=hr_sits.data.shape[3],
            h=hr_sits.data.shape[4],
        )

        coef_a = (
            hr_sits.data.gather(1, second_min_idx)
            - hr_sits.data.gather(1, first_min_idx)
        ) / (second_min_values - first_min_values)

        coef_b = hr_sits.data.gather(1, second_min_idx) - coef_a * second_min_values

        predicted_hr = (
            coef_a
            * repeat(
                lr_sits.doy,
                "b t -> b t c w h",
                c=hr_sits.data.shape[2],
                w=hr_sits.data.shape[3],
                h=hr_sits.data.shape[4],
            )
            + coef_b
        )
        predicted_hr_sits = MonoModalSITS(predicted_hr, lr_sits.doy)

        # Perform high pass filtering on predicted hr sits
        kernel = generate_psf_kernel(
            1.0,
            mtf_res=torch.full(
                (predicted_hr.shape[2],), 1.0, device=predicted_hr.device
            ),
            mtf_fc=torch.full(
                (predicted_hr.shape[2],), self.mtf_for_hpf, device=predicted_hr.device
            ),
        )
        predicted_hr_sits_hpf = sits_high_pass_filtering(predicted_hr_sits, kernel)

        assert lr_sits.mask is not None
        final_hr_mask = torch.logical_or(
            torch.logical_or(
                torch.logical_or(
                    lr_sits.mask,
                    repeat(
                        ~valid,
                        "t -> b t w h",
                        b=predicted_hr.shape[0],
                        w=predicted_hr.shape[-2],
                        h=predicted_hr.shape[-1],
                    ),
                ),
                masked_first,
            ),
            masked_second,
        )

        final_hr = MonoModalSITS(
            lr_sits.data + predicted_hr_sits_hpf.data, lr_sits.doy, final_hr_mask
        )

        return (lr_sits, final_hr)
