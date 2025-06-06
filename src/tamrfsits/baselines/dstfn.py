# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
Encapsulates the dstfn model for testing
"""
import torch
from einops import rearrange

from tamrfsits.baselines.utils import load_dstfn_model
from tamrfsits.core.downsampling import downsample_sits_from_mtf
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.core.utils import find_closest_in_sits


class DSTFNSITSFusion(torch.nn.Module):
    """
    DSTFN model
    """

    def __init__(self, mtf_for_downsampling: float = 0.4):
        """
        Initializer
        """
        super().__init__()
        self.model = load_dstfn_model()
        self.mtf_for_downsampling = mtf_for_downsampling

    def forward(
        self, lr_sits: MonoModalSITS, hr_sits: MonoModalSITS
    ) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Inference function
        """
        # Separate pan and spectral bands in landsat
        lr_pan_sits = MonoModalSITS(
            lr_sits.data[:, :, 6:, ...], lr_sits.doy, lr_sits.mask
        )
        lr_sits = MonoModalSITS(lr_sits.data[:, :, :6, ...], lr_sits.doy, lr_sits.mask)

        # Locate closest hr data
        closest_hr_sits = find_closest_in_sits(
            hr_sits, lr_sits.doy, allow_same_day=False
        )

        # Downsample back lr_sits to 30m
        lr_sits = downsample_sits_from_mtf(
            lr_sits,
            res=1.0,
            mtf_res=torch.full((6,), 1.0, device=lr_sits.data.device),
            mtf_fc=torch.full(
                (6,), self.mtf_for_downsampling, device=lr_sits.data.device
            ),
            factor=2.0,
        )

        # rearrange sits data for model
        lr_data = rearrange(lr_sits.data, "b t c w h -> (b t) c w h")
        lr_pan_data = rearrange(lr_pan_sits.data, "b t c w h -> (b t) c w h")
        closest_hr_data = rearrange(closest_hr_sits.data, "b t c w h -> (b t) c w h")

        if not closest_hr_data.shape[0]:
            pred_hr = torch.full_like(closest_hr_data, torch.nan)
        else:
            pred_hr = torch.cat(
                [
                    self.model(a[None, ...], b[None, ...], c[None, ...])
                    for a, b, c in zip(closest_hr_data, lr_data, lr_pan_data)
                ]
            )

        pred_hr = rearrange(
            pred_hr, "(b t) c w h -> b t c w h", b=lr_sits.data.shape[0]
        )

        pred_hr_sits = MonoModalSITS(pred_hr, lr_sits.doy)
        return lr_sits, pred_hr_sits
