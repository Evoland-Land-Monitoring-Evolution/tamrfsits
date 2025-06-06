# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Tests for validation.metrics module
"""


import torch

from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.validation.metrics import (
    closest_date_in_sits,
    derive_reference_mask,
    frr_referenceless,
    per_date_clear_pixel_rate,
    per_date_masked_rmse,
    per_date_per_band_brisque,
)

from .tests_utils import generate_monomodal_sits


def test_per_date_masked_rmse():
    """
    Test the per_date_masked_rmse function
    """
    ref_sits = generate_monomodal_sits()

    pred_data = ref_sits.data.clone()
    pred_data += 10.0

    pred_sits = MonoModalSITS(pred_data, ref_sits.doy)
    assert ref_sits.mask is not None
    rmse_mask = derive_reference_mask(
        ref_sits.mask, nb_features=ref_sits.shape()[2], spatial_margin=1
    )

    rmse = per_date_masked_rmse(pred_sits.data, ref_sits.data, mask=rmse_mask)

    assert rmse.shape == torch.Size([pred_data.shape[1], pred_data.shape[2]])
    assert torch.all(~torch.isnan(rmse))

    rmse_mask = derive_reference_mask(
        ref_sits.mask, nb_features=ref_sits.shape()[2], spatial_margin=None
    )

    rmse = per_date_masked_rmse(pred_sits.data, ref_sits.data, mask=rmse_mask)

    assert rmse.shape == torch.Size([pred_data.shape[1], pred_data.shape[2]])
    assert torch.all(~torch.isnan(rmse))


def test_per_date_clear_pixel_rate():
    """
    Test the per_date_masked_rmse function
    """
    ref_sits = generate_monomodal_sits()
    assert ref_sits.mask is not None
    clear_pixel_rate = per_date_clear_pixel_rate(ref_sits.mask)

    assert clear_pixel_rate.shape == torch.Size([ref_sits.shape()[1]])
    assert torch.all(~torch.isnan(clear_pixel_rate))
    assert torch.all(0.0 <= clear_pixel_rate)
    assert torch.all(clear_pixel_rate <= 1.0)


def test_closest_date_in_sits():
    """
    Test the closest_date_in_sits function
    """
    doy = torch.tensor([1, 2])
    test_doy = torch.tensor([0, 1])

    cdoy = closest_date_in_sits(doy, test_doy)

    assert torch.all(cdoy == torch.Tensor([1, 0]))

    cdoy_all = closest_date_in_sits(doy)

    assert cdoy_all.shape == torch.Size([365])


def test_per_date_per_band_brisque():
    """
    Test the per_date_per_band_brisque
    """
    pred_sits = generate_monomodal_sits()

    output = per_date_per_band_brisque(pred_sits.data)

    assert torch.all(~torch.isnan(output))
    assert torch.all(output >= 0.0)


def test_frr_referenceless():
    """
    Test the frr referenceless function
    """
    ref_sits = generate_monomodal_sits(batch=1)
    pred_sits = generate_monomodal_sits(batch=1)

    frr = frr_referenceless(pred_sits.data, ref_sits.data)
    assert frr is not None
    assert frr[0].shape == torch.Size((10, 4))
    assert torch.all(~torch.isnan(frr[0]))
