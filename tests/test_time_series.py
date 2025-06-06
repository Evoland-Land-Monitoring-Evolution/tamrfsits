#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Test for the SITS class representing time-series
"""

import pytest
import torch
from pytest import raises

from tamrfsits.core.time_series import (
    SITS,
    MonoModalSITS,
    cat_monomodal_sits,
    constant_pad_end,
    crop_sits,
    pad_acquisition_time,
    sits_where,
    subset_doy_monomodal_sits,
)

from .tests_utils import generate_monomodal_sits


def test_constant_pad_end() -> None:
    """
    Test the time padding function
    """
    data = torch.rand((32, 5, 4, 16, 16))

    padded_data = constant_pad_end(data, dim=1, length=10, fill_value=0.0)

    assert padded_data is not None
    assert padded_data.shape == (32, 10, 4, 16, 16)


def test_sits_class() -> None:
    """
    Test the sits class
    """
    doy = torch.rand((32, 5))
    data = torch.rand((32, 5, 4, 16, 16))
    mask = torch.rand((32, 5, 16, 16)) > 0.0

    # Should pass
    SITS(data, doy, mask)

    # Should raise Exception since doy is not same shape
    doy = torch.rand((32, 1))
    with pytest.raises(AssertionError):
        SITS(data, doy, mask)

    # Should raise exception since number of dimensions is not enough
    doy = torch.rand(32)
    with pytest.raises(AssertionError):
        SITS(data, doy, mask)


def test_monomodal_sits_class() -> None:
    """
    Test the sits class
    """
    doy = torch.rand((32, 5))
    data = torch.rand((32, 5, 4, 16, 16))
    mask = torch.rand((32, 5, 16, 16)) > 0.0

    # Should pass
    MonoModalSITS(data, doy, mask)


def test_pad_acquisition_time() -> None:
    """
    Test the pad acquisition time function
    """
    doy = torch.rand((32, 5))
    data = torch.rand((32, 5, 4, 16, 16))
    mask = torch.rand((32, 5, 16, 16)) > 0.0

    # Should pass
    mono_sits = MonoModalSITS(data, doy, mask)

    padded_mono_sits = pad_acquisition_time(mono_sits, 10)

    assert isinstance(padded_mono_sits, MonoModalSITS)
    assert padded_mono_sits.doy.shape[1] == 10
    assert pad_acquisition_time(mono_sits, 4) is None


def test_cat_monomodal_sits() -> None:
    """
    Test the concatenation of monomodal sits
    """
    nb_doy = 5
    batch_size = 32
    w = h = 16
    nb_channels1 = 4
    nb_channels2 = 6
    doy = torch.rand((batch_size, nb_doy))
    data1 = torch.rand((batch_size, nb_doy, nb_channels1, w, h))
    data2 = torch.rand((batch_size, nb_doy, nb_channels2, w, h))
    mask = torch.rand((batch_size, nb_doy, w, h)) > 0.0

    sits1 = MonoModalSITS(data1, doy, mask)
    sits2 = MonoModalSITS(data2, doy, mask)

    out = cat_monomodal_sits([sits1, sits2])

    assert out.data.shape[2] == nb_channels1 + nb_channels2


def test_subset_doy_monomodal_sits() -> None:
    """
    Test merge monomodal sits function
    """
    # Build first sits
    doy1 = torch.tensor([[0, 2, 5, 6, 8], [0, 2, 5, 6, 8]])
    data1 = torch.randint(10, 20, (2, 5, 1, 1, 1))
    mask1 = torch.rand((2, 5, 1, 1)) > 0.0
    sits1 = MonoModalSITS(data1, doy1, mask1)

    doy2 = torch.tensor([1, 2, 3, 5])

    fill_value = -10.0
    out_sits = subset_doy_monomodal_sits(sits1, target_doy=doy2, fill_value=fill_value)

    print(f"{data1[:, :, 0, 0, 0]=}")
    print(f"{doy1=}")
    print(f"{doy2=}")
    print(f"{out_sits.data[:, :, 0, 0, 0]=}")

    assert out_sits.doy.shape[1] == doy2.shape[0]
    assert out_sits.mask is not None
    # Missing days in doy1
    for day_idx in (0, 2):
        assert torch.all(out_sits.mask[:, day_idx, ...])
        print(out_sits.data[:, day_idx, :, 0, 0])
        assert torch.all(out_sits.data[:, day_idx, ...] == fill_value)

    # Available days in doy1
    for day_idx, source_day_idx in ((1, 1), (3, 2)):
        print(day_idx, source_day_idx)
        print(out_sits.data[:, day_idx, ...])
        print(sits1.data[:, source_day_idx, ...])
        assert torch.all(
            out_sits.data[:, day_idx, ...] == sits1.data[:, source_day_idx, ...]
        )

        assert sits1.mask is not None and torch.all(
            out_sits.mask[:, day_idx, ...] == sits1.mask[:, source_day_idx, ...]
        )


def test_sits_where() -> None:
    """
    Test the sits_where function
    """
    doy1 = torch.tensor([[0, 1, 2, 5, 6]])
    data1 = torch.full((1, 5, 4, 1, 1), 10)
    data2 = torch.full((1, 5, 4, 1, 1), 20)

    mask1 = torch.full((1, 5, 1, 1), False)
    mask1[:, 2:, ...] = True
    mask2 = torch.full((1, 5, 1, 1), False)
    mask2[:, 4:, ...] = True

    sits1 = MonoModalSITS(data1, doy1, mask1)
    sits2 = MonoModalSITS(data2, doy1, mask2)
    assert sits1.mask is not None
    assert sits2.mask is not None
    print(sits1.data[0, :, 0, 0, 0])
    print(sits1.mask[0, :, 0, 0])
    print(sits2.data[0, :, 0, 0, 0])
    print(sits2.mask[0, :, 0, 0])
    output_sits = sits_where(sits1, sits2)
    assert output_sits.mask is not None
    print(output_sits.data[0, :, 0, 0, 0])
    print(output_sits.mask[0, :, 0, 0])
    assert torch.all(output_sits.data[:, :2, ...] == 10)
    assert torch.all(output_sits.data[:, 3:4, ...] == 20)
    assert torch.all(output_sits.mask[:, 4:, ...])

    sits2 = MonoModalSITS(data2, doy1)

    _ = sits_where(sits1, sits2)


def test_crop_sits():
    """
    Test the crop_sits free function
    """
    sits = generate_monomodal_sits(width=32, masked=True)
    sits_nomask = generate_monomodal_sits(width=32, masked=False)
    margin = 2
    out_sits = crop_sits(sits, margin=margin)
    out_sits_nomask = crop_sits(sits_nomask, margin=margin)

    assert out_sits.shape()[-1] == sits.shape()[-1] - 2 * margin
    assert out_sits.shape()[-2] == sits.shape()[-2] - 2 * margin
    assert out_sits_nomask.mask is None

    with raises(ValueError):
        _ = crop_sits(sits, margin=16)
        _ = crop_sits(sits, margin=10000)
