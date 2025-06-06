#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Tests utilities of model.torch.utils
"""

import torch

from tamrfsits.core.utils import (
    add_ndvi_to_sits,
    combine_sits,
    doy_unique,
    split_sits_features,
)

from .tests_utils import generate_monomodal_sits


def test_doy_unique() -> None:
    """
    Test the unique doy function
    """
    doy1 = torch.randint(0, 10, (2, 5))
    doy2 = torch.randint(0, 10, (2, 6))

    doy1 = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    doy2 = torch.tensor([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])

    out_doy, out_mask = doy_unique(doy1, doy2)

    assert torch.all(
        out_doy == torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0], [0, 1, 2, 3, 4, 5, 6, 7]])
    )

    assert torch.all(
        out_mask
        == torch.tensor(
            [
                [True, True, True, True, True, True, True, False],
                [True, True, True, True, True, True, True, True],
            ]
        )
    )


def test_combine_sits():
    """
    Try the full pipeline
    """
    batch = 1
    nb_hr_doy = 20
    nb_lr_doy = 40
    nb_features = 2
    width = 32
    hr_sits = generate_monomodal_sits(
        batch=batch,
        nb_doy=nb_hr_doy,
        nb_features=nb_features,
        width=width,
        masked=True,
        max_doy=100,
    )
    lr_sits = generate_monomodal_sits(
        batch=batch,
        nb_doy=nb_lr_doy,
        nb_features=nb_features,
        width=width,
        masked=True,
        max_doy=100,
    )

    _ = combine_sits(hr_sits, lr_sits)


def test_split_sits_features():
    """
    Test the split_sits_features free function
    """
    sits = generate_monomodal_sits(nb_features=16)

    sits1, sits2 = split_sits_features(sits, 10)

    assert sits1.shape()[2] == 10
    assert sits2.shape()[2] == 6


def test_add_ndvi_to_sits():
    """
    Test for the add_ndvi_to_sits method
    """
    hr_sits = generate_monomodal_sits(
        batch=1,
        nb_doy=20,
        nb_features=4,
        width=32,
        masked=True,
        max_doy=100,
    )

    hr_sits_with_ndvi = add_ndvi_to_sits(hr_sits)

    assert hr_sits_with_ndvi.shape()[2] == 5
