#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Tests of missing data imputer
"""
import warnings

import torch

from tamrfsits.components.missing_data_imputer import MissingDataImputer

from .tests_utils import generate_monomodal_sits


def test_missing_data_imputer() -> None:
    """
    Test MissingDataImputer class
    """
    nb_features = 5
    sits = generate_monomodal_sits(nb_features=nb_features, masked=True)

    net_fixed = MissingDataImputer(nb_features=nb_features)
    net_learnable = MissingDataImputer(
        nb_features=nb_features,
        learnable_token=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly(check_nan=True):
            out_sits = net_fixed(sits)

    assert torch.all(sits.mask == out_sits.mask)

    assert torch.isnan(out_sits.data).sum() == 0

    # No backprop test here since there is no learnable parameters in fixed mode
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly(check_nan=True):
            out_sits = net_learnable(sits)

    assert torch.all(out_sits.mask == sits.mask)

    assert torch.isnan(out_sits.data).sum() == 0

    out_sits.data.sum().backward()

    for params in net_learnable.parameters():
        assert params.grad is not None
        assert torch.isnan(params.grad).sum() == 0
