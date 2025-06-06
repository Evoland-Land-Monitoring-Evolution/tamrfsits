#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Test positional encoders module
"""

import torch

from tamrfsits.components.positional_encoder import (
    FixedPositionalEncoder,
    LearnablePositionalEncoder,
    PositionalEncodingMode,
)

from .tests_utils import generate_monomodal_sits


def test_fixed_positional_encoder():
    """
    Test the positional encoder class
    """
    nb_features = 32
    sits = generate_monomodal_sits(nb_features=nb_features, masked=True)

    pe = FixedPositionalEncoder(nb_features=32)
    out = pe(sits)

    assert sits.data.shape == out.data.shape

    assert torch.isnan(out.data).sum() == 0

    pe = FixedPositionalEncoder(nb_features=12, mode=PositionalEncodingMode.YIELD)
    out = pe(sits)

    assert out.data.shape[2] == 12

    assert torch.isnan(out.data).sum() == 0

    pe = FixedPositionalEncoder(nb_features=12, mode=PositionalEncodingMode.CONCAT)
    out = pe(sits)

    assert out.data.shape[2] == 44

    assert torch.isnan(out.data).sum() == 0


def test_learnable_positional_encoder():
    """
    Test the LearnablePositionalEncoder class
    """

    nb_features = 32
    sits = generate_monomodal_sits(nb_features=nb_features, masked=True)

    pe = LearnablePositionalEncoder(nb_features=nb_features)
    out = pe(sits)

    assert sits.data.shape == out.data.shape

    assert torch.isnan(out.data).sum() == 0

    pe = LearnablePositionalEncoder(nb_features=12, mode=PositionalEncodingMode.YIELD)
    out = pe(sits)

    assert out.data.shape[2] == 12

    assert torch.isnan(out.data).sum() == 0

    pe = LearnablePositionalEncoder(nb_features=12, mode=PositionalEncodingMode.CONCAT)
    out = pe(sits)

    assert out.data.shape[2] == 44

    assert torch.isnan(out.data).sum() == 0
