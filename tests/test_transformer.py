#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Tests of the transformer module
"""

import warnings

import torch

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.components.positional_encoder import FixedPositionalEncoder
from tamrfsits.components.transformer import (
    SITSMultiHeadAttention,
    SITSTransformerEncoder,
    TemporalDecoder,
    TemporalEncoder,
)
from tamrfsits.core.mlp import MLP, MLPConfig

from .tests_utils import generate_monomodal_sits


def test_sits_transformer_encoder():
    """
    Test the SITSTransformerEncoder module
    """
    sits_masked = generate_monomodal_sits(masked=True, nb_features=16)
    sits_nomask = generate_monomodal_sits(masked=False, nb_features=16)

    model = SITSTransformerEncoder(feature_dim=16)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly(check_nan=True):
            output_masked = model(sits_masked)
            output_nomask = model(sits_nomask)

    assert output_masked.mask is None
    assert output_nomask.mask is None
    assert torch.all(~torch.isnan(output_masked.data))
    assert torch.all(~torch.isnan(output_nomask.data))

    output_masked.data.sum().backward()

    for params in model.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0


def test_sits_multihead_attention():
    """
    Test the SITSMultiHeadAttention class
    """
    sits_masked = generate_monomodal_sits(masked=True, nb_features=16)
    sits_nomask = generate_monomodal_sits(masked=False, nb_features=16)

    query_sits = generate_monomodal_sits(masked=False, nb_features=16)

    model = SITSMultiHeadAttention(feature_dim=16, number_of_heads=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly(check_nan=True):
            output_masked = model(sits_masked, query_sits)
            output_nomask = model(sits_nomask, query_sits)

    assert output_masked.mask is not None
    assert output_nomask.mask is None
    assert torch.all(~torch.isnan(output_masked.data))
    assert torch.all(~torch.isnan(output_nomask.data))

    output_masked.data.sum().backward()

    for params in model.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0


def test_temporal_encoder():
    """
    Test the TemporalEncoder class
    """
    sits1 = generate_monomodal_sits(masked=False, nb_features=16)
    sits2 = generate_monomodal_sits(masked=False, nb_features=16)

    encoder = DateWiseSITSModule(MLP(MLPConfig(16, 16)))

    pos_encoding = FixedPositionalEncoder(nb_features=16)

    model = TemporalEncoder(
        hr_encoder=encoder,
        lr_encoder=encoder,
        temporal_positional_encoder=pos_encoding,
        token_size=16,
        sensor_token_size=2,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly(check_nan=True):
            output = model(sits1, sits2)

    assert output.mask is None
    assert torch.all(~torch.isnan(output.data))

    output.data.sum().backward()

    for params in model.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0


def test_temporal_decoder():
    """
    Test the TemporalDecoder class
    """
    # 18 in order to include sensor token size
    encoded = generate_monomodal_sits(masked=False, nb_features=18)

    query = torch.rand((10,))

    decoder = DateWiseSITSModule(MLP(MLPConfig(18, 4)))

    pos_encoding = FixedPositionalEncoder(nb_features=16)

    model = TemporalDecoder(
        decoder=decoder,
        temporal_positional_encoder=pos_encoding,
        nb_heads=2,
        token_size=16,
        sensor_token_size=2,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly(check_nan=True):
            output = model(encoded, query)

    assert output.mask is None
    assert torch.all(~torch.isnan(output.data))

    output.data.sum().backward()

    for params in model.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0
