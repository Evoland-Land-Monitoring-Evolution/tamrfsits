#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Tests of datewise module
"""
import warnings

import pytest
import torch

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.core import carn, esrgan, mlp, unet

from .tests_utils import generate_monomodal_sits


@pytest.mark.parametrize(
    "model",
    [
        mlp.MLP(mlp.MLPConfig(4, 16, hidden_layers=[8, 8])),
        carn.CARN(carn.CARNConfig(4, 16)),
        esrgan.ESRGANGenerator(
            4,
            out_nb_bands=16,
            upsampling_factor=2.0,
            num_basic_blocks=1,
            latent_size=32,
        ),
        esrgan.ESRSpatialEncoder(
            4,
            upsampling_factor=2.0,
            num_basic_blocks=1,
            latent_size=16,
            growth_channels=8,
            upsampling_mode="pixel_shuffle",
        ),
        esrgan.ESRSpatialEncoder(
            4,
            upsampling_factor=2.0,
            num_basic_blocks=1,
            latent_size=16,
            growth_channels=8,
            upsampling_mode="bicubic",
        ),
        esrgan.ESRSpatialDecoder(
            latent_size=4,
            out_nb_bands=16,
        ),
        unet.SRUNet(in_channels=4, out_channels=16),
        unet.SRUNet(in_channels=4, out_channels=16, scale_factor=3.0),
    ],
    ids=[
        "mlp",
        "carn",
        "ersgan",
        "esrenc_ps",
        "esrenc_interp",
        "esrdec",
        "unet",
        "srunet",
    ],
)
def test_datewise_module(model: torch.nn.Module):
    """
    Test the datewise module
    """
    sits_masked = generate_monomodal_sits(nb_features=4, masked=True, batch=4, width=8)
    sits_nomask = generate_monomodal_sits(nb_features=4, masked=False, batch=4, width=8)

    # Instantiate module
    datewise_model = DateWiseSITSModule(model, compiled=False)
    datewise_modelx2 = DateWiseSITSModule(model, upsampling_factor=2, compiled=False)
    datewise_modelx2_early = DateWiseSITSModule(
        model, upsampling_factor=2, early_upsampling=True, compiled=False
    )

    # Call forward pass
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        with torch.autograd.detect_anomaly(check_nan=True):
            output_masked = datewise_model(sits_masked)
            output_nomask = datewise_model(sits_nomask)
            output_masked_x2 = datewise_modelx2(sits_masked)
            output_nomask_x2 = datewise_modelx2(sits_nomask)
            output_masked_x2_early = datewise_modelx2_early(sits_masked)
            output_nomask_x2_early = datewise_modelx2_early(sits_nomask)

    assert output_masked.mask is not None
    assert output_nomask.mask is None
    assert torch.all(~torch.isnan(output_masked.data))
    assert torch.all(~torch.isnan(output_nomask.data))
    assert output_masked_x2.mask is not None
    assert output_nomask_x2.mask is None
    assert torch.all(~torch.isnan(output_masked_x2.data))
    assert torch.all(~torch.isnan(output_nomask_x2.data))
    assert output_masked_x2_early.mask is not None
    assert output_nomask_x2_early.mask is None
    assert torch.all(~torch.isnan(output_masked_x2_early.data))
    assert torch.all(~torch.isnan(output_nomask_x2_early.data))

    output_masked.data.sum().backward()
    output_masked_x2.data.sum().backward()
    output_masked_x2_early.data.sum().backward()

    for params in datewise_model.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0

    for params in datewise_modelx2.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0

    for params in datewise_modelx2_early.parameters():
        if params.requires_grad:
            assert params.grad is not None
            assert torch.isnan(params.grad).sum() == 0
