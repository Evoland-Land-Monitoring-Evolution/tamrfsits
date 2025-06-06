#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Test the training module
"""

import pytest
import pytorch_lightning as pl
import torch

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.components.positional_encoder import FixedPositionalEncoder
from tamrfsits.components.transformer import TemporalDecoder, TemporalEncoder
from tamrfsits.core.cca import DatewiseLinearRegressionLoss
from tamrfsits.core.mlp import MLP, MLPConfig
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.tasks.base import (
    MTFParameters,
    OptimizationParameters,
    StandardizationParameters,
)
from tamrfsits.tasks.interpolation.training_module import (
    ResolutionForCCA,
    TemporalInterpolationTrainingModule,
    TemporalInterpolationTrainingModuleParameters,
    sits_mae_clr_splitter,
)

from .tests_utils import FakeHRLRDatamodule


def test_sits_mae_clr_splitter():
    """
    Test the sits_mae_clr_splitter function
    """
    target_doy = torch.tensor([[0, 1, 2, 3, 4]])
    input_doy = torch.tensor([[0, 1, 2]])
    sits = MonoModalSITS(torch.rand((1, 5, 1, 1, 1)), doy=target_doy)
    sits_mae_output, sits_clr_output = sits_mae_clr_splitter(
        sits, input_doy, target_doy
    )

    assert torch.all(sits_mae_output.doy == torch.tensor([[3, 4]]))
    assert torch.all(sits_clr_output.doy == torch.tensor([[0, 1, 2]]))


@pytest.mark.requires_gpu
def test_training_module():
    """
    Prototype for training loop
    """
    nb_hr_features = 8
    nb_lr_features = 7
    resolution_ratio = 3
    nb_latent_features = 64

    data_module = FakeHRLRDatamodule(
        batch_size=1,
        nb_doys=10,
        nb_train_samples=16,
        nb_val_samples=8,
        nb_test_samples=4,
        nb_hr_features=nb_hr_features,
        nb_lr_features=nb_lr_features,
        resolution_ratio=resolution_ratio,
        max_doy=15,
        clear_doy_proba=0.5,
        lr_width=16,
        masked=True,
    )

    opt_params = OptimizationParameters(
        learning_rate=0.001, t_0=200, t_mult=2, loss=torch.nn.MSELoss()
    )

    std_params = StandardizationParameters(
        hr_mean=(0.0,) * nb_hr_features,
        hr_std=(1.0,) * nb_hr_features,
        lr_mean=(0.0,) * nb_lr_features,
        lr_std=(1.0,) * nb_lr_features,
    )

    mtf_params = MTFParameters(
        lr_mtf=(0.2,) * nb_lr_features,
        hr_mtf=(0.2,) * nb_hr_features,
        lr_resolution=(resolution_ratio,) * nb_lr_features,
        hr_resolution=(1.0,) * nb_hr_features,
        output_resolution=1.0,
    )

    config = TemporalInterpolationTrainingModuleParameters(
        optimization=opt_params,
        standardization=std_params,
        mtfs=mtf_params,
        resolutions_for_cca_loss=(ResolutionForCCA(1.0, 3.0),),
    )

    hr_encoder = DateWiseSITSModule(
        MLP(MLPConfig(nb_hr_features, nb_latent_features, hidden_layers=[])),
    )
    lr_encoder = DateWiseSITSModule(
        MLP(MLPConfig(nb_lr_features, nb_latent_features, hidden_layers=[])),
        upsampling_factor=resolution_ratio,
    )
    decoder = DateWiseSITSModule(
        MLP(
            MLPConfig(
                nb_latent_features + 2,
                nb_lr_features + nb_hr_features,
                hidden_layers=[],
            )
        )
    )

    pos_encoding = FixedPositionalEncoder(nb_features=nb_latent_features)

    temporal_encoder = TemporalEncoder(
        hr_encoder=hr_encoder,
        lr_encoder=lr_encoder,
        temporal_positional_encoder=pos_encoding,
        token_size=nb_latent_features,
        nb_layers=1,
        sensor_token_size=2,
    )

    temporal_decoder = TemporalDecoder(
        decoder=decoder,
        temporal_positional_encoder=pos_encoding,
        token_size=nb_latent_features,
        nb_heads=1,
        sensor_token_size=2,
    )

    cca_loss = DatewiseLinearRegressionLoss(loss=torch.nn.MSELoss())

    training_module = TemporalInterpolationTrainingModule(
        config=config,
        encoder=temporal_encoder,
        decoder=temporal_decoder,
        cca_loss=cca_loss,
    )

    model_summary_cb = pl.callbacks.RichModelSummary(max_depth=-1)
    progress_bar_cb = pl.callbacks.RichProgressBar(leave=True)

    trainer = pl.Trainer(
        max_epochs=1,
        val_check_interval=0.5,
        callbacks=[model_summary_cb, progress_bar_cb],
        log_every_n_steps=1,
        accelerator="gpu",
        precision="bf16",
    )

    trainer.fit(training_module, data_module)

    # Generate samples from testing set
    test_dataloader = data_module.test_dataloader()
    lr_sits, hr_sits = next(iter(test_dataloader))
    target_doy = torch.rand(10)
    out_lr_sits, out_hr_sits = training_module.predict(
        lr_sits, hr_sits, hr_query_doy=hr_sits.doy, lr_query_doy=hr_sits.doy
    )

    assert out_lr_sits is not None
    assert out_hr_sits is not None
    assert lr_sits is not None
    assert hr_sits is not None
    assert out_lr_sits.shape()[-1] == out_hr_sits.shape()[-1] == hr_sits.shape()[-1]
    assert out_lr_sits.shape()[-2] == out_hr_sits.shape()[-2] == hr_sits.shape()[-2]

    out_lr_sits, out_hr_sits = training_module.predict(
        lr_sits,
        hr_sits,
        lr_query_doy=target_doy,
        hr_query_doy=target_doy,
        downscale=False,
    )
    assert out_lr_sits is not None
    assert out_hr_sits is not None

    assert out_lr_sits.shape()[-1] == out_hr_sits.shape()[-1] == hr_sits.shape()[-1]
    assert out_lr_sits.shape()[-2] == out_hr_sits.shape()[-2] == hr_sits.shape()[-2]
    assert out_lr_sits.shape()[1] == 10
    assert out_hr_sits.shape()[1] == 10

    out_lr_sits, _ = training_module.predict(
        lr_sits,
        hr_sits,
        lr_query_doy=target_doy,
        hr_query_doy=target_doy,
        downscale=True,
    )
    assert out_lr_sits is not None

    assert out_lr_sits.shape()[-1] == lr_sits.shape()[-1]
    assert out_lr_sits.shape()[-2] == lr_sits.shape()[-2]
