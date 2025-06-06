# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
Test the baselines for fusion
"""

import torch
from einops import rearrange

from tamrfsits.baselines.naive import NaiveSITSFusion
from tamrfsits.baselines.sen2like import Sen2LikeFusion
from tamrfsits.baselines.stair import STAIRSITSFusion
from tamrfsits.baselines.utils import load_dsen2_model, load_dstfn_model

from .tests_utils import FakeHRLRDatamodule


def test_naive_fusion():
    """
    Test NaiveSITSFusion class
    """
    nb_hr_features = 8
    nb_lr_features = 7
    resolution_ratio = 3

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

    # Get the dataloader
    dataloader = data_module.train_dataloader()

    # Get one batch
    batch = next(iter(dataloader))

    # Unpack batch
    lr_sits, hr_sits = batch

    # Derive target doys
    target_doy = torch.unique(torch.cat((lr_sits.doy, hr_sits.doy)))

    # Build the NaiveSITSFusion instance
    naive_sits_fusion = NaiveSITSFusion(lr_upsampling_factor=resolution_ratio)

    out_lr, out_hr = naive_sits_fusion(lr_sits, hr_sits, target_doy)

    assert out_lr.doy.shape == out_hr.doy.shape
    assert out_lr.doy.shape[1] == target_doy.shape[0]
    assert out_lr.mask is None
    assert out_hr.mask is None
    for i in (0, 1, 3, 4):
        assert out_lr.data.shape[i] == out_hr.data.shape[i]

    for i in (0, 2, 3, 4):
        assert out_hr.data.shape[i] == hr_sits.data.shape[i]

    assert out_lr.data.shape[2] == lr_sits.data.shape[2]

    assert torch.all(~torch.isnan(out_lr.data))
    assert torch.all(~torch.isnan(out_hr.data))


def test_stair_fusion():
    """
    Test STAIRSITSFusion class
    """
    nb_hr_features = 4
    nb_lr_features = 4
    resolution_ratio = 3

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

    # Get the dataloader
    dataloader = data_module.train_dataloader()

    # Get one batch
    batch = next(iter(dataloader))

    # Unpack batch
    lr_sits, hr_sits = batch

    # Build the NaiveSITSFusion instance
    stair_sits_fusion = STAIRSITSFusion(lr_upsampling_factor=resolution_ratio)

    out_lr, out_hr = stair_sits_fusion(lr_sits, hr_sits)

    for i in (0, 3, 4):
        assert out_lr.data.shape[i] == out_hr.data.shape[i]

    for i in (0, 2, 3, 4):
        assert out_hr.data.shape[i] == hr_sits.data.shape[i]

    assert torch.all(~torch.isnan(out_lr.data))
    assert torch.all(~torch.isnan(out_hr.data))


def test_sen2like_fusion():
    """
    Test sen2like fusion class
    """
    nb_hr_features = 4
    nb_lr_features = 4
    resolution_ratio = 3

    data_module = FakeHRLRDatamodule(
        batch_size=1,
        nb_doys=10,
        nb_hr_doys=16,
        nb_train_samples=16,
        nb_val_samples=8,
        nb_test_samples=4,
        nb_hr_features=nb_hr_features,
        nb_lr_features=nb_lr_features,
        resolution_ratio=resolution_ratio,
        max_doy=30,
        clear_doy_proba=0.5,
        lr_width=16,
        masked=True,
    )

    # Get the dataloader
    dataloader = data_module.train_dataloader()

    # Get one batch
    batch = next(iter(dataloader))

    # Unpack batch
    lr_sits, hr_sits = batch

    # Build the NaiveSITSFusion instance
    sen2like_sits_fusion = Sen2LikeFusion(lr_upsampling_factor=resolution_ratio)

    out_lr, out_hr = sen2like_sits_fusion(lr_sits, hr_sits)

    assert out_hr.mask is not None
    for i in (0, 3, 4):
        assert out_lr.data.shape[i] == out_hr.data.shape[i]

    for i in (0, 2, 3, 4):
        assert out_hr.data.shape[i] == hr_sits.data.shape[i]

    assert torch.all(~torch.isnan(out_lr.data))

    out_hr_data_flatten = rearrange(out_hr.data, "b t c w h -> (b t w h) c")

    assert torch.all(~torch.isnan(out_hr_data_flatten[~out_hr.mask.flatten()]))


def test_dsen2():
    """
    Test the dsen2 dependency
    """
    s2_10m = torch.rand((10, 4, 128, 128))
    s2_20m = torch.rand((10, 6, 128, 128))
    model = load_dsen2_model()
    with torch.inference_mode():
        pred = model(s2_10m, s2_20m)

    assert torch.all(~torch.isnan(pred))


def test_dstfn():
    """
    Test the dstfn pre-trained network
    """

    s2 = torch.rand((1, 6, 150, 150))
    ls = torch.rand((1, 6, 50, 50))
    ls_pan = torch.rand((1, 1, 100, 100))

    model = load_dstfn_model()
    with torch.inference_mode():
        out = model(s2, ls, ls_pan)

    assert torch.all(~torch.isnan(out))
