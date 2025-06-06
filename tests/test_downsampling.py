#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Tests downsampling manager module
"""

import os

import pytest
import rasterio as rio  # type: ignore
import torch
from affine import Affine  # type: ignore

from tamrfsits.core.downsampling import (
    convolve_sits_with_psf,
    downsample_sits,
    downsample_sits_from_mtf,
    generate_psf_kernel,
    sits_gradient_magnitude,
    sits_high_pass_filtering,
)
from tamrfsits.core.time_series import MonoModalSITS

from .tests_utils import (
    generate_monomodal_sits,
    get_ls2s2_dataset_path,
    get_tests_output_path,
)


def test_generate_psf_kernel():
    """
    Test the generate_psf_kernel free function
    """
    mtf = torch.tensor([0.1, 0.1], requires_grad=True)
    mtf_res = torch.tensor([1.0, 2.0], requires_grad=False)
    kernel = generate_psf_kernel(1.0, mtf_res, mtf)

    kernel.sum().backward()

    assert mtf._grad is not None


def test_convolve_sits_with_psf():
    """
    Test the convolve sits with psf free function
    """
    sits = generate_monomodal_sits(masked=True)

    mtf = torch.tensor([0.1] * sits.shape()[2], requires_grad=True)
    mtf_res = torch.tensor([1.0] * sits.shape()[2], requires_grad=False)

    kernel = generate_psf_kernel(1.0, mtf_res, mtf)

    convolved_sits = convolve_sits_with_psf(sits, kernel)

    convolved_sits.data.sum().backward()

    assert mtf._grad is not None


def test_sits_high_pass_filtering():
    """
    Test the sits_high_pass_filtering free function
    """
    sits = generate_monomodal_sits(masked=True)

    mtf = torch.tensor([0.1] * sits.shape()[2], requires_grad=True)
    mtf_res = torch.tensor([1.0] * sits.shape()[2], requires_grad=False)

    kernel = generate_psf_kernel(1.0, mtf_res, mtf)

    hpf_sits = sits_high_pass_filtering(sits, kernel)

    assert torch.isnan(hpf_sits.data).sum() == 0


def test_downsample_sits():
    """
    Test the downsample_sits free function
    """
    sits = generate_monomodal_sits(masked=True)

    ds_sits = downsample_sits(sits, factor=2.0)

    assert sits.shape()[-1] / 2 == ds_sits.shape()[-1]


def test_downsample_sits_from_mtf():
    """
    Test the downsample_sits_from_mtf free function
    """
    sits = generate_monomodal_sits(masked=True)
    mtf = torch.tensor([0.1] * sits.shape()[2], requires_grad=True)
    mtf_res = torch.tensor([1.0] * sits.shape()[2], requires_grad=False)

    ds_sits = downsample_sits_from_mtf(
        sits, res=10.0, mtf_res=mtf_res, mtf_fc=mtf, factor=2.0
    )

    assert sits.shape()[-1] / 2 == ds_sits.shape()[-1]


def test_sits_gradient_magnitude():
    """
    Test the sits_gradient_magnitude function
    """
    sits = generate_monomodal_sits(masked=True)

    laplace_sits = sits_gradient_magnitude(sits)

    assert torch.all(~torch.isnan(laplace_sits.data))


@pytest.mark.requires_data
def test_on_real_images():
    """
    Test
    """
    with rio.open(
        os.path.join(
            get_ls2s2_dataset_path(),
            "../test/31UFS_12/sentinel2/20220308/sentinel2_bands_20220308.tif",
        )
    ) as ds:
        h2 = torch.tensor(ds.read(), dtype=torch.float32)
        ds_transform = ds.transform
        ds_crs = ds.crs
    sits = MonoModalSITS(data=h2[None, None, 2:3, ...], doy=torch.full((1, 1), 0))
    mtf = torch.tensor([0.2] * sits.shape()[2])
    mtf_res = torch.tensor([90.0] * sits.shape()[2])

    factor = 3.0
    ds_sits = downsample_sits_from_mtf(
        sits, res=10.0, mtf_res=mtf_res, mtf_fc=mtf, factor=factor
    )

    kernel = generate_psf_kernel(1.0, mtf_res, mtf)

    hpf_sits = sits_high_pass_filtering(sits, kernel)
    convolve_sits = convolve_sits_with_psf(sits, kernel)

    assert torch.all(~torch.isnan(hpf_sits.data))
    assert torch.all(~torch.isnan(ds_sits.data))
    assert torch.all(~torch.isnan(convolve_sits.data))

    profile_pred = {
        "driver": "GTiff",
        "height": ds_sits.shape()[-1],
        "width": ds_sits.shape()[-2],
        "count": ds_sits.shape()[-3],
        "dtype": rio.float32,
        "transform": ds_transform * Affine.scale(factor),
        "crs": ds_crs,
    }

    with rio.open(
        os.path.join(get_tests_output_path(), "test_downsample_sits_from_mtf.tif"),
        "w",
        **profile_pred,
    ) as ds:
        ds.write(ds_sits.data[0, 0, ...].detach().numpy())
