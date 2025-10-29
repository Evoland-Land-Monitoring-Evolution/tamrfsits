#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Tests utilities of model.torch.cca_loss
"""
import os
from dataclasses import dataclass
from time import perf_counter

import pytest
import rasterio as rio  # type: ignore
import torch
from einops import parse_shape, rearrange
from piq import LPIPS  # type: ignore

from tamrfsits.core.cca import (
    DatewiseLinearRegressionLoss,
    HighPassFilteringMode,
    batch_covariance,
    batched_weighted_product,
    vectorized_linear_regression_stable,
)
from tamrfsits.core.downsampling import (
    generate_psf_kernel,
    generic_downscale,
    tensor_high_pass_filtering,
)
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.validation.utils import MeasureExecTime

from .tests_utils import (
    generate_monomodal_sits,
    get_ls2s2_dataset_path,
    get_tests_output_path,
)


def test_batch_covariance():
    """
    Test the batch covariance function
    """
    data = torch.rand((10, 100, 8))
    mask = torch.rand((10, 100)) > 0.5

    cov = batch_covariance(data, mask)

    assert torch.all(~torch.isnan(cov))


def test_batched_weighed_product():
    """ """
    data1 = torch.rand((10, 8, 100))
    data2 = torch.rand((10, 100, 5))
    mask = torch.rand((10, 100)) > 0.5

    out = batched_weighted_product(data1, data2, mask)
    assert out.shape == torch.Size([10, 8, 5])
    assert torch.all(~torch.isnan(out))

    cov = batched_weighted_product(data1, data1.transpose(-1, -2), mask)

    assert cov.shape == torch.Size([10, 8, 8])
    assert torch.all(~torch.isnan(cov))


def test_lpips():
    """
    Test the Lpips loss
    """
    data1 = torch.rand((10, 3, 32, 32))
    data2 = torch.rand((10, 3, 32, 32))
    lpips = LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])

    lpips_compiled = torch.compile(lpips)

    out = lpips(data1, data2)
    out_c = lpips_compiled(data1, data2)

    assert ~torch.isnan(out)
    assert ~torch.isnan(out_c)


@pytest.mark.requires_gpu
def test_vectorized_linear_regression_stable():
    """
    Vectorized linear regression
    """
    src = 0.5 * torch.rand((2, 1000, 8), device=torch.device("cuda")) + 0.2

    solution = torch.rand((2, 8, 5), device=torch.device("cuda"))
    tgt = 1.2 * torch.matmul(src, solution) + 0.7

    mask = torch.rand((2, 1000), device=torch.device("cuda")) > 0.5

    for bias in (True, False):
        output = vectorized_linear_regression_stable(
            src, tgt, mask, precision=torch.float64, bias=bias
        )
        assert torch.all(~torch.isnan(output))

        error = (tgt - output).std(dim=1)

        assert torch.all(error < 1.0e-6)

    mask[1, :] = False

    output = vectorized_linear_regression_stable(
        src, tgt, mask, precision=torch.float64
    )

    assert torch.all(torch.isnan(output[1, :]))
    assert torch.all(~torch.isnan(output[0, :]))

    mask[:, :] = False

    output = vectorized_linear_regression_stable(
        src, tgt, mask, precision=torch.float64
    )

    assert torch.all(torch.isnan(output))


@pytest.mark.requires_gpu
@pytest.mark.requires_data
def test_linear_regression():
    """
    Test the linear regression code
    """
    with rio.open(
        os.path.join(
            get_ls2s2_dataset_path(),
            "../test/31UFS_12/sentinel2/20220308/sentinel2_bands_20220308.tif",
        )
    ) as ds:
        s2_data = torch.tensor(ds.read(), dtype=torch.float32) / 10000
        ds_transform = ds.transform
        ds_crs = ds.crs

    s2_data_gradx, s2_data_grady = torch.gradient(s2_data, dim=(-1, -2))
    s2_data = s2_data_gradx**2 + s2_data_grady**2
    s2_10m = s2_data[:4, ...]
    s2_20m = s2_data[4:, ...]

    s2_10m_flat = rearrange(s2_10m, "c w h -> (w h) c")
    s2_20m_shape = parse_shape(s2_20m, "c w h")
    s2_20m_flat = rearrange(s2_20m, "c w h -> (w h) c")

    s2_10m_transformed_flat = vectorized_linear_regression_stable(
        s2_10m_flat, s2_20m_flat, msk=torch.ones_like(s2_10m_flat)
    )
    s2_10m_transformed = rearrange(
        s2_10m_transformed_flat, "(w h) c -> c w h", **s2_20m_shape
    )

    profile_pred = {
        "driver": "GTiff",
        "height": s2_20m_shape["h"],
        "width": s2_20m_shape["h"],
        "count": s2_20m_shape["c"],
        "dtype": rio.float32,
        "transform": ds_transform,
        "crs": ds_crs,
    }

    with rio.open(
        os.path.join(get_tests_output_path(), "test_linear_reg.tif"),
        "w",
        **profile_pred,
    ) as ds:
        ds.write(10000 * s2_10m_transformed)

    assert s2_10m_transformed is not None and torch.all(
        ~torch.isnan(s2_10m_transformed)
    )


@pytest.mark.requires_gpu
@pytest.mark.requires_data
def test_linear_regression_tir():
    """
    Test the linear regression code
    """
    with rio.open(
        os.path.join(
            get_ls2s2_dataset_path(),
            "../test/31UFS_12/sentinel2/20220729/sentinel2_bands_20220729.tif",
        )
    ) as ds:
        s2_data = torch.tensor(ds.read(), dtype=torch.float32) / 10000

    with rio.open(
        os.path.join(
            get_ls2s2_dataset_path(),
            "../test/31UFS_12/landsat/20220729/landsat_bands_20220729.tif",
        )
    ) as ds:
        ls_data = torch.tensor(ds.read(), dtype=torch.float32) / 10
        ds_transform = ds.transform
        ds_crs = ds.crs

    s2_data = generic_downscale(s2_data[None, ...], factor=3.0, mtf=0.00001)[0, ...]
    s2_data = s2_data[[0, 1, 2, 6], :165, :165]
    # s2_ndvi = (s2_data[3] - s2_data[2]) / ((s2_data[2] + s2_data[3]))
    # s2_data = torch.cat((s2_data, s2_ndvi[None, ...]))
    # s2_data = ls_data[:7, :165, :165].clone()
    s2_data_shape = parse_shape(s2_data, "c w h")
    ls_data = ls_data[7:, :165, :165]

    mtf_t = torch.tensor([0.1] * s2_data.shape[0])
    factor_t = torch.tensor([1.0] * s2_data.shape[0])
    psf_kernel = generate_psf_kernel(1.0, factor_t, mtf_t, 3)
    s2_data = tensor_high_pass_filtering(s2_data[None, ...], psf_kernel)[0, ...]
    mtf_t = torch.tensor([0.1] * ls_data.shape[0])
    factor_t = torch.tensor([1.0] * ls_data.shape[0])
    psf_kernel = generate_psf_kernel(1.0, factor_t, mtf_t, 3)
    ls_data = tensor_high_pass_filtering(ls_data[None, ...], psf_kernel)[0, ...]

    # ls_10m = torch.nn.functional.interpolate(
    #     ls_data[None, ...],
    #     scale_factor=3.0,
    #     mode="bicubic",
    # )[0, ...]
    # s2_data = tensor_gradient_magnitude(s2_data)
    # ls_data = tensor_gradient_magnitude(ls_data)
    s2_data_flat = rearrange(s2_data, "c w h -> 1 (w h) c")
    ls_data_flat = rearrange(ls_data, "c w h -> 1 (w h) c")
    s2_data_transformed_flat = vectorized_linear_regression_stable(
        s2_data_flat,
        ls_data_flat,
        msk=torch.ones_like(s2_data_flat[:, :, -1]),
        bias=True,
    )

    s2_data_transformed = rearrange(
        s2_data_transformed_flat,
        "1 (w h) c -> c w h",
        w=s2_data_shape["w"],
        h=s2_data_shape["h"],
        c=1,
    )
    profile_pred = {
        "driver": "GTiff",
        "height": s2_data_shape["h"],
        "width": s2_data_shape["h"],
        "count": 1,
        "dtype": rio.float32,
        "transform": ds_transform,
        "crs": ds_crs,
    }

    with rio.open(
        os.path.join(get_tests_output_path(), "test_linear_reg_tir.tif"),
        "w",
        **profile_pred,
    ) as ds:
        ds.write(s2_data_transformed)
    with rio.open(
        os.path.join(get_tests_output_path(), "test_linear_reg_tir_ref.tif"),
        "w",
        **profile_pred,
    ) as ds:
        ds.write(ls_data)
    profile_pred = {
        "driver": "GTiff",
        "height": s2_data_shape["h"],
        "width": s2_data_shape["h"],
        "count": s2_data_shape["c"],
        "dtype": rio.float32,
        "transform": ds_transform,
        "crs": ds_crs,
    }
    with rio.open(
        os.path.join(get_tests_output_path(), "test_linear_reg_tir_input.tif"),
        "w",
        **profile_pred,
    ) as ds:
        ds.write(s2_data)

    assert s2_data_transformed is not None and torch.all(
        ~torch.isnan(s2_data_transformed)
    )


@dataclass
class TestConfig:
    """
    Configuration for the test of DatewiseLinearCCALoss class
    """

    high_pass_filtering_mode: HighPassFilteringMode = HighPassFilteringMode.PRE
    use_lpips: bool = False
    mtf_for_high_pass_filtering: float | None = None
    per_date: bool = True


@pytest.mark.parametrize(
    "config",
    [
        TestConfig(HighPassFilteringMode.NO, False, None, True),
        TestConfig(HighPassFilteringMode.NO, False, None, False),
        TestConfig(HighPassFilteringMode.PRE, False, None, True),
        TestConfig(HighPassFilteringMode.PRE, False, 0.1, False),
        TestConfig(HighPassFilteringMode.PRE, False, None, True),
        TestConfig(HighPassFilteringMode.PRE, False, 0.1, False),
        TestConfig(HighPassFilteringMode.POST, False, None),
        TestConfig(HighPassFilteringMode.POST, False, 0.1),
        TestConfig(HighPassFilteringMode.POST, False, None, True),
        TestConfig(HighPassFilteringMode.POST, False, 0.1, False),
        TestConfig(HighPassFilteringMode.NO, True, None, True),
        TestConfig(HighPassFilteringMode.NO, True, None, False),
    ],
)
def test_datewise_linear_regression_loss(config: TestConfig):
    """
    Test the DatewiseLinearRegressionLoss
    """
    loss = DatewiseLinearRegressionLoss(
        loss=torch.nn.MSELoss(),
        mtf_for_high_pass_filtering=config.mtf_for_high_pass_filtering,
    )

    pred_sits = generate_monomodal_sits(nb_features=4, width=64, nb_doy=10, batch=1)
    target_sits = generate_monomodal_sits(nb_features=3, width=64, nb_doy=10, batch=1)

    target_sits = MonoModalSITS(
        torch.cat((2 * pred_sits.data, target_sits.data), dim=2),
        target_sits.doy,
        target_sits.mask,
    )

    with MeasureExecTime(label="cca_loss", log_time=True):
        loss_value = loss(
            pred_sits,
            target_sits,
            high_pass_filtering_mode=config.high_pass_filtering_mode,
            use_lpips=config.use_lpips,
            per_date=config.per_date,
        )

    assert loss_value is not None and torch.isnan(loss_value).sum() == 0


@pytest.mark.parametrize(
    "config",
    [
        TestConfig(HighPassFilteringMode.NO, False, 0.1, False),
    ],
)
@pytest.mark.requires_gpu
@pytest.mark.performances
def test_datewise_linear_regression_loss_performances(config: TestConfig):
    """
    Test performance of datewise linear cca loss
    """

    loss = DatewiseLinearRegressionLoss(
        loss=torch.nn.MSELoss(),
        mtf_for_high_pass_filtering=config.mtf_for_high_pass_filtering,
    )

    pred_sits = generate_monomodal_sits(nb_doy=50, max_doy=50, nb_features=8).to(
        device=torch.device("cuda")
    )
    target_sits = generate_monomodal_sits(nb_doy=50, max_doy=50, nb_features=8).to(
        device=torch.device("cuda")
    )
    torch.cuda.synchronize(device=torch.device("cuda"))
    start = perf_counter()
    loss(pred_sits, target_sits)
    torch.cuda.synchronize(device=torch.device("cuda"))
    stop = perf_counter()
    exec_time = stop - start
    print(f"{exec_time=}")
    assert exec_time < 0.2
