#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
This module handles everything related to MTF filtering and downsampling
"""

from functools import lru_cache

import numpy as np
import torch
from einops import parse_shape, rearrange, repeat
from numpy import sqrt as np_sqrt
from torch import exp, pi, sqrt  # pylint: disable=no-name-in-module

from tamrfsits.core.time_series import MonoModalSITS


@lru_cache
def generate_psf_kernel(
    res: float,
    mtf_res: torch.Tensor,
    mtf_fc: torch.Tensor,
    half_kernel_width: int | None = None,
) -> torch.Tensor:
    """
    Generate a psf convolution kernel for each pair of mtf_res / mtf_fc
    """
    assert mtf_res.shape == mtf_fc.shape
    # Make sure mtf is never greater than 1. or lower than 0.
    mtf_fc = torch.clip(mtf_fc, 0.0, 1.0)
    fc = 0.5 / mtf_res
    sigma = sqrt(-torch.log(mtf_fc) / 2) / (torch.pi * fc)
    if half_kernel_width is None:
        half_kernel_width = int(np.ceil(torch.max(mtf_res).cpu().numpy() / (res)))
    kernel = torch.zeros(
        (mtf_fc.shape[0], 2 * half_kernel_width + 1, 2 * half_kernel_width + 1),
        device=mtf_fc.device,
        dtype=mtf_fc.dtype,
    )
    inside_constant_factor = 1 / (2 * sigma * sigma)
    outside_constant_factor = 1 / (sigma * np_sqrt(2 * pi))
    for i in range(0, half_kernel_width + 1):
        squared_i = i**2
        half_kernel_width_minus_i = half_kernel_width - i
        half_kernel_width_plus_i = half_kernel_width + i
        for j in range(0, half_kernel_width + 1):
            dist = res * np_sqrt(squared_i + j**2)
            psf = exp(-(dist * dist) * inside_constant_factor) * outside_constant_factor
            kernel[:, half_kernel_width_minus_i, half_kernel_width - j] = psf
            kernel[:, half_kernel_width_minus_i, half_kernel_width + j] = psf
            kernel[:, half_kernel_width_plus_i, half_kernel_width + j] = psf
            kernel[:, half_kernel_width_plus_i, half_kernel_width - j] = psf

    kernel = kernel / torch.sum(kernel, dim=(-1, -2), keepdim=True)
    # If mtf_fc is 1., yield dirac kernel
    kernel[mtf_fc == 1.0, ...] = 0.0
    kernel[mtf_fc == 1.0, half_kernel_width, half_kernel_width] = 1.0

    return kernel


def generic_downscale(
    data: torch.Tensor,
    factor: float = 2.0,
    mtf: float = 0.1,
    padding="valid",
    mode: str = "bicubic",
    hkw: int = 3,
):
    """
    Downsample patches with proper aliasing filtering
    """
    # Generate psf kernel for target MTF
    mtf_t = torch.tensor([mtf] * data.shape[1], device=data.device)
    factor_t = torch.tensor([factor] * data.shape[1], device=data.device)
    psf_kernel = generate_psf_kernel(1.0, factor_t, mtf_t, hkw)

    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data, (hkw, hkw, hkw, hkw), mode="reflect")
    data = torch.nn.functional.conv2d(  # pylint: disable=not-callable
        data,
        repeat(psf_kernel, "c w h ->  c 1 w h"),
        groups=data.shape[1],
        padding=padding,
    )
    # Downsample with nearest neighbors
    data = torch.nn.functional.interpolate(data, scale_factor=1 / factor, mode=mode)
    return data


def convolve_tensor_with_psf(data: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve tensor with the generated psf kernel
    """
    # Retrieve half kernel width from kernel
    half_kernel_width = kernel.shape[-1] // 2

    # Colapse time and batch dim
    data_shape = parse_shape(data, "b c w h")

    # Pad by half_kernel_width
    data = torch.nn.functional.pad(data, (half_kernel_width,) * 4, mode="reflect")

    # Convolve with kernel
    data = torch.nn.functional.conv2d(  # pylint: disable=not-callable
        data,
        repeat(kernel, "c w h ->  c 1 w h"),
        padding="valid",
        groups=data_shape["c"],
    )

    return data


def convolve_sits_with_psf(sits: MonoModalSITS, kernel: torch.Tensor) -> MonoModalSITS:
    """
    Convolve sits with the generated psf kernel
    """
    # Colapse time and batch dim
    data_shape = parse_shape(sits.data, "b t c w h")
    data = rearrange(sits.data, "b t c w h -> (b t) c w h")

    # Convolve with psf
    data = convolve_tensor_with_psf(data, kernel)

    # unfold batch and time dimension
    data = rearrange(data, "(b t) c w h -> b t c w h", **data_shape)

    return MonoModalSITS(data, sits.doy, sits.mask)


def tensor_high_pass_filtering(
    data: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor:
    """
    Perform High Pass Filtering on a tensor
    """
    data_lpf = convolve_tensor_with_psf(data, kernel)

    return data - data_lpf


def sits_high_pass_filtering(
    sits: MonoModalSITS, kernel: torch.Tensor
) -> MonoModalSITS:
    """
    High Pass Filtering with provided kernel
    """
    convolved_sits = convolve_sits_with_psf(sits, kernel)

    return MonoModalSITS(sits.data - convolved_sits.data, sits.doy, sits.mask)


def tensor_gradient_magnitude(data: torch.Tensor) -> torch.Tensor:
    """
    Use squared gradient_magnitude operator to detect edges
    """
    first_order_x, first_order_y = torch.gradient(data, dim=(-1, -2))

    gradient_magnitude = first_order_x**2 + first_order_y**2

    return gradient_magnitude


def sits_gradient_magnitude(sits: MonoModalSITS) -> MonoModalSITS:
    """
    Use squared gradient_magnitude operator to detect edges
    """
    # Colapse time and batch dim
    data_shape = parse_shape(sits.data, "b t c w h")
    data = rearrange(sits.data, "b t c w h -> (b t) c w h")

    gradient_magnitude = tensor_gradient_magnitude(data)

    gradient_magnitude = rearrange(
        gradient_magnitude, "(b t) c w h -> b t c w h", **data_shape
    )

    return MonoModalSITS(gradient_magnitude, sits.doy, sits.mask)


def downsample_sits(
    sits: MonoModalSITS, factor: float, mode: str = "bicubic"
) -> MonoModalSITS:
    """
    Downsample SITS by the given factor
    """
    # Early exit
    if factor == 1.0:
        return sits

    # Colapse time and batch dim
    data_shape = parse_shape(sits.data, "b t c w h")
    data = rearrange(sits.data, "b t c w h -> (b t) c w h")

    # Apply downsampling to sits data
    dtype = data.dtype
    data = torch.nn.functional.interpolate(
        data.to(dtype=torch.float32), scale_factor=1 / factor, mode=mode
    ).to(dtype=dtype)

    # Unfold time and batch dim
    data = rearrange(
        data,
        "(b t) c w h -> b t c w h",
        b=data_shape["b"],
        t=data_shape["t"],
        c=data_shape["c"],
    )

    # If mask is not None, also downsample mask
    mask: torch.Tensor | None = None
    if sits.mask is not None:
        mask_shape = parse_shape(sits.mask, "b t w h")
        mask = rearrange(sits.mask, "b t w h -> (b t) w h")
        mask = (
            torch.nn.functional.interpolate(
                mask[:, None, ...].to(dtype=torch.float32),
                scale_factor=1 / factor,
                mode="nearest",
            )[:, 0, ...]
            > 0
        )

        # Unfold time and batch dim
        mask = rearrange(
            mask, "(b t) w h -> b t w h", b=mask_shape["b"], t=mask_shape["t"]
        )

    return MonoModalSITS(data, sits.doy, mask)


def downsample_sits_from_mtf(
    sits: MonoModalSITS,
    res: float,
    mtf_res: torch.Tensor,
    mtf_fc: torch.Tensor,
    factor: float,
    mode: str = "bicubic",
) -> MonoModalSITS:
    """
    Downsample SITS by the given factor
    """
    # Generate the corresponding kernels
    kernel = generate_psf_kernel(res, mtf_res, mtf_fc)

    # Filter sits with kernel
    filtered_sits = convolve_sits_with_psf(sits, kernel)

    # No need for downsampling if factor is 1.
    if factor == 1.0:
        return filtered_sits

    # Downsample sits
    return downsample_sits(filtered_sits, factor=factor, mode=mode)
