# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
This module contains function related to validation metrics.
"""

import math
from typing import cast

import torch
from einops import parse_shape, rearrange, repeat
from piq import brisque  # type: ignore

from tamrfsits.core.utils import patchify
from tamrfsits.validation.fda import compute_fft_profile, compute_frr_referenceless


def derive_reference_mask(
    mask: torch.Tensor, nb_features: int, spatial_margin: int | None = None
) -> torch.Tensor:
    """
    Compute the mask to apply for metrics
    """
    assert spatial_margin is None or 2 * spatial_margin < mask.shape[-1]
    assert spatial_margin is None or 2 * spatial_margin < mask.shape[-2]

    # Reshape mask to data shape
    mask = repeat(~mask, "b t w h -> b t c w h", c=nb_features).clone()

    # Mask spatial margin if required
    if spatial_margin is not None:
        mask[:, :, :, -spatial_margin:, :] = False
        mask[:, :, :, :spatial_margin, :] = False
        mask[:, :, :, :, -spatial_margin:] = False
        mask[:, :, :, :, :spatial_margin] = False

    return mask


def per_date_per_band_brisque(
    pred: torch.Tensor,
    shift: float = 0.0,
    scale: float = 1.0,
    data_range: float = 1.0,
    chunk_size: int = 100,
) -> torch.Tensor:
    """
    Compute brisque score per date and per band
    """
    pred[torch.isnan(pred)] = 0.0
    output = torch.full((pred.shape[1], pred.shape[2]), 100.0, device=pred.device)
    clipped_pred = torch.clip(scale * (shift + pred), 0.0, data_range)

    pred_shape = parse_shape(clipped_pred, "b t c w h")
    clipped_pred = rearrange(clipped_pred, "b t c w h -> (b t c) 1 w h")
    if clipped_pred.numel() == 0:
        return torch.zeros((pred_shape["t"], pred_shape["c"]), device=pred.device)
    try:
        chunks = int(math.ceil(clipped_pred.shape[0] / chunk_size))
        output = torch.cat(
            tuple(
                brisque(
                    t,
                    data_range=data_range,
                    reduction="none",
                )
                for t in torch.chunk(clipped_pred, chunks=chunks, dim=0)
            ),
            dim=0,
        )

    except AssertionError:
        # If we can not compute BRISQUE, assign the worst score
        output = torch.full((clipped_pred.shape[0],), 100.0, device=pred.device)

    output = rearrange(
        output,
        "(b t c) -> b t c",
        c=pred_shape["c"],
        b=pred_shape["b"],
        t=pred_shape["t"],
    )
    return torch.nanmean(output, dim=0)


def per_date_masked_rmse(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    """
    Compute RMSE per band and per date
    """

    assert pred.shape == mask.shape
    assert ref.shape == mask.shape

    mask = torch.logical_and(mask, ~torch.isnan(pred))
    pred[torch.isnan(pred)] = 0.0

    # Compute squared diff
    diff_squared = (pred - ref) ** 2
    # Average over spatial and batch dimensions
    sum_of_squared_error = (diff_squared * mask).sum(dim=(0, -1, -2))

    # Ensure that denom is never null
    if not normalize:
        denom = mask.sum(dim=(0, -1, -2))
    else:
        denom = ((mask * ref) ** 2).sum(dim=(0, -1, -2))

    sum_of_squared_error[denom == 0] = torch.nan

    return torch.sqrt(sum_of_squared_error / denom)


def per_date_masked_sam(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    """
    Compute Spectral Angle Mapper per date
    """
    assert pred.shape == mask.shape
    assert ref.shape == mask.shape
    mask = torch.logical_and(mask, ~torch.isnan(pred))
    pred[torch.isnan(pred)] = 0.0

    # Compute squared diff
    dot_products = torch.matmul(
        rearrange(pred, "b t c w h -> b t w h 1 c"),
        rearrange(ref, "b t c w h -> b t w h c 1"),
    )[:, :, :, :, 0, 0]

    denom = torch.sqrt(((mask * ref) ** 2).sum(dim=(2,)))
    denom *= torch.sqrt(((mask * pred) ** 2).sum(dim=(2,)))

    sam = torch.acos(dot_products / denom)
    sam[~mask[:, :, 0, ...]] = torch.nan
    return torch.nanmean(sam, dim=(0, -1, -2))


def per_date_clear_pixel_rate(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute per date clear pixel rate of reference data
    """
    nb_pixels = mask.shape[0] * mask.shape[-1] * mask.shape[-2]
    return (~mask).sum(dim=(0, -1, -2)) / nb_pixels


def closest_date_in_sits(
    doy: torch.Tensor,
    test_doy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    For each test_doy, compute the distance in days to closest day in doy
    """
    if test_doy is None:
        test_doy = torch.arange(0, 365)
    sits_doy = torch.unique(doy).ravel()
    if doy.shape[0] == 0:
        return torch.empty((test_doy.shape[0],), device=doy.device)
    return torch.min(
        torch.abs(test_doy[None, :].to(device=sits_doy.device) - sits_doy[:, None]),
        dim=0,
    )[0]


def sits_density(
    sits_doy: torch.Tensor, all_doy: torch.Tensor, bandwidth: float = 10.0
) -> torch.Tensor:
    """
    Compute local density with respect to all doy
    """
    if all_doy.shape[0] == 0:
        return torch.zeros_like(sits_doy)
    return torch.exp(-((closest_date_in_sits(all_doy, sits_doy) / bandwidth) ** 2))


def frr_referenceless(
    pred: torch.Tensor,
    ref: torch.Tensor,
    start_idx: int = 1,
    s: int | None = None,
    patch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Compute per band, per date FRR without reference
    """
    if s is None:
        if patch_size is not None:
            s = 2 * (min(pred.shape[-1], patch_size))
        else:
            s = 2 * pred.shape[-1]
    assert pred.shape[0] == 1
    assert ref.shape[0] == 1
    pred[torch.isnan(pred)] = 0.0
    pred = rearrange(pred, "b t c w h -> (b t) c w h")
    ref = rearrange(ref, "b t c w h -> (b t) c w h")
    if pred.numel() == 0:
        return None

    if patch_size is not None and pred.shape[-1] > patch_size:
        pred_patches = patchify(pred[:, None, ...], patch_size=patch_size).flatten(
            0, 1
        )[:, :, 0, ...]
        ref_patches = patchify(ref[:, None, ...], patch_size=patch_size).flatten(0, 1)[
            :, :, 0, ...
        ]
    else:
        pred_patches = pred[None, ...]
        ref_patches = ref[None, ...]

    pred_outputs = [compute_fft_profile(p, s=s) for p in pred_patches]
    freqs = pred_outputs[0][0]
    pred_prof = cast(
        torch.Tensor, sum(p[1] for p in pred_outputs) / pred_patches.shape[0]
    )
    ref_prof = cast(
        torch.Tensor,
        (
            sum(compute_fft_profile(p, s=s)[1] for p in ref_patches)
            / pred_patches.shape[0]
        ),
    )

    pred_logprof = 10 * torch.log10(pred_prof[:, start_idx:, ...]) - 10 * torch.log10(
        pred_prof[:, start_idx : start_idx + 1, ...]
    )
    ref_logprof = 10 * torch.log10(ref_prof[:, start_idx:, ...]) - 10 * torch.log10(
        ref_prof[:, start_idx : start_idx + 1, ...]
    )

    frr = compute_frr_referenceless(pred_logprof, ref_logprof)

    return frr, pred_logprof, ref_logprof, freqs[start_idx:]
