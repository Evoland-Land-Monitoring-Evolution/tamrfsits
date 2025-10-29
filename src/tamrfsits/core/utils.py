# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains generic utility functions
"""

from typing import Any, Protocol

import numpy as np
import torch
from einops import rearrange, repeat

from tamrfsits.core.time_series import MonoModalSITS, subset_doy_monomodal_sits


# pylint: disable=too-few-public-methods
class CompiledTorchModule(Protocol):
    """
    Useful for anotating type of compiled model
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


def strip_masks(sits: MonoModalSITS):
    """
    Remove mask from sits
    """
    return MonoModalSITS(sits.data, sits.doy)


def standardize_sits(
    sits: MonoModalSITS,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale: float | None = None,
    clip: bool = False,
) -> MonoModalSITS:
    """
    Standardization function
    """
    data = sits.data
    if scale is not None:
        data = sits.data / scale

    if clip:
        data = torch.clip(data, -0.2, 1.0)
    data = (data - mean[None, None, :, None, None]) / std[None, None, :, None, None]

    return MonoModalSITS(data, sits.doy, sits.mask)


def unstandardize_sits(
    sits: MonoModalSITS,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale: float | None = None,
) -> MonoModalSITS:
    """
    Standardization function
    """
    data = sits.data

    data = (data * std[None, None, :, None, None]) + mean[None, None, :, None, None]

    if scale is not None:
        data = data * scale

    return MonoModalSITS(data, sits.doy, sits.mask)


def strip_masked_patches(sits: MonoModalSITS) -> MonoModalSITS:
    """
    Remove patches that are masked
    """
    if sits.mask is None:
        return sits
    assert sits.shape()[0] == 1, "Implementation only works for b=1"
    valid_doys = (~sits.mask).sum(dim=(0, -1, -2)) > 0

    return MonoModalSITS(
        sits.data[:, valid_doys, :, :, :],
        sits.doy[:, valid_doys],
        sits.mask[:, valid_doys, :, :],
    )


def mask_sits_by_doy_mask(sits: MonoModalSITS, doy_mask: torch.Tensor) -> MonoModalSITS:
    """
    Mask the sits according to doy to be masked
    """
    assert doy_mask.shape == sits.doy.shape

    out_mask: torch.Tensor

    if sits.mask is not None:
        out_mask = sits.mask.clone()
    else:
        out_mask = torch.zeros_like(sits.data[:, 0, ...])

    doy_mask = repeat(
        doy_mask, "b t -> b t w h", w=out_mask.shape[-2], h=out_mask.shape[-1]
    )

    out_mask[doy_mask] = True

    return MonoModalSITS(sits.data, sits.doy, out_mask)


def doy_unique(
    doy1: torch.Tensor, doy2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive unique doy vector for each element in batch
    """
    assert doy1.shape[0] == doy2.shape[0]

    unique_doys = [
        torch.unique(
            torch.cat([doy1[b : b + 1, :].flatten(), doy2[b : b + 1, :].flatten()])
        )
        for b in range(doy1.shape[0])
    ]
    max_doy_length = max(doy.numel() for doy in unique_doys)
    unique_doys = [torch.sort(doy)[0] for doy in unique_doys]
    out_doys = torch.stack(
        [
            torch.nn.functional.pad(
                doy, (0, max_doy_length - doy.numel()), mode="constant", value=0
            )
            for doy in unique_doys
        ]
    )
    out_mask = torch.stack(
        [
            torch.nn.functional.pad(
                torch.full_like(doy, True, dtype=torch.bool),
                (0, max_doy_length - doy.numel()),
                mode="constant",
                value=False,
            )
            for doy in unique_doys
        ]
    )
    return out_doys, out_mask


def elementwise_subset_doy_monomodal_sits(
    sits: MonoModalSITS,
    target_doy: torch.Tensor,
    fill_value: float | bool | int = 0,
):
    """
    Elementwise version of subset_doy_monomodal_sits
    """
    if target_doy.dim() == 1:
        target_doy = repeat(target_doy, "t -> b t", b=sits.data.shape[0])
    assert target_doy.shape[0] == sits.data.shape[0]

    output_sits = [
        subset_doy_monomodal_sits(
            MonoModalSITS(
                sits.data[b : b + 1, ...],
                sits.doy[b : b + 1, ...],
                sits.mask[b : b + 1, ...] if sits.mask is not None else None,
            ),
            target_doy=target_doy[b, ...],
            fill_value=fill_value,
        )
        for b in range(target_doy.shape[0])
    ]

    return MonoModalSITS(
        torch.cat([s.data for s in output_sits], dim=0),
        torch.cat([s.doy for s in output_sits], dim=0),
        (
            torch.cat([s.mask for s in output_sits if s.mask is not None], dim=0)
            if sits.mask is not None
            else None
        ),
    )


def combine_sits(
    sits1: MonoModalSITS,
    sits2: MonoModalSITS,
    fill_value: float = 0.0,
) -> MonoModalSITS:
    """
    Combine sits together, using sits1 if available else sits2 else the
    sum, on the union of doy per batch
    """

    # Get unique doys for each batch
    unique_doys_per_batch, unique_doys_per_batch_mask = doy_unique(sits1.doy, sits2.doy)

    # Subset each sits on the set of unique doys per batch
    sits1_target_doy = subset_doy_monomodal_sits(
        sits1,
        target_doy=unique_doys_per_batch,
        fill_value=fill_value,
    )
    sits2_target_doy = subset_doy_monomodal_sits(
        sits2,
        target_doy=unique_doys_per_batch,
        fill_value=fill_value,
    )
    assert sits1_target_doy.mask is not None
    assert sits2_target_doy.mask is not None

    # Here we mask padded doy for each batch
    output_mask1 = repeat(
        unique_doys_per_batch_mask,
        "b t -> b t c w h",
        c=sits1_target_doy.data.shape[2],
        w=sits1_target_doy.data.shape[3],
        h=sits1_target_doy.data.shape[4],
    )
    sits1_target_doy.data[~output_mask1] = fill_value
    output_mask1 = repeat(
        unique_doys_per_batch_mask,
        "b t -> b t w h",
        w=sits1_target_doy.data.shape[3],
        h=sits1_target_doy.data.shape[4],
    )
    sits1_target_doy.mask[~output_mask1] = True

    output_mask2 = repeat(
        unique_doys_per_batch_mask,
        "b t -> b t c w h",
        c=sits2_target_doy.data.shape[2],
        w=sits2_target_doy.data.shape[3],
        h=sits2_target_doy.data.shape[4],
    )
    sits2_target_doy.data[~output_mask2] = fill_value
    output_mask2 = repeat(
        unique_doys_per_batch_mask,
        "b t -> b t w h",
        w=sits2_target_doy.data.shape[3],
        h=sits2_target_doy.data.shape[4],
    )
    sits2_target_doy.mask[~output_mask2] = True

    # Now, we will combine both sits by averaging if both sits are valid for a
    # given doy
    mask_sum = (~(sits2_target_doy.mask)).to(dtype=torch.int8) + (
        ~(sits1_target_doy.mask)
    ).to(dtype=torch.int8)

    # Ensure no division by zero
    mask_sum_no_zeros = torch.where(mask_sum > 0, mask_sum, 1)

    sits1_weight = (~(sits1_target_doy.mask)) / mask_sum_no_zeros
    sits2_weight = (~(sits2_target_doy.mask)) / mask_sum_no_zeros

    sits1_weight = repeat(
        sits1_weight, "b t w h -> b t c w h", c=sits1_target_doy.shape()[2]
    )
    sits2_weight = repeat(
        sits2_weight, "b t w h -> b t c w h", c=sits2_target_doy.shape()[2]
    )

    output_data = (
        sits1_weight * sits1_target_doy.data + sits2_weight * sits2_target_doy.data
    )
    output_mask = mask_sum == 0

    return MonoModalSITS(output_data, sits1_target_doy.doy, mask=output_mask)


def split_sits_features(
    sits: MonoModalSITS, split_point: int
) -> tuple[MonoModalSITS, MonoModalSITS]:
    """
    Split sits into two sits
    """
    return MonoModalSITS(
        sits.data[:, :, :split_point, ...], sits.doy, sits.mask
    ), MonoModalSITS(sits.data[:, :, split_point:, ...], sits.doy, sits.mask)


def common_doys(doy1: torch.Tensor, doy2: torch.Tensor) -> torch.Tensor:
    """
    Return a 1d tensor of common doys between sits1 and sits2
    """

    doy2_mask = torch.isin(doy2, doy1)
    return torch.unique(doy2[doy2_mask])


def uncommon_doys(doy1: torch.Tensor, doy2: torch.Tensor) -> torch.Tensor:
    """
    Return a 1d tensor of uncommon doys between sits1 and sits2
    """

    doy2_mask = ~torch.isin(doy2, doy1)
    return torch.unique(doy2[doy2_mask])


def add_ndvi_to_sits(
    sits: MonoModalSITS,
    red_band_idx: int = 2,
    nir_band_idx: int = 3,
    red_band_mean_std: tuple[float, float] = (0.0, 1.0),
    nir_band_mean_std: tuple[float, float] = (0.0, 1.0),
    epsilon: float = 1e-6,
):
    """
    Append NDVI to sits channels
    """
    data = sits.data
    assert data.shape[2] > red_band_idx
    assert data.shape[2] > nir_band_idx

    unscaled_red = (
        red_band_mean_std[0] + red_band_mean_std[1] * data[:, :, red_band_idx, ...]
    )

    unscaled_nir = (
        nir_band_mean_std[0] + nir_band_mean_std[1] * data[:, :, nir_band_idx, ...]
    )

    ndvi = (unscaled_nir - unscaled_red) / (unscaled_red + unscaled_nir + epsilon)

    data = torch.cat((data, ndvi[:, :, None, ...]), dim=2)

    return MonoModalSITS(data, sits.doy, sits.mask)


def patchify(
    data: torch.Tensor,
    patch_size: int = 32,
    margin: int = 0,
    padding_mode: str = "constant",
    padding_value: int = 0,
    spatial_dim1: int = -1,
    spatial_dim2: int = -2,
) -> torch.Tensor:
    """
    Create a patch view on an image Tensor
    data: Tensor of shape [C,W,H] (C: number of channels,
                                   W: image width, H: image height)
    :param patch_size: Size of the square patch
    :param margin: Overlap of patches on each side
    :param padding_mode: Mode for padding on left/bottom end
    :param padding_value: Value of padding if padding_mode is 'constant'
    return: Tensor of shape [PX, PY, C, PW, PH]
            (PX: patch x idx, PY: patch y idx, C: number of channels,
             PW: patch width, PH: patch height) Lastes patch might be padded
    """
    # First, move spatial dims to the end
    data = data.transpose(spatial_dim1, -2).transpose(spatial_dim2, -1)

    padding_left = margin
    padding_right = (
        margin + int(np.ceil(data.shape[-2] / patch_size) * patch_size) - data.shape[-2]
    )
    padding_top = margin
    padding_bottom = (
        margin + int(np.ceil(data.shape[-1] / patch_size) * patch_size) - data.shape[-1]
    )

    out = (
        torch.nn.functional.pad(
            data[None, ...],
            [padding_top, padding_bottom, padding_left, padding_right],
            mode=padding_mode,
            value=padding_value,
        )[0, ...]
        .unfold(-2, patch_size + 2 * margin, patch_size)
        .unfold(-2, patch_size + 2 * margin, patch_size)
    )

    return rearrange(out, "b t c n m w h -> n m b t c w h")


def find_closest_in_sits(
    source_sits: MonoModalSITS,
    target_doy: torch.Tensor,
    max_masked_rate: float = 0.5,
    allow_same_day: bool = True,
) -> MonoModalSITS:
    """
    Look-up closest element in source_sits from target_doy
    """
    # Make predictions based on 2 most recent clear s2 dates for each landsat date
    doy_diff_matrix = torch.abs(target_doy[0, None, :] - source_sits.doy[0, :, None])

    above_max_doy_value = 10000

    if source_sits.mask is not None:
        mask_rate = source_sits.mask.sum(dim=(-1, -2)) / (
            source_sits.mask.shape[-1] * source_sits.mask.shape[-2]
        )
        # print(hr_mask_rate.shape, hr_mask_rate)
        doy_diff_matrix[mask_rate[0, ...] > max_masked_rate, :] = above_max_doy_value
    if not allow_same_day:
        doy_diff_matrix[doy_diff_matrix == 0] = above_max_doy_value

    # Get first min
    _, first_min_idx = torch.min(doy_diff_matrix, dim=0)
    out_doys = source_sits.doy.gather(1, first_min_idx[None, ...])

    first_min_idx = repeat(
        first_min_idx,
        "t -> b t w h",
        b=source_sits.data.shape[0],
        w=source_sits.data.shape[3],
        h=source_sits.data.shape[4],
    )
    out_mask: torch.Tensor | None = None

    if source_sits.mask is not None:
        out_mask = source_sits.mask.gather(1, first_min_idx)

    first_min_idx = repeat(
        first_min_idx,
        "b t w h -> b t c w h",
        c=source_sits.data.shape[2],
    )

    out_data = source_sits.data.gather(1, first_min_idx)
    return MonoModalSITS(out_data, out_doys, out_mask)
