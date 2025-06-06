#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Data structure to store SITS tensors
"""

from dataclasses import dataclass
from typing import cast

import torch
from einops import pack, parse_shape, rearrange, repeat
from torch import all as torch_all  # pylint: disable=no-name-in-module
from torch import eq as torch_eq  # pylint: disable=no-name-in-module
from typing_extensions import Self


def constant_pad_end(
    data: torch.Tensor, dim: int, length: int, fill_value: float | int | bool
) -> torch.Tensor | None:
    """
    Pad given dim at the end
    If data dim is larger than length, returns None
    """

    if data.shape[dim] < length:
        padding_length = length - data.shape[dim]
        padding_shape = list(data.shape)
        padding_shape[dim] = padding_length
        padd_data = torch.full(
            padding_shape, fill_value, dtype=data.dtype, device=data.device
        )
        return torch.cat((data, padd_data), dim=dim)
    if data.shape[dim] == length:
        return data
    return None


class SITS:
    """
    Base class for SITS data
    """

    def __init__(
        self, data: torch.Tensor, doy: torch.Tensor, mask: None | torch.Tensor = None
    ):
        """
        data: measurements
        doy: acquisition times
        mask : Equal to 1 if measurement is missing and 0 otherwise, element-wise
        """
        # Ensure at least one feature dimension
        assert data.dim() > 2
        self.data = data

        if mask is not None:
            # mask has at least 2 dimension
            assert mask.dim() >= 2
            # Two first dimensions of data and mask are the same
            assert mask.shape[:2] == data.shape[:2]
            # Mask should be boolean
            assert mask.dtype == torch.bool
        self.mask = mask

        # doy has at least 2 dims
        assert doy.dim() >= 2
        # First dimensions of doy and data should match
        assert doy.shape[:2] == data.shape[:2]

        # Ensure that all tensors are on the same device
        assert doy.device == data.device
        assert mask is None or mask.device == data.device

        self.doy = doy

    def is_masked(self) -> bool:
        """
        True if SITS contain a mask
        """
        return self.mask is not None

    def shape(self) -> torch.Size:
        """
        Get internal shape
        """
        return self.data.shape

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Self:
        """
        Move sits to another device
        """
        self.data = self.data.to(device=device, dtype=dtype)

        if self.mask is not None:
            # We do not allow to cast mask dtype,
            # since it should always be of type torch.bool
            self.mask = self.mask.to(device=device)
        self.doy = self.doy.to(device=device, dtype=dtype)

        return self

    def pin_memory(self) -> Self:
        """
        Pin memory
        """
        self.data = self.data.pin_memory()
        self.doy = self.doy.pin_memory()
        if self.mask is not None:
            self.mask = self.mask.pin_memory()

        return self


@dataclass
class TargetDOYStrategy:
    """Params to generate target DOYs"""

    first_day: int
    last_day: int
    step: int


class MonoModalSITS(SITS):
    """
    Class representing a monomodal SITS
    """

    def __init__(
        self, data: torch.Tensor, doy: torch.Tensor, mask: None | torch.Tensor = None
    ):
        """
        data shape : b t c w h
        mask shape : b t w h
        doy shape : b t
        """
        # Verify MonoModalSITS constraints
        if mask is not None:
            assert mask.dim() == data.dim() - 1
            assert mask.shape[2:] == data.shape[3:]
        assert doy.dim() == 2
        super().__init__(data, doy, mask)

    def trim(self, fill_value: float | int | bool = 0.0) -> Self:
        """
        Trim masked doy for each element in the batch to reduce the doy
        dimension to the maximum number of valid values over the batch.

        Inplace operation
        """
        if self.mask is None:
            return self

        # parse data shape
        data_shape = parse_shape(self.data, "b t c h w")

        # Find which dates are completely masked for some samples in batch dim
        doy_masked = torch_all(rearrange(self.mask, "b t h w -> b t (h w)"), dim=-1)
        # move those dates at the end of doy dim

        _, ind = torch.sort(doy_masked, dim=-1, descending=False)

        # Derive max length from the max number of valid dates
        max_length = int(torch.max(torch.sum(~doy_masked, dim=-1)).item())
        assert max_length > 0, "SITS is fully masked"

        # Rearange data to move invalid dates at the end of doy dim
        trim_data = torch.full_like(self.data, fill_value)
        trim_data.scatter_(
            1,
            repeat(
                ind,
                "b t -> b t c w h",
                c=data_shape["c"],
                w=data_shape["w"],
                h=data_shape["h"],
            ),
            self.data,
        )
        # Trim data tensor
        self.data = trim_data[:, :max_length, ...]

        # Rearange data to move invalid dates at the end of doy dim
        trim_mask = torch.full_like(self.mask, True)
        trim_mask.scatter_(
            1,
            repeat(ind, "b t -> b t w h", w=data_shape["w"], h=data_shape["h"]),
            self.mask,
        )
        # Trim mask tensor
        self.mask = trim_mask[:, :max_length, ...]

        # Rearange doy to move invalid dates at the end of doy dim
        trim_doy = torch.full_like(self.doy, fill_value)
        trim_doy.scatter_(1, ind, self.doy)
        # Trim doy tensor

        self.doy = trim_doy[:, :max_length, ...]

        return self


def detach(sits: MonoModalSITS) -> MonoModalSITS:
    """
    Detach contained tensors
    """
    return MonoModalSITS(
        sits.data.detach(),
        sits.doy.detach(),
        sits.mask.detach() if sits.mask is not None else None,
    )


def crop_sits(sits: MonoModalSITS, margin: int) -> MonoModalSITS:
    """
    Spatially crop sits according to margin
    """
    if 2 * margin + 1 >= sits.shape()[-1] or 2 * margin + 1 >= sits.shape()[-2]:
        raise ValueError(margin)
    return MonoModalSITS(
        sits.data[:, :, :, margin:-margin, margin:-margin],
        sits.doy,
        None if sits.mask is None else sits.mask[:, :, margin:-margin, margin:-margin],
    )


def pad_acquisition_time(
    sits: MonoModalSITS, doy_length: int, fill_value: float = 0.0
) -> MonoModalSITS | None:
    """
    Increase the size of the time_dimension to doy_length for
    all tensors in the SITS

    If the length of the time dimension is greater than doy_length, returns None
    """
    padded_data = constant_pad_end(sits.data, 1, doy_length, fill_value)
    padded_doy = constant_pad_end(sits.doy, 1, doy_length, fill_value)
    if padded_data is None or padded_doy is None:
        return None

    padded_mask: torch.Tensor | None = None
    if sits.mask is not None:
        padded_mask = constant_pad_end(sits.mask, 1, doy_length, True)
        if padded_mask is None:
            return None

    return MonoModalSITS(padded_data, padded_doy, padded_mask)


def cat_monomodal_sits(sits: list[MonoModalSITS]) -> MonoModalSITS:
    """
    Concatenate monodal sits along the channel dimension
    All doy and mask tensor must match exactly
    """
    # Preconditions
    # at least one element
    assert sits

    # Early exit
    if len(sits) == 1:
        return sits[0]

    # All shape matches except in channel dimension
    # Either all sits are masked or none of them are
    data_shape = parse_shape(sits[0].data, "b t c w h")

    mask = sits[0].mask
    doy = sits[0].doy

    if mask is None:
        for s in sits:
            assert s.mask is None
    else:
        for s in sits:
            assert s.mask is not None
            assert torch_all(
                torch_eq(doy, s.doy)
            ), "Doy should be the same for all input sits"

    for s in sits:
        current_data_shape = parse_shape(s.data, "b t c w h")
        for key in ("b", "t", "w", "h"):
            assert (
                current_data_shape[key] == data_shape[key]
            ), f"Data shape mismatch for dimension {key} ({current_data_shape[key]} !=\
            {data_shape[key]}"
    if mask is not None:
        mask = cast(torch.Tensor, sum(s.mask for s in sits)) > 0

    return MonoModalSITS(pack([s.data for s in sits], "b t * h w")[0], doy, mask)


def subset_doy_monomodal_sits(
    sits: MonoModalSITS,
    target_doy: torch.Tensor | None = None,
    fill_value: float | bool | int = 0,
) -> MonoModalSITS:
    """
    This function uses target_doy as the new set of doy for every batch in sits
    If acquisition is not available in sits, it is masked and filled with fill_value

    if target_doy is None, the union set of all doy in sits is used.
    """
    # variable to hold non-optional target_doy
    nopt_target_doy: torch.Tensor

    if target_doy is None:
        # find intersection of all doy
        nopt_target_doy = torch.unique(sits.doy)
    else:
        nopt_target_doy = target_doy
    if nopt_target_doy.dim() == 1:
        nopt_target_doy = repeat(nopt_target_doy, "t -> b t", b=sits.shape()[0])
    assert nopt_target_doy.dim() == 2
    assert nopt_target_doy.shape[0] == sits.doy.shape[0]

    # Store shape of input data
    data_shape = parse_shape(sits.data, "b t c w h")

    # Build ouptut data shape (with new size for dimension t, and all
    # other dimensions collapsed)
    out_data = torch.full(
        (
            data_shape["c"] * data_shape["w"] * data_shape["h"],
            data_shape["b"],
            nopt_target_doy.shape[1],
        ),
        fill_value,
        dtype=sits.data.dtype,
        device=sits.data.device,
    )

    # Also build output mask shape
    out_mask = torch.full(
        (
            data_shape["w"] * data_shape["h"],
            data_shape["b"],
            nopt_target_doy.shape[1],
        ),
        True,
        dtype=torch.bool,
        device=sits.data.device,
    )

    # Rearrange input doy into (b c w h) t 1 in preparation for torch.where call
    in_doy = repeat(sits.doy, "b t -> b t 1 1")

    # Rearrange target_doy into (1 1 t) in preparation for torch.where call
    nopt_target_doy = rearrange(nopt_target_doy, "b t -> 1 1 b t")
    # Use torch.where to find matching indices in all 3 dimensions
    cross_matrix = in_doy == nopt_target_doy

    # Forbid cross-batch attention
    for b1 in range(in_doy.shape[0]):
        for b2 in range(nopt_target_doy.shape[2]):
            if b1 != b2:
                cross_matrix[b1, :, b2, :] = False

    idx = torch.where(cross_matrix)
    # Copy sits.data to the correct location in out_data

    out_data[:, idx[2], idx[3]] = rearrange(sits.data, "b t c w h -> (c w h) b t")[
        :, idx[0], idx[1]
    ]
    # Rearrange out_data into final shape
    out_data = rearrange(
        out_data,
        "(c w h) b t -> b t c w h",
        b=data_shape["b"],
        c=data_shape["c"],
        w=data_shape["w"],
        h=data_shape["h"],
    )

    # Now rearrange in_doy the same way to process mask (no c dimension)
    del data_shape["c"]  # We do not need c anymore

    # Either we fill with available mask
    if sits.mask is not None:
        out_mask[:, idx[2], idx[3]] = rearrange(sits.mask, "b t h w -> (h w) b t")[
            :, idx[0], idx[1]
        ]
    else:
        # Or we fill with unmasked value if no mask is available in input series
        out_mask[:, idx[2], idx[3]] = False

    # Rearrange mask to final shape
    out_mask = rearrange(
        out_mask,
        "(w h) b t -> b t w h",
        b=data_shape["b"],
        w=data_shape["w"],
        h=data_shape["h"],
    )
    nopt_target_doy = rearrange(nopt_target_doy, "1 1 b t -> b t")

    return MonoModalSITS(out_data, nopt_target_doy, out_mask)


def sits_where(sits1: MonoModalSITS, sits2: MonoModalSITS) -> MonoModalSITS:
    """
    sits1 or sits2 where sits1 is masked and not sits2
    """
    # If there is no mask in sits1, return sits1 itself
    if sits1.mask is None:
        return sits1

    assert sits1.data.shape == sits2.data.shape

    # Build where mask
    where_mask = sits1.mask

    # If there is a mask in sits2, use it
    if sits2.mask is not None:
        where_mask = torch.logical_and(where_mask, ~sits2.mask)

    out_mask = None
    if sits2.mask is not None:
        out_mask = torch.logical_and(sits1.mask, sits2.mask)

    where_mask = repeat(where_mask, "b t w h -> b t c w h", c=sits1.data.shape[2])

    out_data = torch.where(where_mask, sits2.data, sits1.data)

    return MonoModalSITS(out_data, sits1.doy, out_mask)
