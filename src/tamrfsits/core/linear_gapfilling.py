# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
torch based implementation of linear gapfilling, support gradient back-propagation
(module adapted to MonoModalSITS from mtan_s1s2_classif)
"""
import logging
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import Tensor

from tamrfsits.core.time_series import MonoModalSITS

my_logger = logging.getLogger(__name__)


def flatten_trailing(
    data: torch.Tensor, batch_dim: int = 0, first_trailing_dim: int = 3
) -> tuple[torch.Tensor, torch.Size]:
    """
    Merge all dimensions starting at first_trailing_dim into batch dimension
    """
    # Save initial shape
    initial_shape = data.shape

    # Swap last usefull dim and batch dim
    data = data.transpose(batch_dim, first_trailing_dim - 1)
    # Flatten batch dim (at position first_trailing_dim-1 and all
    # remaining trailing dims
    data = data.flatten(first_trailing_dim - 1, -1)
    # Restore flatten dim as batch dimension
    return data.transpose(first_trailing_dim - 1, batch_dim), initial_shape


def unflatten_trailing(
    data: torch.Tensor, output_shape: torch.Size | tuple[int, ...], batch_dim: int = 0
) -> torch.Tensor:
    """
    Unflatten trailing dims from batch dim
    """
    # Get last dim idx
    last_dim_idx = len(data.shape) - 1

    # Swap last dimension and batch_dim
    data = data.transpose(batch_dim, last_dim_idx)

    # Unflatten last dimension to output_shape
    data = data.unflatten(last_dim_idx, output_shape)

    # Swap back batch_dim and last dim
    return data.transpose(batch_dim, last_dim_idx)


# From https://github.com/pytorch/pytorch/issues/61474
def nanmax(
    tensor: Tensor, dim: int = 0, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """
    torch implementation of nanmax
    """
    min_value = torch.finfo(tensor.dtype).min
    values, idx = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    # This is a bit dangerous but only way to manage cases with all NaNs in some slices
    values[values == min_value] = torch.nan
    return values, idx


# From https://github.com/pytorch/pytorch/issues/61474
def nanmin(
    tensor: Tensor, dim: int = 0, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """
    torch implementation of nanmin
    """
    max_value = torch.finfo(tensor.dtype).max
    values, idx = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    # This is a bit dangerous but only way to manage cases with all NaNs in some slices
    values[values == max_value] = torch.nan
    return values, idx


def _prepare_data(sits: MonoModalSITS) -> tuple[MonoModalSITS, torch.Size]:
    """
    Internal method for linear_gapfilling

    Prepare data:
    - Flatten all trailing dims afther first two (batch and time)
    - Set masked doy to nan

    returns flatten tensors and input shape
    """
    # Keep track of input data shape for later use
    data_flat, input_shape = flatten_trailing(sits.data, first_trailing_dim=3)

    doy_flat, _ = flatten_trailing(
        repeat(sits.doy, "b t -> b t w h", w=input_shape[-1], h=input_shape[-2]),
        first_trailing_dim=2,
    )

    # convert doy to float
    doy_flat = doy_flat.to(data_flat.dtype)

    # Handle optional mask
    mask_flat = None
    if sits.mask is not None:
        mask_flat, _ = flatten_trailing(sits.mask, first_trailing_dim=2)

        # Set masked doys to NaN
        doy_flat[mask_flat] = torch.nan

    return MonoModalSITS(data_flat, doy_flat, mask_flat), input_shape


@dataclass
class InterpolationWeights:
    """
    Placeholder for interpolation weights
    """

    w_a: Tensor
    idx_a: Tensor
    w_b: Tensor
    idx_b: Tensor


def _compute_weights(doy: Tensor, target_doy: Tensor) -> InterpolationWeights:
    """
    Internal method for linear_gapfilling

    Compute weights and indices for linear interpolation.

    returns left_weights, left_indices, righ_weights, right_indices
    """
    # Compute the interpolation matrix from doy, target_doy and masks
    xt_source_doy = doy[:, :, None, None]
    my_logger.debug(target_doy.shape)
    xt_target_doy = rearrange(target_doy, "n -> 1 1 1 n")

    doy_diff = xt_source_doy - xt_target_doy

    # Now doy_diff shape is [B,T,N,F]
    doy_diff = doy_diff.transpose(-2, -1)

    # We look for closest source_doy on both sides of target_doy
    pos_doy_diff = doy_diff.clone()
    pos_doy_diff[pos_doy_diff < 0.0] = torch.nan
    closests_pos_doy, closests_pos_idx = nanmin(pos_doy_diff, dim=1)

    neg_doy_diff = doy_diff.clone()
    neg_doy_diff[neg_doy_diff >= 0.0] = torch.nan
    closests_neg_doy, closests_neg_idx = nanmax(neg_doy_diff, dim=1)

    # Next, we compute interpolation weights from doy values of
    # closest sample before and after target doy
    width = closests_pos_doy - closests_neg_doy
    w_a = torch.abs(closests_pos_doy) / width
    w_b = torch.abs(closests_neg_doy) / width

    # In cases where there are no samples before or after a sample,
    # the weight of the missing sample is set to 0. and the weight of
    # remaining sample is set to 1 (extrapolation by closest value)
    w_a[
        torch.logical_and(~torch.isnan(closests_neg_doy), torch.isnan(closests_pos_doy))
    ] = 1.0
    w_b[
        torch.logical_and(~torch.isnan(closests_neg_doy), torch.isnan(closests_pos_doy))
    ] = 0.0
    w_b[
        torch.logical_and(~torch.isnan(closests_pos_doy), torch.isnan(closests_neg_doy))
    ] = 1.0
    w_a[
        torch.logical_and(~torch.isnan(closests_pos_doy), torch.isnan(closests_neg_doy))
    ] = 0.0

    return InterpolationWeights(w_a, closests_neg_idx, w_b, closests_pos_idx)


def _interpolate(data: Tensor, weights: InterpolationWeights) -> Tensor:
    """
    Internal method for linear_gapfilling

    Performs linear interpolation from weights and indices, using torch.gather
    """
    # We use gather to retrieve the values corresponding to position
    # of samples before and after from the data buffer
    #
    # Here is a simpke example of how it works:
    #
    # In [1]: data = torch.randint(0,10,(2,5))
    #
    # In [2]: data
    # Out[2]:
    # tensor([[1, 2, 1, 2, 4],
    #         [2, 3, 0, 4, 5]])
    #
    # In [3]: min_v, min_idx = torch.min(data,dim=1, keepdim=True)
    #
    # In [4]: min_v
    # Out[4]:
    # tensor([[1],
    #         [0]])
    #
    # In [5]: min_idx
    # Out[5]:
    # tensor([[0],
    #         [2]])
    #
    # In [6]: torch.gather(data,1, min_idx)
    # Out[6]:
    # tensor([[1],
    #         [0]])
    v_a = torch.gather(
        data, 1, repeat(weights.idx_a, "b t 1 -> b t c", c=data.shape[-1])
    )
    v_b = torch.gather(
        data, 1, repeat(weights.idx_b, "b t 1 -> b t c", c=data.shape[-1])
    )

    # Interpolated data is the linear combination using weights and values
    return (
        repeat(weights.w_a, "b t 1 -> b t c", c=data.shape[-1]) * v_a
        + repeat(weights.w_b, "b t 1 -> b t c", c=data.shape[-1]) * v_b
    )


def _prepare_outputs(
    interpolated_data: Tensor,
    target_doy: Tensor,
    input_shape: torch.Size,
) -> MonoModalSITS:
    """
    Internal method for linear_gapfilling

    Unflatten and build output sits object
    """
    # unflatten trailing dims
    interpolated_data = unflatten_trailing(
        interpolated_data, input_shape[0:1] + input_shape[3:]
    )
    out_doy = repeat(target_doy, "t -> b t", b=interpolated_data.shape[0])

    return MonoModalSITS(data=interpolated_data, doy=out_doy, mask=None)


def linear_gapfilling(sits: MonoModalSITS, target_doy: Tensor) -> MonoModalSITS:
    """
    Linear gapfilling function
    data: Input data tensor, shape [B,T, ...]
    mask: Input mask tensor, shape[B,T,...] (same as data).
    (value is True for missing data)
    doy: Input doy tensor, shape [B,T, ...]
    target_doy: Target doy tensor, shape [B,N,...]
    return: Resampled Tensor of shape [B,N,...]
    """
    # Check preconditions
    if not target_doy.is_floating_point():
        target_doy = target_doy.to(dtype=sits.data.dtype)

    # prepare data
    sits_flat, input_data_shape = _prepare_data(sits)
    # Compute interpolation weights
    weights = _compute_weights(sits_flat.doy, target_doy)
    # Apply interpolation weights and restore trailing dimensions
    interpolated_data = _interpolate(sits_flat.data, weights)

    assert interpolated_data.shape[-1] == sits.data.shape[2]

    # Final unflatten to initial trailing dims
    return _prepare_outputs(interpolated_data, target_doy, input_data_shape)
