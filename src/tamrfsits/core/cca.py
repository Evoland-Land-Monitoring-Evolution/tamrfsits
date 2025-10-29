# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the DatewiseCCALoss
"""


from enum import Enum

import torch
from einops import parse_shape, rearrange, repeat
from piq import LPIPS  # type: ignore
from torch._C import _LinAlgError  # type: ignore

from tamrfsits.core.downsampling import (
    generate_psf_kernel,
    sits_gradient_magnitude,
    sits_high_pass_filtering,
    tensor_gradient_magnitude,
    tensor_high_pass_filtering,
)
from tamrfsits.core.lpips import MaskedLPIPSLoss
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.core.utils import CompiledTorchModule

MINIMUM_NUMBER_OF_SAMPLES_FOR_REGRESSION = 10


def batched_weighted_product(
    data1: torch.Tensor, data2: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute batched weighted product A^T B
    """
    assert data1.dim() == data2.dim() == 3
    assert weights.dim() == 2
    assert weights.shape == data2.shape[:2]
    assert data1.shape[0] == data2.shape[0]
    assert data1.shape[2] == data2.shape[1]

    weights1 = repeat(weights, "b m -> b n m", n=data1.shape[1])
    weights2 = repeat(weights, "b m -> b m n", n=data2.shape[2])

    return torch.matmul(data1 * weights1, data2 * weights2)


def batch_covariance(data: torch.Tensor, weights: torch.Tensor):
    """
    Compute batched covariance of data

    solution from: https://stackoverflow.com/
    questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
    """
    assert data.dim() == 3
    assert weights.dim() == 2
    assert weights.shape == data.shape[:2]
    data_shape = parse_shape(data, "b n c")
    mean = torch.mean(data, 1, keepdim=True)
    diffs = data - mean
    diffs = rearrange(diffs, "b n c -> (b n) c")
    prods = torch.bmm(repeat(diffs, "b c -> b c 1"), repeat(diffs, "b c -> b 1 c"))
    prods = rearrange(prods, "(b n) c d -> b n c d", **data_shape, d=data_shape["c"])
    weights = repeat(weights, "b n -> b n 1 1")
    prods *= weights
    bcov = prods.sum(dim=1) / (weights.sum(dim=1) - 1)  # Unbiased estimate
    return bcov  # (B, C, C)


def masked_mean_std(
    data: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Masked standard deviation and mean
    """
    data_mean = torch.sum(data * mask, dim=1) / torch.sum(mask, dim=1)
    data_mean_of_squares = torch.sum((data * mask) ** 2, dim=1) / (
        torch.sum(mask, dim=1) - 1
    )
    data_std = torch.sqrt(data_mean_of_squares - data_mean**2)
    return data_mean, data_std


def vectorized_linear_regression_stable(
    src: torch.Tensor,
    tgt: torch.Tensor,
    msk: torch.Tensor,
    regularisation_weight: float | None = 1e-3,
    precision: torch.dtype | None = torch.float64,
    bias: bool = False,
) -> torch.Tensor:
    """
    Vectorized version of linear regression, more stable version
    """
    assert msk.shape == tgt.shape[:2]
    assert msk.shape == src.shape[:2]
    assert src.shape[:2] == tgt.shape[:2]
    assert src.shape[1] > 0
    # assert torch.all(torch.sum(msk, dim=1) > 2)
    # Store shapes for later
    src_shape = parse_shape(src, "b n c")
    tgt_shape = parse_shape(tgt, "b n c")

    # extended mask
    src_msk = repeat(msk, "b n -> b n c", c=src_shape["c"])
    tgt_msk = repeat(msk, "b n -> b n c", c=tgt_shape["c"])
    # Compute mean and std of src and tgt
    src_mean, src_std = masked_mean_std(src, src_msk)
    tgt_mean, tgt_std = masked_mean_std(tgt, tgt_msk)

    # assert torch.all(~torch.isnan(src_mean))
    # assert torch.all(~torch.isnan(tgt_mean))
    # assert torch.all(~torch.isnan(src_std))
    # assert torch.all(~torch.isnan(tgt_std))

    # Standardize src and tgt
    src_bar = (src - repeat(src_mean, "b c -> b n c", n=src_shape["n"])) / repeat(
        src_std, "b c -> b n c", n=src_shape["n"]
    )

    # Use bias if required
    if bias:
        src_bar = torch.cat((src_bar, torch.ones_like(src_bar[:, :, 0:1])), dim=-1)

    tgt_bar = (tgt - repeat(tgt_mean, "b c -> b n c", n=tgt_shape["n"])) / repeat(
        tgt_std, "b c -> b n c", n=tgt_shape["n"]
    )

    # assert torch.all(~torch.isnan(src_bar))
    # assert torch.all(~torch.isnan(tgt_bar))

    src_cov = batched_weighted_product(src_bar.transpose(-1, -2), src_bar, msk)
    srct_tgt = batched_weighted_product(src_bar.transpose(-1, -2), tgt_bar, msk)

    if regularisation_weight is not None:
        # Ridge regression
        src_cov[:, ...] += regularisation_weight * torch.eye(
            src_cov.shape[-1], device=src_cov.device
        )

    if precision is not None:
        src_cov = src_cov.to(dtype=precision)
        srct_tgt = srct_tgt.to(dtype=precision)

    # pylint: disable=not-callable
    cov_rank = torch.linalg.matrix_rank(src_cov, hermitian=True)
    cov_not_full_rank_mask = cov_rank < src_cov.shape[-1]

    lstsq_output = torch.linalg.lstsq(src_cov, srct_tgt)  # pylint: disable=not-callable
    coefs = lstsq_output.solution.detach().to(src.dtype)

    output = torch.matmul(src_bar, coefs)
    output *= repeat(tgt_std, "b c -> b n c", n=src_shape["n"])
    output += repeat(tgt_mean, "b c -> b n c", n=src_shape["n"])

    # Batch for which cov is not full rank are explicitely set to NaN
    output[cov_not_full_rank_mask, ...] = torch.nan

    return output


class HighPassFilteringMode(Enum):
    """
    Enum representing the high past filtering mode
    """

    NO = "NO"
    PRE = "PRE"
    POST = "POST"


class DatewiseLinearRegressionLoss(torch.nn.Module):
    """
    Implement multiscale datewise linear cca
    """

    def __init__(
        self,
        loss: torch.nn.Module,
        max_masked_rate: float = 0.5,
        loss_security_threshold: float = 10.0,
        mtf_for_high_pass_filtering: float | None = None,
    ):
        """
        Initializer
        """
        super().__init__()

        self.lpips: CompiledTorchModule = (
            MaskedLPIPSLoss(LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0]))
            .requires_grad_(False)
            .eval()
        )
        self.lpips.compile()

        self.loss = loss
        self.max_masked_rate = max_masked_rate
        self.loss_security_threshold = loss_security_threshold
        self.hpf_kernel: torch.Tensor | None = None
        # If mtf is not None, generate the PSF kernel
        if mtf_for_high_pass_filtering is not None:
            self.hpf_kernel = generate_psf_kernel(
                1.0,
                mtf_res=torch.ones((1,)),
                mtf_fc=torch.full((1,), mtf_for_high_pass_filtering),
            )

    def _build_inputs(
        self,
        pred_sits: MonoModalSITS,
        target_sits: MonoModalSITS,
        high_pass_filtering: bool = True,
    ) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Derive inputs for the loss
        """
        # (optional) Step 1: perform gradient magnitude on both sits
        if high_pass_filtering:
            if self.hpf_kernel is not None:
                # Ensure that we have correct device and precision
                self.hpf_kernel = self.hpf_kernel.to(
                    device=pred_sits.data.device, dtype=pred_sits.data.dtype
                )
                pred_sits_for_regression = sits_high_pass_filtering(
                    pred_sits,
                    repeat(self.hpf_kernel, "1 w h-> c w h", c=pred_sits.data.shape[2]),
                )
                target_sits_for_regression = sits_high_pass_filtering(
                    target_sits,
                    repeat(
                        self.hpf_kernel, "1 w h -> c w h", c=target_sits.data.shape[2]
                    ),
                )

            else:
                pred_sits_for_regression = sits_gradient_magnitude(pred_sits)
                target_sits_for_regression = sits_gradient_magnitude(target_sits)
        else:
            pred_sits_for_regression = pred_sits
            target_sits_for_regression = target_sits

        return pred_sits_for_regression, target_sits_for_regression

    def _build_validity_mask(self, target_sits: MonoModalSITS) -> torch.Tensor:
        """
        Derive the validity mask
        """
        if target_sits.mask is None:
            loss_mask = torch.full(
                (
                    target_sits.shape()[0],
                    target_sits.shape()[1],
                    target_sits.shape()[3],
                    target_sits.shape()[4],
                ),
                True,
                device=target_sits.data.device,
            )
        else:
            loss_mask = ~target_sits.mask

        return loss_mask

    def _perform_linear_regression(
        self,
        pred_data: torch.Tensor,
        target_data: torch.Tensor,
        loss_mask: torch.Tensor,
        per_date: bool = False,
    ) -> torch.Tensor | None:
        """
        Perform the linear regression step
        """
        # Store pred_shape for future use
        pred_shape = parse_shape(pred_data, "b t c w h")

        if per_date:
            # Now rearrange data and mask
            pred_data = rearrange(pred_data, "b t c w h -> (b t) (w h) c")
            target_data = rearrange(target_data, "b t c w h -> (b t) (w h) c")
            loss_mask = rearrange(loss_mask, "b t w h -> (b t) (w h)")

        else:
            # Now, rearrange again to collapse spatial dimensions
            pred_data = rearrange(pred_data, "b t c w h -> b (t w h) c")
            target_data = rearrange(target_data, "b t c w h -> b (t w h) c")
            loss_mask = rearrange(loss_mask, "b t w h -> b (t w h)")

        # Ensure we do not have an empty tensor here
        if loss_mask.shape[0] > 0 and torch.any(
            torch.sum(loss_mask, dim=-1)
            > MINIMUM_NUMBER_OF_SAMPLES_FOR_REGRESSION
            * pred_data.shape[-1]
            * target_data.shape[-1]  # At least 10 samples per variable
        ):
            # Perform linear regression
            try:
                linear_reg_output_data = vectorized_linear_regression_stable(
                    target_data, pred_data, loss_mask, bias=True
                )

            except _LinAlgError:
                return None

            if per_date:
                return rearrange(
                    linear_reg_output_data, "(b t) (w h) c -> b t c w h", **pred_shape
                )
            return rearrange(
                linear_reg_output_data, "b (t w h) c -> b t c w h", **pred_shape
            )
        return None

    def _post_high_pass_filtering(
        self, pred_data: torch.Tensor, linear_reg_output_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optionnal post HPF
        """
        pred_shape = parse_shape(pred_data, "b t c w h")

        # Optionally recombine images and apply hpf
        linear_reg_output_data = rearrange(
            linear_reg_output_data,
            "b t c w h  -> (b t) c w h",
        )
        pred_data = rearrange(
            pred_data,
            "b t c w h -> (b t) c w h",
        )
        if self.hpf_kernel is not None:
            # Ensure that we have correct device and precision
            self.hpf_kernel = self.hpf_kernel.to(
                device=pred_data.device, dtype=pred_data.dtype
            )
            linear_reg_output_data = tensor_high_pass_filtering(
                linear_reg_output_data,
                repeat(self.hpf_kernel, "1 w h-> c w h", c=pred_data.shape[1]),
            )
            pred_data = tensor_high_pass_filtering(
                pred_data,
                repeat(self.hpf_kernel, "1 w h-> c w h", c=pred_data.shape[1]),
            )
        else:
            linear_reg_output_data = tensor_gradient_magnitude(linear_reg_output_data)
            pred_data = tensor_gradient_magnitude(pred_data)

        return rearrange(
            pred_data, "(b t) c w h -> b t c w h", **pred_shape
        ), rearrange(linear_reg_output_data, "(b t) c w h -> b t c w h", **pred_shape)

    def _compute_standard_loss(
        self,
        pred_data: torch.Tensor,
        linear_reg_output_data: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Compute the standard loss
        """
        linear_reg_output_data = rearrange(
            linear_reg_output_data,
            "b t c w h  -> (b t w h) c",
        )
        pred_data = rearrange(
            pred_data,
            "b t c w h -> (b t w h) c",
        )
        loss_mask = rearrange(
            loss_mask,
            "b t w h -> (b t w h)",
        )
        # Discard dates for which there are nans
        no_nan_mask = torch.isnan(linear_reg_output_data).sum(dim=-1) == 0

        if no_nan_mask.sum() == 0:
            return None

        linear_reg_output_data = linear_reg_output_data[no_nan_mask, ...]
        pred_data = pred_data[no_nan_mask, ...]
        loss_mask = loss_mask[no_nan_mask, ...]

        loss_value = self.loss(
            pred_data[loss_mask],
            linear_reg_output_data[loss_mask],
        )

        return loss_value

    def _compute_lpips_loss(
        self,
        pred_data: torch.Tensor,
        linear_reg_output_data: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Compute the lpips loss
        """
        pred_shape = parse_shape(pred_data, "b t c w h")
        linear_reg_output_data = rearrange(
            linear_reg_output_data,
            "b t c w h  -> (b t c) w h",
        )
        pred_data = rearrange(
            pred_data,
            "b t c w h -> (b t c) w h",
        )
        loss_mask = rearrange(
            repeat(loss_mask, "b t w  h -> b t c w h", c=pred_shape["c"]),
            "b t c w h -> (b t c) w h",
        )
        # Discard dates for which there are nans
        no_nan_mask = torch.isnan(linear_reg_output_data).sum(dim=(-1, -2)) == 0

        if no_nan_mask.sum() == 0:
            return None

        linear_reg_output_data = linear_reg_output_data[no_nan_mask, ...]
        pred_data = pred_data[no_nan_mask, ...]
        loss_mask = loss_mask[no_nan_mask, ...]

        # Extend to 3 bands for LPIPS
        linear_reg_output_data = repeat(linear_reg_output_data, "b w h -> b 3 w h")
        pred_data = repeat(pred_data, "b w h -> b 3 w h")

        loss_value = self.lpips(pred_data, linear_reg_output_data, loss_mask)

        return loss_value

    def forward(
        self,
        pred_sits: MonoModalSITS,
        target_sits: MonoModalSITS,
        per_date: bool = True,
        high_pass_filtering_mode: HighPassFilteringMode = HighPassFilteringMode.PRE,
        use_lpips: bool = False,
    ) -> torch.Tensor | None:
        """
        Compute loss value on all date and element in batch
        """
        # Early exit
        if 0 in (pred_sits.shape()[1], target_sits.shape()[1]):
            return None

        # Derive inputs for the loss (either HPF, gradient magnitude or data themselves)
        pred_sits_for_regression, target_sits_for_regression = self._build_inputs(
            pred_sits,
            target_sits,
            high_pass_filtering_mode == HighPassFilteringMode.PRE,
        )

        # Compute the loss mask
        loss_mask = self._build_validity_mask(target_sits)

        pred_data = pred_sits_for_regression.data
        target_data = target_sits_for_regression.data

        # Compute clear pixel rate
        clear_rate = loss_mask.sum(dim=(0, -1, -2)) / (
            loss_mask.shape[-1] * loss_mask.shape[-2]
        )

        # Discard whole dates that have not enough clear pixels
        loss_mask = loss_mask[:, clear_rate >= 1.0 - self.max_masked_rate, ...]
        pred_data = pred_data[:, clear_rate >= 1.0 - self.max_masked_rate, ...]
        target_data = target_data[:, clear_rate >= 1.0 - self.max_masked_rate, ...]

        # Perform the linear regression to adapt target to prediction
        linear_reg_output_data = self._perform_linear_regression(
            pred_data, target_data, loss_mask, per_date
        )

        if linear_reg_output_data is None:
            return None

        # Optional post high pass filtering
        if high_pass_filtering_mode == HighPassFilteringMode.POST:
            pred_data, linear_reg_output_data = self._post_high_pass_filtering(
                pred_data, linear_reg_output_data
            )
        # Handle LPIPS option
        if use_lpips:
            loss_value = self._compute_lpips_loss(
                pred_data, linear_reg_output_data, loss_mask
            )
        else:
            loss_value = self._compute_standard_loss(
                pred_data, linear_reg_output_data, loss_mask
            )
        # Safeguard to avoid returning none
        if (
            loss_value is None
            or torch.isnan(loss_value)
            or loss_value > self.loss_security_threshold
        ):
            return None

        return loss_value
