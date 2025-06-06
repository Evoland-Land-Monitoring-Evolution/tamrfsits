# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the tests for lpips module
"""
import torch
from piq import LPIPS  # type: ignore

from tamrfsits.core.lpips import MaskedLPIPSLoss


def test_masked_lpips_loss():
    """
    Test for the MaskedLPIPSLoss class
    """
    data1 = torch.rand((10, 3, 128, 128))
    data2 = torch.rand((10, 3, 128, 128))
    mask = torch.zeros((10, 128, 128), dtype=torch.bool)

    mask[:5, :32:, :] = True
    mask[5:6, :, :] = True
    mask[6:, :, 64:] = True

    loss = MaskedLPIPSLoss(
        LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0]).requires_grad_(False).eval()
    )

    assert loss(data1, data2, torch.zeros((10, 128, 128), dtype=torch.bool)) is None

    loss_value = loss(data1, data2, mask)

    assert torch.all(~torch.isnan(loss_value))
    assert torch.all(~torch.isnan(data1))
    assert torch.all(~torch.isnan(data2))


def test_masked_lpips_all_valid_equals_lpips():
    """
    Test that masked lpips with all valid is equal to lpips
    """
    data1 = torch.rand((10, 3, 128, 128))
    data2 = torch.rand((10, 3, 128, 128))
    mask = torch.ones((10, 128, 128), dtype=torch.bool)

    masked_lpips = MaskedLPIPSLoss(
        LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0]).requires_grad_(False).eval()
    )
    lpips = LPIPS(mean=[0, 0, 0], std=[1.0, 1.0, 1.0]).requires_grad_(False).eval()

    masked_lpips_out = masked_lpips(data1, data2, mask)

    lpips_out = lpips(data1, data2)

    assert masked_lpips_out == lpips_out
