# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Tests of the fda module
"""

import torch

from tamrfsits.validation.fda import compute_fft_profile, compute_frr_referenceless


def test_compute_fft_profile():
    """
    Test the compute_fft_profile function
    """
    data = torch.rand((2, 5, 16, 16))
    freqs, profs = compute_fft_profile(data, s=16)

    assert freqs.shape == torch.Size((4,))
    assert profs.shape == torch.Size((2, 4, 5))


def test_compute_frr_referenceless():
    """
    Test the compute_frr_referenceless function
    """
    data1 = torch.rand((2, 5, 16, 16))
    data2 = torch.rand((2, 5, 16, 16))
    data2[:, :, 8:, 8:] += 10.0
    _, prof1 = compute_fft_profile(data1, s=16)
    _, prof2 = compute_fft_profile(data2, s=16)
    frr = compute_frr_referenceless(prof1, prof2)

    assert frr.shape == torch.Size((2, 5))
    assert torch.all(~torch.isnan(frr))
