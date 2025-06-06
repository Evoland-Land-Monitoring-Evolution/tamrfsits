# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Tests for the decorrelation module
"""

import torch

from tamrfsits.core.decorrelation import decorrelation_loss

from .tests_utils import generate_monomodal_sits


def test_decorrelation_loss():
    """
    Test the DecorrelationLoss class
    """
    sits = generate_monomodal_sits(masked=False)

    out = decorrelation_loss(sits)

    assert out and torch.all(~torch.isnan(out))
