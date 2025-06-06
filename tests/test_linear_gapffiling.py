# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import torch

from tamrfsits.core.linear_gapfilling import linear_gapfilling

from .tests_utils import generate_monomodal_sits


def test_linear_gapfilling():
    """
    Test the linear gapffiling function
    """
    sits = generate_monomodal_sits()
    target_doy = torch.arange(
        sits.doy.min().item(), sits.doy.max().item(), dtype=torch.float
    )

    out = linear_gapfilling(sits, target_doy)

    assert torch.all(~torch.isnan(out.doy))
