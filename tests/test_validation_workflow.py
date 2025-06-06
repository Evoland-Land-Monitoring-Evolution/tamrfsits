# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Contains tests for the validation.workflow module
"""

import pandas as pd
import torch

from tamrfsits.validation.workflow import TimeSeriesTestResult, to_pandas


def test_to_pandas():
    """
    Test the to_pandas method
    """
    res1 = TimeSeriesTestResult(
        name="site1",
        sensor="sensor",
        rmse=torch.rand((2, 4)),
        doy=torch.randint(0, 4, (2,)),
        clear_doy=torch.rand((2,)) > 0.5,
        brisque=torch.rand((2, 4)),
        target_brisque=torch.rand((2, 4)),
        frr=torch.rand((2, 4)),
        clear_pixel_rate=torch.rand((2,)),
        local_density=torch.ones((2,)),
        local_joint_density=torch.ones((2,)),
        closest_doy=torch.zeros((2,)),
        closest_joint_doy=torch.zeros((2,)),
        ref_prof=torch.zeros((2, 10, 10)),
        pred_prof=torch.zeros((2, 10, 10)),
        freqs=torch.zeros((10,)),
        pred_pixel_rate=torch.zeros((2,)),
    )

    res2 = TimeSeriesTestResult(
        name="site2",
        sensor="sensor",
        rmse=torch.rand((2, 4)),
        doy=torch.randint(0, 4, (2,)),
        clear_doy=torch.rand((2,)) > 0.5,
        brisque=torch.rand((2, 4)),
        target_brisque=torch.rand((2, 4)),
        frr=torch.rand((2, 4)),
        clear_pixel_rate=torch.rand((2,)),
        local_density=torch.ones((2,)),
        local_joint_density=torch.ones((2,)),
        closest_doy=torch.zeros((2,)),
        closest_joint_doy=torch.zeros((2,)),
        ref_prof=torch.zeros((2, 10, 10)),
        pred_prof=torch.zeros((2, 10, 10)),
        freqs=torch.zeros((10,)),
        pred_pixel_rate=torch.zeros((2,)),
    )

    df = to_pandas([res1, res2], band_labels=["b0", "b1", "b2", "b3"])

    assert len(df) == 4
    print(df.columns)
    assert (
        pd.Index(
            [
                "name",
                "doy",
                "clear_doy",
                "clear_pixel_rate",
                "closest_doy",
                "closest_joint_doy",
                "local_density",
                "local_joint_density",
                "pred_pixel_rate",
                "rmse_b0",
                "rmse_b1",
                "rmse_b2",
                "rmse_b3",
                "brisque_b0",
                "brisque_b1",
                "brisque_b2",
                "brisque_b3",
                "frr_b0",
                "frr_b1",
                "frr_b2",
                "frr_b3",
                "target_brisque_b0",
                "target_brisque_b1",
                "target_brisque_b2",
                "target_brisque_b3",
            ]
        )
        == df.columns
    ).all()
