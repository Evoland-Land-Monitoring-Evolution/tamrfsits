# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests related to the sen2venus_fusion datamodule
"""

import os

import pytest

from tamrfsits.data.joint_sits_datamodule import (
    JointSITSDataModule,
    JointSITSDataModuleConfig,
)

from .tests_utils import get_ls2s2_dataset_path


@pytest.mark.requires_data
def test_ls2s2_datamodule() -> None:
    """
    Test the sen2venus datamodule
    """

    config = JointSITSDataModuleConfig(
        training_sits=[os.path.join(get_ls2s2_dataset_path(), "30SUF_21")],
        testing_sits=[os.path.join(get_ls2s2_dataset_path(), "31UER_57")],
        hr_sensor="sentinel2",
        lr_sensor="landsat",
        hr_resolution=10.0,
        lr_resolution=30.0,
    )

    dm = JointSITSDataModule(config)

    dm.val_dataloader()
    dm.train_dataloader()
    dm.test_dataloader()


@pytest.mark.requires_data
def test_ls2s2_datamodule_pan() -> None:
    """
    Test the sen2venus datamodule
    """

    config = JointSITSDataModuleConfig(
        training_sits=[os.path.join(get_ls2s2_dataset_path(), "30SUF_21")],
        testing_sits=[os.path.join(get_ls2s2_dataset_path(), "31UER_57")],
        hr_sensor="sentinel2",
        lr_sensor="landsat",
        lr_index_files=("index.csv", "index_pan.csv"),
        hr_resolution=10.0,
        lr_resolution=10.0,
    )

    dm = JointSITSDataModule(config)

    dm.val_dataloader()
    dm.train_dataloader()
    dm.test_dataloader()


@pytest.mark.requires_data
def test_ls2s2_datamodule_hr_bands_lr_bands() -> None:
    """
    Test the sen2venus datamodule
    """

    config = JointSITSDataModuleConfig(
        training_sits=[os.path.join(get_ls2s2_dataset_path(), "30SUF_21")],
        testing_sits=[os.path.join(get_ls2s2_dataset_path(), "31UER_57")],
        hr_sensor="sentinel2",
        lr_sensor="landsat",
        hr_resolution=10.0,
        lr_resolution=30.0,
        hr_bands=([0, 1],),
        lr_bands=([0, 1],),
    )

    dm = JointSITSDataModule(config)

    dm.val_dataloader()
    dm.train_dataloader()
    dm.test_dataloader()
