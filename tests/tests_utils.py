#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
utilities for tests
"""
import os
import random

import pytorch_lightning as pl
import torch

from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.data.joint_sits_dataset import collate_fn


def get_ls2s2_dataset_path():
    """ """
    if "LS2S2_DATASET_PATH" in os.environ:
        return os.environ["LS2S2_DATASET_PATH"]
    return "/work/CESBIO/projects/RELEO/datasets/ls2s2_fusion/data/train/"


def get_tests_output_path():
    """ """
    if "TESTS_OUTPUT_PATH" in os.environ:
        return os.environ["TESTS_OUPTUT_PATH"]
    return "/work/scratch/data/" + os.environ["USER"] + "/tmp/"


def generate_monomodal_sits(
    batch: int = 16,
    nb_doy: int = 10,
    nb_features: int = 4,
    width: int = 32,
    masked: bool = True,
    max_doy: int = 20,
    clear_doy_proba: float = 0.5,
) -> MonoModalSITS:
    """
    Generate a fake monomodal sits
    """
    possible_doys = list(range(max_doy))
    doy_list = [
        torch.sort(torch.tensor(random.sample(possible_doys, nb_doy)))[0]
        for i in range(batch)
    ]
    doy = torch.stack(doy_list).to(dtype=torch.int16)
    data = torch.rand((batch, nb_doy, nb_features, width, width))
    mask: torch.Tensor | None = None
    if masked:
        mask = torch.zeros((batch, nb_doy, width, width)).to(dtype=torch.bool)

        for i in range(batch):
            for d in range(nb_doy):
                if random.random() > clear_doy_proba:
                    # This date is masked
                    if random.random() > 0.5:
                        # Fully masked
                        mask[i, d, ...] = True
                    else:
                        # Partially masked
                        start_x = random.randint(0, width)
                        end_x = random.randint(start_x, width)
                        start_y = random.randint(0, width)
                        end_y = random.randint(start_y, width)
                        mask[i, d, start_x:end_x, start_y:end_y] = True

    # Should pass
    return MonoModalSITS(data, doy, mask)


class FakeHRLRDataset(torch.utils.data.Dataset):
    """
    Generate fake data

    Wraps the generate_monomodal_sits in a torch.utils.Dataset subclass
    """

    def __init__(
        self,
        nb_samples: int = 1000,
        nb_doy: int = 10,
        nb_hr_doy: int | None = None,
        nb_hr_features: int = 4,
        nb_lr_features: int = 8,
        lr_width: int = 32,
        resolution_ratio: int = 2,
        masked: bool = True,
        max_doy: int = 20,
        clear_doy_proba: float = 0.5,
    ):
        """
        Initializer
        """
        super().__init__()
        self.nb_samples = nb_samples
        self.nb_doy = nb_doy
        self.nb_hr_doy = nb_hr_doy if nb_hr_doy is not None else nb_doy
        self.nb_hr_features = nb_hr_features
        self.nb_lr_features = nb_lr_features
        self.lr_width = lr_width
        self.resolution_ratio = resolution_ratio
        self.masked = masked
        self.max_doy = max_doy
        self.clear_doy_proba = clear_doy_proba

    def __len__(self) -> int:
        """
        implement len()
        """
        return self.nb_samples

    def __getitem__(self, idx: int) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Implement get_item
        """
        hr_sits = generate_monomodal_sits(
            batch=1,
            nb_doy=self.nb_hr_doy,
            nb_features=self.nb_hr_features,
            width=self.lr_width * self.resolution_ratio,
            max_doy=self.max_doy,
            clear_doy_proba=self.clear_doy_proba,
        )

        lr_sits = generate_monomodal_sits(
            batch=1,
            nb_doy=self.nb_doy,
            nb_features=self.nb_lr_features,
            width=self.lr_width,
            max_doy=self.max_doy,
            clear_doy_proba=self.clear_doy_proba,
        )

        return (lr_sits, hr_sits)


class FakeHRLRDatamodule(pl.LightningDataModule):
    """
    Fake datamodule
    """

    def __init__(
        self,
        batch_size: int = 4,
        nb_train_samples: int = 1000,
        nb_val_samples: int = 1000,
        nb_test_samples: int = 1000,
        nb_doys: int = 10,
        nb_hr_doys: int | None = None,
        nb_hr_features: int = 4,
        nb_lr_features: int = 8,
        lr_width: int = 32,
        resolution_ratio: int = 2,
        masked: bool = True,
        max_doy: int = 20,
        clear_doy_proba: float = 0.5,
    ):
        """
        Initializer
        """
        super().__init__()

        self.batch_size = batch_size

        self.train_dataset = FakeHRLRDataset(
            nb_train_samples,
            nb_doys,
            nb_hr_doys,
            nb_hr_features,
            nb_lr_features,
            lr_width,
            resolution_ratio,
            masked,
            max_doy,
            clear_doy_proba,
        )

        self.val_dataset = FakeHRLRDataset(
            nb_samples=nb_val_samples,
            nb_doy=nb_doys,
            nb_hr_doy=nb_hr_doys,
            nb_hr_features=nb_hr_features,
            nb_lr_features=nb_lr_features,
            lr_width=lr_width,
            resolution_ratio=resolution_ratio,
            masked=masked,
            max_doy=max_doy,
            clear_doy_proba=clear_doy_proba,
        )

        self.test_dataset = FakeHRLRDataset(
            nb_samples=nb_test_samples,
            nb_doy=nb_doys,
            nb_hr_features=nb_hr_features,
            nb_lr_features=nb_lr_features,
            lr_width=lr_width,
            resolution_ratio=resolution_ratio,
            masked=masked,
            max_doy=max_doy,
            clear_doy_proba=clear_doy_proba,
        )

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=1,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Return val dataloader
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Return test dataloader
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
