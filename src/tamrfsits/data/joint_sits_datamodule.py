# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the lightning datamodule for the sen2venus dataset
"""
import logging
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore

from tamrfsits.data.joint_sits_dataset import (
    DEFAULT_INDEX_TUPLE,
    MultiSITSDataset,
    collate_fn,
)


@dataclass(frozen=True)
class JointSITSDataModuleConfig:
    """
    Datamodule config
    """

    training_sits: list[str]
    testing_sits: list[str]
    patch_size: int = 60
    lr_sensor: str = "sentinel2"
    hr_sensor: str = "landsat"
    hr_index_files: tuple[str, ...] | None = DEFAULT_INDEX_TUPLE
    lr_index_files: tuple[str, ...] | None = DEFAULT_INDEX_TUPLE
    lr_resolution: float = 30.0
    hr_resolution: float = 10.0
    hr_bands: tuple[list[int] | None, ...] = (None,)
    lr_bands: tuple[list[int] | None, ...] = (None,)
    mtf_for_downsampling: float = 0.1
    dt_orig: str | None = "2019.01.01"
    time_slices_in_days: int | None = None
    min_nb_dates: int = 4
    max_nb_dates: int | None = None
    conjunctions_only: bool = False
    batch_size: int = 32
    testing_validation_batch_size: int = 128
    random_reduce_rate: float | None = None
    testing_random_reduce_rate: float | None = None
    validation_indices: list[int] | None = None
    validation_ratio: float = 0.1
    train_ratio: float = 0.9
    num_workers: int = 1
    prefetch_factor: int | None = None
    cache_dir: str | None = None


class JointSITSDataModule(pl.LightningDataModule):
    """
    Datamodule for the sen2venus fusion dataset
    """

    def __init__(self, config: JointSITSDataModuleConfig):
        """
        Initializer
        """
        super().__init__()
        # self.save_hyperparameters()

        self.config = config

        # Type annotations
        self.training_dataset: torch.utils.data.Dataset
        self.validation_dataset: torch.utils.data.Dataset

        self.testing_dataset = MultiSITSDataset(
            self.config.testing_sits,
            patch_size=self.config.patch_size,
            hr_sensor=self.config.hr_sensor,
            lr_sensor=self.config.lr_sensor,
            hr_index_files=self.config.hr_index_files,
            lr_index_files=self.config.lr_index_files,
            lr_resolution=self.config.lr_resolution,
            hr_resolution=self.config.hr_resolution,
            hr_bands=self.config.hr_bands,
            lr_bands=self.config.lr_bands,
            mtf_for_downsampling=self.config.mtf_for_downsampling,
            dt_orig=self.config.dt_orig,
            time_slices_in_days=self.config.time_slices_in_days,
            min_nb_dates=self.config.min_nb_dates,
            conjunctions_only=self.config.conjunctions_only,
            cache_dir=self.config.cache_dir,
            random_reduce_rate=self.config.testing_random_reduce_rate,
        )

        remaining_dataset = MultiSITSDataset(
            self.config.training_sits,
            patch_size=self.config.patch_size,
            hr_sensor=self.config.hr_sensor,
            lr_sensor=self.config.lr_sensor,
            hr_index_files=self.config.hr_index_files,
            lr_index_files=self.config.lr_index_files,
            lr_resolution=self.config.lr_resolution,
            hr_resolution=self.config.hr_resolution,
            hr_bands=self.config.hr_bands,
            lr_bands=self.config.lr_bands,
            mtf_for_downsampling=self.config.mtf_for_downsampling,
            dt_orig=self.config.dt_orig,
            time_slices_in_days=self.config.time_slices_in_days,
            min_nb_dates=self.config.min_nb_dates,
            max_nb_dates=self.config.max_nb_dates,
            conjunctions_only=self.config.conjunctions_only,
            cache_dir=self.config.cache_dir,
            random_reduce_rate=self.config.random_reduce_rate,
        )

        if config.validation_indices is not None:
            logging.info(
                "Using a fixed subset of training set for validation:\
                {}",
                config.validation_indices,
            )
            all_indices = list(range(len(remaining_dataset)))
            training_indices = [
                idx for idx in all_indices if idx not in config.validation_indices
            ]
            self.training_dataset = torch.utils.data.Subset(
                remaining_dataset, training_indices
            )
            self.validation_dataset = torch.utils.data.Subset(
                remaining_dataset, config.validation_indices
            )
        else:
            logging.info("Generating a random split for training and validation sets")
            if config.validation_ratio + config.train_ratio < 1:
                (
                    self.training_dataset,
                    self.validation_dataset,
                    _,
                ) = torch.utils.data.random_split(
                    remaining_dataset,
                    [
                        config.train_ratio,
                        config.validation_ratio,
                        (1 - config.train_ratio - config.validation_ratio),
                    ],
                )
            else:
                (
                    self.training_dataset,
                    self.validation_dataset,
                ) = torch.utils.data.random_split(
                    remaining_dataset, [config.train_ratio, config.validation_ratio]
                )
            logging.info(
                "Generated indices for validation: {}", self.validation_dataset.indices
            )
        logging.info("{} training patches available", len(self.training_dataset))
        logging.info("{} validation patches available", len(self.validation_dataset))
        logging.info("{} testing patches available", len(self.testing_dataset))

    def train_dataloader(self):
        """
        Return train dataloaded (reset every time this method is called)
        """
        return StatefulDataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Return validation data loader (never reset)
        """
        return StatefulDataLoader(
            self.validation_dataset,
            batch_size=self.config.testing_validation_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Return test data loader (never reset)
        """
        return StatefulDataLoader(
            self.testing_dataset,
            batch_size=self.config.testing_validation_batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )
