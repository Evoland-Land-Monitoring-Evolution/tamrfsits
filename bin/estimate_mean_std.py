#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Training script for tamrfsits
"""
import logging

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig
from tqdm import tqdm

from tamrfsits.data.joint_sits_datamodule import JointSITSDataModule


@hydra.main(version_base=None, config_path="../hydra/", config_name="train.yaml")
def main(config: DictConfig):
    """
    Main hydra
    """

    if config.get("loglevel"):
        # Configure logging
        numeric_level = getattr(logging, config.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {config.loglevel}")

        logging.basicConfig(
            level=numeric_level,
            datefmt="%y-%m-%d %H:%M:%S",
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )

    # load samples
    # Be sure not to use random_reduce_rate or max_nb_dates
    data_module_cfg = hydra.utils.instantiate(
        config.datamodule.data_module.config, random_reduce_rate=None, max_nb_dates=None
    )

    # Get data loader
    data_loader = JointSITSDataModule(data_module_cfg).train_dataloader()

    nb_batch = 2000

    data_iter = iter(data_loader)

    hr_sum_tensor: torch.Tensor | None = None
    hr_squared_sum_tensor: torch.Tensor | None = None
    nb_hr_samples = 0
    lr_sum_tensor: torch.Tensor | None = None
    lr_squared_sum_tensor: torch.Tensor | None = None
    nb_lr_samples = 0

    for _ in tqdm(range(nb_batch), total=nb_batch, desc="Sampling data"):
        # Get one sample
        batch = next(data_iter)

        # Unpack
        lr_sits, hr_sits = batch
        lr_data = rearrange(lr_sits.data, "b t c w h -> (b t w h) c")
        lr_mask = rearrange(lr_sits.mask, "b t w h -> (b t w h)")
        lr_data = lr_data[~lr_mask, :] / 10000.0
        if lr_data.shape[0]:
            if lr_sum_tensor is None:
                lr_sum_tensor = lr_data.sum(dim=0)
                lr_squared_sum_tensor = (lr_data.shape[0] - 1) * (
                    lr_data.std(dim=0) ** 2
                )
            else:
                lr_sum_tensor += lr_data.sum(dim=0)
                lr_squared_sum_tensor += (lr_data.shape[0] - 1) * (
                    lr_data.std(dim=0) ** 2
                )
            nb_lr_samples += lr_data.shape[0]

        hr_data = rearrange(hr_sits.data, "b t c w h -> (b t w h) c")
        hr_mask = rearrange(hr_sits.mask, "b t w h -> (b t w h)")
        hr_data = hr_data[~hr_mask, :] / 10000.0
        if hr_data.shape:
            if hr_sum_tensor is None:
                hr_sum_tensor = hr_data.sum(dim=0)
                hr_squared_sum_tensor = (hr_data.shape[0] - 1) * (
                    hr_data.std(dim=0) ** 2
                )
            else:
                hr_sum_tensor += hr_data.sum(dim=0)
                hr_squared_sum_tensor += (hr_data.shape[0] - 1) * (
                    hr_data.std(dim=0) ** 2
                )
            nb_hr_samples += hr_data.shape[0]

    lr_mean = lr_sum_tensor / nb_lr_samples
    hr_mean = hr_sum_tensor / nb_hr_samples
    lr_std = torch.sqrt(lr_squared_sum_tensor / (nb_lr_samples - nb_batch))
    hr_std = torch.sqrt(hr_squared_sum_tensor / (nb_hr_samples - nb_batch))

    print(f"LR number of samples: {nb_lr_samples}")
    print(f"mean: {lr_mean}")
    print(f"std: {lr_std}")
    print(f"HR number of samples: {nb_hr_samples}")
    print(f"mean: {hr_mean}")
    print(f"std: {hr_std}")


if __name__ == "__main__":
    main()
