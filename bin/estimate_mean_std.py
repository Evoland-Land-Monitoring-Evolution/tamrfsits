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

    lr_samples_list: list[torch.Tensor] = []
    hr_samples_list: list[torch.Tensor] = []

    for _ in tqdm(range(nb_batch), total=nb_batch, desc="Sampling data"):
        # Get one sample
        batch = next(data_iter)

        # Unpack
        lr_sits, hr_sits = batch
        lr_data = rearrange(lr_sits.data, "b t c w h -> (b t w h) c")
        lr_mask = rearrange(lr_sits.mask, "b t w h -> (b t w h)")
        lr_samples_list.append(lr_data[~lr_mask, :])
        hr_data = rearrange(hr_sits.data, "b t c w h -> (b t w h) c")
        hr_mask = rearrange(hr_sits.mask, "b t w h -> (b t w h)")
        hr_samples_list.append(hr_data[~hr_mask, :])

    lr_samples = torch.cat(lr_samples_list)
    hr_samples = torch.cat(hr_samples_list)

    lr_samples = lr_samples / 10000.0
    hr_samples = hr_samples / 10000.0

    hr_mean = hr_samples.mean(dim=0).numpy()
    lr_mean = lr_samples.mean(dim=0).numpy()
    hr_std = hr_samples.std(dim=0).numpy()
    lr_std = lr_samples.std(dim=0).numpy()

    print(f"LR number of samples: {lr_samples.shape[0]}")
    print(f"mean: {lr_mean}")
    print(f"std: {lr_std}")
    print(f"HR number of samples: {hr_samples.shape[0]=}")
    print(f"mean: {hr_mean}")
    print(f"std: {hr_std}")


if __name__ == "__main__":
    main()
