#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import rasterio as rio  # type: ignore
import torch
from affine import Affine  # type: ignore

from hydra import compose, initialize_config_dir
from tamrfsits.data.joint_sits_dataset import SingleSITSDataset


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(os.path.basename(__file__), description="")

    parser.add_argument(
        "--config", "-cfg", type=str, help="Path to hydra config", required=True
    )

    parser.add_argument("--ts", type=str, help="Path to time-series", required=True)

    parser.add_argument(
        "--output", type=str, help="Path to output folder", required=True
    )

    parser.add_argument(
        "--width", help="Width of the square patch to infer", default=512, type=int
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for sample patches selection"
    )

    return parser


def main():
    """
    Main method
    """
    # configure logging at the root level of Lightning
    # Define logger (from https://github.com/ashleve/lightning-hydra-template/blob
    # /a4b5299c26468e98cd264d3b57932adac491618a/src/testing_pipeline.py)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Find on which device to run
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
    logging.info("Processing will happen on device %s", dev)

    # We instantiate the checkpoint configuration
    with initialize_config_dir(version_base=None, config_dir=args.config):
        config = compose(config_name="config.yaml")
        pl.seed_everything(args.seed)
        if config.get("mat_mul_precision"):
            torch.set_float32_matmul_precision(config.get("mat_mul_precision"))

    dataset = SingleSITSDataset(ts_path=args.ts, patch_size=args.width)

    # Read one patch
    lr_sits, hr_sits = dataset[0]

    for t in range(hr_sits.shape()[1]):
        geotransform = (0, 0.5, 0.0, 0, 0.0, -0.5)
        transform = Affine.from_gdal(*geotransform)

        doy = hr_sits.doy[0, t]

        profile = {
            "driver": "GTiff",
            "height": hr_sits.shape()[-1],
            "width": hr_sits.shape()[-2],
            "count": hr_sits.shape()[2],
            "dtype": np.int16,
            "transform": transform,
        }

        Path(os.path.join(args.output, "hr")).mkdir(parents=True, exist_ok=True)
        with rio.open(
            os.path.join(args.output, "hr", f"{doy}_hr.tif"), "w", **profile
        ) as ds:
            for band in range(hr_sits.shape()[2]):
                ds.write(
                    hr_sits.data[0, t, band, ...].cpu().numpy().astype(np.int16),
                    band + 1,
                )

    for t in range(lr_sits.shape()[1]):
        geotransform = (0, 1.0, 0.0, 0, 0.0, -1.0)
        transform = Affine.from_gdal(*geotransform)

        doy = lr_sits.doy[0, t]

        profile = {
            "driver": "GTiff",
            "height": lr_sits.shape()[-1],
            "width": lr_sits.shape()[-2],
            "count": lr_sits.shape()[2],
            "dtype": np.int16,
            "transform": transform,
        }

        Path(os.path.join(args.output, "lr")).mkdir(parents=True, exist_ok=True)
        with rio.open(
            os.path.join(args.output, "lr", f"{doy}_lr.tif"), "w", **profile
        ) as ds:
            for band in range(lr_sits.shape()[2]):
                ds.write(
                    lr_sits.data[0, t, band, ...].cpu().numpy().astype(np.int16),
                    band + 1,
                )


if __name__ == "__main__":
    main()
