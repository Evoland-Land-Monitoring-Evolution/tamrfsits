# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This modules contains utilities to build testing scripts
"""

import argparse
import logging
import os
from datetime import timedelta
from functools import cache
from logging import info, warning
from os.path import join
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from affine import Affine  # type: ignore
from einops import parse_shape, rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import zeros as np_zeros
from pandas import to_datetime
from rasterio import open as rio_open  # type: ignore
from torch import Tensor
from torch import argwhere as torch_argwhere  # pylint: disable=no-name-in-module
from torch import clip as torch_clip  # pylint: disable=no-name-in-module
from torchmetrics.functional import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from tqdm import tqdm

from tamrfsits.core.time_series import MonoModalSITS, crop_sits
from tamrfsits.core.utils import patchify
from tamrfsits.data.joint_sits_dataset import SingleSITSDataset
from tamrfsits.tasks.interpolation.training_module import (
    TemporalInterpolationTrainingModule,
)


def parse_model_checkpoint(
    checkpoint: dict, module_name: str, namespace: str = "the_modules"
):
    """
    Helper function to retrieve model parameters from checkpoint
    """
    return {
        k.split(".", maxsplit=2)[2].replace("._orig_mod", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith(f"{namespace}.{module_name}")
    }


@cache
def build_model(
    checkpoint_path: str,
    seed: int,
    config_path: str | None = None,
    device: str | None = None,
):
    """
    Factor model building code common to all testing scripts
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
    # Find on which device to run
    if device is None:
        dev = "cpu"
        if torch.cuda.is_available():
            dev = "cuda"
    else:
        dev = device
    info("Processing will happen on device %s", dev)

    if config_path is None:
        config_path = join(os.path.dirname(checkpoint_path), ".hydra")
        config_path = config_path.replace("checkpoints", "runs")

    # We instantiate the checkpoint configuration
    with initialize_config_dir(version_base=None, config_dir=config_path):
        config = compose(config_name="config.yaml")
        pl.seed_everything(seed)
        if config.get("mat_mul_precision"):
            torch.set_float32_matmul_precision(config.get("mat_mul_precision"))

    model = instantiate(config.training_module.training_module)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=True
    )

    # For each registered modules, read back parameters
    for m in model.the_modules.keys():
        p = parse_model_checkpoint(checkpoint, m)
        if p:
            try:
                model.the_modules[m].load_state_dict(p)
            except RuntimeError as e:
                warning(
                    f"Could not load parameters for module {m}:\
                    mismatch with checkpoint: {e}"
                )
        else:
            info(f"No parameters found for module {m} in checkpoint")

    model = model.to(device=dev)

    return model, config, dev


def get_default_parser(checkpoint_config: bool = True) -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Test script"
    )
    if checkpoint_config:
        parser.add_argument(
            "--checkpoint",
            "-cp",
            type=str,
            help="Path to model checkpoint",
        )
        parser.add_argument(
            "--config", "-cfg", type=str, help="Path to hydra config", required=False
        )

    parser.add_argument(
        "--ts", type=str, help="Path to time-series", required=True, nargs="+"
    )

    parser.add_argument(
        "--output", type=str, help="Path to output folder", required=True
    )

    parser.add_argument(
        "--width", help="Width of the square patch to infer", default=9900, type=int
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for sample patches selection"
    )

    return parser


def write_sits(
    pred_sits: MonoModalSITS,
    target_sits: MonoModalSITS,
    dt_orig: str,
    out_dir: str,
    scale_pred: float = 1.0,
    scale_target: float = 2.0,
    radio_scale_pred: float = 10000.0,
    radio_scale_target: float = 1.0,
    dtype=np.int16,
    mode="",
):
    """
    Estimate performances
    """
    geotransform_pred = (0, scale_pred, 0.0, 0, 0.0, -scale_pred)
    transform_pred = Affine.from_gdal(*geotransform_pred)
    profile_pred = {
        "driver": "GTiff",
        "height": pred_sits.shape()[-1],
        "width": pred_sits.shape()[-2],
        "count": pred_sits.shape()[2],
        "dtype": dtype,
        "transform": transform_pred,
    }

    geotransform_ref = (0, scale_target, 0.0, 0, 0.0, -scale_target)
    transform_ref = Affine.from_gdal(*geotransform_ref)
    profile_ref = {
        "driver": "GTiff",
        "height": target_sits.shape()[-1],
        "width": target_sits.shape()[-2],
        "count": target_sits.shape()[2],
        "dtype": dtype,
        "transform": transform_ref,
    }

    Path(join(out_dir, mode)).mkdir(parents=True, exist_ok=True)
    for t in range(pred_sits.shape()[1]):
        doy = pred_sits.doy[0, t]
        doy_str = (to_datetime(dt_orig) + timedelta(days=doy.item())).strftime(
            "%Y-%m-%d"
        )
        with rio_open(
            join(out_dir, mode, f"{doy}_{doy_str}_{mode}_pred.tif"),
            "w",
            **profile_pred,
        ) as ds:
            for band in range(pred_sits.shape()[2]):
                ds.write(
                    (pred_sits.data[0, t, band, ...] * radio_scale_pred)
                    .cpu()
                    .numpy()
                    .astype(dtype),
                    band + 1,
                )

        with rio_open(
            join(out_dir, mode, f"{doy}_{doy_str}_{mode}_ref.tif"),
            "w",
            **profile_ref,
        ) as ds:
            for band in range(target_sits.shape()[2]):
                ds.write(
                    (target_sits.data[0, t, band, ...] * radio_scale_target)
                    .cpu()
                    .numpy()
                    .astype(dtype),
                    band + 1,
                )


def write_latent(
    latent_sits: MonoModalSITS, dt_orig: str, out_dir: str, mode="", scale: float = 1.0
):
    """
    Write latent sits
    """

    geotransform_latent = (0, scale, 0.0, 0, 0.0, -scale)
    transform_latent = Affine.from_gdal(*geotransform_latent)
    profile_latent = {
        "driver": "GTiff",
        "height": latent_sits.shape()[-1],
        "width": latent_sits.shape()[-2],
        "count": latent_sits.shape()[2],
        "dtype": np.float32,
        "transform": transform_latent,
    }

    Path(join(out_dir, mode)).mkdir(parents=True, exist_ok=True)
    for t in range(latent_sits.shape()[1]):
        doy = latent_sits.doy[0, t]
        doy_str = (to_datetime(dt_orig) + timedelta(days=doy.item())).strftime(
            "%Y-%m-%d"
        )
        with rio_open(
            join(out_dir, mode, f"{doy}_{doy_str}_{mode}.tif"),
            "w",
            **profile_latent,
        ) as ds:
            for band in range(latent_sits.shape()[2]):
                ds.write(
                    (latent_sits.data[0, t, band, ...]).cpu().numpy(),
                    band + 1,
                )


def estimate_performance(
    pred_sits: MonoModalSITS,
    target_sits: MonoModalSITS,
    dt_orig: str,
    mode="",
    spatial_margin: int = 5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Estimate performances
    """
    all_maes: list[list[Tensor]] = []
    all_mapes: list[list[Tensor]] = []
    all_rmses: list[list[Tensor]] = []
    all_mask_percents: list[Tensor] = []
    assert target_sits.mask is not None
    nb_bands = pred_sits.shape()[2]
    maes: list[Tensor] = []
    mapes: list[Tensor] = []
    rmses: list[Tensor] = []

    for t in range(pred_sits.shape()[1]):
        # pylint: disable=W8201
        maes.clear()
        mapes.clear()
        rmses.clear()
        metric_mask = ~target_sits.mask[:, t, ...]
        # Apply spatial margin
        metric_mask[:, :spatial_margin, ...] = False
        metric_mask[:, -spatial_margin:, ...] = False
        metric_mask[:, :, :spatial_margin] = False
        metric_mask[:, :, -spatial_margin:] = False

        for b in range(nb_bands):
            pred_data = pred_sits.data[:, t, b, ...][metric_mask]
            target_data = target_sits.data[:, t, b, ...][metric_mask] / 10000

            mae = mean_absolute_error(pred_data, target_data)
            mape = mean_absolute_percentage_error(pred_data, target_data)
            rmse = mean_squared_error(pred_data, target_data, squared=False)
            maes.append(mae)
            mapes.append(mape)
            rmses.append(rmse)

        doy = pred_sits.doy[0, t]
        doy_str = (to_datetime(dt_orig) + timedelta(days=doy.item())).strftime(
            "%Y-%m-%d"
        )

        mask_percent = (
            target_sits.mask[0, t, ...].sum() / target_sits.mask[0, t, ...].numel()
        )
        print(
            f"{mode}\t{doy_str} ({doy}): RMSE={(sum(rmses)/nb_bands):.4f}\
             , MAE={(sum(maes)/nb_bands):.4f}, MAPE={(sum(mapes)/(100*nb_bands)):.2%},\
             (Masked={mask_percent:.1%})"
        )
        all_maes.append(maes)
        all_rmses.append(rmses)
        all_mapes.append(mapes)
        all_mask_percents.append(mask_percent)
    return (
        torch.tensor(all_rmses),
        torch.tensor(all_maes),
        torch.tensor(all_mapes),
        torch.tensor(all_mask_percents),
    )


def build_dataset(ts_path: str, width: int, config) -> SingleSITSDataset:
    """
    Build the dataset for a single sits
    """
    dataset = SingleSITSDataset(
        ts_path=ts_path,
        patch_size=width,
        dt_orig=config.datamodule.config.dt_orig,
        hr_resolution=config.datamodule.config.hr_resolution,
        hr_bands=(
            config.datamodule.config.hr_bands
            if "hr_bands" in config.datamodule.config
            else (None,)
        ),
        lr_bands=(
            config.datamodule.config.lr_bands
            if "lr_bands" in config.datamodule.config
            else (None,)
        ),
        lr_index_files=(
            config.datamodule.config.lr_index_files
            if "lr_index_files" in config.datamodule.config
            else ("index.csv",)
        ),
        hr_index_files=(
            config.datamodule.config.hr_index_files
            if "hr_index_files" in config.datamodule.config
            else ("index.csv",)
        ),
        hr_sensor=config.datamodule.config.hr_sensor,
        lr_resolution=config.datamodule.config.lr_resolution,
        lr_sensor=config.datamodule.config.lr_sensor,
        mtf_for_downsampling=config.datamodule.config.mtf_for_downsampling,
        random_reduce_rate=None,
        conjunctions_only=False,
        max_nb_dates=None,
    )

    return dataset


class MeasureExecTime:
    """
    A context manager that prints the execution time of the wrapped
    code block.
    """

    # https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    def __init__(self, label: str, log_time: bool):
        """
        Class initializer
        """
        self.label = label
        self.log_time = log_time
        self.start = 0.0
        self.time = 0.0
        self.readout = ""

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, _, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"{self.label} exec time: {self.time:.3f} s"
        if self.log_time:
            print(self.readout)


def generate_animation(
    input_lr: MonoModalSITS | None,
    input_hr: MonoModalSITS | None,
    pred_lr: MonoModalSITS | None,
    pred_hr: MonoModalSITS | None,
    target_lr: MonoModalSITS | None,
    target_hr: MonoModalSITS | None,
    lr_bands: list[int],
    hr_bands: list[int],
    out_file: str,
    figsize: tuple[int, int] = (12, 10),
):
    """
    Generate animation of fused data
    """
    matplotlib.rcParams["animation.embed_limit"] = 2**128
    all_dates = torch.unique(
        torch.cat([s.doy.ravel() for s in (pred_lr, pred_hr) if s is not None])
    )

    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(5, 6, figure=fig)
    ax_input_lr = fig.add_subplot(gs[0:2, 0:2])
    ax_input_lr.set_title("LR input")
    ax_input_lr.set_visible(False)
    ax_pred_lr = fig.add_subplot(gs[0:2, 2:4])
    ax_pred_lr.set_title("LR prediction")
    ax_pred_lr.set_visible(False)
    ax_target_lr = fig.add_subplot(gs[0:2, 4:6])
    ax_target_lr.set_title("LR target")
    ax_target_lr.set_visible(False)

    ax_input_hr = fig.add_subplot(gs[2:4, 0:2])
    ax_input_hr.set_title("HR input")
    ax_input_hr.set_visible(False)
    ax_pred_hr = fig.add_subplot(gs[2:4, 2:4])
    ax_pred_hr.set_title("HR prediction")
    ax_pred_hr.set_visible(False)
    ax_target_hr = fig.add_subplot(gs[2:4, 4:6])
    ax_target_hr.set_title("HR target")
    ax_target_hr.set_visible(False)
    timeline = fig.add_subplot(gs[4:, :])
    timeline.set_title("Timeline")

    yticks = (1, 2, 3, 4, 5)
    yticks_labels = ("Current", "HR target", "HR input", "LR target", "LR input")
    timeline.set_ylabel("")
    timeline.set_yticks(yticks)
    timeline.set_yticklabels(yticks_labels)
    timeline.set_xlabel("Day of year")

    timeline.set_ylim(0.5, 5.5)
    timeline.grid(True)
    if input_lr is not None:
        timeline.scatter(
            input_lr.doy.cpu().numpy()[0, ...],
            y=np.full_like(input_lr.doy.cpu().numpy()[0, ...], 5),
            marker=".",
            color="blue",
        )
    if target_lr is not None:
        timeline.scatter(
            target_lr.doy.cpu().numpy()[0, ...],
            y=np.full_like(target_lr.doy.cpu().numpy()[0, ...], 4),
            marker="*",
            color="blue",
        )
    if input_hr is not None:
        timeline.scatter(
            input_hr.doy.cpu().numpy()[0, ...],
            y=np.full_like(input_hr.doy.cpu().numpy()[0, ...], 3),
            marker=".",
            color="green",
        )
    if target_hr is not None:
        timeline.scatter(
            target_hr.doy.cpu().numpy()[0, ...],
            y=np.full_like(target_hr.doy.cpu().numpy()[0, ...], 2),
            marker="*",
            color="green",
        )

    def update_fig(frame_number):
        d = all_dates[frame_number]
        timeline.scatter(d.cpu().numpy(), y=[1], marker=".", color="red")

        for sits, ax, bands, scale in (
            (input_lr, ax_input_lr, lr_bands, 1500),
            (input_hr, ax_input_hr, hr_bands, 1500),
            (pred_lr, ax_pred_lr, lr_bands, 0.15),
            (pred_hr, ax_pred_hr, hr_bands, 0.15),
            (target_lr, ax_target_lr, lr_bands, 0.15),
            (target_hr, ax_target_hr, hr_bands, 0.15),
        ):
            if sits is not None:
                ax.set_visible(True)
                if d in sits.doy.ravel():
                    d_idx = torch_argwhere(sits.doy == d)[0]
                    rgb = sits.data[0, d_idx[1], bands, ...]
                    rgb = torch_clip(rgb / scale, 0.0, 1.0)
                    ax.imshow(rgb.permute(1, 2, 0).cpu().numpy())
                else:
                    ax.imshow(np_zeros((sits.data.shape[-2], sits.data.shape[-1], 3)))

    ani = animation.FuncAnimation(fig, update_fig, all_dates.shape[0], interval=1000)
    with open(out_file, "w", encoding="utf8") as f:
        print(ani.to_jshtml(), file=f)


def tiled_inference(
    lr_sits: MonoModalSITS | None,
    hr_sits: MonoModalSITS | None,
    model: TemporalInterpolationTrainingModule,
    target_lr_doy: torch.Tensor | None,
    target_hr_doy: torch.Tensor | None,
    width: int = 165,
    margin: int = 18,
    show_progress: bool = True,
):
    """
    Tiled inference
    """
    lr_width = width / 3
    assert lr_width.is_integer()
    lr_width = int(lr_width)
    lr_margin = margin / 3
    assert lr_margin.is_integer()
    lr_margin = int(lr_margin)

    hr_tiled_data = (
        patchify(hr_sits.data, patch_size=width, margin=margin)
        if hr_sits is not None
        else None
    )
    lr_tiled_data = (
        patchify(lr_sits.data, patch_size=lr_width, margin=lr_margin)
        if lr_sits is not None
        else None
    )

    hr_tiled_size = (
        parse_shape(hr_tiled_data, "n m b t c w h")
        if hr_tiled_data is not None
        else None
    )
    lr_tiled_size = (
        parse_shape(lr_tiled_data, "n m b t c w h")
        if lr_tiled_data is not None
        else None
    )

    hr_tiled_data = (
        rearrange(hr_tiled_data, "n m b t c w h -> (n m) b t c w h")
        if hr_tiled_data is not None
        else None
    )
    lr_tiled_data = (
        rearrange(lr_tiled_data, "n m b t c w h -> (n m) b t c w h")
        if lr_tiled_data is not None
        else None
    )

    def inference(current_lr: MonoModalSITS | None, current_hr: MonoModalSITS | None):
        """
        Infer a single tile, handling transfert between cpu and gpu
        """

        device = model.lr_mean.device
        out_lr, out_hr = model.predict(
            current_lr.to(device=device) if current_lr is not None else None,
            current_hr.to(device=device) if current_hr is not None else None,
            lr_query_doy=(
                target_lr_doy.to(device=device) if target_lr_doy is not None else None
            ),
            hr_query_doy=(
                target_hr_doy.to(device=device) if target_hr_doy is not None else None
            ),
            downscale=False,
        )
        return out_lr, out_hr

    if lr_tiled_data is None and hr_tiled_data is not None:
        assert hr_sits is not None
        out = [
            inference(None, MonoModalSITS(hr_data, hr_sits.doy))
            for hr_data in tqdm(
                hr_tiled_data,
                total=hr_tiled_data.shape[0],
                desc="Inference",
                disable=None if show_progress else True,
            )
        ]
    elif lr_tiled_data is not None and hr_tiled_data is None:
        assert lr_sits is not None
        out = [
            inference(MonoModalSITS(lr_data, lr_sits.doy), None)
            for lr_data in tqdm(
                lr_tiled_data,
                total=lr_tiled_data.shape[0],
                desc="Inference",
                disable=None if show_progress else True,
            )
        ]
    else:
        assert hr_tiled_data is not None
        assert lr_tiled_data is not None
        assert lr_sits is not None
        assert hr_sits is not None
        out = [
            inference(
                MonoModalSITS(lr_data, lr_sits.doy), MonoModalSITS(hr_data, hr_sits.doy)
            )
            for lr_data, hr_data in tqdm(
                zip(lr_tiled_data, hr_tiled_data),
                total=lr_tiled_data.shape[0],
                desc="Inference",
                disable=None if show_progress else True,
            )
        ]

    # Crop margin
    out = [
        (
            crop_sits(lr_sits, margin) if lr_sits else None,
            crop_sits(hr_sits, margin) if hr_sits else None,
        )
        for lr_sits, hr_sits in out
    ]

    out_tiled_size = hr_tiled_size if hr_tiled_size is not None else lr_tiled_size
    assert out_tiled_size is not None
    out_lr_sits = (
        MonoModalSITS(
            rearrange(
                torch.stack([lr_sits.data for lr_sits, _ in out]),
                "(n m) b t c w h -> b t c (n w) (m h)",
                n=out_tiled_size["n"],
                m=out_tiled_size["m"],
            ),
            target_lr_doy,
        )
        if target_lr_doy is not None
        else None
    )
    out_hr_sits = (
        MonoModalSITS(
            rearrange(
                torch.stack([hr_sits.data for _, hr_sits in out]),
                "(n m) b t c w h -> b t c (n w) (m h)",
                n=out_tiled_size["n"],
                m=out_tiled_size["m"],
            ),
            target_hr_doy,
        )
        if target_hr_doy is not None
        else None
    )

    return out_lr_sits, out_hr_sits
