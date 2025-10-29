# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
Encapsulate Data Mining Sharpener for testing
"""
import multiprocessing
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import rasterio as rio  # type: ignore
import torch
from affine import Affine  # type: ignore
from pyDMS.pyDMS import DecisionTreeSharpener  # type: ignore
from pyDMS.pyDMSUtils import saveImg  # type: ignore

from tamrfsits.core.downsampling import downsample_sits
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.core.utils import find_closest_in_sits


class HiddenPrints:
    """
    From: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __init__(self):
        self._original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = open(os.devnull, "w", encoding="utf8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def write_image(data: torch.Tensor, file_path: Path, dtype: str, resolution: float):
    """
    Write Image to disk
    """
    geotransform = (0, resolution, 0.0, 0, 0.0, -resolution)
    transform = Affine.from_gdal(*geotransform)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "crs": {"init": "epsg:4326"},  # Dummy CRS
        "dtype": dtype,
        "transform": transform,
    }

    with rio.open(file_path, "w", **profile) as f:
        f.write(data.cpu().numpy())


def dms_single_step_predict(
    lr_data: torch.Tensor,
    hr_data: torch.Tensor,
    lr_mask: torch.Tensor,
    hr_mask: torch.Tensor,
    tmp: str,
    residual_compensation: bool = True,
):
    """
    Single predict step to be called by multiprocessing
    """
    # Step 1: write images to temporary folder
    with NamedTemporaryFile(dir=tmp) as lr_tiff_file:
        with NamedTemporaryFile(dir=tmp) as hr_tiff_file:
            with NamedTemporaryFile(dir=tmp) as mask_file:
                with NamedTemporaryFile(dir=tmp) as out_file:
                    write_image(
                        lr_data,
                        Path(lr_tiff_file.name),
                        dtype="uint16",
                        resolution=9.0,
                    )
                    write_image(
                        hr_data,
                        Path(hr_tiff_file.name),
                        dtype="uint16",
                        resolution=1.0,
                    )
                    write_image(
                        (
                            255
                            * (
                                torch.logical_or(
                                    (lr_mask[None, ...]),
                                    (hr_mask[None, ...]),
                                )
                            ).to(dtype=torch.uint8)
                        ),
                        Path(mask_file.name),
                        dtype="uint8",
                        resolution=9.0,
                    )

                    # Step 2: call DMS
                    opts = {
                        "highResFiles": [hr_tiff_file.name],
                        "lowResFiles": [lr_tiff_file.name],
                        "lowResQualityFiles": [mask_file.name],
                        "lowResGoodQualityFlags": [0],
                        "cvHomogeneityThreshold": 0,
                        "movingWindowSize": 15,
                        "disaggregatingTemperature": True,
                        "perLeafLinearRegression": True,
                        "linearRegressionExtrapolationRatio": 0.25,
                    }
                    with HiddenPrints():
                        disaggregator = DecisionTreeSharpener(**opts)
                        disaggregator.trainSharpener()
                        downscaled_file = disaggregator.applySharpener(
                            hr_tiff_file.name, lr_tiff_file.name
                        )
                        out_data = downscaled_file
                        if residual_compensation:
                            _, out_data = disaggregator.residualAnalysis(
                                downscaled_file,
                                lr_tiff_file.name,
                                doCorrection=True,
                            )

                        saveImg(
                            out_data.GetRasterBand(1).ReadAsArray(),
                            out_data.GetGeoTransform(),
                            out_data.GetProjection(),
                            out_file.name,
                        )

                        # Step 3: Read back result
                        with rio.open(out_file.name, "r") as f:
                            return torch.tensor(f.read())


def dms_predict(
    lst_sits: MonoModalSITS,
    hr_sits: MonoModalSITS,
    tmp_dir: str,
    nb_procs: int = 8,
    residual_compensation: bool = True,
):
    """
    Encapsulate DMS predictions
    """
    # Locate closest hr data
    closest_hr_sits = find_closest_in_sits(hr_sits, lst_sits.doy)
    # Go back to 90m
    lr_sits_up = downsample_sits(
        lst_sits.to(dtype=torch.float32),
        factor=1 / 3.0,
    )

    lst_sits = downsample_sits(
        lst_sits.to(dtype=torch.float32),
        factor=3.0,
    )
    hr_sits_down = downsample_sits(
        closest_hr_sits.to(dtype=torch.float32),
        factor=9.0,
    )
    out_list: list[torch.Tensor] = []

    # For each pair of hr/lst
    with TemporaryDirectory(dir=tmp_dir) as tmp:
        with multiprocessing.Pool(nb_procs) as pool:
            assert lst_sits.mask is not None
            assert hr_sits_down.mask is not None
            out_list = pool.starmap(
                dms_single_step_predict,
                [
                    (
                        lst_sits.data[0, i, ...],
                        closest_hr_sits.data[0, i, ...],
                        lst_sits.mask[0, i, ...],
                        hr_sits_down.mask[0, i, ...],
                        tmp,
                        residual_compensation,
                    )
                    for i in range(lst_sits.shape()[1])
                ],
            )

    # Export results
    out_data = torch.cat([t[None, None, ...] for t in out_list], dim=1)
    out_mask = torch.isnan(out_data[:, :, 0, ...])
    out_mask = torch.logical_or(out_mask, closest_hr_sits.mask)
    # Here, we should add this mask only if residual_compensation mode,
    # but we keep this as is to be able to make a fair comparison
    # between the two modes
    out_mask = torch.logical_or(out_mask, lr_sits_up.mask)
    return MonoModalSITS(
        out_data / 10000.0, lst_sits.doy.to(dtype=torch.int16), out_mask
    )
