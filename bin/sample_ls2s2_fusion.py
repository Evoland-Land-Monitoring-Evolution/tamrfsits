#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

import argparse
import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import cast

import numpy as np
import openeo  # type: ignore
import pandas as pd
import rasterio as rio  # type: ignore
import requests
import xarray as xr
from affine import Affine  # type: ignore
from pyproj.crs import CRS
from scipy.ndimage import (  # type: ignore
    affine_transform,
    binary_closing,
    binary_dilation,
    binary_opening,
)
from tqdm import tqdm

from tamrfsits.data.joint_sits_dataset import SingleSITSDataset

DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]


@dataclass(frozen=True)
class GlobalConfig:
    name: str
    bounds: rio.coords.BoundingBox
    crs: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    folder: str

    def __repr__(self) -> str:
        out = f"aoi: {self.bounds}, crs: {self.crs}, time_frame: \
        {self.start_date} -> {self.end_date}, "
        out += f"output folder: {self.folder}"
        return out


def extract_bitmask(mask: xr.DataArray | np.ndarray, bit: int = 0) -> np.ndarray:
    """
    Extract a binary mask from the nth bit of a bit-encoded mask

    :param mask: the bit encoded mask
    :param bit: the index of the bit to extract
    :return: A binary mask of the nth bit of mask, with the same shape
    """
    if isinstance(mask, xr.DataArray):
        return mask.values.astype(int) >> bit & 1
    return mask.astype(int) >> bit & 1


def download_file(url: str, dest: str):
    """
    From:
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    """
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024**2

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(dest, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


def mask_processing(
    input_mask: np.ndarray, min_object_size: int = 5, dilation: int = 50
):
    """
    Binary morphology to extend masks a bit
    """
    structuring_element = np.ones((min_object_size, min_object_size))
    # Remove smaller elements (often false detections)
    current_mask = binary_opening(
        input_mask, structure=structuring_element, iterations=1
    )
    # Dilate cloud mask
    structuring_element = np.ones((dilation, dilation))
    current_mask = binary_closing(
        current_mask, structure=structuring_element, iterations=1
    )
    current_mask = binary_dilation(
        current_mask, structure=structuring_element, iterations=1
    )

    return current_mask


def dilate_mask(input_mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    """
    Binary morphology to extend masks a bit
    """
    # Dilate cloud mask
    structuring_element = np.ones((dilation, dilation))
    current_mask = binary_dilation(
        input_mask, structure=structuring_element, iterations=1
    )

    return current_mask


def read_json_configuration(json_path: str) -> GlobalConfig:
    # open json file with parameter
    with open(json_path) as f:
        params = json.load(f)

        name = params["name"]
        aoi = params["aoi"]
        start_date = params["start_date"]
        crs = params["crs"]
        end_date = params["end_date"]
        bounds = rio.coords.BoundingBox(
            left=aoi[0], bottom=aoi[1], right=aoi[2], top=aoi[3]
        )
        return GlobalConfig(name, bounds, crs, start_date, end_date, folder="")


def netcdf_to_geotiff(
    arr: xr.Dataset,
    output_path: str,
    bands: list[str],
    sensor: str = "sentinel2",
    what: str = "bands",
    dtype: str = "int16",
    nodata: float = -10000,
    fillna: bool = True,
    scale: float = 1.0,
):
    profile = rio.profiles.DefaultGTiffProfile(count=len(bands))
    profile["crs"] = arr.crs.crs_wkt
    profile["transform"] = Affine.from_gdal(
        arr.x.min().values - 0.5 * (arr.x[1] - arr.x[0]).values,
        (arr.x[1] - arr.x[0]).values,
        0.0,
        arr.y.max().values - 0.5 * (arr.y[1] - arr.y[0]).values,
        0.0,
        (arr.y[1] - arr.y[0]).values,
    )
    profile["width"] = len(arr.x)
    profile["height"] = len(arr.y)
    profile["dtype"] = dtype
    profile["nodata"] = nodata
    if fillna:
        arr = arr.fillna(nodata)

    product_ids = []
    files = []

    for d in arr.t:
        datestamp = str(d.dt.strftime("%Y%m%d").values)
        product_id = f"{sensor}_{datestamp}"
        if not Path(os.path.join(output_path, datestamp)).is_dir():
            Path(os.path.join(output_path, datestamp)).mkdir(parents=True)
        current_path = os.path.join(
            output_path, datestamp, f"{sensor}_{what}_{datestamp}.tif"
        )
        with rio.open(current_path, "w", **profile) as dst_ds:
            for bid, b in enumerate(bands):
                nan_mask = np.isnan(arr.sel(t=d.values)[b].values)
                data_to_write = scale * arr.sel(t=d.values)[b].values

                data_to_write[nan_mask] = nodata
                data_to_write = data_to_write.astype(np.int16)
                dst_ds.write(data_to_write, bid + 1)
        files.append(os.path.relpath(current_path, output_path))
        product_ids.append(product_id)

    return (product_ids, files)


def open_xarray(arr, date, bands=DEFAULT_BANDS):
    return np.concatenate([[arr.sel(t=date)[b].values] for b in bands], 0)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Generate sls2s2 time-series"
    )

    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument(
        "--json",
        type=str,
        nargs="+",
        required=True,
        help="Path to the json file describing the dataset to generate",
    )

    parser.add_argument(
        "--mode",
        choices=["submit", "download", "extract", "clean", "check"],
        help="Processing step",
        default="submit",
    )

    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["sentinel2", "landsat", "landsat_pan"],
        choices=["sentinel2", "landsat", "landsat_pan"],
    )
    parser.add_argument(
        "--output", type=str, help="Ouptut directory for data", required=True
    )

    parser.add_argument(
        "--s2_qa", action="store_true", help="Write QA layers for Sentinel2"
    )

    parser.add_argument(
        "--ls_qa", action="store_true", help="Write QA layers for Landsat"
    )

    parser.add_argument(
        "--disable_animation", action="store_true", help="Disable animation generation"
    )

    parser.add_argument(
        "--min_clear_pixel_rate",
        default=0.25,
        type=float,
        help="Minimum clear pixel rate",
    )

    parser.add_argument(
        "--max_product_cloud_cover",
        type=int,
        default=75,
        help="Maximum cloud cover for product to be included in datacube",
    )

    parser.add_argument(
        "--override", action="store_true", help="Override existing files"
    )

    return parser


def submit(args):
    """
    submit step
    """
    # # First, connect to openeo
    connection = openeo.connect("https://openeo.vito.be/openeo/1.2").authenticate_oidc()

    for json_file in args.json:
        logging.info(f"Processing json file {json_file}")
        job_ids: dict[str, str] = {}
        # Read configuration
        cfg = read_json_configuration(json_file)

        # Create output dir if not already existing
        folder = os.path.join(args.output, cfg.name)
        if not Path(folder).is_dir():
            Path(folder).mkdir(parents=True)

        jobs_json_file = os.path.join(folder, "openeo_jobs.json")
        if Path(jobs_json_file).exists():
            with open(jobs_json_file) as f:
                job_ids = json.load(f)

        # Copy json to output folder
        shutil.copyfile(json_file, os.path.join(folder, os.path.basename(json_file)))

        if "sentinel2" in args.sensors:
            if args.override or "sentinel2" not in job_ids:
                # Build s2 cube
                s2_cube = connection.load_collection(
                    "SENTINEL2_L2A_SENTINELHUB",
                    spatial_extent={
                        "west": cfg.bounds[0],
                        "south": cfg.bounds[1],
                        "east": cfg.bounds[2],
                        "north": cfg.bounds[3],
                        "crs": cfg.crs,
                    },
                    temporal_extent=[cfg.start_date, cfg.end_date],
                    bands=[
                        "B02",
                        "B03",
                        "B04",
                        "B05",
                        "B06",
                        "B07",
                        "B08",
                        "B8A",
                        "B11",
                        "B12",
                        "SCL",
                        "CLM",
                        "CLP",
                        "AOT",
                        "SNW",
                        "dataMask",
                    ],
                    max_cloud_cover=args.max_product_cloud_cover,
                )

                # Build download job
                s2_download_job = (
                    s2_cube.resample_spatial(resolution=10.0, projection=cfg.crs)
                    .save_result("NetCDF")
                    .create_job(title=f"Sentinel2_{cfg.name}")
                )
                s2_download_job.start()
                logging.info(f"Sent job {s2_download_job.job_id}")
                # Keep track of job id
                job_ids["sentinel2"] = s2_download_job.job_id
            else:
                logging.info(
                    f"Sentinel-2 job already exists for json {json_file} and \
                    override is deactivated, skipping"
                )
        if "landsat" in args.sensors:
            if args.override or "landsat" not in job_ids:
                # Build Landsat-8 s2_cube
                ls_cube = connection.load_collection(
                    "LANDSAT8-9_L2",
                    spatial_extent={
                        "west": cfg.bounds[0],
                        "south": cfg.bounds[1],
                        "east": cfg.bounds[2],
                        "north": cfg.bounds[3],
                        "crs": cfg.crs,
                    },
                    temporal_extent=[cfg.start_date, cfg.end_date],
                    bands=[
                        "B01",
                        "B02",
                        "B03",
                        "B04",
                        "B05",
                        "B06",
                        "B07",
                        "B10",
                        "ST_EMIS",
                        "BQA",
                        "ST_CDIST",
                        "QA_RADSAT",
                        "SR_QA_AEROSOL",
                    ],
                    max_cloud_cover=args.max_product_cloud_cover,
                )

                ls_download_job = (
                    ls_cube.resample_spatial(resolution=30.0, projection=cfg.crs)
                    .save_result("NetCDF")
                    .create_job(title=f"Landsat_{cfg.name}")
                )
                ls_download_job.start()
                logging.info(f"Sent job {ls_download_job.job_id}")
                job_ids["landsat"] = ls_download_job.job_id
            else:
                logging.info(
                    f"Landsat job already exists for json {json_file} and \
                    override is deactivated, skipping"
                )

        if "landsat_pan" in args.sensors:
            if args.override or "landsat_pan" not in job_ids:
                # Build Landsat-8 s2_cube
                lspan_cube = connection.load_collection(
                    "LANDSAT8-9_L1",
                    spatial_extent={
                        "west": cfg.bounds[0],
                        "south": cfg.bounds[1],
                        "east": cfg.bounds[2],
                        "north": cfg.bounds[3],
                        "crs": cfg.crs,
                    },
                    temporal_extent=[cfg.start_date, cfg.end_date],
                    bands=["B08"],
                    max_cloud_cover=args.max_product_cloud_cover,
                )

                lspan_download_job = (
                    lspan_cube.resample_spatial(resolution=15.0, projection=cfg.crs)
                    .save_result("NetCDF")
                    .create_job(title=f"Landsat_pan_{cfg.name}")
                )
                lspan_download_job.start()
                logging.info(f"Sent job {lspan_download_job.job_id}")
                job_ids["landsat_pan"] = lspan_download_job.job_id
            else:
                logging.info(
                    f"Landsat PAN job already exists for json {json_file} and \
                    override is deactivated, skipping"
                )

        Path(jobs_json_file).write_text(json.dumps(job_ids))


def download(args):
    """
    Download step
    """
    # # First, connect to openeo
    connection = openeo.connect("https://openeo.vito.be/openeo/1.2").authenticate_oidc()

    for json_file in args.json:
        logging.info(f"Processing json file {json_file}")
        job_ids = {}

        # Read configuration
        cfg = read_json_configuration(json_file)

        # Create output dir if not already existing
        folder = os.path.join(args.output, cfg.name)
        if not Path(folder).is_dir():
            Path(folder).mkdir(parents=True)

        # Download results
        with open(os.path.join(folder, "openeo_jobs.json")) as f:
            job_ids = json.load(f)

        for sensor in args.sensors:
            if sensor not in job_ids:
                raise LookupError(f"Sensor {sensor} not found in Job ids")
            else:
                nc_file = os.path.join(folder, f"{sensor}.nc")
                if not Path(nc_file).exists() or args.override:
                    # Reconnect to jobs
                    job = connection.job(job_ids[sensor])

                    logging.info(f"{sensor} job: {job.status()}")

                    if job.status() != "finished":
                        logging.error(f"{sensor} job is not ready for download")
                    else:
                        nc_file = os.path.join(folder, f"{sensor}.nc")
                        download_file(job.get_results().get_asset().href, nc_file)
                else:
                    logging.info(
                        f"{sensor} already downloaded and override is "
                        "deactivated, skipping"
                    )


def geographical_slicing(arr: xr.Dataset, cfg: dict, crop: int = 990) -> xr.Dataset:
    """
    Perform geographical slicing.
    Applies bounds from cfg if utm zone matches the requested utm zone.
    Otherwise crop the data with provided crop parameter

    """
    retrieved_epsg = CRS(arr.crs.crs_wkt).to_epsg()
    requested_epsg = int(cfg.crs[5:])

    if retrieved_epsg == requested_epsg:
        bounds = cfg.bounds
    else:
        logging.warning(
            f"Retrieved epsg {retrieved_epsg} does not match requested epsg {requested_epsg}"
        )

        bounds = [arr.x[0], arr.y[crop - 1], arr.x[crop - 1], arr.y[0]]

    arr = arr.sel(
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[3], bounds[1]),
    )
    return arr


def extract_s2(s2_nc_file, cfg, folder, args):
    """
    Extract S2 data
    """
    # Load the s2 datacube
    s2_arr = xr.open_dataset(s2_nc_file, mask_and_scale=False)

    s2_arr = geographical_slicing(s2_arr, cfg, 990)

    # Build no-data mask from Scene classification layer
    no_data_mask = s2_arr.SCL.astype(np.uint8).isin([0, 1, 2, 3, 7, 8, 9, 10])
    no_data_mask = np.logical_or(no_data_mask, s2_arr["dataMask"] == 0)
    no_data_mask = np.logical_or(no_data_mask, s2_arr["CLM"] > 0)
    no_data_mask = np.logical_or(no_data_mask, s2_arr["CLP"] > 150)

    s2_arr["snow"] = s2_arr["SCL"] == 11

    mask_stack = []
    # Perform mask dilation since sen2corr mask are very tight
    for t in s2_arr.t:
        current_mask = no_data_mask.sel(t=t).values
        mask_stack.append(
            mask_processing(current_mask, min_object_size=10, dilation=25)
        )

    mask_stack = np.stack(mask_stack)

    nan_mask = np.isnan(s2_arr["B02"].values)
    for b in (
        "B03",
        "B04",
        "B08",
        "B05",
        "B06",
        "B07",
        "B8A",
        "B11",
        "B12",
    ):
        nan_mask = np.logical_or(nan_mask, np.isnan(s2_arr[b].values))

    # Introduce nan mask here since we do not want to dilate nan mask
    s2_arr["no_data"] = ("t", "y", "x"), np.logical_or(nan_mask, mask_stack)

    # Compute clear pixel rate
    del nan_mask
    del mask_stack
    s2_clear_pixel_rate = 1 - (
        s2_arr["no_data"].sum(dim=("x", "y")) / (len(s2_arr.x) * len(s2_arr.y))
    )
    s2_dark_pixel_rate = (s2_arr["SCL"] == 2).sum(dim=("x", "y")) / (
        len(s2_arr.x) * len(s2_arr.y)
    )
    s2_snow_pixel_rate = (s2_arr["SCL"] == 11).sum(dim=("x", "y")) / (
        len(s2_arr.x) * len(s2_arr.y)
    )

    # Create output dir if not already existing
    s2_dir = os.path.join(folder, "sentinel2")
    if not Path(s2_dir).is_dir():
        Path(s2_dir).mkdir(parents=True)

    # if index.csv exists in the folder, delete it to avoid problems
    if Path(os.path.join(s2_dir, "index.csv")).exists():
        Path(os.path.join(s2_dir, "index.csv")).unlink()

    # Build a dataframe with info
    s2_dict = {
        "product_id": ("sentinel2_" + s2_arr.t.dt.strftime("%Y%m%d")).values,
        "acquisition_date": s2_arr.t.dt.strftime("%Y-%m-%d").values,
        "clear_pixel_rate": s2_clear_pixel_rate.values,
        "dark_pixel_rate": s2_dark_pixel_rate,
        "snow_pixel_rate": s2_snow_pixel_rate,
        "valid": s2_clear_pixel_rate.values >= args.min_clear_pixel_rate,
        "bands": [None for _ in range(len(s2_arr.t))],
        "mask": [None for _ in range(len(s2_arr.t))],
        "snow_mask": [None for _ in range(len(s2_arr.t))],
    }
    s2_index = pd.DataFrame(s2_dict).set_index("product_id", drop=False)

    # Filter out invalid dates
    clear_dates = s2_arr.t[s2_clear_pixel_rate >= args.min_clear_pixel_rate]
    if not len(clear_dates):
        logging.warning(f"No clear sentinel2 dates found for AOI {cfg.name}")
        s2_index.to_csv(os.path.join(s2_dir, "index.csv"), sep="\t")
        return

    s2_arr = s2_arr.where(
        s2_arr.t.isin(s2_arr.t[s2_clear_pixel_rate > args.min_clear_pixel_rate]),
        drop=True,
    )

    # Fix nearest neighbours resampling of 20m bands done in OpenEO
    x_offset = -0.25 + divmod((s2_arr.x.min().values - 5) / 20.0, 1.0)[1]
    y_offset = -0.25 + divmod((s2_arr.y.min().values - 5) / 20.0, 1.0)[1]
    for b in (
        "B05",
        "B06",
        "B07",
        "B8A",
        "B11",
        "B12",
    ):
        s2_arr[b] = ("t", "x", "y"), affine_transform(
            s2_arr[b].to_numpy()[:, ::2, ::2],
            np.diag([1.0, 0.5, 0.5]),
            offset=[0.0, y_offset, x_offset],
            output_shape=s2_arr[b].to_numpy().shape,
            order=3,
            prefilter=True,
            mode="reflect",
        )
    product_ids, files = netcdf_to_geotiff(
        s2_arr,
        s2_dir,
        bands=[
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ],
        nodata=-10000,
        fillna=False,
        sensor="sentinel2",
        what="bands",
    )

    if args.s2_qa:
        _, _ = netcdf_to_geotiff(
            s2_arr,
            s2_dir,
            bands=[
                "AOT",
                "CLM",
                "CLP",
                "SCL",
                "dataMask",
            ],
            nodata=-10000,
            fillna=False,
            sensor="sentinel2",
            what="qa",
        )

    s2_index.loc[product_ids, "bands"] = files

    product_ids, files = netcdf_to_geotiff(
        s2_arr,
        s2_dir,
        bands=["no_data"],
        sensor="sentinel2",
        nodata=0,
        dtype="uint8",
        what="mask",
    )
    s2_index.loc[product_ids, "mask"] = files

    product_ids, files = netcdf_to_geotiff(
        s2_arr,
        s2_dir,
        bands=["snow"],
        sensor="sentinel2",
        nodata=0,
        dtype="uint8",
        what="snow_mask",
    )
    s2_index.loc[product_ids, "snow_mask"] = files

    s2_index.to_csv(os.path.join(s2_dir, "index.csv"), sep="\t")


def extract_ls(ls_nc_file, cfg, folder, args):
    """
    Extract Landsat
    """
    # Load the landsat datacube
    ls_arr = xr.open_dataset(ls_nc_file, mask_and_scale=False)

    ls_arr = geographical_slicing(ls_arr, cfg, 330)

    ls_arr["no_data"] = ("t", "y", "x"), (
        extract_bitmask(ls_arr.BQA, 0)  # fill values
        + extract_bitmask(ls_arr.BQA, 1)  # dilated cloud
        + extract_bitmask(ls_arr.BQA, 2)  # cirrus
        + extract_bitmask(ls_arr.BQA, 3)  # cloud
        + extract_bitmask(ls_arr.BQA, 4)  # cloud shadow
        # + (ls_arr.ST_CDIST.values < 1.0).astype(int)  # distance to cloud < 500m
        + (
            extract_bitmask(ls_arr.SR_QA_AEROSOL, 1)  # Check if aerosols are valid
            * extract_bitmask(ls_arr.SR_QA_AEROSOL, 6)
            * extract_bitmask(ls_arr.SR_QA_AEROSOL, 7)
        )  # No high aerosol
        + extract_bitmask(ls_arr.QA_RADSAT, 0)  # no saturation
        + extract_bitmask(ls_arr.QA_RADSAT, 1)
        + extract_bitmask(ls_arr.QA_RADSAT, 2)
        + extract_bitmask(ls_arr.QA_RADSAT, 3)
        + extract_bitmask(ls_arr.QA_RADSAT, 4)
        + extract_bitmask(ls_arr.QA_RADSAT, 5)
        + extract_bitmask(ls_arr.QA_RADSAT, 6)
        + extract_bitmask(ls_arr.QA_RADSAT, 11)  # no terrain occlusion
    ) > 0

    ls_arr["snow"] = ("t", "y", "x"), extract_bitmask(ls_arr.BQA, 5) > 0

    # Build mask of data that are zero
    zero_mask = cast(
        xr.DataArray,
        (
            sum(
                ls_arr[b] == 0
                for b in (
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B10",
                )
            )
            > 0
        ),
    )

    zero_mask_stack: list[np.ndarray] = []
    # Perform mask dilation since sen2corr mask are very tight
    for t in ls_arr.t:
        current_mask = zero_mask.sel(t=t).values
        zero_mask_stack.append(dilate_mask(current_mask, dilation=10))

    zero_mask_stack = np.stack(zero_mask_stack)

    ls_arr["no_data"] = ("t", "y", "x"), np.logical_or(
        ls_arr["no_data"].values, zero_mask_stack
    )

    # Build no-data mask
    nan_mask = np.isnan(ls_arr["B02"].values)
    for b in (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B10",
    ):
        nan_mask = np.logical_or(nan_mask, np.isnan(ls_arr[b].values))

    # Perform mask dilation
    mask_stack = []
    for t in ls_arr.t:
        current_mask = ls_arr["no_data"].sel(t=t).values
        mask_stack.append(mask_processing(current_mask, min_object_size=5, dilation=1))
    mask_stack = np.stack(mask_stack)
    # Introduce nan mask here since we do not want to dilate nan mask
    ls_arr["no_data"] = ("t", "y", "x"), np.logical_or(nan_mask, mask_stack)

    ls_clear_pixel_rate = 1 - (
        ls_arr["no_data"].sum(dim=("x", "y")) / (len(ls_arr.x) * len(ls_arr.y))
    )

    # Compute snow pixel rate
    ls_snow_pixel_rate = (ls_arr["snow"]).sum(dim=("x", "y")) / (
        len(ls_arr.x) * len(ls_arr.y)
    )

    # Create output dir if not already existing
    ls_dir = os.path.join(folder, "landsat")
    if not Path(ls_dir).is_dir():
        Path(ls_dir).mkdir(parents=True)

    # if index.csv exists in the folder, delete it to avoid problems
    if Path(os.path.join(ls_dir, "index.csv")).exists():
        Path(os.path.join(ls_dir, "index.csv")).unlink()

    # Build a dataframe with info
    ls_dict = {
        "product_id": ("landsat_" + ls_arr.t.dt.strftime("%Y%m%d")).values,
        "acquisition_date": ls_arr.t.dt.strftime("%Y-%m-%d").values,
        "clear_pixel_rate": ls_clear_pixel_rate.values,
        "snow_pixel_rate": ls_snow_pixel_rate.values,
        "valid": ls_clear_pixel_rate.values >= args.min_clear_pixel_rate,
        "bands": [None for _ in range(len(ls_arr.t))],
        "mask": [None for _ in range(len(ls_arr.t))],
        "snow_mask": [None for _ in range(len(ls_arr.t))],
    }
    ls_index = pd.DataFrame(ls_dict).set_index("product_id", drop=False)

    # Filter out invalid dates
    clear_dates = ls_arr.t[ls_clear_pixel_rate >= args.min_clear_pixel_rate]
    if not len(clear_dates):
        logging.warning(f"No clear landsat dates found for AOI {cfg.name}")
        ls_index.to_csv(os.path.join(ls_dir, "index.csv"), sep="\t")
        return

    ls_arr = ls_arr.where(
        ls_arr.t.isin(clear_dates),
        drop=True,
    )

    # Ugly patch because B10 is not scaled
    ls_arr["B10"] = ls_arr["B10"] / 1000

    # Fix half pixel grid shift
    for b in (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B10",
    ):
        ls_arr[b] = ("t", "x", "y"), affine_transform(
            ls_arr[b].to_numpy(),
            np.eye(3),
            offset=[0.0, -0.5, -0.5],
            order=3,
            prefilter=True,
            mode="reflect",
        )
    ls_arr["no_data"] = ("t", "x", "y"), affine_transform(
        ls_arr["no_data"].to_numpy(),
        np.eye(3),
        offset=[0.0, -0.5, -0.5],
        order=0,
        prefilter=False,
        mode="reflect",
    )
    ls_arr["snow"] = ("t", "x", "y"), affine_transform(
        ls_arr["snow"].to_numpy(),
        np.eye(3),
        offset=[0.0, -0.5, -0.5],
        order=0,
        prefilter=False,
        mode="reflect",
    )

    product_ids, files = netcdf_to_geotiff(
        ls_arr,
        ls_dir,
        bands=[
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B10",
        ],
        nodata=-10000,
        fillna=False,
        sensor="landsat",
        what="bands",
        scale=10000,
    )

    ls_index.loc[product_ids, "bands"] = files

    product_ids, files = netcdf_to_geotiff(
        ls_arr,
        ls_dir,
        bands=["no_data"],
        sensor="landsat",
        fillna=False,
        scale=1.0,
        nodata=0,
        dtype="uint8",
        what="mask",
    )
    ls_index.loc[product_ids, "mask"] = files

    product_ids, files = netcdf_to_geotiff(
        ls_arr,
        ls_dir,
        bands=["snow"],
        sensor="landsat",
        fillna=False,
        scale=1.0,
        nodata=0,
        dtype="uint8",
        what="snow_mask",
    )
    ls_index.loc[product_ids, "snow_mask"] = files

    if args.ls_qa:
        product_ids, files = netcdf_to_geotiff(
            ls_arr,
            ls_dir,
            bands=["BQA"],
            sensor="landsat",
            fillna=False,
            scale=1.0,
            nodata=0,
            dtype="uint16",
            what="qa",
        )

        product_ids, files = netcdf_to_geotiff(
            ls_arr,
            ls_dir,
            bands=["ST_CDIST"],
            sensor="landsat",
            fillna=False,
            scale=1.0,
            nodata=0,
            dtype="float32",
            what="st_cdist",
        )

        product_ids, files = netcdf_to_geotiff(
            ls_arr,
            ls_dir,
            bands=["QA_RADSAT"],
            sensor="landsat",
            fillna=False,
            scale=1.0,
            nodata=0,
            dtype="uint16",
            what="qa_radsat",
        )
        product_ids, files = netcdf_to_geotiff(
            ls_arr,
            ls_dir,
            bands=["SR_QA_AEROSOL"],
            sensor="landsat",
            fillna=False,
            scale=1.0,
            nodata=0,
            dtype="uint16",
            what="sr_qa_aerosol",
        )

    ls_index.to_csv(os.path.join(ls_dir, "index.csv"), sep="\t")


def extract_ls_pan(ls_pan_nc_file, cfg, folder, args):
    """
    Extract Landsat
    """
    # Load the landsat datacube
    ls_arr = xr.open_dataset(ls_pan_nc_file, mask_and_scale=False)

    ls_arr = geographical_slicing(ls_arr, cfg, 660)

    # Build mask of data that are zero
    zero_mask = (ls_arr["B08"] == 0).values

    # Build no-data mask
    nan_mask = np.isnan(ls_arr["B08"].values)

    ls_arr["no_data"] = ("t", "y", "x"), np.logical_or(nan_mask, zero_mask)

    ls_clear_pixel_rate = 1 - (
        ls_arr["no_data"].sum(dim=("x", "y")) / (len(ls_arr.x) * len(ls_arr.y))
    )

    # Create output dir if not already existing
    ls_dir = os.path.join(folder, "landsat")
    if not Path(ls_dir).is_dir():
        Path(ls_dir).mkdir(parents=True)

    # if index.csv exists in the folder, delete it to avoid problems
    if Path(os.path.join(ls_dir, "index_pan.csv")).exists():
        Path(os.path.join(ls_dir, "index_pan.csv")).unlink()

    # Build a dataframe with info
    ls_dict = {
        "product_id": ("landsat_" + ls_arr.t.dt.strftime("%Y%m%d")).values,
        "acquisition_date": ls_arr.t.dt.strftime("%Y-%m-%d").values,
        "clear_pixel_rate": ls_clear_pixel_rate.values,
        "valid": ls_clear_pixel_rate.values >= args.min_clear_pixel_rate,
        "bands": [None for _ in range(len(ls_arr.t))],
        "mask": [None for _ in range(len(ls_arr.t))],
    }
    ls_index = pd.DataFrame(ls_dict).set_index("product_id")

    # Filter out invalid dates
    clear_dates = ls_arr.t[ls_clear_pixel_rate >= args.min_clear_pixel_rate]
    if not len(clear_dates):
        logging.warning(f"No clear landsat pan dates found for AOI {cfg.name}")
        ls_index.to_csv(os.path.join(ls_dir, "index_pan.csv"), sep="\t")
        return
    ls_arr = ls_arr.where(
        ls_arr.t.isin(clear_dates),
        drop=True,
    )

    ls_arr["B08"] = ("t", "x", "y"), affine_transform(
        ls_arr["B08"].to_numpy(),
        np.eye(3),
        offset=[0.0, -0.5, -0.5],
        order=3,
        prefilter=True,
        mode="reflect",
    )
    ls_arr["no_data"] = ("t", "x", "y"), affine_transform(
        ls_arr["no_data"].to_numpy(),
        np.eye(3),
        offset=[0.0, -0.5, -0.5],
        order=0,
        prefilter=False,
        mode="reflect",
    )
    product_ids, files = netcdf_to_geotiff(
        ls_arr,
        ls_dir,
        bands=["B08"],
        nodata=-10000,
        fillna=False,
        sensor="landsat",
        what="pan",
        scale=10000,
    )

    ls_index.loc[product_ids, "bands"] = files

    product_ids, files = netcdf_to_geotiff(
        ls_arr,
        ls_dir,
        bands=["no_data"],
        sensor="landsat",
        fillna=False,
        scale=1.0,
        nodata=0,
        dtype="uint8",
        what="pan_mask",
    )
    ls_index.loc[product_ids, "mask"] = files

    ls_index.to_csv(os.path.join(ls_dir, "index_pan.csv"), sep="\t")


def extract(args):
    """
    Extract step
    """
    for json_file in args.json:
        # Read configuration
        cfg = read_json_configuration(json_file)

        # Create output dir if not already existing
        folder = os.path.join(args.output, cfg.name)
        if not Path(folder).is_dir():
            Path(folder).mkdir(parents=True)

        logging.info(f"Processing json file {json_file}")
        for sensor in args.sensors:
            nc_file = os.path.join(folder, f"{sensor}.nc")

            match sensor:
                case "sentinel2":
                    if (
                        not Path(
                            os.path.join(folder, "sentinel2", "index.csv")
                        ).exists()
                        or args.override
                    ):
                        extract_s2(nc_file, cfg, folder, args)
                    else:
                        logging.info(
                            "Sentinel-2 data already extracted and override is \
                            deactivated, skipping"
                        )
                case "landsat":
                    if (
                        not Path(os.path.join(folder, "landsat", "index.csv")).exists()
                        or args.override
                    ):
                        extract_ls(nc_file, cfg, folder, args)
                    else:
                        logging.info(
                            "Landsat data already extracted and override is \
                            deactivated, skipping"
                        )

                case "landsat_pan":
                    if (
                        not Path(
                            os.path.join(folder, "landsat", "index_pan.csv")
                        ).exists()
                        or args.override
                    ):
                        extract_ls_pan(nc_file, cfg, folder, args)
                    else:
                        logging.info(
                            "Landsat data already extracted and override is \
                            deactivated, skipping"
                        )


def clean(args):
    """
    Clean landsat pan files for invalid landsat dates
    """
    for json_file in args.json:
        # Read configuration
        cfg = read_json_configuration(json_file)

        # Create output dir if not already existing
        folder = os.path.join(args.output, cfg.name)
        # If folder exists
        if Path(folder).is_dir():
            # Check and load landsat index and landsat_pan index
            ls_index_path = Path(os.path.join(folder, "landsat", "index.csv"))
            if ls_index_path.exists():
                ls_index = pd.read_csv(ls_index_path, sep="\t", index_col="product_id")
                ls_index_pan_path = Path(
                    os.path.join(folder, "landsat", "index_pan.csv")
                )
                if ls_index_pan_path.exists():
                    ls_index_pan = pd.read_csv(
                        ls_index_pan_path, sep="\t", index_col="product_id"
                    )

                    invalid_landsat_products = ls_index[~ls_index.valid]

                    for _, row in invalid_landsat_products.iterrows():
                        data_path = Path(
                            os.path.join(
                                folder,
                                "landsat",
                                row["acquisition_date"].replace("-", ""),
                            )
                        )
                        if data_path.exists():
                            rmtree(data_path)

                    missing_pan_products = ls_index[
                        ~ls_index.index.isin(ls_index_pan.index)
                    ]
                    for _, row in missing_pan_products.iterrows():
                        data_path = Path(
                            os.path.join(
                                folder,
                                "landsat",
                                row["acquisition_date"].replace("-", ""),
                            )
                        )
                        if data_path.exists():
                            rmtree(data_path)
                    ls_index = ls_index[ls_index.index.isin(ls_index_pan.index)]
                    ls_index_pan = ls_index_pan.loc[ls_index.index].copy()
                    ls_index_pan.clear_pixel_rate = ls_index.clear_pixel_rate.values
                    ls_index_pan.valid = ls_index.valid.values

                    ls_index_pan.bands = ls_index_pan.bands.where(ls_index_pan.valid)
                    ls_index_pan["mask"] = ls_index_pan["mask"].where(
                        ls_index_pan.valid
                    )
                    ls_index.to_csv(ls_index_path, sep="\t")
                    ls_index_pan.to_csv(ls_index_pan_path, sep="\t")


def check_index_and_images(folder: str, index_file: str, expected_size: int):
    """
    Helper function to validate each sesor
    """
    index_file_path = os.path.join(folder, index_file)
    index_df = pd.read_csv(index_file_path, sep="\t")
    for _, row in index_df.iterrows():
        if row["valid"]:
            with rio.open(os.path.join(folder, row.bands), "r") as ds:
                assert ds.width == expected_size
                assert ds.height == expected_size
            with rio.open(os.path.join(folder, row["mask"]), "r") as ds:
                assert ds.width == expected_size
                assert ds.height == expected_size


def check(args):
    """
    Perform validity check on all sites by instanciating the dataset
    """
    for json_file in tqdm(args.json, total=len(args.json), desc="Checking time-series"):
        # Read configuration
        cfg = read_json_configuration(json_file)

        # Create output dir if not already existing
        folder = os.path.join(args.output, cfg.name)

        try:
            arr = xr.open_dataset(os.path.join(folder, "sentinel2.nc"))
            retrieved_epsg = CRS(arr.crs.crs_wkt).to_epsg()
            requested_epsg = int(cfg.crs[5:])

            if retrieved_epsg != requested_epsg:
                logging.warning(
                    "%s: retrieved epsg (%s) does not match the requested epsg (%s)",
                    cfg.name,
                    retrieved_epsg,
                    requested_epsg,
                )

            check_index_and_images(os.path.join(folder, "sentinel2"), "index.csv", 990)
            check_index_and_images(os.path.join(folder, "landsat"), "index.csv", 330)
            check_index_and_images(
                os.path.join(folder, "landsat"), "index_pan.csv", 660
            )

            ds = SingleSITSDataset(
                folder,
                patch_size=3300,
                hr_index_files=["index.csv"],
                lr_index_files=["index.csv", "index_pan.csv"],
                lr_resolution=15.0,
            )

            # retrieve first patch
            ds[0]
        except Exception as e:
            logging.warning(f"Site {cfg.name} is invalid ({e})")
            pass


def main(args):
    # Configure logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.loglevel)

    logging.basicConfig(
        level=numeric_level,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

    if args.mode == "submit":
        submit(args)

    elif args.mode == "download":
        download(args)

    elif args.mode == "extract":
        extract(args)

    elif args.mode == "clean":
        clean(args)

    elif args.mode == "check":
        check(args)


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()
    main(args)
