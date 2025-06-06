# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains helper classes related to the dataset
"""
import datetime as dt
import glob
import math
import os
import random
from contextlib import suppress
from functools import reduce
from itertools import chain, pairwise
from os.path import basename, join, normpath
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from numpy import all as np_all
from rasterio import open as rio_open  # type: ignore
from torch import logical_or, tensor  # pylint: disable=no-name-in-module

from tamrfsits.core.downsampling import generic_downscale
from tamrfsits.core.time_series import (
    MonoModalSITS,
    cat_monomodal_sits,
    pad_acquisition_time,
)


def filename_strip(file_path: str) -> str:
    """
    Strip absolute part of filenames in index
    """
    if Path(file_path).is_absolute():
        splits = file_path.rsplit(maxsplit=2, sep="/")
        return "/".join(splits[1:])
    return file_path


DEFAULT_INDEX_TUPLE = ("index.csv",)


class SingleSITSDataset(torch.utils.data.Dataset):
    """
    A map-style dataset handling a single sen2venµs site
    """

    def __init__(
        self,
        ts_path: str,
        patch_size: float = 1200,
        hr_sensor: str = "sentinel2",
        lr_sensor: str = "landsat",
        hr_index_files: tuple[str, ...] | None = DEFAULT_INDEX_TUPLE,
        lr_index_files: tuple[str, ...] | None = DEFAULT_INDEX_TUPLE,
        hr_resolution: float = 10.0,
        lr_resolution: float = 30.0,
        hr_bands: tuple[list[int] | None, ...] = (None,),
        lr_bands: tuple[list[int] | None, ...] = (None,),
        mtf_for_downsampling: float = 0.1,
        dt_orig: str | None = "2018.01.01",
        time_slices_in_days: int | None = None,
        random_reduce_rate: float | None = None,
        min_nb_dates: int = 4,
        max_nb_dates: int | None = None,
        conjunctions_only: bool = False,
    ):
        """
        Class initializer


        """
        self.ts_path = ts_path
        self.lr_resolution = lr_resolution
        self.hr_resolution = hr_resolution
        self.lr_sensor = lr_sensor
        self.hr_sensor = hr_sensor
        self.hr_bands = hr_bands
        self.lr_bands = lr_bands
        self.random_reduce_rate = random_reduce_rate
        self.max_nb_dates = max_nb_dates
        self.patch_size = patch_size
        self.mtf_for_downsampling = mtf_for_downsampling
        self.conjunctions_only = conjunctions_only
        assert Path(self.ts_path).exists(), self.ts_path

        factor = lr_resolution / hr_resolution
        # Detect misconfiguration early
        if math.modf(patch_size / factor)[0] > 0.0:
            raise ValueError(
                f"Patch size {patch_size} can not be divided by scale factor {factor}"
            )

        def build_index(
            sensor: str, index_files: tuple[str, ...] | None = None
        ) -> list[pd.DataFrame]:
            """
            Helper function that reads all index files for given sensor
            """
            if index_files is None:
                # Read LR index: find all csv files
                index_files = tuple(glob.glob(join(ts_path, sensor, "*.csv")))
            else:
                index_files = tuple(
                    os.path.join(ts_path, sensor, f) for f in index_files
                )
            assert index_files, self.ts_path

            # Read them with pandas
            index = [
                pd.read_csv(f, sep="\t", parse_dates=["acquisition_date"])
                for f in index_files
            ]

            # Find common dates to all index files
            common_acquisition_dates = pd.Series(
                sorted(
                    list(set.intersection(*(set(df.acquisition_date) for df in index)))
                )
            )
            index = [
                df[df.acquisition_date.isin(common_acquisition_dates)].sort_values(
                    "acquisition_date"
                )
                for df in index
            ]
            for x, y in pairwise(index):
                # pylint: disable=loop-invariant-statement
                assert len(x) == len(y) == len(common_acquisition_dates)
                assert np_all(
                    x.acquisition_date.values == y.acquisition_date.values
                ), self.ts_path

            # Derive dates that are valid in all dataframes
            valid = reduce(np.logical_and, (df.valid.to_numpy() for df in index))

            # Filter out invalid dates
            out = [df[valid] for df in index]
            for x, y in pairwise(out):
                assert len(x) == len(y)
                assert np_all(x.acquisition_date.values == y.acquisition_date.values)

            return out

        # Read LR and HR indices
        self.lr_index = build_index(lr_sensor, lr_index_files)
        self.hr_index = build_index(hr_sensor, hr_index_files)

        assert self.lr_index
        assert self.hr_index

        # If we retain only conjonctions, filter dataframes
        if conjunctions_only:
            lr_conjunctions_mask = self.lr_index[0].acquisition_date.dt.date.isin(
                self.hr_index[0].acquisition_date.dt.date
            )
            hr_conjunctions_mask = self.hr_index[0].acquisition_date.dt.date.isin(
                self.lr_index[0].acquisition_date.dt.date
            )
            for i, row in enumerate(self.lr_index):
                self.lr_index[i] = row[lr_conjunctions_mask.values]
            for i, row in enumerate(self.hr_index):
                self.hr_index[i] = row[hr_conjunctions_mask.values]

            for lr_index in self.lr_index:
                for hr_index in self.hr_index:
                    assert np_all(
                        lr_index.acquisition_date.dt.date.values
                        == hr_index.acquisition_date.dt.date.values
                    )

        # Derive dt orig date
        if dt_orig is not None:
            self.dt_orig = pd.to_datetime(dt_orig)
        else:
            # If not provided, use first date of first index of hr sensor
            self.dt_orig = self.hr_index[0].iloc[0].acquisition_date

        # Now derive the number of patches
        if self.hr_index[0].empty:
            raise ValueError("No image files found for sensor " + hr_sensor)
        first_hr_date_file = join(
            ts_path, hr_sensor, filename_strip(self.hr_index[0].iloc[0].bands)
        )

        # Derive patch positions and number of patches
        with rio_open(first_hr_date_file, "r") as rio_ds:
            bounds = rio_ds.bounds

        nb_patches_x = int(np.floor((bounds[2] - bounds[0]) / patch_size))
        nb_patches_y = int(np.floor((bounds[3] - bounds[1]) / patch_size))

        pul_x, pul_y = np.meshgrid(
            np.linspace(bounds[0], bounds[2], nb_patches_x, endpoint=False),
            np.linspace(bounds[1], bounds[3], nb_patches_y, endpoint=False),
        )
        self.pul_x: np.ndarray = pul_x.ravel()
        self.pul_y: np.ndarray = pul_y.ravel()

        # Derive time range
        full_time_range = (
            min(
                df.acquisition_date.min() for df in chain(self.lr_index, self.hr_index)
            ),
            max(
                df.acquisition_date.max() for df in chain(self.lr_index, self.hr_index)
            ),
        )

        # Generate time_slices if required
        if time_slices_in_days is not None:
            nb_slices = int(
                np.ceil((full_time_range[1] - self.dt_orig).days / time_slices_in_days)
            )
            self.time_slices: list[dt.date] = [
                (
                    self.dt_orig + dt.timedelta(s * time_slices_in_days),
                    self.dt_orig + dt.timedelta((s + 1) * time_slices_in_days),
                )
                for s in range(nb_slices)
            ]

            # Filter out slices that do not have enough dates
            self.time_slices = [
                s
                for s in self.time_slices
                if len(
                    self.lr_index[0][
                        np.logical_and(
                            self.lr_index[0].acquisition_date >= s[0],
                            self.lr_index[0].acquisition_date < s[1],
                        )
                    ]
                )
                > min_nb_dates
                and len(
                    self.hr_index[0][
                        np.logical_and(
                            self.hr_index[0].acquisition_date >= s[0],
                            self.hr_index[0].acquisition_date < s[1],
                        )
                    ]
                )
                > min_nb_dates
            ]

        else:
            self.time_slices = [full_time_range]

    def __len__(self) -> int:
        """
        Dataset length
        """
        return len(self.pul_x) * len(self.time_slices)

    def load_state_dict(self, state_dict):
        """
        load the state_dict
        """
        random.setstate(state_dict["random_state"])

    def state_dict(self):
        """
        Save state_dict
        """
        return {"random_state": random.getstate()}

    def _read_time_series(
        self,
        patch: tuple[float, float, float, float],
        csv: pd.DataFrame,
        sensor: str = "sentinel2",
        resolution: float | None = None,
        bands: list[int] | None = None,
    ) -> MonoModalSITS:
        """
        Read one time series
        """
        data: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        doys: list[int] = []

        if csv.empty:
            raise ValueError("No image files found for sensor " + sensor)

        bands_for_rasterio = [b + 1 for b in bands] if bands else None
        with rio_open(
            join(self.ts_path, sensor, filename_strip(csv.iloc[0].bands))
        ) as rio_ds:
            source_resolution = rio_ds.transform.a
            window = rio_ds.window(*patch)
        for _, row in csv.iterrows():
            bands_file = join(self.ts_path, sensor, filename_strip(row.bands))
            mask_file = join(self.ts_path, sensor, filename_strip(row["mask"]))
            with rio_open(bands_file, "r") as rio_ds:
                current_data = tensor(
                    rio_ds.read(indexes=bands_for_rasterio, window=window)
                )
                data.append(current_data)

            with rio_open(mask_file, "r") as rio_ds:
                current_mask = tensor(rio_ds.read(window=window) > 0)
                # Add to mask pixel that have value of 0 (which may be missing in the mask)
                current_mask = logical_or(
                    (current_data == 0).sum(dim=0) > 0, current_mask
                )
                masks.append(current_mask)

            doys.append((row.acquisition_date - self.dt_orig).days)
        assert data and masks and doys
        data_stack = torch.stack(data)
        mask_stack = torch.stack(masks)
        doy_tensor = tensor(doys)

        # Filter data_stack that are fully masked for this patch
        # fully_mask_stacked_timesteps = (~mask_stack).sum(dim=(1, 2, 3)) == 0
        # data_stack = data_stack[~fully_mask_stacked_timesteps, :, :, :]
        # doy_tensor = doy_tensor[~fully_mask_stacked_timesteps]
        # mask_stack = mask_stack[~fully_mask_stacked_timesteps, :, :]

        # Resampling_Factor if required
        if resolution is not None:
            assert source_resolution is not None
            resampling_factor = source_resolution / resolution
            if resampling_factor > 1.0:
                data_stack = torch.nn.functional.interpolate(
                    data_stack.to(dtype=torch.float32),
                    mode="bicubic",
                    scale_factor=resampling_factor,
                    align_corners=False,
                ).to(dtype=torch.short)
                mask_stack = (
                    torch.nn.functional.interpolate(
                        mask_stack.to(dtype=torch.float32),
                        mode="nearest",
                        scale_factor=resampling_factor,
                    )
                    > 0
                )
            elif resampling_factor < 1.0:
                data_stack = generic_downscale(
                    data_stack.to(dtype=torch.float),
                    mtf=self.mtf_for_downsampling,
                    padding="valid",
                ).to(dtype=torch.int16)
                mask_stack = (
                    torch.nn.functional.interpolate(
                        mask_stack.to(dtype=torch.float32),
                        mode="nearest",
                        scale_factor=resampling_factor,
                    )
                    > 0
                )
        data_stack = data_stack[None, ...]
        mask_stack = mask_stack[None, :, 0, :, :]
        doy_tensor = doy_tensor[None, ...]
        mask_stack = mask_stack > 0  # We need a boolean mask

        return MonoModalSITS(data_stack, doy_tensor, mask_stack)

    def __getitem__(self, idx: int) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Read one item of the dataset
        """
        # Retrieve current patch and time slice idx
        slice_idx = idx % len(self.time_slices)
        patch_idx = idx // len(self.time_slices)

        # Get coordinates for current patch
        current_pul_x = self.pul_x[patch_idx]
        current_pul_y = self.pul_y[patch_idx]

        # Derive current patch
        current_patch = (
            current_pul_x,
            current_pul_y,
            current_pul_x + self.patch_size,
            current_pul_y + self.patch_size,
        )

        def read_sits_closure(
            sits_index: list[pd.DataFrame],
            resolution: float,
            sensor: str,
            bands: tuple[list[int] | None, ...] = (None,),
            nb_dates: int | None = None,
        ) -> MonoModalSITS:
            """
            Closure that factors the reading code for lr / hr sits
            """
            # Note this is incompatible with several index files with different dates
            if sits_index[0].empty:
                raise ValueError("No image files found for sensor " + sensor)
            all_dates = list(range(len(sits_index[0])))
            random.shuffle(all_dates)

            if self.random_reduce_rate is not None:
                nb_kept_dates = int(np.ceil(len(all_dates) * self.random_reduce_rate))
            else:
                nb_kept_dates = len(all_dates)
            if nb_dates is not None:
                nb_kept_dates = min(nb_dates, nb_kept_dates)

            kept_dates = sorted(all_dates[:nb_kept_dates])

            sits_index = [csv.iloc[kept_dates] for csv in sits_index]

            # Read sits
            sits_list = [
                self._read_time_series(
                    current_patch,
                    df[
                        np.logical_and(
                            df.acquisition_date >= self.time_slices[slice_idx][0],
                            df.acquisition_date < self.time_slices[slice_idx][1],
                        )
                    ],
                    resolution=resolution,
                    sensor=sensor,
                    bands=b,
                )
                for df, b in zip(sits_index, bands)
            ]

            # Ensure that all masks are available
            for s in sits_list:
                assert s.mask is not None

            # Compute a joint mask by union of all masks
            # The assert is safe since checked above
            joint_mask = reduce(
                logical_or, (s.mask for s in sits_list if s.mask is not None)
            )

            # Update mask of sits
            for s in sits_list:
                s.mask = joint_mask

            out_sits = cat_monomodal_sits(sits_list)
            assert torch.all(~torch.isnan(out_sits.data))
            return out_sits

        # Actually read the hr and lr sits
        nb_hr_dates: int | None = None
        nb_lr_dates: int | None = None

        if self.max_nb_dates is not None:
            total_nb_dates = len(self.lr_index[0]) + len(self.hr_index[0])
            nb_hr_dates = int(
                self.max_nb_dates * (float(len(self.hr_index[0])) / total_nb_dates)
            )
            nb_lr_dates = int(
                self.max_nb_dates * (float(len(self.lr_index[0])) / total_nb_dates)
            )

        lr_sits = read_sits_closure(
            self.lr_index,
            self.lr_resolution,
            self.lr_sensor,
            bands=self.lr_bands,
            nb_dates=nb_lr_dates,
        )
        hr_sits = read_sits_closure(
            self.hr_index,
            self.hr_resolution,
            self.hr_sensor,
            bands=self.hr_bands,
            nb_dates=nb_hr_dates,
        )

        # Ensure that conjonctions is enforced
        if self.conjunctions_only:
            valid_lr = torch.isin(lr_sits.doy, hr_sits.doy)[0, :]
            lr_sits = MonoModalSITS(
                lr_sits.data[:, valid_lr, ...],
                lr_sits.doy[:, valid_lr],
                None if lr_sits.mask is None else lr_sits.mask[:, valid_lr, ...],
            )
            valid_hr = torch.isin(hr_sits.doy, lr_sits.doy)[0, :]
            hr_sits = MonoModalSITS(
                hr_sits.data[:, valid_hr, ...],
                hr_sits.doy[:, valid_hr],
                None if hr_sits.mask is None else hr_sits.mask[:, valid_hr, ...],
            )
        return lr_sits, hr_sits


class MultiSITSDataset(torch.utils.data.Dataset):
    """
    A dataset that aggregates all single site datasets
    """

    def __init__(
        self,
        ts_paths: list[str],
        patch_size: int = 1200,
        hr_sensor: str = "sentinel2",
        lr_sensor: str = "landsat",
        hr_index_files: tuple[str, ...] | None = DEFAULT_INDEX_TUPLE,
        lr_index_files: tuple[str, ...] | None = DEFAULT_INDEX_TUPLE,
        lr_resolution: float = 30.0,
        hr_resolution: float = 10.0,
        hr_bands: tuple[list[int] | None, ...] = (None,),
        lr_bands: tuple[list[int] | None, ...] = (None,),
        mtf_for_downsampling: float = 0.1,
        dt_orig: str | None = "2018.01.01",
        time_slices_in_days: int | None = None,
        random_reduce_rate: float | None = None,
        min_nb_dates: int = 4,
        max_nb_dates: int | None = None,
        cache_dir: str | None = None,
        conjunctions_only: bool = False,
    ):
        """
        Initializer
        """
        super().__init__()

        single_datasets: list[SingleSITSDataset | OnDiskCacheWrapper] = []
        cache_id = f"{patch_size}_{lr_resolution}__{hr_resolution}_\
                    {mtf_for_downsampling}_{dt_orig}_{time_slices_in_days}\
                    _{min_nb_dates}"

        for sits_path in ts_paths:
            # Pass if sits is empty
            with suppress(IndexError):
                current_ds: SingleSITSDataset = SingleSITSDataset(
                    sits_path,
                    patch_size,
                    hr_sensor,
                    lr_sensor,
                    hr_index_files,
                    lr_index_files,
                    hr_resolution,
                    lr_resolution,
                    hr_bands,
                    lr_bands,
                    mtf_for_downsampling,
                    dt_orig,
                    time_slices_in_days,
                    random_reduce_rate,
                    min_nb_dates,
                    max_nb_dates,
                    conjunctions_only,
                )

                # Perflint warning disabled since this is only called once at
                # beginning of training
                if cache_dir is not None:  # pylint: disable=loop-invariant-statement
                    current_cache_dir = join(cache_dir, cache_id)
                    cached_current_ds = OnDiskCacheWrapper(
                        current_ds,
                        join(
                            current_cache_dir,
                            basename(normpath(sits_path)),
                        ),
                    )
                    single_datasets.append(cached_current_ds)
                else:
                    single_datasets.append(current_ds)

        self.dataset: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset(
            single_datasets
        )

    def __len__(self):
        """
        return dataset length
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        return nth element
        """
        return self.dataset[idx]

    def load_state_dict(self, state_dict):
        """
        load the state_dict
        """
        random.setstate(state_dict["random_state"])

    def state_dict(self):
        """
        Save state_dict
        """
        return {"random_state": random.getstate()}


def cat_sits_batch(sits: list[MonoModalSITS]) -> MonoModalSITS:
    """
    Tells DataLoader how to collate BatchData
    """
    # Preconditions
    assert sits

    # Pad sits to max doy length
    max_doy_dim = max(s.doy.shape[1] for s in sits)
    sits = [
        cast(MonoModalSITS, pad_acquisition_time(s, doy_length=max_doy_dim))
        for s in sits
    ]

    if sits[0].mask is not None:
        for s in sits:
            assert s.mask is not None
    out_data = torch.cat([s.data for s in sits], dim=0)
    out_doy = torch.cat([s.doy for s in sits], dim=0)
    out_mask: torch.Tensor | None

    if sits[0].mask is not None:
        out_mask = torch.cat([s.mask for s in sits if s.mask is not None], dim=0)

    return MonoModalSITS(out_data, out_doy, out_mask)


def collate_fn(
    sits: list[tuple[MonoModalSITS, MonoModalSITS]]
) -> tuple[MonoModalSITS, MonoModalSITS]:
    """
    Concatenate several tuples of lr and hr sits to a single tuple of lr and hr sits
    """
    return cat_sits_batch([s[0] for s in sits]), cat_sits_batch([s[1] for s in sits])


class OnDiskCacheWrapper(torch.utils.data.Dataset):
    """
    A wrapper that allows persistent caching of dataset on disk
    """

    def __init__(self, ds: SingleSITSDataset | MultiSITSDataset, cache_dir: str):
        """ """
        super().__init__()

        self.dataset = ds
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        """
        Dataset length
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[MonoModalSITS, MonoModalSITS]:
        """
        Read data from cache or cache it and return it
        """
        sample_cache_folder = join(self.cache_dir, f"sample_{idx}")

        if Path(sample_cache_folder).exists():
            hr_data = torch.load(
                join(sample_cache_folder, "hr_data.pt"), weights_only=True
            )
            lr_data = torch.load(
                join(sample_cache_folder, "lr_data.pt"), weights_only=True
            )
            hr_mask = torch.load(
                join(sample_cache_folder, "hr_mask.pt"), weights_only=True
            )
            lr_mask = torch.load(
                join(sample_cache_folder, "lr_mask.pt"), weights_only=True
            )
            hr_doy = torch.load(
                join(sample_cache_folder, "hr_doy.pt"), weights_only=True
            )
            lr_doy = torch.load(
                join(sample_cache_folder, "lr_doy.pt"), weights_only=True
            )

            return MonoModalSITS(lr_data, lr_doy, lr_mask), MonoModalSITS(
                hr_data, hr_doy, hr_mask
            )

        # If not in cache, read it
        lr_sits, hr_sits = self.dataset[idx]

        Path(sample_cache_folder).mkdir(parents=True, exist_ok=True)
        torch.save(
            lr_sits.data,
            join(sample_cache_folder, "lr_data.pt"),
        )
        torch.save(
            hr_sits.data,
            join(sample_cache_folder, "hr_data.pt"),
        )
        torch.save(
            lr_sits.mask,
            join(sample_cache_folder, "lr_mask.pt"),
        )
        torch.save(
            hr_sits.mask,
            join(sample_cache_folder, "hr_mask.pt"),
        )
        torch.save(
            lr_sits.doy,
            join(sample_cache_folder, "lr_doy.pt"),
        )
        torch.save(
            hr_sits.doy,
            join(sample_cache_folder, "hr_doy.pt"),
        )

        return lr_sits, hr_sits
