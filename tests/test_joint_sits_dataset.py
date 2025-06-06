# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests related to the sen2venus_fusion dataset
"""


import os
from tempfile import TemporaryDirectory

import pytest
import torch

from tamrfsits.data import joint_sits_dataset

from .tests_utils import get_ls2s2_dataset_path


@pytest.mark.requires_data
def test_joint_sits_dataset_single_site() -> None:
    """
    Test single site dataset
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(ts_path=dataset_path)

    ls_sits, s2_sits = ds[0]

    assert s2_sits.shape()[-2] == 3 * ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_joint_sits_dataset_single_site_ls2s2() -> None:
    """
    Test single site dataset
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(
        ts_path=dataset_path,
        hr_sensor="sentinel2",
        lr_sensor="landsat",
        hr_resolution=10.0,
        lr_resolution=30.0,
        lr_bands=([0, 1, 2],),
        patch_size=600.0,
    )

    ls_sits, s2_sits = ds[0]

    assert s2_sits.shape()[-2] == 3 * ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_joint_sits_dataset_single_site_ls2s2_pan() -> None:
    """
    Test single site dataset
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(
        ts_path=dataset_path,
        hr_sensor="sentinel2",
        lr_sensor="landsat",
        lr_index_files=("index.csv", "index_pan.csv"),
        hr_resolution=10.0,
        lr_resolution=10.0,
        lr_bands=([0, 1, 2], None),
        patch_size=600.0,
    )

    ls_sits, s2_sits = ds[0]

    assert s2_sits.shape()[-2] == ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_joint_sits_dataset_single_site_slice() -> None:
    """
    Test single site dataset
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(
        ts_path=dataset_path, time_slices_in_days=90, min_nb_dates=1
    )

    ls_sits, s2_sits = ds[0]

    assert s2_sits.shape()[-2] == 3 * ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_joint_sits_dataset_single_site_conjunctions() -> None:
    """
    Test single site dataset
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(
        ts_path=dataset_path,
        time_slices_in_days=90,
        min_nb_dates=1,
        conjunctions_only=True,
    )

    ls_sits, s2_sits = ds[0]
    print(ls_sits.doy, s2_sits.doy)
    assert torch.all(s2_sits.doy == ls_sits.doy)
    assert s2_sits.shape()[-2] == 3 * ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_dataset_cache() -> None:
    """
    Test caching mechanism
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(ts_path=dataset_path)

    with TemporaryDirectory() as tmpdir:
        cached_ds = joint_sits_dataset.OnDiskCacheWrapper(ds, tmpdir)

        not_cached = ds[0]
        cached_first_reach = cached_ds[0]
        cached_second_reach = cached_ds[0]

        for i in range(2):
            current_not_cached = not_cached[i]
            current_cached_first_reach = cached_first_reach[i]
            current_cached_second_reach = cached_second_reach[i]
            assert current_not_cached.mask is not None
            assert current_cached_first_reach.mask is not None
            assert current_cached_second_reach.mask is not None

            assert torch.equal(current_not_cached.data, current_cached_first_reach.data)
            assert torch.equal(current_not_cached.mask, current_cached_first_reach.mask)
            assert torch.equal(current_not_cached.doy, current_cached_first_reach.doy)
            assert torch.equal(
                current_not_cached.data, current_cached_second_reach.data
            )
            assert torch.equal(
                current_not_cached.mask, current_cached_second_reach.mask
            )
            assert torch.equal(current_not_cached.doy, current_cached_second_reach.doy)


@pytest.mark.requires_data
def test_joint_sits_dataset_single_site_downsample() -> None:
    """
    Test single site dataset
    """
    dataset_path = os.path.join(get_ls2s2_dataset_path(), "30SUF_21")
    ds = joint_sits_dataset.SingleSITSDataset(ts_path=dataset_path, lr_resolution=60.0)

    ls_sits, s2_sits = ds[0]

    assert s2_sits.shape()[-2] == 6 * ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_joint_sits_dataset_multi_sites() -> None:
    """
    Test multiple site dataset
    """
    dataset_paths = [
        os.path.join(get_ls2s2_dataset_path(), s) for s in ("30SUF_21", "30SUF_12")
    ]
    ds = joint_sits_dataset.MultiSITSDataset(
        ts_paths=dataset_paths, hr_bands=([0, 1, 2],)
    )

    ls_sits, s2_sits = ds[0]
    ls_sits, s2_sits = ds[len(ds) - 1]

    assert s2_sits.shape()[-2] == 3 * ls_sits.shape()[-2]

    state_dict = ds.state_dict()

    ds.load_state_dict(state_dict)


@pytest.mark.requires_data
def test_joint_sits_dataset_multi_sites_cache() -> None:
    """
    Test multiple site dataset
    """
    dataset_paths = [
        os.path.join(get_ls2s2_dataset_path(), s) for s in ("30SUF_21", "30SUF_12")
    ]
    with TemporaryDirectory() as tmpdir:
        ds = joint_sits_dataset.MultiSITSDataset(
            ts_paths=dataset_paths, cache_dir=tmpdir
        )

    ls_sits, s2_sits = ds[0]
    ls_sits, s2_sits = ds[len(ds) - 1]

    assert s2_sits.shape()[-2] == 3 * ls_sits.shape()[-2]


@pytest.mark.requires_data
def test_joint_sits_dataset_multi_sites_dataloader() -> None:
    """
    Test dataloader with collate_fn
    """
    dataset_paths = [
        os.path.join(get_ls2s2_dataset_path(), s) for s in ("30SUF_21", "30SUF_12")
    ]
    ds = joint_sits_dataset.MultiSITSDataset(ts_paths=dataset_paths)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        collate_fn=joint_sits_dataset.collate_fn,
        prefetch_factor=1,
        pin_memory=True,
    )

    _ = next(iter(dl))
