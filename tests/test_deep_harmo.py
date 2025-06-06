# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests to deep-harmo model
"""

import os

import pytest
import rasterio as rio  # type: ignore
import torch
from transformers import AutoModel  # type: ignore

from tamrfsits.data import joint_sits_dataset

from .tests_utils import get_ls2s2_dataset_path, get_tests_output_path


@pytest.mark.requires_data
def test_deep_harmo_inference() -> None:
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
        lr_bands=(None, None),
        patch_size=3300.0,
        conjunctions_only=True,
    )

    ls_sits, s2_sits = ds[0]

    ensembles = {
        "5depth-upsample": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-5depth-upsample", trust_remote_code=True
        )
        .eval()
        .cuda(),
        "5depth-shuffle": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-5depth-shuffle", trust_remote_code=True
        )
        .eval()
        .cuda(),
        "6depth-upsample": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-6depth-upsample", trust_remote_code=True
        )
        .eval()
        .cuda(),
        "6depth-shuffle": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-5depth-shuffle", trust_remote_code=True
        )
        .eval()
        .cuda(),
    }

    ls_ms = ls_sits.data[:, 0, [1, 2, 3, 4, 5, 6], ...] / 10000.0
    ls_pan = ls_sits.data[:, 0, 8:9, ...] / 10000.0

    with torch.no_grad():
        out = sum(
            [model(ls_ms.cuda(), ls_pan.cuda()) for model in ensembles.values()]
        ) / len(ensembles)

    profile_pred = {
        "driver": "GTiff",
        "height": out.shape[-1],
        "width": out.shape[-2],
        "count": out.shape[1],
        "dtype": rio.uint16,
    }

    with rio.open(
        os.path.join(get_tests_output_path(), "deep_harmo_out.tif"), "w", **profile_pred
    ) as rio_ds:
        rio_ds.write(10000 * out[0, ...].cpu().numpy())

    profile_pred = {
        "driver": "GTiff",
        "height": s2_sits.data.shape[-1],
        "width": s2_sits.data.shape[-2],
        "count": s2_sits.data.shape[2],
        "dtype": rio.uint16,
    }

    with rio.open(
        os.path.join(get_tests_output_path(), "deep_harmo_ref.tif"), "w", **profile_pred
    ) as rio_ds:
        rio_ds.write(s2_sits.data[0, 0, ...])
