#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Tests of hydra configuration
"""
import glob
import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate


def generate_variants(
    modules_and_targets: list[tuple[str, str | None]], datamodule: str | None = None
) -> list[tuple[str, str, str | None, str]]:
    """
    This function lists all variants for a given hydra module
    """
    config_path = "../hydra"
    output: list[tuple[str, str, str | None, str]] = []
    for module, target in modules_and_targets:
        print(f"Checking {os.path.join(config_path, module)}")
        variants = glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                config_path,
                module,
                "*.yaml",
            )
        )
        if datamodule is not None:
            datamodule_variants = [datamodule]
        else:
            datamodule_variants = glob.glob(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    config_path,
                    "datamodule",
                    "*.yaml",
                )
            )

        for v in variants:
            for dm_v in datamodule_variants:
                output.append((module, Path(v).stem, target, Path(dm_v).stem))
    return output


def hydra_instantiate(module: str, variant: str, target: str | None, datamodule: str):
    """
    Test hydra instanciation of a single target
    """
    config_path = "../hydra"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"{module}={variant}", f"datamodule={datamodule}"],
        )
        # In that case try to instanciate every 1st level
        # key
        if target is None:
            for k in cfg[module]:
                instantiate(cfg[module][k])
        # In that case instantiate the given target
        elif target != "":
            instantiate(cfg[module][target])
        # In that case instantiate the module
        else:
            instantiate(cfg[module])


@pytest.mark.check_hydra
@pytest.mark.parametrize(
    "module,variant,target,datamodule",
    generate_variants(
        [
            ("hr_encoder", "model"),
            ("lr_encoder", "model"),
            ("positional_encoding", "model"),
            ("decoder", "model"),
            ("callbacks", None),
            ("loggers", None),
            ("training_module", "training_module"),
        ],
        datamodule="ls2s2_2022",
    ),
)
def test_hydra_instantiate(
    module: str, variant: str, target: str | None, datamodule: str
):
    """
    Tests that can be run everyhwere
    """
    hydra_instantiate(module, variant, target, datamodule)


@pytest.mark.check_hydra
@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "module,variant,target,datamodule",
    generate_variants(
        [("trainer", "")],
        datamodule="ls2s2_2022",
    ),
)
def test_hydra_instantiate_gpu(
    module: str, variant: str, target: str | None, datamodule: str
):
    """
    Tests that requires gpu
    """
    hydra_instantiate(module, variant, target, datamodule)


@pytest.mark.check_hydra
@pytest.mark.requires_data
@pytest.mark.parametrize(
    "module,variant,target,datamodule",
    generate_variants(
        [("datamodule", "data_module")],
        datamodule="ls2s2_2022",
    ),
)
def test_hydra_instantiate_data(
    module: str, variant: str, target: str | None, datamodule: str
):
    """
    Tests that requires date
    """
    hydra_instantiate(module, variant, target, datamodule)
