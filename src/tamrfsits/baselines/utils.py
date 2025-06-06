# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
Utility module to load some of the third-party network
"""
import os
import pathlib
import sys
from functools import cache

import torch
from remote_sensing_processor.sentinel2.superres import supres  # type: ignore
from transformers import AutoModel  # type: ignore

from tamrfsits.baselines import network_dstfn_v1


@cache
def load_dsen2_model(map_location: str = "cpu") -> torch.nn.Module:
    """
    Load the pre-trained dsen2 model
    """
    mdl_path = pathlib.Path(supres.__file__).parents[0].joinpath("weights/")

    dsen2_mdl_path = mdl_path.joinpath("L2A20M.pt")

    return torch.jit.load(dsen2_mdl_path, map_location=map_location)


@cache
def load_dstfn_model(map_location: str = "cpu") -> torch.nn.Module:
    """
    Load the pre-trained dstfn model
    """
    # The checkpoint expects network_dstfn_v1 to be a top level module
    # This hacks allows to load it
    sys.modules["network_dstfn_v1"] = network_dstfn_v1
    model = torch.load(
        os.path.join(
            pathlib.Path(network_dstfn_v1.__file__).parent.resolve(),
            "model_100.pth",
        ),
        map_location=map_location,
    )
    # Undo the hack
    sys.modules.pop("network_dstfn_v1", None)

    return model


@cache
def load_deepharmo_ensemble(device: torch.DeviceObjType | str = "cpu"):
    """
    Load the deep-harmo ensemble of models
    """
    ensembles = {
        "5depth-upsample": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-5depth-upsample", trust_remote_code=True
        )
        .eval()
        .to(device=device),
        "5depth-shuffle": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-5depth-shuffle", trust_remote_code=True
        )
        .eval()
        .to(device=device),
        "6depth-upsample": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-6depth-upsample", trust_remote_code=True
        )
        .eval()
        .to(device=device),
        "6depth-shuffle": AutoModel.from_pretrained(
            "venkatesh-thiru/s2l8h-UNet-5depth-shuffle", trust_remote_code=True
        )
        .eval()
        .to(device=device),
    }

    return ensembles
