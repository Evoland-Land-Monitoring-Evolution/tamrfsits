#!/usr/bin/env python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Generate test metrics over full test dataset
"""
import logging
import os
import random
from enum import Enum
from itertools import starmap
from pathlib import Path
from shutil import copyfile
from typing import cast

import pandas as pd
import torch
import yaml
from einops import rearrange
from tqdm import tqdm

from tamrfsits.baselines.dms import dms_predict
from tamrfsits.baselines.dstfn import DSTFNSITSFusion
from tamrfsits.baselines.naive import NaiveSITSFusion
from tamrfsits.baselines.sen2like import Sen2LikeFusion
from tamrfsits.baselines.stair import STAIRSITSFusion
from tamrfsits.baselines.utils import load_deepharmo_ensemble, load_dsen2_model
from tamrfsits.core.time_series import (
    MonoModalSITS,
    crop_sits,
    subset_doy_monomodal_sits,
)
from tamrfsits.data.joint_sits_dataset import SingleSITSDataset
from tamrfsits.validation.strategy import (
    TestingConfiguration,
    ValidationParameters,
    ValidationStrategy,
    generate_configurations,
)
from tamrfsits.validation.utils import (
    MeasureExecTime,
    build_model,
    generate_animation,
    get_default_parser,
    write_sits,
)
from tamrfsits.validation.workflow import (
    TimeSeriesTestResult,
    compute_metrics,
    get_tensor,
    model_predict,
    to_pandas,
    write_log_profiles,
)

HR_RES = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 20.0, 20.0, 20.0]
HR_MTFS = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3, 0.3]
HR_LABELS = ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]
LR_RES = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 90.0]
LR_MTFS = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2]
LR_LABELS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "lst"]
DSEN2_SCALE = 2000.0


class Algorithm(Enum):
    TAMRFSITS = "TAMRFSITS"
    DSEN2 = "DSEN2"
    DEEPHARMO = "DEEPHARMO"
    STAIR = "STAIR"
    SEN2LIKE = "SEN2LIKE"
    NAIVE = "NAIVE"
    DSTFN = "DSTFN"
    DMS = "DMS"


def get_processed_bands(algorithm: Algorithm) -> tuple[list[int], list[int]]:
    """
    Return the list of processed band for lr and hr sits
    """
    if algorithm in (Algorithm.TAMRFSITS, Algorithm.NAIVE):
        return (
            list(range(8)),
            list(range(10)),
        )
    if algorithm in (Algorithm.STAIR, Algorithm.SEN2LIKE):
        lr_bands = [1, 2, 3, 4, 5, 6]
        hr_bands = [0, 1, 2, 7, 8, 9]
        return (
            lr_bands,
            hr_bands,
        )
    if algorithm is Algorithm.DSEN2:
        return (
            list(range(8)),
            list(range(10)),
        )
    if algorithm is Algorithm.DEEPHARMO:
        lr_bands = [1, 2, 3, 4, 5, 6]
        hr_bands = [0, 1, 2, 7, 8, 9]
        return (
            lr_bands,
            hr_bands,
        )
    if algorithm is Algorithm.DSTFN:
        lr_bands = [1, 2, 3, 4, 5, 6]
        hr_bands = [0, 1, 2, 7, 8, 9]

        return (
            lr_bands,
            hr_bands,
        )
    if algorithm is Algorithm.DMS:
        lr_bands = [7]
        hr_bands = [0, 1, 2, 3]
        return (lr_bands, hr_bands)
    raise ValueError(algorithm)


def build_dataset(ts: str, args) -> tuple[SingleSITSDataset, list[int], list[int]]:
    """
    Build The dataset according to the algorithm being benchmarked
    """
    algorithm = Algorithm(args.algorithm)

    if algorithm in (Algorithm.TAMRFSITS, Algorithm.NAIVE):
        return (
            SingleSITSDataset(
                ts, lr_resolution=30.0, patch_size=args.width, dt_orig="2022.01.01"
            ),
            list(range(8)),
            list(range(10)),
        )
    if algorithm in (Algorithm.STAIR, Algorithm.SEN2LIKE):
        lr_bands = [1, 2, 3, 4, 5, 6]
        hr_bands = [0, 1, 2, 7, 8, 9]
        return (
            SingleSITSDataset(
                ts,
                lr_index_files=("index.csv",),
                lr_bands=(lr_bands,),
                hr_bands=(hr_bands,),
                lr_resolution=30.0,
                conjunctions_only=False,
                dt_orig="2022.01.01",
                patch_size=args.width,
            ),
            lr_bands,
            hr_bands,
        )
    if algorithm is Algorithm.DSEN2:
        return (
            SingleSITSDataset(
                ts,
                patch_size=args.width,
                dt_orig="2022.01.01",
                lr_resolution=10.0,
            ),
            list(range(8)),
            list(range(10)),
        )
    if algorithm is Algorithm.DEEPHARMO:
        lr_bands = [1, 2, 3, 4, 5, 6]
        hr_bands = [0, 1, 2, 7, 8, 9]
        return (
            SingleSITSDataset(
                ts,
                patch_size=args.width,
                lr_index_files=("index.csv", "index_pan.csv"),
                lr_bands=(lr_bands, None),
                hr_bands=(hr_bands,),
                lr_resolution=10.0,
                dt_orig="2022.01.01",
                conjunctions_only=True,
            ),
            lr_bands,
            hr_bands,
        )
    if algorithm is Algorithm.DSTFN:
        lr_bands = [1, 2, 3, 4, 5, 6]
        hr_bands = [0, 1, 2, 7, 8, 9]

        return (
            SingleSITSDataset(
                ts,
                lr_index_files=("index.csv", "index_pan.csv"),
                lr_bands=(lr_bands, None),
                hr_bands=(hr_bands,),
                lr_resolution=15.0,
                dt_orig="2022.01.01",
                patch_size=args.width,
            ),
            lr_bands,
            hr_bands,
        )
    if algorithm is Algorithm.DMS:
        lr_bands = [7]
        hr_bands = [0, 1, 2, 3]

        return (
            SingleSITSDataset(
                ts,
                lr_bands=(lr_bands,),
                hr_bands=(hr_bands,),
                dt_orig="2022.01.01",
                patch_size=args.width,
            ),
            lr_bands,
            hr_bands,
        )

    raise ValueError(algorithm)


def get_strategy(args) -> ValidationParameters:
    """
    Build The dataset according to the algorithm being benchmarked
    """
    algorithm = Algorithm(args.algorithm)

    if algorithm in (Algorithm.TAMRFSITS, Algorithm.NAIVE):
        return ValidationParameters(
            strategy=ValidationStrategy(args.strategy),
            rate_for_random_strategy=args.mask_rate_for_random_strategy,
            forecast_doy_start=args.forecast_doy_start,
            gaps_size=args.gaps_size,
        )
    if algorithm in (Algorithm.SEN2LIKE, Algorithm.DSTFN):
        return ValidationParameters(strategy=ValidationStrategy.CONJLRuHR2HR)
    if algorithm is Algorithm.STAIR:
        return ValidationParameters(strategy=ValidationStrategy.ALL2CONJHR)
    if algorithm is Algorithm.DEEPHARMO:
        return ValidationParameters(strategy=ValidationStrategy.CONJLR2HR)
    if algorithm is Algorithm.DSEN2:
        return ValidationParameters(strategy=ValidationStrategy.ALLHR2ALLHR)
    if algorithm is Algorithm.DMS:
        return ValidationParameters(strategy=ValidationStrategy.ALL)
    raise ValueError(algorithm)


def compute_predictions(
    test_config: TestingConfiguration, device: str, args
) -> tuple[MonoModalSITS | None, MonoModalSITS | None]:
    """
    Perform inference according to model
    """
    algorithm = Algorithm(args.algorithm)

    if algorithm is Algorithm.TAMRFSITS:
        # Build model
        model, config, dev = build_model(
            args.checkpoint, args.seed, args.config, device=device
        )
        # Make predictions
        return model_predict(
            test_config,
            model,
            show_progress=args.show_subtile_progress,
            subtile_width=args.subtile_width,
        )

    elif algorithm is Algorithm.NAIVE:
        assert test_config.lr_input is not None
        assert test_config.hr_input is not None
        assert test_config.lr_target is not None
        assert test_config.hr_target is not None
        model = NaiveSITSFusion(lr_upsampling_factor=3.0)
        lr_input = MonoModalSITS(
            test_config.lr_input.data / 10000.0,
            test_config.lr_input.doy,
            test_config.lr_input.mask,
        )
        hr_input = MonoModalSITS(
            test_config.hr_input.data / 10000.0,
            test_config.hr_input.doy,
            test_config.hr_input.mask,
        )
        pred_lr = (
            model(lr_input, hr_input, test_config.lr_target.doy.ravel())[0]
            if (test_config.lr_target.doy.shape[1] and lr_input.doy.shape[1])
            else None
        )
        pred_hr = (
            model(lr_input, hr_input, test_config.hr_target.doy.ravel())[1]
            if (test_config.hr_target.doy.shape[1] and hr_input.doy.shape[1])
            else None
        )

        return pred_lr, pred_hr

    elif algorithm in (Algorithm.STAIR, Algorithm.SEN2LIKE, Algorithm.DSTFN):
        assert test_config.lr_input is not None
        assert test_config.hr_input is not None
        # Make predictions
        lr_input = MonoModalSITS(
            test_config.lr_input.data / 10000.0,
            test_config.lr_input.doy,
            test_config.lr_input.mask,
        )
        hr_input = MonoModalSITS(
            test_config.hr_input.data / 10000.0,
            test_config.hr_input.doy,
            test_config.hr_input.mask,
        )
        if algorithm is Algorithm.STAIR:
            model = STAIRSITSFusion(lr_upsampling_factor=3.0)
        elif algorithm is Algorithm.SEN2LIKE:
            model = Sen2LikeFusion(
                lr_upsampling_factor=3.0,
                mtf_for_hpf=args.sen2like_hpf_mtf,
                max_masked_rate_for_fusion=args.sen2like_max_masked_rate,
            )
        else:
            model = DSTFNSITSFusion().to(device=device)

        with torch.inference_mode():
            _, pred_hr = model(lr_input, hr_input)

        return None, pred_hr

    elif algorithm is Algorithm.DEEPHARMO:
        assert test_config.lr_input is not None
        lr_input = MonoModalSITS(
            test_config.lr_input.data / 10000.0,
            test_config.lr_input.doy,
            test_config.lr_input.mask,
        )
        # rearrange sits data for model
        lr_data = rearrange(lr_input.data, "b t c w h -> (b t) c w h")

        ensembles = load_deepharmo_ensemble(device=device)

        with torch.inference_mode():
            pred_hr = sum(
                [
                    model(lr_data[:, :7, ...], lr_data[:, 7:, ...])
                    for model in ensembles.values()
                ]
            ) / len(ensembles)

        pred_hr = rearrange(
            pred_hr, "(b t) c w h -> b t c w h", b=lr_input.data.shape[0]
        )

        pred_hr_sits = MonoModalSITS(pred_hr, lr_input.doy, lr_input.mask)

        return None, pred_hr_sits

    elif algorithm is Algorithm.DSEN2:
        assert test_config.hr_input is not None
        model = load_dsen2_model(map_location=device)
        # rearrange sits data for model
        hr_data = (
            rearrange(test_config.hr_input.data, "b t c w h -> (b t) c w h")
            / DSEN2_SCALE
        )
        with torch.inference_mode():
            pred_hr = (
                torch.cat(
                    list(
                        starmap(
                            model,
                            zip(
                                torch.tensor_split(hr_data[:, [0, 1, 2, 6], ...], 10),
                                torch.tensor_split(
                                    hr_data[:, [3, 4, 5, 7, 8, 9], ...], 10
                                ),
                            ),
                        )
                    )
                )
                / 5.0
            )

        pred_hr = torch.cat((hr_data[:, [0, 1, 2, 6], ...] / 5.0, pred_hr), dim=1)
        # Band permutation to recover the correct order
        pred_hr = pred_hr[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9], ...]
        pred_hr = rearrange(
            pred_hr, "(b t) c w h -> b t c w h", b=test_config.hr_input.data.shape[0]
        )

        pred_hr_sits = MonoModalSITS(
            pred_hr, test_config.hr_input.doy, test_config.hr_input.mask
        )
        return None, pred_hr_sits

    elif algorithm is Algorithm.DMS:
        assert test_config.lr_input is not None
        assert test_config.hr_input is not None
        pred_lst = dms_predict(
            test_config.lr_input,
            test_config.hr_input,
            tmp_dir=args.dms_tmp_dir,
            nb_procs=args.dms_nb_procs,
            residual_compensation=args.dms_rescomp,
        )
        return pred_lst, None
    raise ValueError(algorithm)


def process_time_series(
    ts: str, validation_parameters: ValidationParameters, args, device: str = "cpu"
) -> tuple[TimeSeriesTestResult | None, TimeSeriesTestResult | None]:
    """
    This function processes a single time-series and output the metrics
    """
    name = os.path.basename(os.path.normpath(ts))
    # First, build the dataset
    # batch_size=10000 to ensure that we read all patches at once
    dataset, lr_bands, hr_bands = build_dataset(ts, args)

    with MeasureExecTime("Processing time-series", args.profile):
        batch = dataset[args.patch_idx]
        if Algorithm(args.algorithm) is not Algorithm.DMS:
            batch = (batch[0].to(device=device), batch[1].to(device=device))
        # Generate all possible testing configurations
        random.seed(args.seed)
        test_config = next(
            iter(generate_configurations(batch, parameters=validation_parameters))
        )
        # Apply normalization for testing
        test_config.normalize_for_tests()

        with MeasureExecTime("Model inference", args.profile):
            pred_lr, pred_hr = compute_predictions(
                test_config, device=device, args=args
            )
        if not args.disable_metrics:
            with MeasureExecTime("Compute metrics", args.profile):
                # Compute metrics
                lr_result, hr_result = compute_metrics(
                    pred_lr,
                    pred_hr,
                    test_config,
                    get_tensor([LR_RES[i] for i in lr_bands], device=device),
                    get_tensor([LR_MTFS[i] for i in lr_bands], device=device),
                    get_tensor([HR_RES[i] for i in hr_bands], device=device),
                    get_tensor([HR_MTFS[i] for i in hr_bands], device=device),
                    name=name,
                    margin=args.margin,
                    profile=args.profile,
                )
        if args.write_images or args.generate_animations:
            # Crop the sits for optional rendering
            pred_lr = crop_sits(pred_lr, margin=int(args.margin)) if pred_lr else None
            test_config.lr_input = (
                crop_sits(test_config.lr_input, margin=int(args.margin / 3))
                if test_config.lr_input
                else None
            )
            test_config.lr_target = (
                crop_sits(test_config.lr_target, margin=int(args.margin / 3))
                if test_config.lr_target
                else None
            )
            pred_hr = crop_sits(pred_hr, margin=int(args.margin)) if pred_hr else None
            test_config.hr_input = (
                crop_sits(test_config.hr_input, margin=int(args.margin))
                if test_config.hr_input
                else None
            )
            test_config.hr_target = (
                crop_sits(test_config.hr_target, margin=int(args.margin))
                if test_config.hr_target
                else None
            )

        if args.write_images:
            with MeasureExecTime("Writing outputs", args.profile):
                lst_idx = [
                    i for i in range(len(lr_bands)) if LR_LABELS[lr_bands[i]] == "lst"
                ]
                write_predictions(
                    pred_lr,
                    pred_hr,
                    name,
                    str(dataset.dt_orig),
                    test_config,
                    lst_idx[0] if lst_idx else None,
                    args,
                )

        if args.generate_animations:
            with MeasureExecTime("Generate Animation", args.profile):
                # Determine color composition
                lr_bands_labels = [LR_LABELS[i] for i in lr_bands]
                hr_bands_labels = [HR_LABELS[i] for i in hr_bands]
                lr_bands_for_rendering = [
                    lr_bands_labels.index(b) for b in ("b4", "b3", "b2")
                ]
                hr_bands_for_rendering = [
                    hr_bands_labels.index(b) for b in ("b4", "b3", "b2")
                ]
                out_path = Path(args.output) / "animations"
                out_path.mkdir(parents=True, exist_ok=True)
                generate_animation(
                    test_config.lr_input,
                    test_config.hr_input,
                    pred_lr,
                    pred_hr,
                    test_config.lr_target,
                    test_config.hr_target,
                    lr_bands_for_rendering,
                    hr_bands_for_rendering,
                    out_file=str(out_path / f"{name}.html"),
                )
    if not args.disable_metrics:
        return lr_result, hr_result
    return None, None


def write_predictions(
    pred_lr: MonoModalSITS | None,
    pred_hr: MonoModalSITS | None,
    name: str,
    dt_orig: str,
    test_config: TestingConfiguration,
    lst_idx: int | None,
    args,
):
    """
    Write predictions if required
    """
    output_dir = Path(args.output) / "predictions" / name

    if test_config.lr_target is not None and pred_lr is not None:
        resolution_ratio = (
            float(pred_lr.shape()[-1]) / test_config.lr_target.shape()[-1]
        )
        # Denormalize TIR for image writing
        test_config.lr_target.data[:, :, lst_idx, ...] /= 1000.0

        assert test_config.lr_input is not None
        masked_lr_target_doys = test_config.lr_target.doy[
            ~torch.isin(test_config.lr_target.doy, test_config.lr_input.doy)
        ]
        if masked_lr_target_doys.numel() > 0:
            write_sits(
                subset_doy_monomodal_sits(
                    pred_lr,
                    masked_lr_target_doys,
                ),
                subset_doy_monomodal_sits(
                    test_config.lr_target,
                    masked_lr_target_doys,
                ),
                out_dir=str(output_dir),
                dt_orig=dt_orig,
                mode=f"lr_mae_{args.strategy}_{int(100*args.mask_rate_for_random_strategy)}_{args.forecast_doy_start}",
                radio_scale_target=10000.0,
                scale_pred=1.0,
                scale_target=resolution_ratio,
            )

        clear_lr_target_doys = test_config.lr_input.doy[
            torch.isin(test_config.lr_input.doy, pred_lr.doy)
        ]
        if clear_lr_target_doys.numel() > 0:
            write_sits(
                subset_doy_monomodal_sits(pred_lr, clear_lr_target_doys),
                test_config.lr_input,
                out_dir=str(output_dir),
                dt_orig=dt_orig,
                mode=f"lr_clr_{args.strategy}_{int(100*args.mask_rate_for_random_strategy)}_{args.forecast_doy_start}",
                scale_pred=1.0,
                scale_target=resolution_ratio,
            )

        # Renormalize TIR for metrics
        test_config.lr_target.data[:, :, lst_idx, ...] *= 1000.0

    if test_config.hr_target is not None and pred_hr is not None:
        if test_config.hr_input is not None:
            masked_hr_target_doys = test_config.hr_target.doy[
                ~torch.isin(test_config.hr_target.doy, test_config.hr_input.doy)
            ]
        else:
            masked_hr_target_doys = test_config.hr_target.doy
        if masked_hr_target_doys.numel() > 0:
            write_sits(
                subset_doy_monomodal_sits(
                    pred_hr,
                    masked_hr_target_doys,
                ),
                subset_doy_monomodal_sits(
                    test_config.hr_target,
                    masked_hr_target_doys,
                ),
                out_dir=str(output_dir),
                dt_orig=dt_orig,
                mode=f"hr_mae_{args.strategy}_{int(100*args.mask_rate_for_random_strategy)}_{args.forecast_doy_start}",
                radio_scale_target=10000.0,
                scale_pred=1.0,
                scale_target=1.0,
            )
    if test_config.hr_input is not None and pred_hr is not None:
        clear_hr_target_doys = test_config.hr_input.doy[
            torch.isin(test_config.hr_input.doy, pred_hr.doy)
        ]
        if clear_hr_target_doys.numel():
            write_sits(
                subset_doy_monomodal_sits(pred_hr, clear_hr_target_doys),
                test_config.hr_input,
                out_dir=str(output_dir),
                dt_orig=dt_orig,
                mode=f"hr_clr_{args.strategy}_{int(100*args.mask_rate_for_random_strategy)}_{args.forecast_doy_start}",
                scale_pred=1.0,
                scale_target=1.0,
            )


def main():
    """
    Main method
    """
    # Parser arguments
    parser = get_default_parser()
    parser.add_argument(
        "--device", required=False, type=str, help="On which device to run the model"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Print processing time information"
    )
    parser.add_argument(
        "--margin", default=30, help="Margin for evaluation and rendering"
    )
    parser.add_argument(
        "--disable_metrics", action="store_true", help="Disable metrics computation"
    )
    parser.add_argument(
        "--patch_idx",
        default=0,
        type=int,
        help="Which sub-patch to load if width is < 9900",
    )
    parser.add_argument(
        "--subtile_width",
        default=165,
        type=int,
        help="The subtile width used for TAMRFSITS inference",
    )

    parser.add_argument(
        "--show_subtile_progress",
        action="store_true",
        help="Show tqdm progress bar for subtile inference",
    )
    parser.add_argument(
        "--algorithm",
        choices=[a.value for a in Algorithm],
        default=Algorithm.TAMRFSITS,
        help="Which algorithm to test",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ValidationStrategy],
        default="ALL",
        help="Strategy used for validation",
    )
    parser.add_argument(
        "--mask_rate_for_random_strategy",
        type=float,
        default=0.5,
        help="Masking rate for the random masking strategy",
    )
    parser.add_argument(
        "--forecast_doy_start",
        type=float,
        default=183,
        help="Doy for start of forecasting in the forecast strategy",
    )
    parser.add_argument(
        "--gaps_size", default=30, type=int, help="Size of GAPS for the GAPS strategy"
    )
    parser.add_argument(
        "--sen2like_hpf_mtf",
        type=float,
        default=0.4,
        help="MTF for sen2like high pass filtering",
    )

    parser.add_argument(
        "--sen2like_max_masked_rate",
        type=float,
        default=0.5,
        help="Max masked rate for sen2like",
    )

    parser.add_argument(
        "--dms_tmp_dir",
        type=str,
        default="/tmp",
        help="Tmp directory where to write images when using DMS",
    )
    parser.add_argument(
        "--dms_nb_procs",
        type=int,
        default=8,
        help="Number of process for DMS prediction",
    )
    parser.add_argument(
        "--dms_rescomp",
        action="store_true",
        help="Perform residual compensation in DMS",
    )
    parser.add_argument(
        "--write_images", action="store_true", help="If true write predictions to disk"
    )

    parser.add_argument(
        "--generate_animations",
        action="store_true",
        help="Generate HTML animation for prediction",
    )

    args = parser.parse_args()

    # Dump useful arguments
    Path(args.output).mkdir(exist_ok=True)

    if args.device is None:
        dev = "cpu"
        if torch.cuda.is_available():
            dev = "cuda"
    else:
        dev = args.device

    if "TORCH_NUM_THREADS" in os.environ:
        logging.info(
            f'Setting number of CPU threads for PyTorch to {os.environ["TORCH_NUM_THREADS"]}'
        )
        torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))

    if Algorithm(args.algorithm) is Algorithm.TAMRFSITS:
        copyfile(args.checkpoint, os.path.join(args.output, "model.ckpt"))
        with open(os.path.join(args.output, "args.yaml"), "w") as f:
            yaml.dump(args, f)

    lr_results: list[TimeSeriesTestResult] = []
    hr_results: list[TimeSeriesTestResult] = []

    # Retrieve the validation strategy
    validation_parameters = get_strategy(args)

    for ts in tqdm(args.ts, total=len(args.ts), desc="Processing SITS ..."):
        try:
            lr_result, hr_result = process_time_series(
                ts,
                validation_parameters,
                args,
                dev,
            )
            assert lr_result is not None
            assert hr_result is not None
            lr_results.append(lr_result)
            hr_results.append(hr_result)
        except FileNotFoundError as e:
            logging.error(e)
        except ValueError as e:
            logging.error(e)

    lr_bands, hr_bands = get_processed_bands(Algorithm(args.algorithm))
    lr_bands_labels = [LR_LABELS[i] for i in lr_bands]
    hr_bands_labels = [HR_LABELS[i] for i in hr_bands]

    lr_results_df: pd.DataFrame | None = None
    if lr_results:
        lr_results = [r for r in lr_results if r is not None]
        if lr_results:
            lr_results_df = to_pandas(lr_results, lr_bands_labels).sort_values(by="doy")
            lr_results_df.to_csv(os.path.join(args.output, "lr_metrics.csv"), sep="\t")
    hr_results_df: pd.DataFrame | None = None
    if hr_results:
        hr_results = [r for r in hr_results if r is not None]
        if hr_results:
            hr_results_df = to_pandas(hr_results, hr_bands_labels).sort_values(by="doy")
            hr_results_df.to_csv(os.path.join(args.output, "hr_metrics.csv"), sep="\t")

    print("Masked dates performances:")
    if lr_results_df is not None and (~lr_results_df.clear_doy).sum() > 0:
        for band in lr_bands_labels:
            rmse = lr_results_df[f"rmse_{band}"][~lr_results_df.clear_doy].mean()
            brisque = lr_results_df[f"brisque_{band}"][~lr_results_df.clear_doy].mean()
            frr = lr_results_df[f"frr_{band}"][~lr_results_df.clear_doy].mean()
            print(
                f"LR {band}:\tRMSE={rmse:.3f},\tBRISQUE={brisque:.2f},\tFRR={frr:.2f}"
            )

        lr_masked_pred_prof = cast(
            torch.Tensor,
            sum(
                [
                    torch.nanmean(m.pred_prof[~m.clear_doy, ...], dim=0)
                    for m in lr_results
                ]
            )
            / max(len(lr_results), 1),
        )

        lr_masked_ref_prof = cast(
            torch.Tensor,
            sum(
                [
                    torch.nanmean(m.ref_prof[~m.clear_doy, ...], dim=0)
                    for m in lr_results
                ]
            )
            / max(len(lr_results), 1),
        )

        write_log_profiles(
            lr_masked_pred_prof,
            lr_masked_ref_prof,
            freqs=lr_results[0].freqs,
            outfile=os.path.join(args.output, "lr_masked_freq_restoration.pdf"),
            labels=lr_bands_labels,
        )
        torch.save(
            lr_masked_pred_prof, os.path.join(args.output, "lr_masked_pred_prof.pt")
        )
        torch.save(
            lr_masked_ref_prof, os.path.join(args.output, "lr_masked_ref_prof.pt")
        )
    if hr_results_df is not None and (~hr_results_df.clear_doy).sum() > 0:
        for band in hr_bands_labels:
            rmse = hr_results_df[f"rmse_{band}"][~hr_results_df.clear_doy].mean()
            brisque = hr_results_df[f"brisque_{band}"][~hr_results_df.clear_doy].mean()
            frr = hr_results_df[f"frr_{band}"][~hr_results_df.clear_doy].mean()
            print(
                f"HR {band}:\tRMSE={rmse:.3f},\tBRISQUE={brisque:.2f},\tFRR={frr:.2f}"
            )
        hr_masked_pred_prof = cast(
            torch.Tensor,
            sum(
                [
                    torch.nanmean(m.pred_prof[~m.clear_doy, ...], dim=0)
                    for m in hr_results
                ]
            )
            / max(len(hr_results), 1),
        )

        hr_masked_ref_prof = cast(
            torch.Tensor,
            sum(
                [
                    torch.nanmean(m.ref_prof[~m.clear_doy, ...], dim=0)
                    for m in hr_results
                ]
            )
            / max(len(hr_results), 1),
        )

        write_log_profiles(
            hr_masked_pred_prof,
            hr_masked_ref_prof,
            freqs=hr_results[0].freqs,
            outfile=os.path.join(args.output, "hr_masked_freq_restoration.pdf"),
            labels=hr_bands_labels,
        )
        torch.save(
            hr_masked_pred_prof, os.path.join(args.output, "hr_masked_pred_prof.pt")
        )
        torch.save(
            hr_masked_ref_prof, os.path.join(args.output, "hr_masked_ref_prof.pt")
        )

    print("Clear dates performances:")
    if lr_results_df is not None and lr_results_df.clear_doy.sum() > 0:
        for band in lr_bands_labels:
            rmse = lr_results_df[f"rmse_{band}"][lr_results_df.clear_doy].mean()
            brisque = lr_results_df[f"brisque_{band}"][lr_results_df.clear_doy].mean()
            frr = lr_results_df[f"frr_{band}"][lr_results_df.clear_doy].mean()
            print(
                f"LR {band}:\tRMSE={rmse:.3f},\tBRISQUE={brisque:.2f},\tFRR={frr:.2f}"
            )

        lr_clear_pred_prof = cast(
            torch.Tensor,
            sum(
                [
                    torch.nanmean(m.pred_prof[m.clear_doy, ...], dim=0)
                    for m in lr_results
                ]
            )
            / max(len(lr_results), 1),
        )

        lr_clear_ref_prof = cast(
            torch.Tensor,
            sum(
                [torch.nanmean(m.ref_prof[m.clear_doy, ...], dim=0) for m in lr_results]
            )
            / max(len(lr_results), 1),
        )

        write_log_profiles(
            lr_clear_pred_prof,
            lr_clear_ref_prof,
            freqs=lr_results[0].freqs,
            outfile=os.path.join(args.output, "lr_clear_freq_restoration.pdf"),
            labels=lr_bands_labels,
        )
        torch.save(
            lr_clear_pred_prof, os.path.join(args.output, "lr_clear_pred_prof.pt")
        )
        torch.save(lr_clear_ref_prof, os.path.join(args.output, "lr_clear_ref_prof.pt"))

    if hr_results_df is not None and hr_results_df.clear_doy.sum() > 0:
        for band in hr_bands_labels:
            rmse = hr_results_df[f"rmse_{band}"][hr_results_df.clear_doy].mean()
            brisque = hr_results_df[f"brisque_{band}"][hr_results_df.clear_doy].mean()
            frr = hr_results_df[f"frr_{band}"][hr_results_df.clear_doy].mean()
            print(
                f"HR {band}:\tRMSE={rmse:.3f},\tBRISQUE={brisque:.2f},\tFRR={frr:.2f}"
            )
        hr_clear_pred_prof = cast(
            torch.Tensor,
            sum(
                [
                    torch.nanmean(m.pred_prof[m.clear_doy, ...], dim=0)
                    for m in hr_results
                ]
            )
            / max(len(hr_results), 1),
        )

        hr_clear_ref_prof = cast(
            torch.Tensor,
            sum(
                [torch.nanmean(m.ref_prof[m.clear_doy, ...], dim=0) for m in hr_results]
            )
            / max(len(hr_results), 1),
        )
        write_log_profiles(
            hr_clear_pred_prof,
            hr_clear_ref_prof,
            freqs=hr_results[0].freqs,
            outfile=os.path.join(args.output, "hr_clear_freq_restoration.pdf"),
            labels=hr_bands_labels,
        )

        torch.save(
            hr_clear_pred_prof, os.path.join(args.output, "hr_clear_pred_prof.pt")
        )
        torch.save(hr_clear_ref_prof, os.path.join(args.output, "hr_clear_ref_prof.pt"))


if __name__ == "__main__":
    main()
