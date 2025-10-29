#!/usr/bin/env python
# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

"""
Script providing benchmarking of computational cost
"""

import argparse
import logging
import math
import os
import random
import resource

import pandas
import torch

from tamrfsits.components.datewise import DateWiseSITSModule
from tamrfsits.components.positional_encoder import FixedPositionalEncoder
from tamrfsits.components.transformer import TemporalDecoder, TemporalEncoder
from tamrfsits.core.esrgan import ESRSpatialDecoder, ESRSpatialEncoder
from tamrfsits.core.time_series import MonoModalSITS
from tamrfsits.validation.utils import MeasureExecTime


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Benchmarking script"
    )

    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument("--output", type=str, help="Path to output file", required=True)

    parser.add_argument("--lr_tile_size", default=55, type=int, help="LR tile size")
    parser.add_argument(
        "--nb_input_dates",
        nargs="+",
        type=int,
        default=(8, 16, 32, 64, 128),
        help="Number of input dates",
    )
    parser.add_argument(
        "--nb_target_dates",
        nargs="+",
        type=int,
        default=(12, 24),
        help="Number of target dates",
    )
    parser.add_argument(
        "--nb_runs", type=int, default=5, help="Number of runs for measure"
    )
    parser.add_argument("--gpu", action="store_true", help="Benchmark on GPU")

    return parser


def build_model() -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Build the model
    """
    hr_encoder = DateWiseSITSModule(
        ESRSpatialEncoder(
            in_nb_bands=10,
            upsampling_factor=1.0,
            latent_size=64,
            num_basic_blocks=1,
            growth_channels=32,
            residual_scaling=0.2,
            upsampling_mode="pixel_shuffle",
        )
    )
    lr_encoder = DateWiseSITSModule(
        ESRSpatialEncoder(
            in_nb_bands=8,
            upsampling_factor=3,
            upsampling_base=3,
            latent_size=64,
            num_basic_blocks=1,
            growth_channels=32,
            residual_scaling=0.2,
            upsampling_mode="bicubic",
        )
    )

    encoder = TemporalEncoder(
        hr_encoder=hr_encoder,
        lr_encoder=lr_encoder,
        temporal_positional_encoder=FixedPositionalEncoder(
            nb_features=64, mode="CONCAT", div=365.0
        ),
        token_size=64,
        nb_heads=4,
        nb_layers=3,
        dim_feedforward=256,
        sensor_token_size=8,
    )

    decoder = TemporalDecoder(
        decoder=DateWiseSITSModule(
            model=ESRSpatialDecoder(latent_size=136, out_nb_bands=18)
        ),
        temporal_positional_encoder=FixedPositionalEncoder(
            nb_features=64, mode="CONCAT", div=365.0
        ),
        nb_heads=4,
        token_size=64,
        sensor_token_size=8,
    )

    return encoder, decoder


def generate_monomodal_sits(
    batch: int = 16,
    nb_doy: int = 10,
    nb_features: int = 4,
    width: int = 32,
    masked: bool = True,
    max_doy: int = 20,
    clear_doy_proba: float = 0.5,
    device: str = "cpu",
) -> MonoModalSITS:
    """
    Generate a fake monomodal sits
    """
    possible_doys = list(range(max_doy))
    doy_list = [
        torch.sort(torch.tensor(random.sample(possible_doys, nb_doy), device=device))[0]
        for i in range(batch)
    ]
    doy = torch.stack(doy_list).to(dtype=torch.int16)
    data = torch.rand((batch, nb_doy, nb_features, width, width), device=device)
    mask: torch.Tensor | None = None
    if masked:
        mask = torch.zeros((batch, nb_doy, width, width), device=device).to(
            dtype=torch.bool
        )

        for i in range(batch):
            for d in range(nb_doy):
                if random.random() > clear_doy_proba:
                    # This date is masked
                    if random.random() > 0.5:
                        # Fully masked
                        mask[i, d, ...] = True
                    else:
                        # Partially masked
                        start_x = random.randint(0, width)
                        end_x = random.randint(start_x, width)
                        start_y = random.randint(0, width)
                        end_y = random.randint(start_y, width)
                        mask[i, d, start_x:end_x, start_y:end_y] = True

    # Should pass
    return MonoModalSITS(data, doy, mask)


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

    device = "cpu"
    if args.gpu:
        device = "cuda"

    # Get encoder and decoder
    encoder, decoder = build_model()

    encoder = encoder.to(device=device).eval()
    decoder = decoder.to(device=device).eval()

    hr_to_lr_ratio = 5.0 / 8.0

    max_total_nb_dates = max(args.nb_input_dates)
    max_output_nb_dates = max(args.nb_target_dates)
    max_hr_nb_dates = int(math.floor(max_total_nb_dates / (1 + hr_to_lr_ratio)) + 0.5)
    max_lr_nb_dates = max_total_nb_dates - max_hr_nb_dates
    target_doys = torch.linspace(0, 365.0, max_output_nb_dates, device=device)

    logging.info(
        f"Max HR dates: {max_hr_nb_dates}, max LR dates: {max_lr_nb_dates}",
    )

    lr_sits = generate_monomodal_sits(
        batch=1,
        width=args.lr_tile_size,
        nb_doy=max_lr_nb_dates,
        nb_features=8,
        max_doy=365,
        masked=False,
        device=device,
    )

    hr_sits = generate_monomodal_sits(
        batch=1,
        width=3 * args.lr_tile_size,
        nb_doy=max_hr_nb_dates,
        nb_features=10,
        max_doy=365,
        masked=False,
        device=device,
    )
    decoder_records: list[list[float]] = []
    cpu_memory_offset = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    warmed_up = False

    for total_nb_dates in args.nb_input_dates:
        for out_nb_dates in args.nb_target_dates:
            current_target_doys = target_doys[:out_nb_dates]

            nb_hr_dates = int(math.floor(total_nb_dates / (1 + hr_to_lr_ratio)))
            nb_lr_dates = total_nb_dates - nb_hr_dates
            current_hr_sits = MonoModalSITS(
                hr_sits.data[:, :nb_hr_dates, ...],
                hr_sits.doy[:, :nb_hr_dates],
            )

            current_lr_sits = MonoModalSITS(
                lr_sits.data[:, :nb_lr_dates, ...],
                lr_sits.doy[:, :nb_lr_dates],
            )

            logging.info(
                f"Total dates:{total_nb_dates}, hr dates: {nb_hr_dates}, lr dates:\
                {nb_lr_dates}, out nb dates: {out_nb_dates}"
            )
            if not warmed_up:
                logging.info("Warming up ...")
                with torch.no_grad():
                    encoded = encoder(current_lr_sits, current_hr_sits)
                    decoder(encoded, current_target_doys)
                warmed_up = True

            total_time = 0
            cpu_memory = 0
            cuda_memory = 0
            for _ in range(args.nb_runs):
                torch.cuda.empty_cache()

                with MeasureExecTime(label="inference", log_time=False) as chrono:
                    with torch.no_grad():
                        encoded = encoder(current_lr_sits, current_hr_sits)
                        decoder(encoded, current_target_doys)
                        torch.cuda.synchronize()
                total_time += chrono.time
                cuda_memory += torch.cuda.max_memory_allocated()
                cpu_memory += resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                cpu_memory -= cpu_memory_offset
            total_time /= args.nb_runs
            cpu_memory /= args.nb_runs
            cuda_memory /= args.nb_runs
            decoder_records.append(
                [
                    total_nb_dates,
                    out_nb_dates,
                    total_time,
                    cuda_memory / (1024**2),
                    cpu_memory / (1024),
                ]
            )

    decoder_df = pandas.DataFrame(
        data=decoder_records,
        columns=[
            "nb_input_dates",
            "nb_target_dates",
            "time",
            "cuda_memory",
            "cpu_memory",
        ],
    ).sort_values(by="nb_target_dates")

    decoder_df["time_per_mp"] = (
        (10**6) * decoder_df["time"] / ((3 * args.lr_tile_size) ** 2)
    )
    decoder_df["cuda_memory_per_mp"] = (
        (10**6) * decoder_df["cuda_memory"] / ((3 * args.lr_tile_size) ** 2)
    )
    decoder_df["cpu_memory_per_mp"] = (
        (10**6) * decoder_df["cpu_memory"] / ((3 * args.lr_tile_size) ** 2)
    )
    decoder_df.to_csv(args.output, sep="\t")

    print(decoder_df)


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()
    main(args)
