import argparse
import os
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio  # type: ignore
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.rcParams["font.size"] = 10


def read_img(
    img_path: str, bands: list[int], downscale_factor: float = 0.5
) -> np.ndarray:
    """ """
    with rio.open(img_path, "r") as ds:
        data = ds.read(
            out_shape=(
                ds.count,
                int(ds.height * downscale_factor),
                int(ds.width * downscale_factor),
            ),
            resampling=rio.enums.Resampling.bilinear,
        )
        return data[bands, ...]


def read_msk(img_path: str, downscale_factor: float = 0.5) -> np.ndarray:
    """ """
    with rio.open(img_path, "r") as ds:
        data = ds.read(
            out_shape=(
                ds.count,
                int(ds.height * downscale_factor),
                int(ds.width * downscale_factor),
            ),
            resampling=rio.enums.Resampling.nearest,
        )
        return data


def generate_tir_synopsis(
    index_file: str,
    synopsis_path: str,
    nb_img_per_rows: int = 8,
    downscale_factor: float = 0.5,
):
    """
    Generate time series synopsis for fast checking
    """
    df = pd.read_csv(index_file, sep="\t")
    data_dir = os.path.dirname(index_file)
    # Strip invalid products
    df = df[df.valid]

    if not len(df):
        print(f"index file {index_file} has no valid images")

    # Number of rows
    nb_rows = int(np.ceil(len(df) / nb_img_per_rows))
    nb_cols = min(nb_img_per_rows, len(df))
    base_size = 2.5
    figsize = (nb_cols * base_size, nb_rows * base_size)

    fig, axs = plt.subplots(
        nb_rows, nb_cols, layout="constrained", figsize=figsize, squeeze=False
    )

    raveled_axes = axs.ravel()

    for idx, (_, row) in enumerate(df.iterrows()):
        img = read_img(
            os.path.join(data_dir, row.bands),
            bands=[7],
            downscale_factor=downscale_factor,
        )
        img = img / 10.0  # To Kelvin
        img -= 273.15  # To celsius
        msk = read_msk(
            os.path.join(data_dir, row["mask"]),
            downscale_factor=downscale_factor,
        )
        valid_samples = img[msk == 0]
        min_value = np.quantile(valid_samples, 0.05)
        max_value = np.quantile(valid_samples, 0.95)

        cax = raveled_axes[idx].imshow(
            img[0, ...], vmin=min_value, vmax=max_value, resample=True
        )
        fig.colorbar(
            cax,
            ax=raveled_axes[idx],
            location="bottom",
            shrink=0.7,
            extend="both",
        )

        msk = msk.astype(float)
        msk[msk == 0] = np.nan
        raveled_axes[idx].imshow(msk[0, ...], vmax=1.0, vmin=0.0, cmap="OrRd")

        raveled_axes[idx].set_title(
            f"{row.acquisition_date}, {row.clear_pixel_rate:.2%} CPR"
        )
        raveled_axes[idx].set_xticks([])
        raveled_axes[idx].set_yticks([])

    for idx in range(len(df), len(raveled_axes)):
        raveled_axes[idx].set_visible(False)
    fig.savefig(synopsis_path, format="png", bbox_inches="tight")
    plt.close()


def main():
    """
    Main method
    """

    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Test script"
    )

    parser.add_argument(
        "--ts", type=str, help="Path to time-series", required=True, nargs="+"
    )

    parser.add_argument("--output", type=str, help="Path to output", required=True)

    args = parser.parse_args()  # Parser arguments
    Path(args.output).mkdir(exist_ok=True, parents=True)
    for ts in tqdm(args.ts, total=len(args.ts), desc="Generating synopsis"):
        ts_name = os.path.basename(os.path.normpath(ts))
        with suppress(FileNotFoundError):
            generate_tir_synopsis(
                os.path.join(ts, "landsat", "index.csv"),
                os.path.join(args.output, f"{ts_name}_landsat_synopsis.png"),
                downscale_factor=0.5,
            )


if __name__ == "__main__":
    main()
