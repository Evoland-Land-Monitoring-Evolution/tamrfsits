# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
This module contains helper functions to plot graphs in order to analyse the results
"""

import os
from typing import cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray as np_ndarray

S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B11", "B12")
LS_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "LST")


def read_results(results_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read results in pd DataFrames
    """

    lr_df = pd.read_csv(
        os.path.join(results_path, "lr_metrics.csv"), sep="\t", index_col=False
    )
    del lr_df["Unnamed: 0"]
    lr_df = lr_df.rename(columns={"clear_doy": "clear"})
    lr_df["status"] = "clear"
    lr_df["status"].where(lr_df.clear, "masked", inplace=True)

    ls_bands = LS_BANDS  # Fix W8202
    for b in ls_bands:
        lr_df[f"brisque_{b.lower()}"].where(
            lr_df[f"brisque_{b.lower()}"] < 100.0, inplace=True
        )

    current_bands: tuple[str, ...] = ("b1", "b2", "b3", "b4", "b5", "b6", "b7")
    for metric in ("frr", "rmse", "brisque"):
        # pylint: disable=loop-invariant-statement
        lr_df[metric + "_30m"] = sum(
            lr_df[f"{metric}_{b}"] for b in current_bands
        ) / len(current_bands)

    hr_df = pd.read_csv(
        os.path.join(results_path, "hr_metrics.csv"), sep="\t", index_col=False
    )
    del hr_df["Unnamed: 0"]
    hr_df = hr_df.rename(columns={"clear_doy": "clear"})
    hr_df["status"] = "clear"
    hr_df["status"].where(hr_df.clear, "masked", inplace=True)

    current_bands = ("b2", "b3", "b4", "b8")
    for metric in ("frr", "rmse", "brisque"):
        # pylint: disable=loop-invariant-statement
        hr_df[metric + "_10m"] = sum(
            hr_df[f"{metric}_{b}"] for b in current_bands
        ) / len(current_bands)
    current_bands = ("b5", "b6", "b7", "b8a", "b11", "b12")
    for metric in ("frr", "rmse", "brisque"):
        # pylint: disable=loop-invariant-statement
        hr_df[metric + "_20m"] = sum(
            hr_df[f"{metric}_{b}"] for b in current_bands
        ) / len(current_bands)

    return lr_df, hr_df


def catplot_per_site(
    df: pd.DataFrame,
    metric: str,
    out_file: str,
    title: str | None = None,
    font_scale: int = 2,
    ylim: tuple[float, float] | None = None,
    **kwargs,
):
    """
    Seaborn categorical plot organised per site
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=font_scale)
    sites = np.unique(df.name)
    g = sns.catplot(data=df, x="name", y=metric, **kwargs)
    g.set_xticklabels(labels=sites, rotation=45)
    g.set_ylabels(metric)
    if ylim is not None:
        g.axes[0, 0].set_ylim(ylim)
    g.set_xlabels(None)
    if title is not None:
        g.set(title=title)
    g.despine(left=True, bottom=True)
    sns.move_legend(
        g, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=False
    )

    g.savefig(out_file, bbox_inches="tight")
    return out_file


def metric_violintplot(
    df: pd.DataFrame,
    bands: list[str],
    title: str,
    out_fname: str,
    ylabel: str | None = None,
    labels: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    metric: str = "rmse",
    figsize: tuple[int, int] = (10, 5),
):
    """
    Seaborn violin plot per band
    """
    if labels is None:
        labels = bands
    if ylabel is None:
        ylabel = metric
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(figsize=figsize)

    custom_df = df[[f"{metric}_{band}" for band in bands] + ["clear"]]
    custom_df = custom_df.melt(id_vars=["clear"], var_name="band", value_name="metric")
    custom_df["band"] = custom_df.band.str.split("_").str.get(1)

    sns.violinplot(
        data=custom_df,
        x="band",
        y="metric",
        hue="clear",
        bw_adjust=1.0,
        inner="quart",
        cut=1.0,
        linewidth=1,
        palette="Set2",
        split=True,
        scale_hue=False,
        scale="width",
        gap=0.1,
        gridsize=200,
        dodge=True,
    )
    axes.grid(True)
    axes.set_ylabel(ylabel)
    axes.set_xticklabels(labels)
    axes.set_ylabel("Site")
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_title(title)
    sns.despine(left=True, bottom=True)
    fig.savefig(out_fname, bbox_inches="tight")

    return out_fname


def compare_violintplot(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    xp_labels: tuple[str, str],
    bands: list[str],
    title: str,
    out_fname: str,
    ylim: tuple[float, float] | None = None,
    metric: str = "rmse",
    figsize: tuple[int, int] = (10, 5),
):
    """
    Violin plot analysis of given metric
    """

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(figsize=figsize)

    df1["xp"] = xp_labels[0]
    df2["xp"] = xp_labels[1]

    joint_df = pd.concat([df1, df2])

    custom_df = joint_df[[f"{metric}_{band}" for band in bands] + ["clear", "xp"]]
    custom_df = custom_df.melt(
        id_vars=["clear", "xp"], var_name="band", value_name="metric"
    )
    custom_df["band"] = custom_df.band.str.split("_").str.get(1)

    sns.violinplot(
        data=custom_df,
        x="band",
        y="metric",
        hue="xp",
        bw_adjust=1.0,
        inner="quart",
        cut=1.0,
        linewidth=1,
        palette="Set2",
        split=True,
        scale_hue=False,
        scale="width",
        gap=0.1,
        gridsize=200,
        dodge=True,
    )
    axes.grid(True)
    axes.set_ylabel(metric)
    axes.set_xlabel("AOI")
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_title(title)
    sns.despine(left=True, bottom=True)
    fig.savefig(out_fname, bbox_inches="tight")

    return out_fname


def build_result_table(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """
    Build a result table agregated per band an per metric
    """
    clear_lr = df[df.clear]
    masked_lr = df[~df.clear]
    print(masked_lr)
    out_df_content: dict[str, str | list[str] | np_ndarray] = {"bands": labels}

    for metric in ("rmse", "brisque", "frr"):
        for label, cdf in zip(["clear", "masked"], [clear_lr, masked_lr]):
            mean = cdf[[f"{metric}_{b.lower()}" for b in labels]].mean()
            std = cdf[[f"{metric}_{b.lower()}" for b in labels]].std()
            out_df_content[f"{metric}_{label}_mean"] = cast(np_ndarray, mean.values)
            out_df_content[f"{metric}_{label}_std"] = cast(np_ndarray, std.values)

    out_df = pd.DataFrame(out_df_content).set_index("bands")
    return out_df
