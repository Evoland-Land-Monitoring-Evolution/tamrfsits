# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Contains classes and functions related to the validation workflow
"""

from dataclasses import dataclass

import pandas as pd
import torch
from matplotlib import pyplot as plt

from tamrfsits.core.downsampling import downsample_sits, downsample_sits_from_mtf
from tamrfsits.core.time_series import (
    MonoModalSITS,
    crop_sits,
    subset_doy_monomodal_sits,
)
from tamrfsits.tasks.interpolation.training_module import (
    TemporalInterpolationTrainingModule,
)
from tamrfsits.validation.metrics import (
    closest_date_in_sits,
    derive_reference_mask,
    frr_referenceless,
    per_date_clear_pixel_rate,
    per_date_masked_rmse,
    per_date_per_band_brisque,
    sits_density,
)
from tamrfsits.validation.strategy import TestingConfiguration
from tamrfsits.validation.utils import MeasureExecTime, tiled_inference


@dataclass(frozen=True)
class TimeSeriesTestResult:
    """
    Aggregate metrics per time-series
    """

    name: str
    sensor: str
    rmse: torch.Tensor
    brisque: torch.Tensor
    frr: torch.Tensor
    target_brisque: torch.Tensor
    doy: torch.Tensor
    clear_doy: torch.Tensor
    clear_pixel_rate: torch.Tensor
    closest_doy: torch.Tensor
    closest_joint_doy: torch.Tensor
    local_density: torch.Tensor
    local_joint_density: torch.Tensor
    ref_prof: torch.Tensor
    pred_prof: torch.Tensor
    freqs: torch.Tensor
    pred_pixel_rate: torch.Tensor

    def __post_init__(self):
        """
        Check data integrity
        """
        assert self.rmse.dim() == 2
        assert self.clear_doy.dtype == torch.bool
        assert self.rmse.shape == self.brisque.shape
        assert self.rmse.shape == self.target_brisque.shape
        nb_doys = self.rmse.shape[0]

        for t in (
            self.doy,
            self.clear_doy,
            self.clear_pixel_rate,
            self.local_density,
            self.local_joint_density,
        ):
            assert t.dim() == 1
            assert t.shape[0] == nb_doys
        assert torch.all(self.closest_doy[self.clear_doy] == 0.0)
        assert torch.all(self.closest_joint_doy[self.clear_doy] == 0.0)
        assert torch.all(self.local_density[self.clear_doy] == 1.0)
        assert torch.all(self.local_joint_density[self.clear_doy] == 1.0)


def to_pandas(
    results: list[TimeSeriesTestResult], band_labels: list[str]
) -> pd.DataFrame:
    """
    Converts the results to a pandas dataframe for analysis
    """
    data_dict_list: list[dict] = []
    assert results
    nb_channels = results[0].rmse.shape[1]
    band_labels_str = [f"_{b}" for b in band_labels]
    # Loop on results and doy
    for result in results:
        assert result.rmse.shape[1] == nb_channels
        for idx, doy in enumerate(result.doy):
            current_dict = {
                "name": result.name,
                "doy": doy.item(),
                "clear_doy": result.clear_doy[idx].item(),
                "clear_pixel_rate": result.clear_pixel_rate[idx].item(),
                "closest_doy": result.closest_doy[idx].item(),
                "closest_joint_doy": result.closest_joint_doy[idx].item(),
                "local_density": result.local_density[idx].item(),
                "local_joint_density": result.local_joint_density[idx].item(),
                "pred_pixel_rate": result.pred_pixel_rate[idx].item(),
            }
            current_rmse = result.rmse[idx, ...]
            current_brisque = result.brisque[idx, ...]
            current_target_brisque = result.target_brisque[idx, ...]
            current_frr = result.frr[idx, ...]
            # pylint: disable=loop-invariant-statement
            rmse_dict = {
                "rmse" + label: current_rmse[bidx].item()
                for bidx, label in enumerate(band_labels_str)
            }
            # pylint: disable=loop-invariant-statement
            brisque_dict = {
                "brisque" + label: current_brisque[bidx].item()
                for bidx, label in enumerate(band_labels_str)
            }
            # pylint: disable=loop-invariant-statement
            target_brisque_dict = {
                "target_brisque" + label: current_target_brisque[bidx].item()
                for bidx, label in enumerate(band_labels_str)
            }
            # pylint: disable=loop-invariant-statement
            frr_dict = {
                "frr" + label: current_frr[bidx].item()
                for bidx, label in enumerate(band_labels_str)
            }

            # Update current dict with band dict
            current_dict.update(rmse_dict)
            current_dict.update(brisque_dict)
            current_dict.update(frr_dict)
            current_dict.update(target_brisque_dict)
            data_dict_list.append(current_dict)

    return pd.DataFrame(data=data_dict_list)


def model_predict(
    test_config: TestingConfiguration,
    model: TemporalInterpolationTrainingModule,
    subtile_width: int = 165,
    all_doys: bool = False,
    show_progress: bool = True,
) -> tuple[MonoModalSITS, MonoModalSITS]:
    """
    Performs model inference for given test configuration
    """
    if all_doys:
        all_doys_tensor = torch.unique(
            torch.cat(
                [
                    s.doy.ravel()
                    for s in (test_config.lr_target, test_config.hr_target)
                    if s is not None
                ]
            )
        )
        return tiled_inference(
            test_config.lr_input,
            test_config.hr_input,
            model,
            all_doys_tensor,
            all_doys_tensor,
            show_progress=show_progress,
            width=subtile_width,
        )

    return tiled_inference(
        test_config.lr_input,
        test_config.hr_input,
        model,
        test_config.lr_target.doy if test_config.lr_target is not None else None,
        test_config.hr_target.doy if test_config.hr_target is not None else None,
        show_progress=show_progress,
        width=subtile_width,
    )


def get_tensor(values: list[float], device: torch.device | str) -> torch.Tensor:
    """
    Cached access to tensor on GPU
    """
    return torch.tensor(values, device=device)


ComputeMetricsResultType = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def compute_lr_metrics(
    pred_lr: MonoModalSITS,
    target_lr: MonoModalSITS,
    mtf_res: torch.Tensor,
    mtf_fc: torch.Tensor,
    margin: int = 5,
    profile: bool = False,
) -> ComputeMetricsResultType:
    """
    Compute LR metrics
    """
    with MeasureExecTime("Prepare data for metrics", profile):
        # Up-scaled input sits for FDA
        target_lr_up = downsample_sits(
            target_lr,
            factor=float(target_lr.shape()[-1] / pred_lr.shape()[-1]),
        )
        # Downscaled sits for RMSE
        pred_lr_ds = downsample_sits_from_mtf(
            pred_lr,
            10.0,
            mtf_res=mtf_res,
            mtf_fc=mtf_fc,
            factor=float(pred_lr.shape()[-1]) / target_lr.shape()[-1],
        )

        assert target_lr.mask is not None
        lr_ref_mask = derive_reference_mask(
            target_lr.mask,
            nb_features=target_lr.shape()[2],
            spatial_margin=margin,
        )
        # We assume batch_size of 1
        assert pred_lr.doy.shape[0] == 1

    with MeasureExecTime("RMSE metric", profile):
        # If TIR is produced, compute metrics in °K
        if pred_lr_ds.data.shape[2] >= 8:
            pred_lr_ds.data[:, :, 7, ...] *= 1000

        lr_rmse = per_date_masked_rmse(pred_lr_ds.data, target_lr.data, lr_ref_mask)

    pred_lr = crop_sits(pred_lr, margin=margin)
    target_lr_up = crop_sits(target_lr_up, margin=margin)

    with MeasureExecTime("FRR metric", profile):
        if pred_lr.data.shape[2] >= 8:
            pred_lr.data[:, :, 7, ...] /= 1000
        if target_lr_up.data.shape[2] >= 8:
            target_lr_up.data[:, :, 7, ...] /= 1000.0
        frr_referenceless_result = frr_referenceless(
            pred_lr.data, target_lr_up.data, patch_size=990
        )
        assert frr_referenceless_result is not None
        lr_frr, pred_lr_prof, ref_lr_prof, freqs = frr_referenceless_result

    with MeasureExecTime("BRISQUE metric", profile):
        if target_lr.data.shape[2] >= 8:
            # Here we scale TIR data for each date with the range of LST
            # from the reference, in order to get a meaningful BRISQUE value
            target_lr.data[:, :, 7, ...] /= 1000.0
            min_lst = (
                target_lr.data[:, :, 7, ...]
                .min(dim=-1, keepdim=True)[0]
                .min(dim=-2, keepdim=True)[0]
            )
            max_lst = (
                target_lr.data[:, :, 7, ...]
                .max(dim=-1, keepdim=True)[0]
                .max(dim=-2, keepdim=True)[0]
            )
            target_lr.data[:, :, 7, ...] = (target_lr.data[:, :, 7, ...] - min_lst) / (
                max_lst - min_lst
            )
            pred_lr.data[:, :, 7, ...] = (
                1000.0 * pred_lr.data[:, :, 7, ...] - min_lst
            ) / (max_lst - min_lst)
        lr_brisque = per_date_per_band_brisque(pred_lr.data)
        lr_target_brisque = per_date_per_band_brisque(target_lr.data)

        return (
            lr_rmse,
            lr_brisque,
            lr_frr,
            lr_target_brisque,
            pred_lr_prof,
            ref_lr_prof,
            freqs,
        )


def compute_hr_metrics(
    pred_hr: MonoModalSITS,
    target_hr: MonoModalSITS,
    mtf_res: torch.Tensor,
    mtf_fc: torch.Tensor,
    margin: int = 5,
    profile: bool = False,
) -> ComputeMetricsResultType:
    """
    Compute metrics from given predictions and test_configuration
    """
    with MeasureExecTime("Prepare data for metrics", profile):
        target_hr_up = target_hr
        # Downscaled sits for RMSE

        pred_hr_ds = downsample_sits_from_mtf(
            pred_hr,
            10.0,
            mtf_res=mtf_res,
            mtf_fc=mtf_fc,
            factor=float(pred_hr.shape()[-1]) / target_hr.shape()[-1],
        )

        # Union of all possible doys
        assert target_hr.mask is not None
        hr_ref_mask = derive_reference_mask(
            target_hr.mask,
            nb_features=target_hr.shape()[2],
            spatial_margin=margin,
        )
        # We assume batch_size of 1
        assert pred_hr.doy.shape[0] == 1

    with MeasureExecTime("RMSE metric", profile):
        hr_rmse = per_date_masked_rmse(pred_hr_ds.data, target_hr.data, hr_ref_mask)

    pred_hr = crop_sits(pred_hr, margin=margin)
    target_hr_up = crop_sits(target_hr_up, margin=margin)

    with MeasureExecTime("FRR metric", profile):
        frr_referenceless_results = frr_referenceless(
            pred_hr.data, target_hr_up.data, patch_size=990
        )
        assert frr_referenceless_results is not None
        hr_frr, pred_hr_prof, ref_hr_prof, freqs = frr_referenceless_results

    with MeasureExecTime("BRISQUE metric", profile):
        hr_brisque = per_date_per_band_brisque(pred_hr.data)
        hr_target_brisque = per_date_per_band_brisque(target_hr.data)

    return (
        hr_rmse,
        hr_brisque,
        hr_frr,
        hr_target_brisque,
        pred_hr_prof,
        ref_hr_prof,
        freqs,
    )


def compute_metrics(
    pred_lr: MonoModalSITS | None,
    pred_hr: MonoModalSITS | None,
    test_config: TestingConfiguration,
    lr_mtf_res: torch.Tensor,
    lr_mtf_fc: torch.Tensor,
    hr_mtf_res: torch.Tensor,
    hr_mtf_fc: torch.Tensor,
    name: str = "",
    margin: int = 5,
    profile: bool = False,
) -> tuple[TimeSeriesTestResult | None, TimeSeriesTestResult | None]:
    """
    Compute metrics from given predictions and test_configuration
    """
    # Union of all possible doys
    all_doys = torch.unique(
        torch.cat(
            [
                s.doy
                for s in (test_config.lr_input, test_config.hr_input)
                if s is not None
            ],
            dim=1,
        )
    )
    lr_test_results: TimeSeriesTestResult | None = None
    if pred_lr is not None and test_config.lr_target is not None:
        assert test_config.lr_input is not None
        lr_clear_doy = pred_lr.doy[
            0, torch.isin(pred_lr.doy[0, ...], test_config.lr_input.doy)
        ]
        lr_masked_doy = pred_lr.doy[
            0, ~torch.isin(pred_lr.doy[0, ...], test_config.lr_input.doy)
        ]

        results = [
            (
                *compute_lr_metrics(
                    subset_doy_monomodal_sits(pred_lr, doys),
                    subset_doy_monomodal_sits(test_config.lr_target, doys),
                    lr_mtf_res,
                    lr_mtf_fc,
                    margin=margin,
                    profile=profile,
                ),
                torch.full_like(doys, clear_value, dtype=torch.bool),
            )
            for doys, clear_value in zip((lr_clear_doy, lr_masked_doy), (True, False))
            if doys.numel()
        ]
        if not results:
            raise ValueError("Found no predicted dates for metrics computation")
        lr_rmse = torch.cat([r[0] for r in results])
        lr_brisque = torch.cat([r[1] for r in results])
        lr_frr = torch.cat([r[2] for r in results])
        lr_target_brisque = torch.cat([r[3] for r in results])
        pred_lr_prof = torch.cat([r[4] for r in results])
        ref_lr_prof = torch.cat([r[5] for r in results])
        freqs = results[0][6]
        lr_clear_doy_mask = torch.cat([r[7] for r in results])
        if test_config.lr_input is not None and test_config.lr_target is not None:
            all_lr_doys = torch.cat(
                [d for d in (lr_clear_doy, lr_masked_doy) if d is not None]
            )
            lr_closest_doy = closest_date_in_sits(
                test_config.lr_input.doy[0, ...], all_lr_doys
            )
            lr_local_density = sits_density(
                all_lr_doys, test_config.lr_input.doy[0, ...]
            )
            lr_closest_joint_doy = closest_date_in_sits(all_doys, all_lr_doys)
            lr_local_joint_density = sits_density(all_lr_doys, all_doys)
            assert test_config.lr_target.mask is not None
            lr_clear_pixel_rate = per_date_clear_pixel_rate(test_config.lr_target.mask)

            lr_test_results = TimeSeriesTestResult(
                name=name,
                sensor="landsat",
                doy=all_lr_doys,
                clear_doy=lr_clear_doy_mask,
                rmse=lr_rmse,
                frr=lr_frr,
                brisque=lr_brisque,
                target_brisque=lr_target_brisque,
                closest_doy=lr_closest_doy,
                closest_joint_doy=lr_closest_joint_doy,
                local_density=lr_local_density,
                local_joint_density=lr_local_joint_density,
                clear_pixel_rate=lr_clear_pixel_rate,
                pred_pixel_rate=torch.ones_like(lr_clear_pixel_rate),
                pred_prof=pred_lr_prof,
                ref_prof=ref_lr_prof,
                freqs=freqs,
            )

    hr_test_results: TimeSeriesTestResult | None = None
    if pred_hr is not None and test_config.hr_target is not None:
        hr_clear_doy: None | torch.Tensor = None
        if test_config.hr_input is not None:
            hr_clear_doy = pred_hr.doy[
                0, torch.isin(pred_hr.doy[0, ...], test_config.hr_input.doy)
            ]
            hr_masked_doy = pred_hr.doy[
                0, ~torch.isin(pred_hr.doy[0, ...], test_config.hr_input.doy)
            ]
        else:
            hr_masked_doy = pred_hr.doy[0, ...]

        results = [
            (
                *compute_hr_metrics(
                    subset_doy_monomodal_sits(pred_hr, doys),
                    subset_doy_monomodal_sits(test_config.hr_target, doys),
                    hr_mtf_res,
                    hr_mtf_fc,
                    margin=margin,
                    profile=profile,
                ),
                torch.full_like(doys, clear_value, dtype=torch.bool),
            )
            for doys, clear_value in zip((hr_clear_doy, hr_masked_doy), (True, False))
            if doys is not None and doys.numel()
        ]
        if not results:
            raise ValueError("Found no predicted dates for metrics computation")

        hr_rmse = torch.cat([r[0] for r in results])
        hr_brisque = torch.cat([r[1] for r in results])
        hr_frr = torch.cat([r[2] for r in results])
        hr_target_brisque = torch.cat([r[3] for r in results])
        pred_hr_prof = torch.cat([r[4] for r in results])
        ref_hr_prof = torch.cat([r[5] for r in results])
        freqs = results[0][6]
        hr_clear_doy_mask = torch.cat([r[7] for r in results])
        if test_config.hr_input is not None and test_config.hr_target is not None:
            all_hr_doys = torch.cat(
                [d for d in (hr_clear_doy, hr_masked_doy) if d is not None]
            )
            hr_closest_doy = closest_date_in_sits(
                test_config.hr_input.doy[0, ...],
                all_hr_doys,
            )
            hr_local_density = sits_density(
                all_hr_doys,
                test_config.hr_input.doy[0, ...],
            )
            hr_closest_joint_doy = closest_date_in_sits(all_doys, all_hr_doys)
            hr_local_joint_density = sits_density(all_hr_doys, all_doys)

        else:
            all_hr_doys = hr_masked_doy
            hr_closest_doy = (
                hr_local_density
            ) = hr_closest_joint_doy = hr_local_joint_density = torch.full_like(
                test_config.hr_target.doy[0, ...], torch.nan, dtype=torch.float
            )

        assert test_config.hr_target.mask is not None
        hr_clear_pixel_rate = per_date_clear_pixel_rate(test_config.hr_target.mask)

        hr_test_results = TimeSeriesTestResult(
            name=name,
            sensor="landsat",
            doy=all_hr_doys,
            clear_doy=hr_clear_doy_mask,
            rmse=hr_rmse,
            frr=hr_frr,
            brisque=hr_brisque,
            target_brisque=hr_target_brisque,
            closest_doy=hr_closest_doy,
            closest_joint_doy=hr_closest_joint_doy,
            local_density=hr_local_density,
            local_joint_density=hr_local_joint_density,
            clear_pixel_rate=hr_clear_pixel_rate,
            pred_pixel_rate=torch.ones_like(hr_clear_pixel_rate),
            pred_prof=pred_hr_prof,
            ref_prof=ref_hr_prof,
            freqs=freqs,
        )

    return lr_test_results, hr_test_results


def write_log_profiles(
    pred_prof: torch.Tensor,
    ref_prof: torch.Tensor,
    freqs: torch.Tensor,
    outfile: str,
    labels: list[str] | None = None,
):
    """
    Write the log profile figure
    """
    if labels is None:
        labels = [f"B{i+1}" for i in range(pred_prof.shape[-1])]
    nrows = min(pred_prof.shape[-1], 2)
    ncols = max(pred_prof.shape[-1] // 2, 1)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
    )

    axes = axes.ravel()

    for i in range(pred_prof.shape[-1]):
        axes[i].plot(freqs.cpu().numpy(), pred_prof[:, i].cpu().numpy(), color="red")
        axes[i].plot(freqs.cpu().numpy(), ref_prof[:, i].cpu().numpy(), color="blue")
        axes[i].grid(True)
        axes[i].set_ylabel("Signal attenuation (dB)")
        axes[i].set_xlabel("Spatial freq. (1/px)")
        axes[i].set_title(labels[i])
    #        axes[i].set_ylim([-40, 0])
    fig.savefig(outfile, format="pdf", bbox_inches="tight")
