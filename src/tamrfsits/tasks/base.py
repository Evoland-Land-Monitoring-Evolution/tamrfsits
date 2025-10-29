# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
This is the base class for training all model parts
"""
import random
from dataclasses import dataclass
from logging import warning
from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from torch import all as torch_all  # pylint: disable=no-name-in-module
from torch import cat as torch_cat  # pylint: disable=no-name-in-module
from torch import full_like as torch_full_like  # pylint: disable=no-name-in-module
from torch import isnan as torch_isnan  # pylint: disable=no-name-in-module
from torch import nanmean as torch_nanmean  # pylint: disable=no-name-in-module

from tamrfsits.core.cca import DatewiseLinearRegressionLoss, HighPassFilteringMode
from tamrfsits.core.downsampling import downsample_sits, downsample_sits_from_mtf
from tamrfsits.core.time_series import MonoModalSITS, crop_sits
from tamrfsits.core.utils import add_ndvi_to_sits, standardize_sits
from tamrfsits.validation.metrics import (
    derive_reference_mask,
    frr_referenceless,
    per_date_masked_rmse,
    per_date_per_band_brisque,
)


@dataclass(frozen=True)
class OptimizationParameters:
    """
    Holds the optimization parameters
    """

    learning_rate: float
    t_0: float
    t_mult: float
    loss: torch.nn.Module
    weight_decay: float = 0.0
    nb_warmup_steps: int = 0
    minimum_learning_rate: float = 0.0


@dataclass(frozen=True)
class StandardizationParameters:
    """
    Holds the standardization parameters
    """

    hr_mean: tuple[float, ...]
    hr_std: tuple[float, ...]
    lr_mean: tuple[float, ...]
    lr_std: tuple[float, ...]


@dataclass(frozen=True)
class MTFParameters:
    """
    Holds the mtf parameters
    """

    hr_mtf: tuple[float, ...]
    hr_resolution: tuple[float, ...]
    lr_mtf: tuple[float, ...]
    lr_resolution: tuple[float, ...]
    output_resolution: float
    learnable_mtfs: bool = True


@dataclass(frozen=True)
class ResolutionForCCA:
    """
    Represent pairs of resolution for CCA
    """

    ref_resolution: float
    pred_resolution: float
    per_date: bool = False
    high_pass_filtering_mode: HighPassFilteringMode = HighPassFilteringMode.NO
    use_lpips: bool = True
    include_hr_ndvi: bool = False
    red_band_idx: int = 2
    nir_band_idx: int = 3
    mtf_for_downsampling: float = 0.4
    scale: float = 1.0


class BaseTrainingModule(pl.LightningModule):
    # pylint: disable=too-many-ancestors
    """
    Lightning wrapper to train the fusion module
    """

    def __init__(
        self,
        optimization: OptimizationParameters,
        standardization: StandardizationParameters,
        mtfs: MTFParameters,
        loss_spatial_margin: int = 5,
        cca_loss_weight: float = 1.0,
        min_clear_pixels_rate_for_data_loss: float = 0.5,
        max_diff_for_data_loss: float | None = None,
        cca_loss: DatewiseLinearRegressionLoss | None = None,
        validation_random_seed: int = 42,
    ):
        """
        initializer
        """

        super().__init__()

        # Placeholder for modules
        self.the_modules = torch.nn.ModuleDict()

        self.current_random_state: None | tuple[Any, ...] = None
        self.optimization_parameters = optimization
        self.learning_rate = optimization.learning_rate
        self.standardization_parameters = standardization
        self.mtf_parameters = mtfs
        self.loss_spatial_margin = loss_spatial_margin
        self.cca_loss_weight = cca_loss_weight
        self.cca_loss_instance = cca_loss
        self.max_diff_for_data_loss = max_diff_for_data_loss
        self.min_clear_pixels_rate_for_data_loss = min_clear_pixels_rate_for_data_loss
        self.validation_random_seed = validation_random_seed

        # Register mean / std buffer
        self.register_buffer(
            "hr_mean",
            torch.tensor(standardization.hr_mean, requires_grad=False),
        )
        self.register_buffer(
            "hr_std",
            torch.tensor(standardization.hr_std, requires_grad=False),
        )
        self.register_buffer(
            "lr_mean",
            torch.tensor(standardization.lr_mean, requires_grad=False),
        )
        self.register_buffer(
            "lr_std",
            torch.tensor(standardization.lr_std, requires_grad=False),
        )

        # Register MTFs / resolution
        if mtfs.learnable_mtfs:
            self.register_parameter(
                "hr_mtf",
                torch.nn.Parameter(torch.tensor(mtfs.hr_mtf), requires_grad=True),
            )
            self.register_parameter(
                "lr_mtf",
                torch.nn.Parameter(torch.tensor(mtfs.lr_mtf), requires_grad=True),
            )

        else:
            self.register_buffer(
                "hr_mtf", torch.tensor(mtfs.hr_mtf, requires_grad=False)
            )
            self.register_buffer(
                "lr_mtf", torch.tensor(mtfs.lr_mtf, requires_grad=False)
            )

        self.register_buffer(
            "hr_resolution", torch.tensor(mtfs.hr_resolution, requires_grad=False)
        )

        self.register_buffer(
            "lr_resolution",
            torch.tensor(mtfs.lr_resolution, requires_grad=False),
        )

    def data_loss(
        self,
        pred_sits: MonoModalSITS,
        target_sits: MonoModalSITS,
        loss_bands: list[int] | None = None,
        context: str = "training",
        modality: str = "hr2lr",
        margin: int = 5,
    ) -> torch.Tensor | None:
        """
        Data loss
        """
        assert torch_all(pred_sits.doy == target_sits.doy)
        assert (
            pred_sits.shape() == target_sits.shape()
        ), f"{pred_sits.shape()=} != {target_sits.shape()=}"

        # Early exit in case of no dates
        if target_sits.shape()[1] == 0:
            return None

        pred_sits = crop_sits(pred_sits, margin)
        target_sits = crop_sits(target_sits, margin)

        pred_data = pred_sits.data
        target_data = target_sits.data

        # Restrict bands on which loss is computed if required
        if loss_bands is not None:
            pred_data = pred_data[:, :, loss_bands, ...]
            target_data = target_data[:, :, loss_bands, ...]

        # Repeat mask to be consistent with augmentations
        loss_mask = torch.ones(
            (
                pred_data.shape[0],
                pred_data.shape[1],
                pred_data.shape[3],
                pred_data.shape[4],
            ),
            dtype=torch.bool,
            device=pred_data.device,
        )
        if target_sits.mask is not None:
            loss_mask = torch.logical_and(loss_mask, ~target_sits.mask)
        if pred_sits.mask is not None:
            loss_mask = torch.logical_and(loss_mask, ~pred_sits.mask)

        # We extend the loss_mask tot the number of channels
        loss_mask = repeat(
            loss_mask, "b t w h -> b t c w h", c=pred_data.shape[2]
        ).clone()
        # print(
        #     f"{modality}, init, margin={margin},
        #     shape={pred_sits.shape()}, valid pixels
        #     rate={loss_mask.sum()/loss_mask.numel()}"
        # )
        # We compute loss on pixel from the training mask that are not
        # masked in target sits

        # We completely mask patches for which the
        # min_clear_pixels_rate_for_data_loss condition is not met

        nb_pixels = loss_mask.shape[-1] * loss_mask.shape[-2]
        clear_rate = loss_mask.sum(dim=(-1, -2)) / nb_pixels
        invalid_patches = clear_rate < self.min_clear_pixels_rate_for_data_loss
        loss_mask = loss_mask.clone()  # To avoid warning about index_put_
        loss_mask[invalid_patches, :, :] = False
        # Testing hard filtering of outliers in loss
        if self.max_diff_for_data_loss is not None:
            loss_mask = torch.logical_and(
                loss_mask,
                torch.abs(target_data - pred_data) <= self.max_diff_for_data_loss,
            )

        # print(
        #     f"{modality}, max diff, margin={margin},
        #     shape={pred_sits.shape()}, valid pixels
        #     rate={loss_mask.sum()/loss_mask.numel()}"
        # )
        # In that case we do not have any pixel to compute loss on
        if (
            not pred_sits.shape()[1]
            or not target_sits.shape()[1]
            or torch_all(~loss_mask)
        ):
            return None

        assert (torch_isnan(pred_data[loss_mask])).sum() == 0

        # We compute loss on selected pixels
        loss_value = self.optimization_parameters.loss(
            pred_data[loss_mask], target_data[loss_mask]
        )
        assert ~torch_isnan(loss_value)

        self.log(
            context + f"/{modality}_loss",
            loss_value,
            batch_size=pred_sits.shape()[0],
            sync_dist=context in ("validation", "test"),
        )

        return loss_value

    def triplet_loss(
        self,
        pred_sits: MonoModalSITS,
        target_sits: MonoModalSITS,
        loss_bands: list[int] | None = None,
        context: str = "training",
        modality: str = "hr2lr",
        doy_distance_threshold: float = 20.0,
        triplet_margin: float = 0.1,
        spatial_margin: int = 5,
        swap: bool = True,
    ) -> torch.Tensor | None:
        """
        Triplet loss to avoid interpolating clouds

        This loss enforces that predicted pixel that have a masked
        target reference should be closer to the closest clear target
        pixel in time than to the masked pixels

        Step1 : find pixel that are clear in prediction but masked in target

        Step2 : identify closest target pixel in time for each of them


        """
        assert torch_all(pred_sits.doy == target_sits.doy)
        assert (
            pred_sits.shape() == target_sits.shape()
        ), f"{pred_sits.shape()=} != {target_sits.shape()=}"

        # Early exit in case of no dates
        if target_sits.shape()[1] == 0:
            return None

        pred_sits = crop_sits(pred_sits, spatial_margin)
        target_sits = crop_sits(target_sits, spatial_margin)

        pred_data = pred_sits.data
        target_data = target_sits.data

        # Restrict bands on which loss is computed if required
        if loss_bands is not None:
            pred_data = pred_data[:, :, loss_bands, ...]
            target_data = target_data[:, :, loss_bands, ...]

        loss_mask = target_sits.mask
        assert loss_mask is not None

        # Loss now contains all pixels that are clear in prediction but not in target
        if pred_sits.mask is not None:
            loss_mask = torch.logical_and(loss_mask, ~pred_sits.mask)

        # Avoid bad target_data range
        # Set no-data values to zero
        target_data_clean = (
            target_data  # torch.masked_fill(target_data, target_data < -10.0, 0.0)
        )

        # Now, we look for closest clear pixels

        # First, extent the doy tensors
        pred_doy = repeat(
            pred_sits.doy,
            "b t -> b t w h",
            w=pred_sits.shape()[-2],
            h=pred_sits.shape()[-1],
        )
        target_doy = repeat(
            target_sits.doy,
            "b t -> b t w h",
            w=pred_sits.shape()[-2],
            h=pred_sits.shape()[-1],
        )

        # Now, doy of masked pixels will be set to a large value
        assert target_sits.mask is not None
        target_doy_masked = torch.masked_fill(target_doy, target_sits.mask, -10000)

        # Next, we compute the distance matrix between pred_doy and target_doy
        doy_distance_matrix = torch.abs(
            pred_doy[:, None, ...] - target_doy_masked[:, :, None, ...]
        )

        # From this matrix, we can identify the closest clear pixel in target_data
        closest_target_pixel_doy_distance, closest_target_pixel_idx = torch.min(
            doy_distance_matrix, dim=1
        )

        closest_target_pixel_idx = repeat(
            closest_target_pixel_idx, "b t w h -> b t c w h", c=target_data.shape[2]
        )
        closest_target_data = torch.gather(
            target_data_clean, 1, closest_target_pixel_idx
        )

        # Only consider pixels that have closest clear doy below the user-defined treshold
        loss_mask = torch.logical_and(
            loss_mask,
            closest_target_pixel_doy_distance < doy_distance_threshold,
        )

        if torch.any(loss_mask):
            loss_values = torch.nn.functional.triplet_margin_loss(
                rearrange(pred_data, "b t c w h -> (b t w h c) 1"),
                rearrange(closest_target_data, "b t c w h -> (b t w h c) 1"),
                rearrange(target_data_clean, "b t c w h -> (b t w h c) 1"),
                margin=triplet_margin,
                reduction="none",
                p=2,
                swap=swap,
            )

            loss_value = torch.mean(
                loss_values[
                    repeat(loss_mask, "b t w h -> (b t w h c)", c=pred_data.shape[2])
                ]
            )
            self.log(
                context + f"/{modality}_triplet_loss",
                loss_value,
                batch_size=pred_sits.shape()[0],
                sync_dist=context in ("validation", "test"),
            )

            return loss_value
        return None

    def cca_loss(
        self,
        input_hr: MonoModalSITS,
        pred_lr: MonoModalSITS,
        pred_hr: MonoModalSITS,
        cca_config: ResolutionForCCA,
        label: str = "",
        context: str = "training",
    ) -> torch.Tensor | None:
        """
        Compute cca loss
        """
        if self.cca_loss_instance is None:
            warning(
                "cca_loss() method has been called by self.cca_loss instance is None"
            )
            return None
        # early exit
        if input_hr.shape()[1] == 0:
            return None
        # Derive bands to use as reference
        hr_ref_bands = self.hr_resolution <= cca_config.ref_resolution
        hr_pred_bands = self.hr_resolution == cca_config.pred_resolution
        lr_pred_bands = self.lr_resolution == cca_config.pred_resolution

        # Compute scale factor to got to resolution
        scale_factor = self.mtf_parameters.output_resolution / cca_config.ref_resolution

        # Resample everything to resolution
        if scale_factor == 1.0:
            ref_hr_sits = MonoModalSITS(
                input_hr.data[:, :, hr_ref_bands, ...], input_hr.doy, input_hr.mask
            )
        else:
            ref_hr_sits = downsample_sits_from_mtf(
                MonoModalSITS(
                    input_hr.data[:, :, hr_ref_bands, ...], input_hr.doy, input_hr.mask
                ),
                cca_config.ref_resolution,
                mtf_res=self.hr_resolution[hr_ref_bands],
                mtf_fc=torch_full_like(
                    self.hr_resolution[hr_ref_bands],
                    cca_config.mtf_for_downsampling,
                    device=input_hr.data.device,
                ),
                factor=scale_factor,
            )
        pred_hr_sits: MonoModalSITS | None = None
        if torch.any(hr_pred_bands):
            if scale_factor == 1.0:
                pred_hr_sits = MonoModalSITS(
                    pred_hr.data[:, :, hr_pred_bands, ...], input_hr.doy, input_hr.mask
                )
            else:
                pred_hr_sits = downsample_sits_from_mtf(
                    MonoModalSITS(
                        pred_hr.data[:, :, hr_pred_bands, ...],
                        input_hr.doy,
                        input_hr.mask,
                    ),
                    cca_config.ref_resolution,
                    mtf_res=self.hr_resolution[hr_pred_bands],
                    mtf_fc=torch_full_like(
                        self.hr_resolution[hr_pred_bands],
                        cca_config.mtf_for_downsampling,
                        device=input_hr.data.device,
                    ),
                    factor=scale_factor,
                )
        pred_lr_sits: MonoModalSITS | None = None
        if torch.any(lr_pred_bands):
            if scale_factor == 1.0:
                pred_lr_sits = MonoModalSITS(
                    pred_lr.data[:, :, lr_pred_bands, ...], pred_lr.doy, pred_lr.mask
                )
            else:
                pred_lr_sits = downsample_sits_from_mtf(
                    MonoModalSITS(
                        pred_lr.data[:, :, lr_pred_bands, ...],
                        pred_lr.doy,
                        pred_lr.mask,
                    ),
                    cca_config.ref_resolution,
                    mtf_res=self.lr_resolution[lr_pred_bands],
                    mtf_fc=torch_full_like(
                        self.lr_resolution[lr_pred_bands],
                        cca_config.mtf_for_downsampling,
                        device=pred_lr.data.device,
                    ),
                    factor=scale_factor,
                )

        # Concatenate to form pred_sits and ref_sits
        pred_sits = MonoModalSITS(
            torch_cat(
                [
                    sits.data
                    for sits in (pred_lr_sits, pred_hr_sits)
                    if sits is not None
                ],
                dim=2,
            ),
            pred_lr.doy,
        )

        margin = int(self.loss_spatial_margin / scale_factor)
        # print(f"cca loss margin = {margin}")
        pred_sits = crop_sits(pred_sits, margin)
        ref_hr_sits = crop_sits(ref_hr_sits, margin)

        # if we need to include ndvi in hr reference sits
        if cca_config.include_hr_ndvi:
            ref_hr_sits = add_ndvi_to_sits(
                ref_hr_sits,
                red_band_mean_std=(
                    self.hr_mean[cca_config.red_band_idx],
                    self.hr_std[cca_config.red_band_idx],
                ),
                nir_band_mean_std=(
                    self.hr_mean[cca_config.nir_band_idx],
                    self.hr_std[cca_config.nir_band_idx],
                ),
            )

        # Compute cca loss

        if (
            loss_value := self.cca_loss_instance(
                pred_sits,
                ref_hr_sits,
                per_date=cca_config.per_date,
                use_lpips=cca_config.use_lpips,
                high_pass_filtering_mode=cca_config.high_pass_filtering_mode,
            )
        ) is not None:
            self.log(
                context
                + f"/{label}_cca_{cca_config.ref_resolution}_{cca_config.pred_resolution}_loss",
                cca_config.scale * loss_value,
                batch_size=pred_hr.shape()[0],
                sync_dist=context in ("validation", "test"),
            )
            return cca_config.scale * loss_value
        return None

    def testing_validation_metrics(
        self,
        pred_sits: MonoModalSITS,
        pred_sits_full_res: MonoModalSITS,
        target_sits: MonoModalSITS,
        resolutions: torch.Tensor,
        modality: str = "hr2hr",
    ):
        """
        Compute and log additional metrics
        """
        assert target_sits.mask is not None
        assert torch_all(~torch_isnan(pred_sits.data))
        assert torch_all(~torch_isnan(pred_sits_full_res.data))
        assert torch_all(~torch_isnan(target_sits.data))
        factor = float(target_sits.shape()[-1]) / pred_sits_full_res.shape()[-1]
        if target_sits.shape()[-1] == pred_sits_full_res.shape()[-1]:
            target_sits_full_res = target_sits
        else:
            target_sits_full_res = downsample_sits(
                target_sits,
                factor=factor,
            )
        assert target_sits_full_res.shape() == pred_sits_full_res.shape()

        metric_mask = derive_reference_mask(
            target_sits.mask,
            nb_features=target_sits.data.shape[2],
            spatial_margin=int(self.config.loss_spatial_margin * factor),
        )
        rmse = per_date_masked_rmse(pred_sits.data, target_sits.data, metric_mask)

        brisque = per_date_per_band_brisque(
            crop_sits(pred_sits_full_res, margin=self.config.loss_spatial_margin).data
        )

        frr_output = frr_referenceless(
            crop_sits(pred_sits_full_res, margin=self.config.loss_spatial_margin).data,
            crop_sits(
                target_sits_full_res, margin=self.config.loss_spatial_margin
            ).data,
        )
        if frr_output is not None:
            frr, _, _, _ = frr_output
        else:
            frr = None
        unique_resolutions = torch.unique(resolutions)

        # average over date and batch
        for metric, metric_label in zip(
            (rmse, brisque, frr), ("rmse", "brisque", "frr")
        ):
            if metric is not None and metric.numel() > 0:
                avg_metric = torch_nanmean(metric, dim=0)
                for resolution in unique_resolutions:
                    band_mask = resolutions == resolution.item()
                    metric_value = torch_nanmean(avg_metric[band_mask])
                    if torch_all(~torch_isnan(metric_value)):
                        self.log(
                            f"metrics/{modality}_{resolution}m_{metric_label}",
                            metric_value,
                            batch_size=pred_sits.shape()[0],
                            sync_dist=True,
                        )
                metric_value = torch_nanmean(avg_metric)
                if torch_all(~torch_isnan(metric_value)):
                    self.log(
                        f"metrics/{modality}_{metric_label}",
                        metric_value,
                        batch_size=pred_sits.shape()[0],
                        sync_dist=True,
                    )

    @torch.compile(dynamic=True)
    def preprocessing(
        self, batch_data: tuple[MonoModalSITS, MonoModalSITS]
    ) -> tuple[
        tuple[MonoModalSITS, MonoModalSITS],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Preprocessing (scaling, floating point conversion, etc) goes here
        """
        mean_hr = self.hr_mean
        mean_lr = self.lr_mean
        std_hr = self.hr_std
        std_lr = self.lr_std

        # Standardize
        hr_sits_std = standardize_sits(batch_data[1], mean_hr, std_hr, scale=10000.0)

        lr_sits_std = standardize_sits(
            batch_data[0],
            mean_lr,
            std_lr,
            scale=10000.0,
        )
        return (lr_sits_std, hr_sits_std), (mean_lr, std_lr, mean_hr, std_hr)

    def generic_step(
        self, batch_data: tuple[MonoModalSITS, MonoModalSITS], context: str = "training"
    ) -> torch.Tensor:
        """
        Implement training step
        """
        raise NotImplementedError

    def training_step(  # pylint: disable=arguments-differ
        self, batch_data: tuple[MonoModalSITS, MonoModalSITS], _batch_idx: int
    ) -> torch.Tensor:
        """
        The actual training step
        """
        return self.generic_step(batch_data, context="training")

    def on_validation_start(self):
        """
        Save current seed when validation start
        """
        self.current_random_state = random.getstate()
        random.seed(self.validation_random_seed)

    def on_validation_end(self):
        """
        Reset the random state for training
        """
        assert self.current_random_state is not None
        random.setstate(self.current_random_state)
        self.current_random_state = None

    def validation_step(  # pylint: disable=arguments-differ
        self, batch_data: tuple[MonoModalSITS, MonoModalSITS], _batch_idx: int
    ) -> torch.Tensor:
        """
        The actual training step
        """
        return self.generic_step(batch_data, context="validation")

    def test_step(  # pylint: disable=arguments-differ
        self, batch_data: tuple[MonoModalSITS, MonoModalSITS], _batch_idx: int
    ) -> torch.Tensor:
        """
        The actual training step
        """
        with torch.random.fork_rng(devices=[self.hr_mean.device]):
            torch.manual_seed(42)
            return self.generic_step(batch_data, context="test")

    def configure_optimizers(self):
        """
        Optimizer and scheduler
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
            betas=(0.9, 0.999),
        )
        # pylint: disable=use-tuple-over-list
        optimizers = [optimizer]

        if self.config.optimization.nb_warmup_steps > 0:
            scheduler: torch.optim.lr_scheduler.LRScheduler = (
                torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            total_iters=self.config.optimization.nb_warmup_steps,
                        ),
                        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer,
                            T_0=self.config.optimization.t_0,
                            T_mult=self.config.optimization.t_mult,
                            eta_min=self.config.optimization.minimum_learning_rate,
                        ),
                    ],
                    milestones=[self.config.optimization.nb_warmup_steps],
                )
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.optimization.t_0,
                T_mult=self.config.optimization.t_mult,
                eta_min=self.config.optimization.minimum_learning_rate,
            )

        # pylint: disable=use-tuple-over-list
        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]

        return optimizers, schedulers

    def on_validation_model_zero_grad(self) -> None:
        """
        Small hack to avoid first validation on resume.
        This will NOT work if the gradient accumulation step should be performed at this point.

        Imported from: https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
        """
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(
            self, "_restarting_skip_val_flag", True
        ):
            self.trainer.sanity_checking = True
            # pylint: disable=attribute-defined-outside-init
            self._restarting_skip_val_flag = False
