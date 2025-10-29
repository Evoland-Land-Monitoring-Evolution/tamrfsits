# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This is the lightning module for pre-training spatial encoders and decoders
"""
from dataclasses import dataclass

import torch

from tamrfsits.components.transformer import TemporalDecoder, TemporalEncoder
from tamrfsits.core.cca import DatewiseLinearRegressionLoss
from tamrfsits.core.decorrelation import decorrelation_loss
from tamrfsits.core.downsampling import downsample_sits_from_mtf
from tamrfsits.core.time_series import MonoModalSITS, detach
from tamrfsits.core.utils import (
    doy_unique,
    elementwise_subset_doy_monomodal_sits,
    split_sits_features,
    standardize_sits,
    strip_masks,
    unstandardize_sits,
)
from tamrfsits.tasks.base import (
    BaseTrainingModule,
    MTFParameters,
    OptimizationParameters,
    ResolutionForCCA,
    StandardizationParameters,
)
from tamrfsits.validation.strategy import (
    MAEParameters,
    generate_configurations,
    generate_mae_strategy,
)


@dataclass(frozen=True)
class TemporalInterpolationTrainingModuleParameters:
    """
    Parameters for training module
    """

    optimization: OptimizationParameters
    standardization: StandardizationParameters
    mtfs: MTFParameters
    # Restrict bands taken into account for loss
    hr2lr_loss_bands: list[int] | None = None
    lr2hr_loss_bands: list[int] | None = None
    loss_spatial_margin: int = 5
    cca_loss_weight: float = 0.1
    resolutions_for_cca_loss: tuple[ResolutionForCCA, ...] = (
        ResolutionForCCA(10.0, 20.0, per_date=True),
        ResolutionForCCA(10.0, 30.0, per_date=True),
        ResolutionForCCA(10.0, 90.0, per_date=True),
    )
    # disabled by default
    decorrelation_loss_weight: float | None = None
    min_clear_pixels_rate_for_data_loss: float = 0.5
    validation_random_seed: int = 42
    max_diff_for_data_loss: float | None = None
    mae_parameters: MAEParameters = MAEParameters()
    scale_mae_loss: float | None = 0.5
    cca_loss_on_predictions: bool = False
    use_triplet_loss: bool = False
    triplet_loss_margin_clear: float = 0.00001
    triplet_loss_margin_masked: float = 0.00001
    triplet_loss_doy_distance_threshold: float = 20.0
    triplet_loss_weight: float = 1.0
    compute_metrics: bool = False


# Split between mae and clr outputs
def sits_mae_clr_splitter(
    sits: MonoModalSITS, input_doy: torch.Tensor, target_doy: torch.Tensor
):
    """
    Split sits between clear and masked sits according to clear doys
    """
    mae_doy = target_doy[~torch.isin(target_doy, input_doy)]
    mae_output = elementwise_subset_doy_monomodal_sits(sits, mae_doy)
    clr_output = elementwise_subset_doy_monomodal_sits(sits, input_doy)
    assert mae_output.doy.shape[1] + clr_output.doy.shape[1] == target_doy.shape[1]
    return mae_output, clr_output


class TemporalInterpolationTrainingModule(BaseTrainingModule):
    # pylint: disable=too-many-ancestors
    """
    Lightning wrapper to train the fusion module
    """

    def __init__(
        self,
        config: TemporalInterpolationTrainingModuleParameters,
        encoder: TemporalEncoder,
        decoder: TemporalDecoder,
        cca_loss: DatewiseLinearRegressionLoss | None = None,
    ):
        """
        Initializer
        """
        super().__init__(
            config.optimization,
            config.standardization,
            config.mtfs,
            loss_spatial_margin=config.loss_spatial_margin,
            cca_loss_weight=config.cca_loss_weight,
            cca_loss=cca_loss,
            min_clear_pixels_rate_for_data_loss=config.min_clear_pixels_rate_for_data_loss,
            validation_random_seed=config.validation_random_seed,
            max_diff_for_data_loss=config.max_diff_for_data_loss,
        )
        # self.mem_reporter = MemReporter()
        self.the_modules = torch.nn.ModuleDict({"encoder": encoder, "decoder": decoder})
        self.config = config

    def reconstruction_loss(
        self,
        pred: MonoModalSITS,
        ref: MonoModalSITS,
        resolution: torch.Tensor,
        mtf: torch.Tensor,
        weight: float,
        loss_bands: list[int] | None,
        mean: torch.Tensor,
        std: torch.Tensor,
        modality: str,
        task: str,
        context: str,
    ) -> torch.Tensor | None:
        """
        Compute the reconstruction loss for a single configuration (clr/mae and hr/lr)
        """
        # Initialize output loss
        has_loss = False
        total_loss = torch.zeros((), device=pred.data.device)

        factor = float(pred.shape()[-1]) / ref.shape()[-1]
        # Get back to the initial resolution
        pred_ds = downsample_sits_from_mtf(
            pred,
            self.mtf_parameters.output_resolution,
            mtf_res=resolution,
            mtf_fc=mtf,
            factor=factor,
        )

        # Fidelity loss
        masked_loss = self.data_loss(
            pred_ds,
            ref,
            context=context,
            loss_bands=loss_bands,
            modality=f"{modality}_{task}",
            margin=int(self.loss_spatial_margin / factor),
        )
        if masked_loss is not None:
            self.log(
                context + f"/weighted_{modality}_{task}_loss",
                weight * masked_loss,
                batch_size=pred_ds.shape()[0],
                sync_dist=context in ("validation", "test"),
            )

            has_loss = True
            total_loss += weight * masked_loss

        if self.config.use_triplet_loss:
            if modality == "lr":
                loss_bands = self.config.hr2lr_loss_bands
            triplet_loss = self.triplet_loss(
                pred_ds,
                ref,
                context=context,
                loss_bands=loss_bands,
                modality=f"{modality}_{task}",
                spatial_margin=int(self.loss_spatial_margin / factor),
                triplet_margin=(
                    self.config.triplet_loss_margin_clear
                    if task == "clr"
                    else self.config.triplet_loss_margin_masked
                ),
                doy_distance_threshold=self.config.triplet_loss_doy_distance_threshold,
            )
            if triplet_loss is not None:
                self.log(
                    context + f"/weighted_{modality}_{task}_triplet_loss",
                    weight * self.config.triplet_loss_weight * triplet_loss,
                    batch_size=pred_ds.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )

                has_loss = True
                total_loss += weight * self.config.triplet_loss_weight * triplet_loss

        if context in ("validation", "test") and self.config.compute_metrics:
            # Upsample target for computation of frr
            pred_ustd = unstandardize_sits(pred, mean, std)
            pred_ds_ustd = unstandardize_sits(pred_ds, mean, std)
            target_ustd = unstandardize_sits(ref, mean, std)
            self.testing_validation_metrics(
                pred_ds_ustd,
                pred_ustd,
                target_ustd,
                resolution,
                modality=f"{modality}_{task}",
            )

        if has_loss:
            return total_loss
        return None

    def task_losses(
        self,
        lr_output: MonoModalSITS,
        hr2lr_output: MonoModalSITS,
        lr_target: MonoModalSITS,
        hr_output: MonoModalSITS,
        hr_target: MonoModalSITS,
        lr_loss_bands: list[int] | None,
        hr_loss_bands: list[int] | None,
        total_lr_dates: int = 1,
        total_hr_dates: int = 1,
        task: str = "mae",
        context: str = "training",
    ) -> torch.Tensor | None:
        """
        Compute all the losses for current task (masked/clear)
        """
        lr_weight = float(lr_target.doy.shape[1]) / (total_lr_dates + total_hr_dates)
        hr_weight = float(hr_target.doy.shape[1]) / (total_lr_dates + total_hr_dates)

        # Initialize output loss
        has_loss = False
        total_loss = torch.zeros((), device=lr_output.data.device)

        if self.cca_loss_instance and not self.config.cca_loss_on_predictions:
            cca_has_loss = False
            cca_total_loss = torch.zeros((), device=lr_output.data.device)
            nb_cca_losses = 0
            for cca_config in self.config.resolutions_for_cca_loss:
                cca_loss_value = self.cca_loss(
                    hr_target,
                    hr2lr_output,
                    hr_output,
                    cca_config,
                    context=context,
                    label=task,
                )
                if cca_loss_value is not None:
                    cca_has_loss = True
                    cca_total_loss += cca_loss_value
                    nb_cca_losses += 1
            if cca_has_loss:
                cca_total_loss /= nb_cca_losses
                self.log(
                    context + f"/{task}_cca_loss",
                    cca_total_loss,
                    batch_size=hr_target.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )
                self.log(
                    context + f"/weighted_{task}_cca_loss",
                    self.config.cca_loss_weight * cca_total_loss * hr_weight,
                    batch_size=hr_target.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )

                has_loss = True
                total_loss += self.config.cca_loss_weight * cca_total_loss * hr_weight
        if total_lr_dates > 0:
            # print( f"step {self.global_step}, {task}, lr,
            #     weight={lr_target.doy.shape[1]}/{total_lr_dates}={lr_weight}"
            #     )

            lr_loss = self.reconstruction_loss(
                lr_output,
                lr_target,
                self.lr_resolution,
                self.lr_mtf,
                lr_weight,
                lr_loss_bands,
                self.lr_mean,
                self.lr_std,
                "lr",
                task,
                context,
            )
            if lr_loss is not None:
                has_loss = True
                total_loss += lr_loss

        if total_hr_dates > 0:
            # print( f"step {self.global_step}, {task}, hr,
            #     weight={hr_target.doy.shape[1]}/{total_hr_dates}={hr_weight}"
            #     )

            hr_loss = self.reconstruction_loss(
                hr_output,
                hr_target,
                self.hr_resolution,
                self.hr_mtf,
                hr_weight,
                hr_loss_bands,
                self.hr_mean,
                self.hr_std,
                "hr",
                task,
                context,
            )
            if hr_loss is not None:
                has_loss = True
                total_loss += hr_loss

        if has_loss:
            return total_loss
        return None

    def generic_step(
        self, batch_data: tuple[MonoModalSITS, MonoModalSITS], context: str = "training"
    ):
        """
        Implement training step
        Butterfly cross-reconstruction on conjunctions
        trains hr_encoder, lr_encoder, hr_decoder, lr_decoder
        """

        # Preprocessin of data
        (input_lr, input_hr), _ = self.preprocessing(batch_data)

        # Generation of MAE strategy for current step
        mae_strategy = generate_mae_strategy(self.config.mae_parameters)
        # print(f"step {self.global_step}, mae strategy: {mae_strategy}")
        # Use the strategy to generate data for current step
        data_for_current_step = next(
            iter(generate_configurations((input_lr, input_hr), parameters=mae_strategy))
        )

        assert data_for_current_step.lr_target is not None
        assert data_for_current_step.hr_target is not None
        assert data_for_current_step.lr_input is not None
        assert data_for_current_step.hr_input is not None

        # Compute total number of dates for lr and hr series
        # before_forward_memory = torch.cuda.memory_allocated()
        total_lr_dates = data_for_current_step.lr_target.doy.shape[1]
        total_hr_dates = data_for_current_step.hr_target.doy.shape[1]

        # nb_input_lr_dates = data_for_current_step.lr_input.doy.shape[1]
        # nb_input_hr_dates = data_for_current_step.hr_input.doy.shape[1]

        # derive unique doys for each element in batch
        query_doy, valid_doys_mask = doy_unique(input_lr.doy, input_hr.doy)

        # Ensure that invalid doys do not collide with possible doy values
        query_doy[~valid_doys_mask] = -1

        # Encode joint sits
        encoded = self.the_modules["encoder"](
            strip_masks(data_for_current_step.lr_input),
            strip_masks(data_for_current_step.hr_input),
        )
        # Initialize output loss
        has_loss = False
        total_loss = torch.zeros((), device=encoded.data.device)

        # Compute deccorelation loss if required
        if self.config.decorrelation_loss_weight is not None:
            decorr_loss_value = decorrelation_loss(encoded)
            if decorr_loss_value is not None:
                self.log(
                    context + "/decorrelation_loss",
                    decorr_loss_value,
                    batch_size=input_lr.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )
                self.log(
                    context + "/weighted_decorrelation_loss",
                    self.config.decorrelation_loss_weight * decorr_loss_value,
                    batch_size=input_lr.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )

                has_loss = True
                total_loss += self.config.decorrelation_loss_weight * decorr_loss_value

        # Call decoder on query doys
        decoded = self.the_modules["decoder"](encoded, query_doy)
        lr_output, hr_output = split_sits_features(decoded, input_lr.shape()[2])
        # after_forward_memory = torch.cuda.memory_allocated()

        assert lr_output is not None
        assert hr_output is not None

        if self.config.cca_loss_on_predictions and self.cca_loss is not None:
            cca_total_loss = torch.zeros((), device=lr_output.data.device)
            nb_cca_losses = 0
            cca_has_loss = False
            for cca_config in self.config.resolutions_for_cca_loss:
                cca_loss_value = self.cca_loss(
                    detach(hr_output),
                    lr_output,
                    hr_output,
                    cca_config,
                    context=context,
                    label="pred",
                )
                if cca_loss_value is not None:
                    cca_has_loss = True
                    cca_total_loss += cca_loss_value
                    nb_cca_losses += 1
            if cca_has_loss:
                cca_total_loss /= nb_cca_losses
                self.log(
                    context + "/pred_cca_loss",
                    cca_total_loss,
                    batch_size=hr_output.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )
                self.log(
                    context + "/weighted_pred_cca_loss",
                    self.config.cca_loss_weight * cca_total_loss,
                    batch_size=hr_output.shape()[0],
                    sync_dist=context in ("validation", "test"),
                )

                has_loss = True
                total_loss += self.config.cca_loss_weight * cca_total_loss

        # Split targets and outputs between clear and masked sits
        lr_mae_output, lr_clr_output = sits_mae_clr_splitter(
            lr_output,
            data_for_current_step.lr_input.doy,
            data_for_current_step.lr_target.doy,
        )
        hr_mae_output, hr_clr_output = sits_mae_clr_splitter(
            hr_output,
            data_for_current_step.hr_input.doy,
            data_for_current_step.hr_target.doy,
        )
        lr_mae_target, lr_clr_target = sits_mae_clr_splitter(
            data_for_current_step.lr_target,
            data_for_current_step.lr_input.doy,
            data_for_current_step.lr_target.doy,
        )
        hr_mae_target, hr_clr_target = sits_mae_clr_splitter(
            data_for_current_step.hr_target,
            data_for_current_step.hr_input.doy,
            data_for_current_step.hr_target.doy,
        )

        # Also predict LR at same date than hr_mae_target
        hr2lr_mae_output = elementwise_subset_doy_monomodal_sits(
            lr_output, hr_mae_target.doy
        )
        hr2lr_clr_output = elementwise_subset_doy_monomodal_sits(
            lr_output, hr_clr_target.doy
        )

        mae_loss = self.task_losses(
            lr_mae_output,
            hr2lr_mae_output,
            lr_mae_target,
            hr_mae_output,
            hr_mae_target,
            self.config.hr2lr_loss_bands,
            None,
            total_lr_dates,
            total_hr_dates,
            "mae",
            context,
        )
        if mae_loss is not None:
            self.log(
                context + "/weighted_mae_loss",
                mae_loss
                * (
                    self.config.scale_mae_loss
                    if self.config.scale_mae_loss is not None
                    else 1.0
                ),
                batch_size=input_lr.shape()[0],
                sync_dist=context in ("validation", "test"),
            )

            has_loss = True
            total_loss += mae_loss * (
                self.config.scale_mae_loss
                if self.config.scale_mae_loss is not None
                else 1.0
            )

        clr_loss = self.task_losses(
            lr_clr_output,
            hr2lr_clr_output,
            lr_clr_target,
            hr_clr_output,
            hr_clr_target,
            None,
            None,
            total_lr_dates,
            total_hr_dates,
            "clr",
            context,
        )
        if clr_loss is not None:
            self.log(
                context + "/weighted_clr_loss",
                clr_loss,
                batch_size=input_lr.shape()[0],
                sync_dist=context in ("validation", "test"),
            )
            has_loss = True
            total_loss += clr_loss

        # after_loss_memory = torch.cuda.memory_allocated() with
        # open("mem_vs_nb_dates.log", "a") as f: f.write(
        # f"{nb_input_lr_dates}\t{nb_input_hr_dates}\t{query_doy.shape[1]}
        # \t{before_forward_memory/1e9}\t{after_forward_memory/1e9}\t{after_loss_memory/1e9}\n"
        # )

        if has_loss:
            self.log(
                context + "/total_loss",
                total_loss,
                batch_size=input_lr.shape()[0],
                sync_dist=context in ("validation", "test"),
            )

            return total_loss
        return None

    def predict(
        self,
        lr_sits: MonoModalSITS | None,
        hr_sits: MonoModalSITS | None,
        lr_query_doy: torch.Tensor | None,
        hr_query_doy: torch.Tensor | None,
        downscale: bool = False,
    ) -> tuple[MonoModalSITS | None, MonoModalSITS | None]:
        """
        Inference for self interpolation task
        """
        # Eval mode important because of potential dropout
        encoder = self.the_modules["encoder"].eval()
        decoder = self.the_modules["decoder"].eval()

        hr_sits = (
            standardize_sits(hr_sits, self.hr_mean, self.hr_std, scale=10000.0)
            if hr_sits
            else None
        )
        lr_sits = (
            standardize_sits(lr_sits, self.lr_mean, self.lr_std, scale=10000.0)
            if lr_sits
            else None
        )

        with torch.no_grad():
            encoded = encoder(
                strip_masks(lr_sits) if lr_sits else None,
                strip_masks(hr_sits) if hr_sits else None,
            )
            output_lr: MonoModalSITS | None = None
            if lr_query_doy is not None:
                output_lr, _ = split_sits_features(
                    decoder(
                        encoded,
                        lr_query_doy,
                    ),
                    self.lr_mean.shape[0],
                )
            output_hr: MonoModalSITS | None = None
            if hr_query_doy is not None:
                _, output_hr = split_sits_features(
                    decoder(
                        encoded,
                        hr_query_doy,
                    ),
                    self.lr_mean.shape[0],
                )
            if downscale:
                if output_lr is not None:
                    assert lr_sits is not None
                    output_lr = downsample_sits_from_mtf(
                        output_lr,
                        self.mtf_parameters.output_resolution,
                        mtf_res=self.lr_resolution,
                        mtf_fc=self.lr_mtf,
                        factor=float(output_lr.shape()[-1]) / lr_sits.shape()[-1],
                    )
                if output_hr is not None:
                    assert hr_sits is not None
                    output_hr = downsample_sits_from_mtf(
                        output_hr,
                        self.mtf_parameters.output_resolution,
                        mtf_res=self.hr_resolution,
                        mtf_fc=self.hr_mtf,
                        factor=float(output_hr.shape()[-1]) / hr_sits.shape()[-1],
                    )

        return (
            unstandardize_sits(output_lr, self.lr_mean, self.lr_std)
            if output_lr
            else None
        ), (
            unstandardize_sits(output_hr, self.hr_mean, self.hr_std)
            if output_hr
            else None
        )
