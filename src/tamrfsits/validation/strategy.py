# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Contains classes and function related to MAE training and validation strategies
"""


import math
import random
from collections.abc import Generator
from dataclasses import dataclass
from datetime import date
from enum import Enum

import torch
from torch import logical_and as torch_logical_and  # pylint: disable=no-name-in-module

from tamrfsits.core.time_series import MonoModalSITS, subset_doy_monomodal_sits


class ValidationStrategy(Enum):
    """
    This class represents the validation strategy
    """

    ALL = "ALL"
    RANDOM = "RANDOM"
    GAPS = "GAPS"
    NOHR = "NOHR"
    NOLR = "NOLR"
    FORECAST = "FORECAST"
    BACKCAST = "BACKCAST"
    DEEPHARMO = "DEEPHARMO"
    ALL2CONJHR = "ALL2CONJHR"
    # pylint: disable=invalid-name
    CONJLRuHR2HR = "CONJLRuHR2HR"
    CONJLR2HR = "CONJLR2HR"
    # pylint: disable=invalid-name
    LRuHRNOCONJ2HR = "LRuHRNOCONJ2HR"
    HRNOCONJ2HR = "HRNOCONJ2HR"
    ALLHR2ALLHR = "ALLHR2ALLHR"
    RANDOM_ALL_DOYS = "RANDOM_ALL_DOYS"
    L3A = "L3A"
    L3A_10D = "L3A_10D"
    CONTEXT = "CONTEXT"
    CUSTOM = "CUSTOM"


DEFAULT_MAE_STRATEGIES = (
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.RANDOM,
    ValidationStrategy.NOHR,
    ValidationStrategy.NOLR,
    ValidationStrategy.FORECAST,
    ValidationStrategy.BACKCAST,
    ValidationStrategy.GAPS,
    ValidationStrategy.GAPS,
    ValidationStrategy.GAPS,
    ValidationStrategy.GAPS,
)


@dataclass(frozen=True)
class MAEParameters:
    """
    Parameters for the random generation of MAE strategies
    """

    strategies: tuple[ValidationStrategy, ...] = DEFAULT_MAE_STRATEGIES
    rate_for_random_strategy_range: tuple[float, float] = (0.2, 0.7)
    forecast_doy_start_range: tuple[int, int] = (65, 300)
    gaps_size_range: tuple[int, int] = (30, 90)


@dataclass(frozen=True)
class ValidationParameters:
    """
    Holds the parameters for the masking strategy for validation
    """

    strategy: ValidationStrategy = ValidationStrategy.ALL
    rate_for_random_strategy: float = 0.5
    forecast_doy_start: int = 183
    gaps_size: int = 30
    context_reference_size: int = 60
    context_start: int = 160
    nb_context_images: int = 1
    custom_dates: list[int] | None = None


@dataclass
class TestingConfiguration:
    """
    Model a testing configuration
    """

    lr_input: MonoModalSITS | None
    hr_input: MonoModalSITS | None
    lr_target: MonoModalSITS | None
    hr_target: MonoModalSITS | None

    def normalize_for_tests(self):
        """
        Apply common preprocessing to target data
        """
        # ensure that configuration are always generated with same seed
        if self.lr_target is not None:
            self.lr_target = MonoModalSITS(
                self.lr_target.data / 10000.0, self.lr_target.doy, self.lr_target.mask
            )
            # If TIR is produced, compute metrics in °K
            if self.lr_target.data.shape[2] >= 8:
                self.lr_target.data[:, :, 7, ...] *= 1000
            elif self.lr_target.data.shape[2] == 1:
                self.lr_target.data[:, :, 0, ...] *= 1000
        if self.hr_target is not None:
            self.hr_target = MonoModalSITS(
                self.hr_target.data / 10000.0, self.hr_target.doy, self.hr_target.mask
            )


def mask_sits_by_doy(
    sits: MonoModalSITS, mask: list[int] | torch.Tensor
) -> MonoModalSITS:
    """
    Apply a doy mask to sits
    """
    return MonoModalSITS(
        sits.data[:, mask, ...],
        sits.doy[:, mask],
        sits.mask[:, mask, ...] if sits.mask is not None else None,
    )


def generate_mae_strategy(parameters: MAEParameters) -> ValidationParameters:
    """
    Random generation of the MAE strategy
    """
    return ValidationParameters(
        strategy=random.choice(parameters.strategies),
        rate_for_random_strategy=random.uniform(
            *parameters.rate_for_random_strategy_range
        ),
        forecast_doy_start=random.randint(*parameters.forecast_doy_start_range),
        gaps_size=random.randint(*parameters.gaps_size_range),
    )


def generate_configurations(
    batch: tuple[MonoModalSITS, MonoModalSITS], parameters: ValidationParameters
) -> Generator[TestingConfiguration, None, None]:
    """
    Generate possible configurations for testing
    """
    lr_sits, hr_sits = batch
    assert lr_sits.shape()[0] == hr_sits.shape()[0] == 1

    if parameters.strategy is ValidationStrategy.ALL:
        yield TestingConfiguration(lr_sits, hr_sits, lr_sits, hr_sits)

    elif parameters.strategy is ValidationStrategy.RANDOM_ALL_DOYS:
        # Find dates that we will remove for this batch
        nb_lr_dates = lr_sits.shape()[1]
        nb_hr_dates = hr_sits.shape()[1]

        # Make a shuffled list of all doy indices
        all_lr_dates = list(range(nb_lr_dates))
        random.shuffle(all_lr_dates)
        all_hr_dates = list(range(nb_hr_dates))
        random.shuffle(all_hr_dates)
        # Decide where to split between masked and clear dates
        lr_split_index = int(
            math.ceil(nb_lr_dates * parameters.rate_for_random_strategy)
        )
        hr_split_index = int(
            math.ceil(nb_hr_dates * parameters.rate_for_random_strategy)
        )

        # Make sorted list of which image will be clear and which will be masked
        clear_lr_dates = sorted(all_lr_dates[lr_split_index:])
        clear_hr_dates = sorted(all_hr_dates[hr_split_index:])

        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            subset_doy_monomodal_sits(
                lr_sits, torch.cat((lr_sits.doy, hr_sits.doy), dim=1)
            ),
            subset_doy_monomodal_sits(
                hr_sits, torch.cat((lr_sits.doy, hr_sits.doy), dim=1)
            ),
        )

    elif parameters.strategy is ValidationStrategy.RANDOM:
        # Find dates that we will remove for this batch
        nb_lr_dates = lr_sits.shape()[1]
        nb_hr_dates = hr_sits.shape()[1]

        # Make a shuffled list of all doy indices
        all_lr_dates = list(range(nb_lr_dates))
        random.shuffle(all_lr_dates)
        all_hr_dates = list(range(nb_hr_dates))
        random.shuffle(all_hr_dates)

        # Decide where to split between masked and clear dates
        lr_split_index = int(
            math.ceil(nb_lr_dates * parameters.rate_for_random_strategy)
        )
        hr_split_index = int(
            math.ceil(nb_hr_dates * parameters.rate_for_random_strategy)
        )

        # Make sorted list of which image will be clear and which will be masked
        clear_lr_dates = sorted(all_lr_dates[lr_split_index:])
        clear_hr_dates = sorted(all_hr_dates[hr_split_index:])
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            lr_sits,
            hr_sits,
        )

    elif parameters.strategy is ValidationStrategy.GAPS:
        lr_doys = torch.ones_like(lr_sits.doy, dtype=torch.bool)
        hr_doys = torch.ones_like(hr_sits.doy, dtype=torch.bool)

        for d in range(parameters.gaps_size, 365, 2 * parameters.gaps_size):
            lr_doys[
                torch_logical_and(
                    lr_sits.doy > d, lr_sits.doy < d + parameters.gaps_size
                )
            ] = False
            hr_doys[
                torch_logical_and(
                    hr_sits.doy > d, hr_sits.doy < d + parameters.gaps_size
                )
            ] = False
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, lr_doys[0, ...]),
            mask_sits_by_doy(hr_sits, hr_doys[0, ...]),
            lr_sits,
            hr_sits,
        )
    elif parameters.strategy is ValidationStrategy.NOHR:
        clear_lr_dates = list(range(lr_sits.doy.shape[1]))
        clear_hr_dates = []
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            lr_sits,
            hr_sits,
        )
    elif parameters.strategy is ValidationStrategy.NOLR:
        clear_hr_dates = list(range(hr_sits.doy.shape[1]))
        clear_lr_dates = []
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            lr_sits,
            hr_sits,
        )

    elif parameters.strategy is ValidationStrategy.FORECAST:
        clear_lr_dates = [
            i
            for i in range(lr_sits.doy.shape[1])
            if lr_sits.doy[0, i] <= parameters.forecast_doy_start
        ]
        clear_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if hr_sits.doy[0, i] <= parameters.forecast_doy_start
        ]
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            lr_sits,
            hr_sits,
        )

    elif parameters.strategy is ValidationStrategy.BACKCAST:
        clear_lr_dates = [
            i
            for i in range(lr_sits.doy.shape[1])
            if lr_sits.doy[0, i] >= parameters.forecast_doy_start
        ]
        clear_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if hr_sits.doy[0, i] >= parameters.forecast_doy_start
        ]
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            lr_sits,
            hr_sits,
        )
    elif parameters.strategy is ValidationStrategy.DEEPHARMO:
        clear_lr_dates = list(range(lr_sits.doy.shape[1]))
        clear_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if not torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            lr_sits,
            hr_sits,
        )
    elif parameters.strategy is ValidationStrategy.CONJLRuHR2HR:
        clear_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if not torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        clear_lr_dates = [
            i
            for i in range(lr_sits.doy.shape[1])
            if torch.isin(lr_sits.doy[0, i], hr_sits.doy)
        ]

        target_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            None,
            mask_sits_by_doy(hr_sits, target_hr_dates),
        )
    elif parameters.strategy is ValidationStrategy.CONJLR2HR:
        clear_lr_dates = [
            i
            for i in range(lr_sits.doy.shape[1])
            if torch.isin(lr_sits.doy[0, i], hr_sits.doy)
        ]

        target_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        yield TestingConfiguration(
            mask_sits_by_doy(lr_sits, clear_lr_dates),
            None,
            None,
            mask_sits_by_doy(hr_sits, target_hr_dates),
        )
    elif parameters.strategy is ValidationStrategy.LRuHRNOCONJ2HR:
        clear_lr_dates = list(range(lr_sits.doy.shape[1]))
        clear_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if not torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        target_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        yield TestingConfiguration(
            lr_sits,
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            None,
            mask_sits_by_doy(hr_sits, target_hr_dates),
        )
    elif parameters.strategy is ValidationStrategy.HRNOCONJ2HR:
        clear_lr_dates = list(range(lr_sits.doy.shape[1]))
        clear_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if not torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        target_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        yield TestingConfiguration(
            None,
            mask_sits_by_doy(hr_sits, clear_hr_dates),
            None,
            mask_sits_by_doy(hr_sits, target_hr_dates),
        )
    elif parameters.strategy is ValidationStrategy.ALLHR2ALLHR:
        yield TestingConfiguration(
            None,
            hr_sits,
            None,
            hr_sits,
        )
    elif parameters.strategy is ValidationStrategy.ALL2CONJHR:
        target_hr_dates = [
            i
            for i in range(hr_sits.doy.shape[1])
            if torch.isin(hr_sits.doy[0, i], lr_sits.doy)
        ]
        yield TestingConfiguration(
            lr_sits, hr_sits, None, mask_sits_by_doy(hr_sits, target_hr_dates)
        )
    elif parameters.strategy is ValidationStrategy.L3A:
        target_doy = (
            torch.tensor(
                [
                    [date(2022, i, 15).timetuple().tm_yday for i in range(1, 13)]
                    + [date(2022, i, 1).timetuple().tm_yday for i in range(1, 13)]
                ],
                device=hr_sits.doy.device,
            )
            - 1
        )
        target_doy = torch.sort(target_doy, dim=1)[0]
        dummy_target_data = torch.zeros(
            (
                hr_sits.data.shape[0],
                target_doy.shape[1],
                hr_sits.data.shape[2],
                hr_sits.data.shape[3],
                hr_sits.data.shape[4],
            ),
            device=hr_sits.data.device,
        )
        dummy_target_mask = torch.ones(
            (
                hr_sits.data.shape[0],
                target_doy.shape[1],
                hr_sits.data.shape[3],
                hr_sits.data.shape[4],
            ),
            device=hr_sits.data.device,
            dtype=torch.bool,
        )
        yield TestingConfiguration(
            lr_sits,
            hr_sits,
            None,
            MonoModalSITS(dummy_target_data, target_doy, dummy_target_mask),
        )
    elif parameters.strategy is ValidationStrategy.L3A_10D:
        target_doy = torch.arange(0, 365, 10, device=hr_sits.data.device)[None, :]
        dummy_target_data = torch.zeros(
            (
                hr_sits.data.shape[0],
                target_doy.shape[1],
                hr_sits.data.shape[2],
                hr_sits.data.shape[3],
                hr_sits.data.shape[4],
            ),
            device=hr_sits.data.device,
        )
        dummy_target_mask = torch.ones(
            (
                hr_sits.data.shape[0],
                target_doy.shape[1],
                hr_sits.data.shape[3],
                hr_sits.data.shape[4],
            ),
            device=hr_sits.data.device,
            dtype=torch.bool,
        )
        yield TestingConfiguration(
            lr_sits,
            hr_sits,
            None,
            MonoModalSITS(dummy_target_data, target_doy, dummy_target_mask),
        )
    elif parameters.strategy is ValidationStrategy.CONTEXT:
        reference_end = parameters.context_reference_size + parameters.context_start
        first_idx_lr = torch.argmax(
            lr_sits.doy.masked_fill(lr_sits.doy > parameters.context_start, 0)
        )

        last_idx_lr = torch.argmin(
            lr_sits.doy.masked_fill(lr_sits.doy < reference_end, 365)
        )
        first_idx_hr = torch.argmax(
            hr_sits.doy.masked_fill(hr_sits.doy > parameters.context_start, 0)
        )
        last_idx_hr = torch.argmin(
            hr_sits.doy.masked_fill(hr_sits.doy < reference_end, 365)
        )
        nb_lr_doys = lr_sits.doy.shape[1]
        nb_hr_doys = hr_sits.doy.shape[1]

        lr_target_doy = lr_sits.doy[:, first_idx_lr + 1 : last_idx_lr]
        lr_input_doy = torch.cat(
            (
                lr_sits.doy[
                    :,
                    max(
                        0,
                        first_idx_lr + 1 - parameters.nb_context_images,
                    ) : first_idx_lr
                    + 1,
                ],
                lr_sits.doy[
                    :,
                    last_idx_lr : min(
                        last_idx_lr + parameters.nb_context_images, nb_lr_doys
                    ),
                ],
            ),
            dim=1,
        )
        hr_target_doy = hr_sits.doy[:, first_idx_hr + 1 : last_idx_hr]
        hr_input_doy = torch.cat(
            (
                hr_sits.doy[
                    :,
                    max(
                        0, first_idx_hr + 1 - parameters.nb_context_images
                    ) : first_idx_hr
                    + 1,
                ],
                hr_sits.doy[
                    :,
                    last_idx_hr : min(
                        last_idx_hr + parameters.nb_context_images, nb_hr_doys
                    ),
                ],
            ),
            dim=1,
        )
        yield TestingConfiguration(
            subset_doy_monomodal_sits(lr_sits, lr_input_doy),
            subset_doy_monomodal_sits(hr_sits, hr_input_doy),
            subset_doy_monomodal_sits(
                lr_sits,
                torch.sort(torch.cat((lr_input_doy, lr_target_doy), dim=1), dim=1)[0],
            ),
            subset_doy_monomodal_sits(
                hr_sits,
                torch.sort(torch.cat((hr_input_doy, hr_target_doy), dim=1), dim=1)[0],
            ),
        )
    elif parameters.strategy is ValidationStrategy.CUSTOM:
        assert parameters.custom_dates is not None

        target_doy = torch.tensor(parameters.custom_dates, device=hr_sits.data.device)
        target_doy = target_doy[None, ...]

        yield TestingConfiguration(
            lr_sits,
            hr_sits,
            subset_doy_monomodal_sits(lr_sits, target_doy),
            subset_doy_monomodal_sits(hr_sits, target_doy),
        )

    else:
        raise ValueError(parameters.strategy)
