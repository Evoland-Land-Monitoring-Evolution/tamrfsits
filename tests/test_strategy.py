# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Contains tests for the validation.strategy module
"""
import pytest

from tamrfsits.validation.strategy import (
    MAEParameters,
    ValidationParameters,
    ValidationStrategy,
    generate_configurations,
    generate_mae_strategy,
)

from .tests_utils import generate_monomodal_sits


@pytest.mark.parametrize(
    "parameters",
    [
        ValidationParameters(),
        ValidationParameters(strategy=ValidationStrategy.RANDOM),
        ValidationParameters(strategy=ValidationStrategy.NOHR),
        ValidationParameters(strategy=ValidationStrategy.NOLR),
        ValidationParameters(strategy=ValidationStrategy.FORECAST),
        ValidationParameters(strategy=ValidationStrategy.BACKCAST),
        ValidationParameters(strategy=ValidationStrategy.GAPS),
        ValidationParameters(strategy=ValidationStrategy.DEEPHARMO),
    ],
    ids=["All", "Random", "NoHR", "NoLR", "Forecast", "Backcast", "Gaps", "DeepHarmo"],
)
def test_generate_configurations(parameters: ValidationParameters):
    """
    Test for the testing_configuration_generator function
    """
    lr_sits = generate_monomodal_sits(batch=1)
    hr_sits = generate_monomodal_sits(batch=1)

    confs = list(generate_configurations((lr_sits, hr_sits), parameters=parameters))

    assert len(confs) == 1


def test_generate_mae_strategy():
    """
    Test the generate_mae_strategy function
    """
    parameters = MAEParameters()
    strategy = generate_mae_strategy(MAEParameters())

    assert (
        parameters.rate_for_random_strategy_range[0]
        <= strategy.rate_for_random_strategy
        <= parameters.rate_for_random_strategy_range[1]
    )

    assert (
        parameters.gaps_size_range[0]
        <= strategy.gaps_size
        <= parameters.gaps_size_range[1]
    )

    assert (
        parameters.forecast_doy_start_range[0]
        <= strategy.forecast_doy_start
        <= parameters.forecast_doy_start_range[1]
    )
