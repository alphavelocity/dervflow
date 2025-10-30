# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pytest

from dervflow import utils


def test_validate_option_params_valid():
    is_valid, message = utils.validate_option_params(
        spot=100.0,
        strike=95.0,
        rate=0.03,
        dividend=0.01,
        volatility=0.2,
        time=1.0,
    )
    assert is_valid is True
    assert message is None


def test_validate_option_params_invalid_values():
    is_valid, message = utils.validate_option_params(
        spot=-1.0,
        strike=100.0,
        rate=0.03,
        dividend=0.01,
        volatility=0.2,
        time=1.0,
    )
    assert is_valid is False
    assert "Spot price" in message


def test_validate_option_params_non_finite():
    is_valid, message = utils.validate_option_params(
        spot=float("nan"),
        strike=100.0,
        rate=0.03,
        dividend=0.01,
        volatility=0.2,
        time=1.0,
    )
    assert is_valid is False
    assert "finite" in message


def test_validate_portfolio_weights_valid():
    is_valid, message = utils.validate_portfolio_weights([0.4, 0.6])
    assert is_valid is True
    assert message is None


def test_validate_portfolio_weights_invalid_sum():
    is_valid, message = utils.validate_portfolio_weights([0.2, 0.7])
    assert is_valid is False
    assert "sum" in message


def test_validate_portfolio_weights_negative():
    is_valid, message = utils.validate_portfolio_weights([0.4, -0.1, 0.7])
    assert is_valid is False
    assert "non-negative" in message


def test_annualize_returns_compound():
    returns = np.array([0.01, -0.02, 0.015])
    expected = np.prod(1 + returns) ** (252 / len(returns)) - 1
    assert math.isclose(utils.annualize_returns(returns), expected)


@pytest.mark.parametrize("bad_returns", [[-1.0], [-1.2, 0.1]])
def test_annualize_returns_invalid(bad_returns):
    with pytest.raises(ValueError):
        utils.annualize_returns(bad_returns)


@pytest.mark.parametrize("volatility, periods, expected", [
    (0.1, 12, 0.1 * math.sqrt(12)),
])
def test_annualize_volatility_scalar(volatility, periods, expected):
    assert math.isclose(utils.annualize_volatility(volatility, periods), expected)


def test_annualize_volatility_array():
    returns = np.array([0.01, -0.02, 0.015, 0.005])
    expected = np.std(returns, ddof=1) * math.sqrt(252)
    assert math.isclose(utils.annualize_volatility(returns), expected)


@pytest.mark.parametrize("volatility", [-0.1, -1.0])
def test_annualize_volatility_negative_scalar(volatility):
    with pytest.raises(ValueError):
        utils.annualize_volatility(volatility)


def test_sharpe_ratio_matches_manual():
    returns = np.array([0.01, 0.02, -0.005, 0.015])
    risk_free = 0.001
    excess = returns - risk_free
    expected = np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(252)
    assert math.isclose(utils.sharpe_ratio(returns, risk_free_rate=risk_free), expected)


def test_tracking_error_matches_manual():
    returns = np.array([0.01, 0.015, -0.005, 0.02])
    benchmark = np.array([0.008, 0.012, -0.004, 0.018])
    diff = returns - benchmark
    expected = np.std(diff, ddof=1) * math.sqrt(12)
    assert math.isclose(utils.tracking_error(returns, benchmark, periods_per_year=12), expected)


def test_information_ratio_matches_manual():
    returns = np.array([0.01, 0.015, -0.005, 0.02])
    benchmark = np.array([0.008, 0.012, -0.004, 0.018])
    te = utils.tracking_error(returns, benchmark, periods_per_year=12)
    mean_diff = np.mean(returns - benchmark) * 12
    expected = mean_diff / te
    assert math.isclose(utils.information_ratio(returns, benchmark, periods_per_year=12), expected)


def test_treynor_ratio_matches_manual():
    returns = np.array([0.012, 0.018, -0.004, 0.022, 0.01])
    benchmark = np.array([0.01, 0.015, -0.003, 0.02, 0.008])
    risk_free = 0.001
    periods = 12

    mean_excess = np.mean(returns - risk_free) * periods
    beta_value = np.cov(returns, benchmark, ddof=1)[0, 1] / np.var(benchmark, ddof=1)
    expected = mean_excess / beta_value

    result = utils.treynor_ratio(
        returns,
        benchmark,
        risk_free_rate=risk_free,
        periods_per_year=periods,
    )
    assert math.isclose(result, expected)


def test_beta_matches_manual():
    returns = np.array([0.01, 0.015, -0.005, 0.02])
    benchmark = np.array([0.008, 0.012, -0.004, 0.018])
    cov = np.cov(returns, benchmark, ddof=1)[0, 1]
    var = np.var(benchmark, ddof=1)
    expected = cov / var
    assert math.isclose(utils.beta(returns, benchmark), expected)


def test_moment_ratios_match_manual():
    returns = np.array([0.01, 0.015, -0.005, 0.02, -0.012, 0.03, -0.008, 0.011, 0.007, -0.009])

    mean = returns.mean()
    diff = returns - mean
    n = len(returns)
    std = np.std(returns, ddof=1)
    expected_skew = (n / ((n - 1) * (n - 2))) * np.sum(diff**3) / std**3
    expected_kurt = (
        (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(diff**4) / std**4
        - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    )

    assert math.isclose(utils.skewness(returns), expected_skew)
    assert math.isclose(utils.excess_kurtosis(returns), expected_kurt)

    positive = returns[returns > 0]
    negative = returns[returns < 0]
    expected_gain_loss = positive.mean() / (-negative).mean()
    assert math.isclose(utils.gain_loss_ratio(returns), expected_gain_loss)

    upper = np.quantile(returns, 0.95, method="linear")
    lower = np.quantile(returns, 0.05, method="linear")
    expected_tail = upper / abs(lower)
    assert math.isclose(utils.tail_ratio(returns, percentile=0.95), expected_tail)

    upside = np.maximum(returns, 0).sum() / len(returns)
    downside = np.sqrt((np.minimum(returns, 0) ** 2).sum() / len(returns))
    expected_upr = upside / downside
    assert math.isclose(utils.upside_potential_ratio(returns), expected_upr)


def test_alpha_matches_manual():
    returns = np.array([0.01, 0.015, -0.005, 0.02])
    benchmark = np.array([0.008, 0.012, -0.004, 0.018])
    risk_free = 0.0005
    periods = 252
    beta_value = utils.beta(returns, benchmark)
    excess_port = np.mean(returns - risk_free) * periods
    excess_bench = np.mean(benchmark - risk_free) * periods
    expected = excess_port - beta_value * excess_bench
    assert math.isclose(utils.alpha(returns, benchmark, risk_free_rate=risk_free, periods_per_year=periods), expected)


def test_alpha_with_nan_risk_free_matches_manual():
    returns = np.array([0.01, 0.015, -0.005, 0.02])
    benchmark = np.array([0.008, 0.012, -0.004, 0.018])
    risk_free = np.array([0.0005, np.nan, 0.0004, np.nan])
    periods = 252

    mask = np.isfinite(returns) & np.isfinite(benchmark) & np.isfinite(risk_free)
    filtered_returns = returns[mask]
    filtered_benchmark = benchmark[mask]
    filtered_risk_free = risk_free[mask]

    beta_value = np.cov(filtered_returns, filtered_benchmark, ddof=1)[0, 1] / np.var(
        filtered_benchmark, ddof=1
    )
    excess_port = np.mean(filtered_returns - filtered_risk_free) * periods
    excess_bench = np.mean(filtered_benchmark - filtered_risk_free) * periods
    expected = excess_port - beta_value * excess_bench

    result = utils.alpha(
        returns,
        benchmark,
        risk_free_rate=risk_free,
        periods_per_year=periods,
    )

    assert math.isclose(result, expected)


def test_sortino_ratio_matches_manual():
    returns = np.array([0.01, -0.02, 0.015, 0.005])
    risk_free = 0.0
    target = 0.0
    excess = returns - risk_free
    downside = np.minimum(returns - target, 0.0)
    downside_std = math.sqrt(np.mean(downside ** 2)) * math.sqrt(12)
    expected = np.mean(excess) * 12 / downside_std
    assert math.isclose(
        utils.sortino_ratio(returns, risk_free_rate=risk_free, target_return=target, periods_per_year=12),
        expected,
    )


def test_sortino_ratio_no_downside_returns_inf():
    returns = np.array([0.02, 0.03, 0.01])
    assert utils.sortino_ratio(returns) == math.inf


def test_downside_deviation_zero_when_no_downside():
    returns = np.array([0.02, 0.03, 0.01])
    assert utils.downside_deviation(returns) == 0.0


def test_omega_ratio_matches_manual():
    returns = np.array([0.02, -0.01, 0.015, -0.005, 0.03])
    threshold = 0.0
    gains = np.clip(returns - threshold, 0.0, None)
    losses = np.clip(threshold - returns, 0.0, None)
    expected = gains.mean() / losses.mean()

    assert math.isclose(utils.omega_ratio(returns, threshold), expected)


def test_capture_ratios_match_manual():
    returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.04])
    benchmark = np.array([0.01, -0.02, 0.015, -0.01, 0.005, 0.02])
    periods = 12

    mask_up = benchmark > 0
    port_up_growth = np.prod(1.0 + returns[mask_up])
    bench_up_growth = np.prod(1.0 + benchmark[mask_up])
    up_expected = (
        (port_up_growth ** (periods / mask_up.sum()) - 1.0)
        / (bench_up_growth ** (periods / mask_up.sum()) - 1.0)
    )

    mask_down = benchmark < 0
    port_down_growth = np.prod(1.0 + returns[mask_down])
    bench_down_growth = np.prod(1.0 + benchmark[mask_down])
    down_expected = (
        (port_down_growth ** (periods / mask_down.sum()) - 1.0)
        / (bench_down_growth ** (periods / mask_down.sum()) - 1.0)
    )

    assert math.isclose(
        utils.upside_capture_ratio(returns, benchmark, periods_per_year=periods),
        up_expected,
    )
    assert math.isclose(
        utils.downside_capture_ratio(returns, benchmark, periods_per_year=periods),
        down_expected,
    )


def test_capture_ratio_requires_relevant_benchmark_observations():
    returns = np.array([0.01, 0.02, 0.015])
    benchmark = np.array([-0.01, -0.02, -0.03])

    with pytest.raises(ValueError):
        utils.upside_capture_ratio(returns, benchmark)

    benchmark_pos = np.array([0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        utils.downside_capture_ratio(returns, benchmark_pos)


def test_drawdown_series_and_max_drawdown():
    prices = np.array([100, 110, 105, 120, 115])
    expected = np.array([0.0, 0.0, -5 / 110, 0.0, -5 / 120])
    np.testing.assert_allclose(utils.drawdown_series(prices), expected)
    assert math.isclose(utils.max_drawdown(prices), expected.min())


def test_drawdown_indices_match_manual():
    prices = np.array([100.0, 98.0, 101.0, 99.5, 104.0, 102.0])
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1.0

    expected_pain = np.abs(drawdowns).mean()
    expected_ulcer = np.sqrt((drawdowns**2).mean())

    assert math.isclose(utils.pain_index(prices), expected_pain)
    assert math.isclose(utils.ulcer_index(prices), expected_ulcer)


def test_calmar_ratio_handles_negative_drawdown():
    assert math.isclose(utils.calmar_ratio(0.15, -0.2), 0.75)


@pytest.mark.parametrize("prices", [[0, 1, 2], [-1, 2, 3]])
def test_drawdown_series_invalid_prices(prices):
    with pytest.raises(ValueError):
        utils.drawdown_series(prices)


@pytest.mark.parametrize("returns, risk_free", [
    ([0.01, 0.02], [0.001, 0.002, 0.003]),
])
def test_sharpe_ratio_mismatched_inputs(returns, risk_free):
    with pytest.raises(ValueError):
        utils.sharpe_ratio(returns, risk_free)


@pytest.mark.parametrize("returns, risk_free", [
    ([0.02, 0.03], [0.01]),
])
def test_sortino_ratio_mismatched_inputs(returns, risk_free):
    with pytest.raises(ValueError):
        utils.sortino_ratio(returns, risk_free)


@pytest.mark.parametrize("returns, benchmark", [
    ([0.01, 0.02], [0.015]),
])
def test_tracking_error_mismatched_inputs(returns, benchmark):
    with pytest.raises(ValueError):
        utils.tracking_error(returns, benchmark)


@pytest.mark.parametrize("returns, benchmark", [
    ([0.01], [0.008]),
])
def test_beta_requires_two_observations(returns, benchmark):
    with pytest.raises(ValueError):
        utils.beta(returns, benchmark)


@pytest.mark.parametrize("returns, benchmark, risk_free", [
    ([0.01, 0.02], [0.015], 0.001),
])
def test_alpha_mismatched_inputs(returns, benchmark, risk_free):
    with pytest.raises(ValueError):
        utils.alpha(returns, benchmark, risk_free)


def test_value_at_risk_accepts_percentage_confidence():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    var_decimal = utils.value_at_risk(returns, confidence_level=0.95)
    var_percentage = utils.value_at_risk(returns, confidence_level=95)
    var_string = utils.value_at_risk(returns, confidence_level="95%")

    assert math.isclose(var_decimal, var_percentage)
    assert math.isclose(var_decimal, var_string)


@pytest.mark.parametrize(
    "alias",
    ["parametric", "gaussian", "normal", "variance-covariance", "variance_covariance"],
)
def test_value_at_risk_method_aliases(alias):
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    baseline = utils.value_at_risk(returns, method="parametric")
    alias_result = utils.value_at_risk(returns, method=alias)
    assert math.isclose(baseline, alias_result)


def test_conditional_value_at_risk_accepts_string_confidence():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    cvar_decimal = utils.conditional_value_at_risk(returns, confidence_level=0.9)
    cvar_string = utils.conditional_value_at_risk(returns, confidence_level="90%")
    assert math.isclose(cvar_decimal, cvar_string)


def test_value_at_risk_historical_requires_returns():
    with pytest.raises(ValueError):
        utils.value_at_risk(confidence_level=0.95, method="historical")


def test_value_at_risk_monte_carlo_matches_risk_metrics():
    from dervflow.risk import RiskMetrics

    mean = 0.001
    std_dev = 0.02
    seed = 123
    confidence = 0.975
    simulations = 20_000

    utils_var = utils.value_at_risk(
        confidence_level=confidence,
        method="monte_carlo",
        mean=mean,
        std_dev=std_dev,
        num_simulations=simulations,
        seed=seed,
    )

    rm = RiskMetrics()
    rm_var = rm.var(
        confidence_level=confidence,
        method="monte_carlo",
        mean=mean,
        std_dev=std_dev,
        num_simulations=simulations,
        seed=seed,
    )["var"]

    assert math.isclose(utils_var, rm_var, rel_tol=0.05)


def test_conditional_value_at_risk_monte_carlo_matches_risk_metrics():
    from dervflow.risk import RiskMetrics

    mean = 0.001
    std_dev = 0.02
    seed = 321
    confidence = 0.99
    simulations = 25_000

    utils_cvar = utils.conditional_value_at_risk(
        confidence_level=confidence,
        method="monte_carlo",
        mean=mean,
        std_dev=std_dev,
        num_simulations=simulations,
        seed=seed,
    )

    rm = RiskMetrics()
    rm_cvar = rm.cvar(
        confidence_level=confidence,
        method="monte_carlo",
        mean=mean,
        std_dev=std_dev,
        num_simulations=simulations,
        seed=seed,
    )["cvar"]

    assert math.isclose(utils_cvar, rm_cvar, rel_tol=0.05)


def test_value_at_risk_monte_carlo_requires_parameters():
    with pytest.raises(ValueError):
        utils.value_at_risk(method="monte_carlo")

    with pytest.raises(ValueError):
        utils.value_at_risk(method="monte_carlo", mean=0.0)


def test_conditional_value_at_risk_monte_carlo_requires_parameters():
    with pytest.raises(ValueError):
        utils.conditional_value_at_risk(method="monte_carlo")

    with pytest.raises(ValueError):
        utils.conditional_value_at_risk(method="monte_carlo", std_dev=0.2)
