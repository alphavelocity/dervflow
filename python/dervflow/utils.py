# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rust-backed utility helpers exposed to Python."""

from __future__ import annotations

from dervflow._dervflow import utils as _utils

_DEFAULT_CONFIDENCE = object()


def validate_option_params(spot, strike, rate, dividend, volatility, time):
    """Validate option pricing parameters."""

    return _utils.validate_option_params(spot, strike, rate, dividend, volatility, time)


def validate_portfolio_weights(weights, tolerance: float = 1e-6):
    """Validate a vector of portfolio weights."""

    return _utils.validate_portfolio_weights(weights, tolerance)


def annualize_returns(returns, periods_per_year: int = 252):
    """Convert periodic returns to a compounded annual growth rate."""

    return _utils.annualize_returns(returns, periods_per_year)


def annualize_volatility(volatility, periods_per_year: int = 252):
    """Annualise volatility from a scalar or a series of returns."""

    return _utils.annualize_volatility(volatility, periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year: int = 252):
    """Calculate the annualised Sharpe ratio from periodic returns."""

    return _utils.sharpe_ratio(returns, risk_free_rate, periods_per_year)


def tracking_error(returns, benchmark_returns, periods_per_year: int = 252):
    """Calculate the annualised tracking error between a portfolio and benchmark."""

    return _utils.tracking_error(returns, benchmark_returns, periods_per_year)


def information_ratio(returns, benchmark_returns, periods_per_year: int = 252):
    """Compute the annualised information ratio of a portfolio versus a benchmark."""

    return _utils.information_ratio(returns, benchmark_returns, periods_per_year)


def beta(returns, benchmark_returns):
    """Calculate the beta of returns relative to benchmark_returns."""

    return _utils.beta(returns, benchmark_returns)


def alpha(returns, benchmark_returns, risk_free_rate=0.0, periods_per_year: int = 252):
    """Calculate Jensen's alpha relative to benchmark_returns."""

    return _utils.alpha(returns, benchmark_returns, risk_free_rate, periods_per_year)


def downside_deviation(returns, target_return: float = 0.0, periods_per_year: int = 252):
    """Compute the annualised downside deviation of returns."""

    return _utils.downside_deviation(returns, target_return, periods_per_year)


def sortino_ratio(
    returns,
    risk_free_rate=0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
):
    """Calculate the annualised Sortino ratio."""

    return _utils.sortino_ratio(returns, risk_free_rate, target_return, periods_per_year)


def treynor_ratio(
    returns,
    benchmark_returns,
    risk_free_rate=0.0,
    periods_per_year: int = 252,
):
    """Calculate the Treynor ratio for a portfolio."""

    return _utils.treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year)


def omega_ratio(returns, threshold: float = 0.0):
    """Compute the Omega ratio for a series of returns."""

    return _utils.omega_ratio(returns, threshold)


def skewness(returns):
    """Return the sample skewness of *returns* using the adjusted Fisher moment."""

    return _utils.skewness(returns)


def excess_kurtosis(returns):
    """Return the sample excess kurtosis (Fisher definition) of *returns*."""

    return _utils.excess_kurtosis(returns)


def gain_loss_ratio(returns):
    """Ratio of the average gain to the average loss in absolute value."""

    return _utils.gain_loss_ratio(returns)


def tail_ratio(returns, percentile: float = 0.95):
    """Ratio of upside to downside tail magnitudes at the given percentile."""

    return _utils.tail_ratio(returns, percentile)


def upside_potential_ratio(returns, threshold: float = 0.0):
    """Upside potential ratio of *returns* relative to *threshold*."""

    return _utils.upside_potential_ratio(returns, threshold)


def upside_capture_ratio(returns, benchmark_returns, periods_per_year: int = 252):
    """Return the upside capture ratio of returns versus benchmark_returns."""

    return _utils.upside_capture_ratio(returns, benchmark_returns, periods_per_year)


def downside_capture_ratio(returns, benchmark_returns, periods_per_year: int = 252):
    """Return the downside capture ratio of returns versus benchmark_returns."""

    return _utils.downside_capture_ratio(returns, benchmark_returns, periods_per_year)


def value_at_risk(
    returns=None,
    confidence_level=_DEFAULT_CONFIDENCE,
    method: str = "historical",
    *,
    mean: float | None = None,
    std_dev: float | None = None,
    num_simulations: int = 10_000,
    seed: int | None = None,
    decay: float | None = None,
):
    """Estimate Value at Risk (VaR) for a return series.

    Parameters
    ----------
    returns:
        Historical return observations. Required for ``historical``, ``parametric``
        and ``cornish_fisher`` methods. May be omitted for ``monte_carlo`` where
        ``mean`` and ``std_dev`` define the distribution.
    confidence_level:
        Tail confidence level expressed either as a probability in ``(0, 1)`` or
        a percentage (e.g. ``95`` or ``"95%"``).
    method:
        VaR methodology: ``historical``, ``parametric``, ``cornish_fisher``,
        ``monte_carlo`` or ``ewma``.
    mean, std_dev:
        Distribution parameters required for the Monte Carlo method.
    num_simulations:
        Number of Monte Carlo paths to simulate when ``method='monte_carlo'``.
    seed:
        Optional seed for reproducible Monte Carlo draws.
    decay:
        EWMA decay factor when ``method='ewma'``. Defaults to ``0.94`` if not
        provided.
    """

    if confidence_level is _DEFAULT_CONFIDENCE:
        resolved_confidence = 0.95
    elif confidence_level is None:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    else:
        resolved_confidence = confidence_level

    return _utils.value_at_risk(
        returns,
        resolved_confidence,
        method,
        mean,
        std_dev,
        num_simulations,
        seed,
        decay,
    )


def conditional_value_at_risk(
    returns=None,
    confidence_level=_DEFAULT_CONFIDENCE,
    method: str = "historical",
    *,
    mean: float | None = None,
    std_dev: float | None = None,
    num_simulations: int = 10_000,
    seed: int | None = None,
    decay: float | None = None,
):
    """Estimate Conditional Value at Risk (CVaR) for a return series."""

    if confidence_level is _DEFAULT_CONFIDENCE:
        resolved_confidence = 0.95
    elif confidence_level is None:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    else:
        resolved_confidence = confidence_level

    return _utils.conditional_value_at_risk(
        returns,
        resolved_confidence,
        method,
        mean,
        std_dev,
        num_simulations,
        seed,
        decay,
    )


def drawdown_series(prices):
    """Return the drawdown series for prices."""

    return _utils.drawdown_series(prices)


def max_drawdown(prices):
    """Calculate the maximum drawdown of a price series."""

    return _utils.max_drawdown(prices)


def pain_index(prices):
    """Average magnitude of drawdowns for *prices*."""

    return _utils.pain_index(prices)


def ulcer_index(prices):
    """Root-mean-square drawdown (Ulcer index) for *prices*."""

    return _utils.ulcer_index(prices)


def calmar_ratio(annual_return, max_drawdown_value):
    """Calculate the Calmar ratio (return divided by maximum drawdown)."""

    return _utils.calmar_ratio(annual_return, max_drawdown_value)


__all__ = [
    "validate_option_params",
    "validate_portfolio_weights",
    "annualize_returns",
    "annualize_volatility",
    "sharpe_ratio",
    "tracking_error",
    "information_ratio",
    "beta",
    "alpha",
    "treynor_ratio",
    "downside_deviation",
    "sortino_ratio",
    "omega_ratio",
    "skewness",
    "excess_kurtosis",
    "gain_loss_ratio",
    "tail_ratio",
    "upside_potential_ratio",
    "upside_capture_ratio",
    "downside_capture_ratio",
    "drawdown_series",
    "max_drawdown",
    "pain_index",
    "ulcer_index",
    "calmar_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
]
