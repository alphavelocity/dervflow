# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Risk analytics module

This module provides comprehensive risk analytics and measurement tools:

Classes
-------
GreeksCalculator
    Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga)
    using analytical formulas or numerical differentiation
RiskMetrics
    Calculate portfolio risk metrics including Value at Risk (VaR),
    Conditional Value at Risk (CVaR), and other risk measures

Features
--------
- First-order Greeks: Delta, Vega, Theta, Rho
- Second-order Greeks: Gamma, Vanna, Volga
- Third-order Greeks: Speed, Zomma, Color, Ultima
- Portfolio-level Greeks aggregation
- Value at Risk (VaR) using historical simulation, variance-covariance, and Monte Carlo
- Conditional Value at Risk (CVaR) / Expected Shortfall
- Decomposition of parametric VaR and CVaR into marginal/component contributions
- Maximum drawdown calculation
- Sortino ratio and other risk-adjusted performance metrics
- Numerical Greeks using finite differences with configurable precision
- Portfolio risk decomposition (volatility, risk contributions, diversification ratios)
- Active risk analytics (tracking error, information ratio, active share) and CAPM metrics
 - Active risk analytics (tracking error, information ratio, active share) with active-return decomposition and CAPM metrics

Examples
--------
>>> import numpy as np
>>> from dervflow.risk import GreeksCalculator, RiskMetrics
>>>
>>> # Calculate Greeks for a single option
>>> greeks_calc = GreeksCalculator()
>>> greeks = greeks_calc.calculate(
...     spot=100.0,
...     strike=100.0,
...     rate=0.05,
...     dividend=0.0,
...     volatility=0.2,
...     time=1.0,
...     option_type='call',
...     method='analytical'
... )
>>> print(f"Delta: {greeks['delta']:.4f}")
>>> print(f"Gamma: {greeks['gamma']:.4f}")
>>> print(f"Vega: {greeks['vega']:.4f}")

>>> # Calculate portfolio Greeks
>>> positions = [
...     {'spot': 100.0, 'strike': 100.0, 'volatility': 0.2, 'time': 1.0,
...      'option_type': 'call', 'quantity': 10},
...     {'spot': 100.0, 'strike': 105.0, 'volatility': 0.2, 'time': 1.0,
...      'option_type': 'put', 'quantity': -5},
... ]
>>> portfolio_greeks = greeks_calc.portfolio_greeks(positions, rate=0.05, dividend=0.0)
>>> print(f"Portfolio Delta: {portfolio_greeks['delta']:.4f}")

>>> # Calculate Value at Risk
>>> risk_metrics = RiskMetrics()
>>> returns = np.random.normal(0.001, 0.02, 1000)  # Sample returns
>>> var_95 = risk_metrics.var(returns, confidence=0.95, method='historical')
>>> print(f"95% VaR: {var_95:.4f}")

>>> # Calculate Conditional VaR (Expected Shortfall)
>>> cvar_95 = risk_metrics.cvar(returns, confidence=0.95)
>>> print(f"95% CVaR: {cvar_95:.4f}")

>>> # Calculate maximum drawdown
>>> prices = np.array([100, 105, 103, 108, 102, 110, 107])
>>> max_dd = risk_metrics.max_drawdown(prices)
>>> print(f"Maximum Drawdown: {max_dd:.2%}")

>>> # Calculate Sortino ratio
>>> sortino = risk_metrics.sortino_ratio(returns, risk_free_rate=0.03/252, target_return=0.0)
>>> print(f"Sortino Ratio: {sortino:.4f}")

>>> # Analyse portfolio risk
>>> weights = np.array([0.4, 0.6])
>>> covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
>>> metrics = risk_metrics.portfolio_metrics(weights, covariance, risk_free_rate=0.02)
>>> print(f"Portfolio volatility: {metrics['volatility']:.4f}")
>>> print("Risk contributions:", metrics['risk_contributions']['percentage'])
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dervflow._dervflow import GreeksCalculator, RiskMetrics
else:
    from dervflow._dervflow import GreeksCalculator, RiskMetrics

__all__ = [
    "GreeksCalculator",
    "RiskMetrics",
]
