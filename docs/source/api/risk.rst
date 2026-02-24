Risk Analytics API
==================

.. currentmodule:: dervflow

Risk analytics in :mod:`dervflow` are provided by two high-performance classes
backed by the Rust extension: :class:`GreeksCalculator` for option sensitivities
and :class:`RiskMetrics` for portfolio risk/statistics. Both are imported at the
top level.

GreeksCalculator
----------------

.. autoclass:: GreeksCalculator
   :members: calculate, calculate_extended, portfolio_greeks, portfolio_extended_greeks
   :show-inheritance:

:meth:`calculate` returns first- and second-order Greeks for a single option
using either analytical formulas or finite differences. When higher-order
sensitivities are required (vanna, volga, speed, zomma, etc.) use
:meth:`calculate_extended`. Portfolio aggregation is available through
:meth:`portfolio_greeks` (first-order) and :meth:`portfolio_extended_greeks`
(higher-order) by passing NumPy arrays of per-position inputs.

.. code-block:: python

   import dervflow
   import numpy as np

   calc = dervflow.GreeksCalculator()

   single = calc.calculate(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.0,
       volatility=0.2,
       time_to_maturity=1.0,
       option_type="call",
   )
   portfolio = calc.portfolio_extended_greeks(
       spots=np.array([100.0, 100.0]),
       strikes=np.array([95.0, 105.0]),
       rates=np.array([0.05, 0.05]),
       dividends=np.array([0.02, 0.02]),
       volatilities=np.array([0.25, 0.20]),
       times_to_maturity=np.array([0.5, 1.0]),
       option_types=["call", "put"],
       quantities=np.array([8.0, -3.0]),
   )

RiskMetrics
-----------

.. autoclass:: RiskMetrics
   :members: var, cvar, max_drawdown, sortino_ratio, calmar_ratio, portfolio_metrics, portfolio_var_parametric, portfolio_cvar_parametric
   :show-inheritance:

The :class:`RiskMetrics` helper computes Value at Risk (historical/parametric),
Conditional VaR, drawdowns and a range of risk-adjusted performance statistics.
Portfolio utilities evaluate covariance-driven metrics, marginal risk
contributions and parametric VaR/CVaR estimates.

.. code-block:: python

   import numpy as np
   import dervflow

   metrics = dervflow.RiskMetrics()

   returns = np.random.normal(0.001, 0.02, 1000)
   var_95 = metrics.var(returns, confidence_level=0.95)
   cvar_95 = metrics.cvar(returns, confidence_level=0.95)
   calmar = metrics.calmar_ratio(returns)

   weights = np.array([0.4, 0.6])
   cov = np.array([[0.04, 0.01], [0.01, 0.09]])
   summary = metrics.portfolio_metrics(weights, cov, risk_free_rate=0.02)
   var_param = metrics.portfolio_var_parametric(weights, cov, confidence_level=0.99)
   cvar_param = metrics.portfolio_cvar_parametric(weights, cov, confidence_level=0.99)

See Also
--------

* :doc:`../user_guide/risk_analytics` - comprehensive walkthroughs of risk
  workflows
* :doc:`../theory/stochastic_processes` - mathematical background on stochastic
  calculus and risk measures
