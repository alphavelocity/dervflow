Risk Analytics Guide
====================

DervFlow bundles fast risk analytics implemented in Rust with a thin Python
interface. This guide shows how to compute sensitivities (Greeks), aggregate
portfolio exposure, and evaluate Value at Risk (VaR) and other common metrics.

Greeks Calculation
------------------

Use :class:`dervflow.GreeksCalculator` to obtain numerical Greeks for an option.
The calculator bumps inputs under the hood and works for any payoff supported by
DervFlow's pricers.

.. code-block:: python

   import dervflow

   calc = dervflow.GreeksCalculator()
   greeks = calc.calculate(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.25,
       time_to_maturity=1.0,
       option_type='call',
   )

   print(f"Delta: {greeks['delta']:.4f}")
   print(f"Gamma: {greeks['gamma']:.4f}")
   print(f"Vega:  {greeks['vega']:.4f}")
   print(f"Theta: {greeks['theta']:.4f}")
   print(f"Rho:   {greeks['rho']:.4f}")

Extended Greeks (vanna, volga, speed, etc.) are available via
:meth:`GreeksCalculator.calculate_extended`.

.. code-block:: python

   extended = calc.calculate_extended(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.25,
       time_to_maturity=1.0,
       option_type='call',
   )
   print(f"Vanna: {extended['vanna']:.6f}")
   print(f"Ultima: {extended['ultima']:.6f}")

Portfolio Greeks
----------------

Aggregate exposures across a book by supplying vector inputs. Arrays must be the
same length and ``option_types`` should contain ``'call'`` or ``'put'`` labels.

.. code-block:: python

   import numpy as np
   from dervflow import GreeksCalculator

   calc = GreeksCalculator()

   spots = np.array([100.0, 102.0, 95.0])
   strikes = np.array([100.0, 105.0, 90.0])
   rates = np.full(3, 0.05)
   dividends = np.full(3, 0.02)
   volatilities = np.array([0.25, 0.20, 0.30])
   times = np.array([1.0, 0.75, 0.5])
   option_types = ['call', 'put', 'call']
   quantities = np.array([50.0, -20.0, 35.0])

   portfolio = calc.portfolio_greeks(
       spots,
       strikes,
       rates,
       dividends,
       volatilities,
       times,
       option_types,
       quantities,
   )

   print(f"Portfolio delta: {portfolio['delta']:.2f}")
   print(f"Portfolio vega:  {portfolio['vega']:.2f}")

Value at Risk and Expected Shortfall
------------------------------------

The :class:`dervflow.RiskMetrics` helper provides multiple VaR methodologies and
expected shortfall (CVaR).

Historical, parametric, and Cornish-Fisher VaR operate on historical return
series:

.. code-block:: python

   import numpy as np
   from dervflow import RiskMetrics

   rng = np.random.default_rng(seed=1)
   returns = rng.normal(0.0005, 0.02, size=252)

   risk = RiskMetrics()

   var_hist = risk.var(returns, confidence_level=0.95, method='historical')
   var_param = risk.var(returns, confidence_level=0.95, method='parametric')
   var_cf = risk.var(returns, confidence_level=0.95, method='cornish_fisher')

   print(f"Historical VaR: {var_hist['var']:.2%}")
   print(f"Parametric VaR: {var_param['var']:.2%}")
   print(f"Cornish-Fisher VaR: {var_cf['var']:.2%}")

Monte Carlo VaR draws simulated returns from a normal distribution. Provide the
mean, standard deviation, number of simulations, and optional seed:

.. code-block:: python

   var_mc = risk.var(
       returns=None,
       confidence_level=0.99,
       method='monte_carlo',
       mean=0.0005,
       std_dev=0.02,
       num_simulations=50_000,
       seed=123,
   )
   print(f"Monte Carlo VaR (99%): {var_mc['var']:.2%}")

Expected shortfall (CVaR) is accessed similarly. Historical CVaR reuses the
return series, while Monte Carlo CVaR requires distribution parameters.

.. code-block:: python

   cvar_hist = risk.cvar(returns, confidence_level=0.95)
   cvar_mc = risk.cvar(
       returns=None,
       confidence_level=0.99,
       method='monte_carlo',
       mean=0.0005,
       std_dev=0.02,
       num_simulations=50_000,
       seed=123,
   )

   print(f"Historical CVaR: {cvar_hist['cvar']:.2%}")
   print(f"Monte Carlo CVaR: {cvar_mc['cvar']:.2%}")

Additional Risk Metrics
-----------------------

DervFlow also exposes helper stat metrics for analysing return streams.

``max_drawdown``
   Returns the maximum peak-to-trough decline computed from cumulative returns.

``sortino_ratio``
   Computes the Sortino ratio given returns, a risk-free rate, and optional
   target return for downside deviation.

``calmar_ratio``
   Calculates the Calmar ratio (annualised return divided by maximum drawdown).

.. code-block:: python

   max_dd = risk.max_drawdown(returns)
   sortino = risk.sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0)
   calmar = risk.calmar_ratio(returns, periods_per_year=252)

   print(f"Max drawdown: {max_dd:.2%}")
   print(f"Sortino ratio: {sortino:.2f}")
   print(f"Calmar ratio: {calmar:.2f}")

Next Steps
----------

* :doc:`../api/risk` - API reference for the risk module.
* :doc:`../theory/var` - Mathematical background for the implemented
  risk measures.
