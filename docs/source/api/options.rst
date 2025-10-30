Options Pricing API
===================

.. currentmodule:: dervflow

The :mod:`dervflow` options layer exposes fast pricing models for vanilla and
exotic contracts along with calibration utilities. All classes documented here
are imported at the package top level (``from dervflow import ...``).

Black-Scholes Model
-------------------

.. autoclass:: BlackScholesModel
   :members: price, price_batch, greeks, implied_vol, implied_vol_batch
   :show-inheritance:

The Black-Scholes-Merton engine evaluates European calls and puts using the
closed-form solution. Use :meth:`price` for a single contract or
:meth:`price_batch` to vectorise over NumPy arrays while taking advantage of the
Rust backend's parallelism. Analytical sensitivities are available through
:meth:`greeks`, and the inverse pricing routines :meth:`implied_vol` and
:meth:`implied_vol_batch` recover implied volatilities from observed prices.

.. code-block:: python

   import numpy as np
   import dervflow

   bs = dervflow.BlackScholesModel()

   call = bs.price(spot=100.0, strike=100.0, rate=0.05, dividend=0.02,
                   volatility=0.2, time=1.0, option_type="call")
   chain = bs.price_batch(
       spots=np.full(5, 100.0),
       strikes=np.linspace(90, 110, 5),
       rates=np.full(5, 0.05),
       dividends=np.full(5, 0.02),
       volatilities=np.full(5, 0.2),
       times=np.full(5, 1.0),
       option_types=["call"] * 5,
   )

   greeks = bs.greeks(spot=100.0, strike=100.0, rate=0.05, dividend=0.02,
                      volatility=0.2, time=1.0, option_type="call")

   smile_vols = bs.implied_vol_batch(
       market_prices=np.linspace(8.0, 12.0, 5),
       spots=np.full(5, 100.0),
       strikes=np.linspace(95, 105, 5),
       rates=np.full(5, 0.05),
       dividends=np.full(5, 0.02),
       times=np.full(5, 1.0),
       option_types=["call"] * 5,
   )

Binomial Tree Model
-------------------

.. autoclass:: BinomialTreeModel
   :members: price, price_batch
   :show-inheritance:

The recombining tree implementation supports CRR and Jarrow-Rudd parameter
choices, American early exercise and batch pricing for full option chains.

Monte Carlo Option Pricer
-------------------------

.. autoclass:: MonteCarloOptionPricer
   :members: price_european, price_american
   :show-inheritance:

The Monte Carlo pricer wraps the :class:`~dervflow.MonteCarloEngine` to value
European contracts (enable variance reduction with ``use_antithetic=True``) and Longstaff-Schwartz
American options. Returned dictionaries include both the estimated price and the
standard error of the simulation.

.. code-block:: python

   import dervflow

   mc = dervflow.MonteCarloOptionPricer()
   european = mc.price_european(
       spot=100.0,
       strike=105.0,
       rate=0.03,
       dividend=0.01,
       volatility=0.25,
       time=1.0,
       option_type="call",
       num_paths=200_000,
       use_antithetic=True,
       seed=7,
       parallel=True,
   )

   american = mc.price_american(
       spot=100.0,
       strike=100.0,
       rate=0.03,
       dividend=0.01,
       volatility=0.2,
       time=1.0,
       option_type="put",
       num_paths=100_000,
       num_steps=50,
   )

Exotic Option Models
--------------------

Asian Option
~~~~~~~~~~~~

.. autoclass:: AsianOption
   :members: price_arithmetic, price_arithmetic_stats, price_geometric
   :show-inheritance:

Arithmetic payoffs return a price (and optional path statistics) using Monte
Carlo simulation, while the geometric formulation relies on the closed-form
solution. ``price_arithmetic_stats`` provides both the estimated value and the
simulation variance for more detailed reporting.

Barrier Option
~~~~~~~~~~~~~~

.. autoclass:: BarrierOption
   :members: price
   :show-inheritance:

Supports up/down and in/out configurations with optional rebates for a broad
range of barrier contracts.

Lookback Option
~~~~~~~~~~~~~~~

.. autoclass:: LookbackOption
   :members: price
   :show-inheritance:

Prices fixed and floating strike lookbacks via Monte Carlo simulation.

Digital Option
~~~~~~~~~~~~~~

.. autoclass:: DigitalOption
   :members: price
   :show-inheritance:

Computes payout profiles for cash-or-nothing and asset-or-nothing digitals.

Volatility Surface Utilities
----------------------------

.. autoclass:: VolatilitySurface
   :members: implied_volatility, spot, rate, strikes, maturities, volatilities
   :show-inheritance:

The surface object performs interpolation across strikes and maturities while
exposing the underlying grid data. Construct the surface with observed data and
call :meth:`implied_volatility` to query arbitrary maturities/strikes.

SABR Model
----------

.. autoclass:: SABRModel
   :members: calibrate, implied_volatility, alpha, beta, rho, nu
   :show-inheritance:

The SABR implementation calibrates the model parameters against observed
volatility smiles and evaluates the analytical approximation once calibrated.
The helper accessors (:meth:`alpha`, :meth:`beta`, :meth:`rho`, :meth:`nu`)
return the calibrated parameter values for further analysis.

See Also
--------

* :doc:`monte_carlo` – stochastic process simulation with
  :class:`~dervflow.MonteCarloEngine`
* :doc:`../user_guide/options_pricing` – end-to-end tutorials for pricing
  workflows
