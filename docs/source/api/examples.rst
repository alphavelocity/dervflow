API Usage Examples
==================

This page demonstrates typical usage patterns for the high-level Python API.

Option Pricing
--------------

Single Option
~~~~~~~~~~~~~

.. code-block:: python

   import dervflow

   bs = dervflow.BlackScholesModel()

   call_price = bs.price(spot=100.0, strike=100.0, rate=0.05,
                         dividend=0.02, volatility=0.25, time=1.0,
                         option_type='call')
   put_price = bs.price(100.0, 100.0, 0.05, 0.02, 0.25, 1.0, 'put')

   print(f"Call price: {call_price:.2f}")
   print(f"Put price:  {put_price:.2f}")

Batch Pricing
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   spots = np.full(5, 100.0)
   strikes = np.array([90, 95, 100, 105, 110], dtype=float)
   rates = np.full(5, 0.05, dtype=float)
   dividends = np.full(5, 0.02, dtype=float)
   volatilities = np.full(5, 0.25, dtype=float)
   times = np.full(5, 1.0, dtype=float)
   option_types = ['call'] * 5

   prices = bs.price_batch(spots, strikes, rates, dividends, volatilities,
                           times, option_types)
   print(prices)

Greeks and Portfolio Greeks
---------------------------

.. code-block:: python

   from dervflow import GreeksCalculator
   import numpy as np

   calc = GreeksCalculator()

   single = calc.calculate(spot=100.0, strike=100.0, rate=0.05,
                           dividend=0.02, volatility=0.25,
                           time_to_maturity=1.0, option_type='call')
   print(single['delta'], single['gamma'])

   spots = np.array([100.0, 100.0])
   strikes = np.array([100.0, 105.0])
   rates = np.array([0.05, 0.05])
   dividends = np.array([0.02, 0.02])
   volatilities = np.array([0.25, 0.2])
   times = np.array([1.0, 0.5])
   option_types = ['call', 'put']
   quantities = np.array([10.0, -5.0])

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

Monte Carlo Option Pricing
--------------------------

.. code-block:: python

   from dervflow import MonteCarloOptionPricer

   pricer = MonteCarloOptionPricer()
   mc = pricer.price_european(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.0,
       volatility=0.2,
       time=1.0,
       option_type='call',
       num_paths=100_000,
       use_antithetic=True,
       seed=123,
       parallel=True,
   )
   print(f"MC price: {mc['price']:.2f} ± {mc['std_error']:.4f}")

Portfolio Optimisation
----------------------

.. code-block:: python

   import numpy as np
   from dervflow import PortfolioOptimizer

   rng = np.random.default_rng(seed=42)
   returns = rng.normal(0.001, 0.02, size=(252, 4))
   optimizer = PortfolioOptimizer(returns)

   min_w = np.zeros(4)
   max_w = np.full(4, 0.4)
   target = optimizer.optimize(target_return=0.08,
                               min_weights=min_w,
                               max_weights=max_w)
   print(target['weights'])

Risk Metrics
------------

.. code-block:: python

   import numpy as np
   from dervflow import RiskMetrics

   risk = RiskMetrics()
   rng = np.random.default_rng(seed=0)
   returns = rng.normal(0.0, 0.02, size=1000)

   var = risk.var(returns, confidence_level=0.95, method='historical')
   cvar = risk.cvar(returns, confidence_level=0.95)

   print(f"VaR 95%:  {var['var']:.4%}")
   print(f"CVaR 95%: {cvar['cvar']:.4%}")

Yield Curve Analytics
---------------------

.. code-block:: python

   import numpy as np
   from dervflow import YieldCurve, BondAnalytics

   times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
   rates = np.array([0.02, 0.024, 0.028, 0.033, 0.037])
   curve = YieldCurve(times, rates, method='cubic_spline_natural')

   print(f"5Y discount factor: {curve.discount_factor(5.0):.6f}")

   analytics = BondAnalytics()
   cashflows = analytics.generate_cashflows(maturity=5.0, coupon_rate=0.04,
                                            face_value=100.0, frequency=2)
   price = analytics.bond_price(yield_rate=0.032, cashflows=cashflows)
   print(f"Bond price: {price:.2f}")
