Quick Start Guide
=================

This guide will help you get started with dervflow's core functionality.

Basic Option Pricing
--------------------

Black-Scholes Model
~~~~~~~~~~~~~~~~~~~

The Black-Scholes model is the most common method for pricing European options:

.. code-block:: python

   import dervflow

   # Create a Black-Scholes model instance
   bs_model = dervflow.BlackScholesModel()

   # Price a call option
   call_price = bs_model.price(
       spot=100.0,        # Current stock price
       strike=105.0,      # Strike price
       rate=0.05,         # Risk-free rate (5%)
       dividend=0.02,     # Dividend yield (2%)
       volatility=0.25,   # Volatility (25%)
       time=1.0,          # Time to maturity (1 year)
       option_type='call'
   )
   print(f"Call option price: ${call_price:.2f}")

   # Price a put option
   put_price = bs_model.price(
       spot=100.0,
       strike=105.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.25,
       time=1.0,
       option_type='put'
   )
   print(f"Put option price: ${put_price:.2f}")

Batch Pricing
~~~~~~~~~~~~~

For pricing multiple options efficiently:

.. code-block:: python

   import numpy as np

   # Price options at different strikes
   strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=float)
   spots = np.full(5, 100.0, dtype=float)

   rates = np.full_like(strikes, 0.05, dtype=float)
   dividends = np.full_like(strikes, 0.02, dtype=float)
   volatilities = np.full_like(strikes, 0.25, dtype=float)
   times = np.full_like(strikes, 1.0, dtype=float)
   option_types = ['call'] * len(strikes)

   prices = bs_model.price_batch(
       spots,
       strikes,
       rates,
       dividends,
       volatilities,
       times,
       option_types,
   )
   
   for strike, price in zip(strikes, prices):
       print(f"Strike ${strike:.0f}: ${price:.2f}")

Calculating Greeks
------------------

Greeks measure the sensitivity of option prices to various parameters:

.. code-block:: python

   # Calculate all Greeks for an option
   greeks = bs_model.greeks(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.25,
       time=1.0,
       option_type='call'
   )

   print(f"Delta: {greeks['delta']:.4f}")    # Price sensitivity to spot
   print(f"Gamma: {greeks['gamma']:.4f}")    # Delta sensitivity to spot
   print(f"Vega: {greeks['vega']:.4f}")      # Price sensitivity to volatility
   print(f"Theta: {greeks['theta']:.4f}")    # Time decay
   print(f"Rho: {greeks['rho']:.4f}")        # Interest rate sensitivity

Implied Volatility
------------------

Calculate implied volatility from market prices:

.. code-block:: python

   # Market price of the option
   market_price = 10.45

   # Calculate implied volatility
   iv = bs_model.implied_vol(
       market_price=market_price,
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       time=1.0,
       option_type='call'
   )
   
   print(f"Implied volatility: {iv:.2%}")

Monte Carlo Simulation
----------------------

Simulate price paths and price options:

.. code-block:: python

   # Create Monte Carlo engine
   mc_engine = dervflow.MonteCarloEngine(seed=42)

   # Simulate Geometric Brownian Motion paths
   paths = mc_engine.simulate_gbm(
       s0=100.0,          # Initial price
       mu=0.05,           # Drift (expected return)
       sigma=0.25,        # Volatility
       T=1.0,             # Time horizon
       steps=252,         # Number of time steps
       paths=1000         # Number of paths
   )

   # Paths array has shape (num_paths, steps + 1)
   print(f"Simulated paths shape: {paths.shape}")
   print(
       "Final prices - Mean: $"
       f"{paths[:, -1].mean():.2f}, Std: ${paths[:, -1].std():.2f}"
   )

   # Price European option with Monte Carlo
   mc_pricer = dervflow.MonteCarloOptionPricer()
   result = mc_pricer.price_european(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.25,
       time=1.0,
       option_type='call',
       num_paths=100000,
       use_antithetic=True,   # Use variance reduction
   )

   print(f"MC Price: ${result['price']:.2f} ± ${result['std_error']:.2f}")

Portfolio Optimization
----------------------

Optimize portfolio allocations:

.. code-block:: python

   import numpy as np

   # Generate sample returns (5 assets, 252 days)
   rng = np.random.default_rng(seed=42)
   returns = rng.normal(0.0005, 0.01, size=(252, 5))

   # Create optimizer
   optimizer = dervflow.PortfolioOptimizer(returns)

   # Optimize for target return
   min_weights = np.zeros(returns.shape[1])
   max_weights = np.full(returns.shape[1], 0.4)

   result = optimizer.optimize(
       target_return=0.10,  # 10% annual return
       min_weights=min_weights,
       max_weights=max_weights,
   )

   print(f"Optimal weights: {result['weights']}")
   print(f"Expected return: {result['expected_return']:.2%}")
   print(f"Volatility: {result['volatility']:.2%}")
   print(f"Sharpe ratio: {result['sharpe_ratio']:.2f}")

Risk Metrics
------------

Calculate Value at Risk (VaR):

.. code-block:: python

   # Generate sample returns
   rng = np.random.default_rng(seed=0)
   returns = rng.normal(0.0, 0.02, size=1000)

   # Create risk metrics calculator
   risk_metrics = dervflow.RiskMetrics()

   # Historical VaR
   var_result = risk_metrics.var(
       returns,
       confidence_level=0.95,
       method='historical',
   )
   print(f"95% VaR (Historical): {var_result['var']:.2%}")

   # Conditional VaR (Expected Shortfall)
   cvar_result = risk_metrics.cvar(returns, confidence_level=0.95)
   print(f"95% CVaR: {cvar_result['cvar']:.2%}")

Next Steps
----------

* Explore the :doc:`options_pricing` guide for advanced pricing models
* Learn about :doc:`risk_analytics` for comprehensive risk management
* Check out the :doc:`../api/options` for detailed API documentation
* See the example notebooks in the repository for more complex use cases
