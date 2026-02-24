Monte Carlo API
===============

.. currentmodule:: dervflow

This module provides Monte Carlo simulation and stochastic process modeling through the
:class:`MonteCarloEngine` class.

MonteCarloEngine
----------------

.. autoclass:: MonteCarloEngine
   :show-inheritance:

Usage Notes
-----------

* Pass ``parallel=True`` to leverage Rayon-powered parallel path generation.
* Provide a ``seed`` when constructing :class:`MonteCarloEngine` for reproducible draws.
* Simulation methods return NumPy arrays shaped ``(paths, steps + 1)`` except for
  :meth:`MonteCarloEngine.simulate_correlated`, which returns a list of arrays—one per asset.

Examples
--------

Geometric Brownian Motion
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import dervflow

   mc = dervflow.MonteCarloEngine(seed=7)

   paths = mc.simulate_gbm(
       s0=100.0,
       mu=0.10,
       sigma=0.25,
       T=1.0,
       steps=252,
       paths=1000,
       parallel=True,
   )

   plt.figure(figsize=(12, 6))
   plt.plot(paths[:10].T)
   plt.xlabel("Time Steps")
   plt.ylabel("Price")
   plt.title("Geometric Brownian Motion Paths")
   plt.show()

Ornstein-Uhlenbeck Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ou_paths = mc.simulate_ou(
       x0=0.05,
       theta=0.5,
       mu=0.03,
       sigma=0.01,
       T=1.0,
       steps=252,
       paths=10_000,
   )

Correlated Multi-Asset Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   correlation = np.array([
       [1.0, 0.6, 0.3],
       [0.6, 1.0, 0.4],
       [0.3, 0.4, 1.0],
   ])

   correlated_paths = mc.simulate_correlated(
       initial_values=[100.0, 50.0, 75.0],
       mu_values=[0.10, 0.08, 0.12],
       sigma_values=[0.25, 0.30, 0.20],
       correlation=correlation,
       T=1.0,
       steps=252,
       paths=5000,
   )

   print(len(correlated_paths))  # -> 3 assets
   print(correlated_paths[0].shape)  # (5000, 253)

Option Pricing with Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mc_pricer = dervflow.MonteCarloOptionPricer()

   result = mc_pricer.price_european(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.25,
       time=1.0,
       option_type="call",
       num_paths=100_000,
       antithetic=True,
   )

   print(f"MC Price: ${result['price']:.2f} ± ${result['std_error']:.2f}")

See Also
--------

* :doc:`../user_guide/monte_carlo` - User guide for Monte Carlo simulation
* :doc:`../theory/stochastic_processes` - Mathematical background on stochastic processes
* :doc:`options` - Option pricing methods
