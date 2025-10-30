Monte Carlo Simulation Guide
============================

DervFlow's Monte Carlo facilities combine Rust performance with a convenient
Python interface. Two components cover most workflows:

* :class:`dervflow.MonteCarloEngine` generates stochastic process paths.
* :class:`dervflow.MonteCarloOptionPricer` values European and American options
  using simulation (with optional parallel execution).

Stochastic Process Simulation
-----------------------------

Create an engine with an optional seed for reproducibility. Each simulation
method accepts a ``parallel`` flag that leverages Rayon threads in Rust when
``True``.

Geometric Brownian Motion
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import dervflow
   import numpy as np

   engine = dervflow.MonteCarloEngine(seed=123)

   paths = engine.simulate_gbm(
       s0=100.0,
       mu=0.08,
       sigma=0.25,
       T=1.0,
       steps=252,
       paths=10_000,
       parallel=True,
   )

   # ``paths`` is a NumPy array of shape (paths, steps + 1)
   terminal = paths[:, -1]
   print(f"Terminal price mean: {terminal.mean():.2f}")
   print(f"Terminal price std:  {terminal.std():.2f}")

Ornstein–Uhlenbeck Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ou_paths = engine.simulate_ou(
       x0=0.03,
       theta=0.7,
       mu=0.04,
       sigma=0.015,
       T=2.0,
       steps=500,
       paths=5_000,
   )
   print(f"OU final mean: {ou_paths[:, -1].mean():.4f}")

Interest Rate Models
~~~~~~~~~~~~~~~~~~~~

Cox–Ingersoll–Ross (CIR) and Vasicek models are also available.

.. code-block:: python

   cir_paths = engine.simulate_cir(
       x0=0.02,
       kappa=1.2,
       theta=0.04,
       sigma=0.08,
       T=5.0,
       steps=1_000,
       paths=4_000,
   )
   print(f"CIR minimum rate: {cir_paths.min():.6f}")

   vasicek_paths = engine.simulate_vasicek(
       r0=0.03,
       kappa=0.6,
       theta=0.045,
       sigma=0.012,
       T=3.0,
       steps=600,
       paths=4_000,
   )
   print(f"Vasicek final mean: {vasicek_paths[:, -1].mean():.4f}")

Correlated GBM Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

``simulate_correlated`` produces correlated GBM paths given drifts, volatilities,
initial values, and a correlation matrix. The return value is a list of NumPy
arrays (one per asset) with shape ``(paths, steps)``.

.. code-block:: python

   initial = [100.0, 80.0, 120.0]
   mus = [0.07, 0.05, 0.06]
   sigmas = [0.20, 0.25, 0.18]
   corr = np.array([
       [1.0, 0.5, 0.3],
       [0.5, 1.0, 0.4],
       [0.3, 0.4, 1.0],
   ])

   correlated = engine.simulate_correlated(
       initial,
       mus,
       sigmas,
       corr,
       T=1.0,
       steps=252,
       paths=20_000,
   )
   print(f"Assets simulated: {len(correlated)}")
   print(f"Asset 0 path grid: {correlated[0].shape}")

Monte Carlo Option Pricing
--------------------------

The Monte Carlo option pricer supports European options (with antithetic
variance reduction) and American options priced via the Longstaff–Schwartz
algorithm.

European Options
~~~~~~~~~~~~~~~~

.. code-block:: python

   pricer = dervflow.MonteCarloOptionPricer()

   euro = pricer.price_european(
       spot=100.0,
       strike=105.0,
       rate=0.03,
       dividend=0.0,
       volatility=0.20,
       time=1.0,
       option_type='call',
       num_paths=75_000,
       use_antithetic=True,
       seed=42,
       parallel=True,
   )

   print(f"European call price: {euro['price']:.2f}")
   print(f"Standard error: {euro['std_error']:.4f}")

American Options
~~~~~~~~~~~~~~~~

.. code-block:: python

   american_put = pricer.price_american(
       spot=100.0,
       strike=100.0,
       rate=0.03,
       dividend=0.0,
       volatility=0.25,
       time=1.0,
       option_type='put',
       num_paths=60_000,
       num_steps=50,
       seed=42,
       parallel=True,
   )

   print(f"American put price: {american_put:.2f}")

Tips
----

* Increase ``num_paths`` gradually – Monte Carlo error decreases with
  :math:`\sqrt{N}`.
* Specify ``seed`` for reproducible pricing or path generation.
* Enable ``parallel=True`` for CPU-bound workloads to take advantage of Rust's
  multithreading.

Further Reading
---------------

* :doc:`../api/monte_carlo` – Detailed API reference.
* :doc:`../theory/stochastic_processes` – Mathematical background for the
  supported processes.
