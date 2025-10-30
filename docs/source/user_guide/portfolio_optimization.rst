Portfolio Optimization Guide
============================

This guide covers portfolio construction and optimisation features provided by
dervflow. The Rust core exposes quadratic programming utilities for classic
mean–variance optimisation as well as a risk parity solver. The Python bindings
wrap these primitives in convenient classes that operate on NumPy arrays.

Mean-Variance Optimisation
--------------------------

Mean–variance optimisation, introduced by Harry Markowitz, searches for the
portfolio weights that balance expected return against risk (volatility). The
:class:`dervflow.PortfolioOptimizer` class accepts either a matrix of historical
returns or explicit vectors of expected returns and a covariance matrix.

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import dervflow
   import numpy as np

   # Generate sample daily returns (252 trading days, 4 assets)
   rng = np.random.default_rng(seed=42)
   returns = rng.normal(0.001, 0.02, size=(252, 4))

   # Create optimiser from historical returns (expected returns and covariance
   # are inferred automatically)
   optimizer = dervflow.PortfolioOptimizer(returns)

   # Optimise for minimum variance subject to long-only weights
   min_weights = np.zeros(returns.shape[1])
   max_weights = np.ones(returns.shape[1])

   result = optimizer.optimize(
       min_weights=min_weights,
       max_weights=max_weights,
   )

   print(f"Weights: {result['weights']}")
   print(f"Expected return: {result['expected_return']:.2%}")
   print(f"Volatility: {result['volatility']:.2%}")

Target Return
~~~~~~~~~~~~~

To target a specific expected return, provide ``target_return``. Weight bounds
are optional but recommended in practice.

.. code-block:: python

   target = optimizer.optimize(
       target_return=0.10,              # 10% annual target
       min_weights=np.zeros(4),
       max_weights=np.full(4, 0.5),     # Cap exposure at 50% per asset
   )

   print(f"Target-return weights: {target['weights']}")
   print(f"Portfolio volatility: {target['volatility']:.2%}")

Target Risk or Maximum Sharpe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``target_risk`` to maximise return for a given volatility, or
``risk_free_rate`` to maximise the Sharpe ratio.

.. code-block:: python

   # Target annualised volatility of 15%
   high_risk = optimizer.optimize(target_risk=0.15)

   # Maximum Sharpe ratio with a 2% risk-free rate
   max_sharpe = optimizer.optimize(risk_free_rate=0.02)
   print(f"Sharpe ratio: {max_sharpe['sharpe_ratio']:.2f}")

Efficient Frontier
------------------

Generate a grid of optimal portfolios across a range of target returns.

.. code-block:: python

   frontier = optimizer.efficient_frontier(num_points=30)

   risks = [point['volatility'] for point in frontier]
   returns_ef = [point['expected_return'] for point in frontier]

   import matplotlib.pyplot as plt

   plt.plot(risks, returns_ef, marker='o')
   plt.xlabel('Volatility')
   plt.ylabel('Expected return')
   plt.title('Efficient frontier')
   plt.grid(True)
   plt.show()

Working with Weight Bounds
--------------------------

The optimiser accepts optional ``min_weights`` and ``max_weights`` arrays to
model basic allocation constraints. Each array must have length equal to the
number of assets. The default bounds are 0 and 1 (long-only portfolio).

.. code-block:: python

   lower_bounds = np.array([0.05, 0.05, 0.10, 0.00])
   upper_bounds = np.array([0.40, 0.35, 0.40, 0.25])

   constrained = optimizer.optimize(
       target_return=0.09,
       min_weights=lower_bounds,
       max_weights=upper_bounds,
   )
   print(f"Constrained weights: {constrained['weights']}")

Risk Parity Optimisation
------------------------

Risk parity allocates capital so that each asset contributes equally (or in a
specified proportion) to total portfolio risk. Use the
:class:`dervflow.RiskParityOptimizer` for this style of allocation.

.. code-block:: python

   covariance = np.cov(returns, rowvar=False)
   rp = dervflow.RiskParityOptimizer(covariance)

   # Equal risk contribution
   rp_weights = rp.optimize()
   print(f"Risk parity weights: {rp_weights}")

   # Target contributions (e.g. 50%, 30%, 20%)
   targets = np.array([0.4, 0.3, 0.2, 0.1])
   custom_weights = rp.optimize(target_risk_contributions=targets)

   contributions = rp.risk_contributions(custom_weights)
   print(f"Risk contributions: {contributions}")

Further Reading
---------------

* :doc:`../api/portfolio` – API reference for the portfolio module.
* :doc:`../theory/portfolio_theory` – Mathematical background and derivations.
