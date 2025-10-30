# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monte Carlo simulation module

This module provides a comprehensive Monte Carlo simulation engine for stochastic processes:

Classes
-------
MonteCarloEngine
    High-performance Monte Carlo simulation engine for various stochastic processes
    with parallel processing support

Features
--------
- Geometric Brownian Motion (GBM) simulation
- Ornstein-Uhlenbeck (OU) mean-reverting process
- Cox-Ingersoll-Ross (CIR) interest rate model
- Vasicek interest rate model
- Merton jump-diffusion process
- Kou double exponential jump-diffusion
- Heston stochastic volatility model
- SABR stochastic volatility model
- Correlated multi-asset path generation using Cholesky decomposition
- Quasi-random sequences (Sobol, Halton) for variance reduction
- Antithetic variates variance reduction
- Parallel path generation using Rayon
- Thread-safe random number generation

Examples
--------
>>> import numpy as np
>>> from dervflow.monte_carlo import MonteCarloEngine
>>> 
>>> # Create Monte Carlo engine
>>> mc = MonteCarloEngine()
>>> 
>>> # Simulate Geometric Brownian Motion
>>> gbm_paths = mc.simulate_gbm(
...     s0=100.0,           # Initial price
...     mu=0.05,            # Drift
...     sigma=0.2,          # Volatility
...     T=1.0,              # Time horizon
...     steps=252,          # Number of time steps
...     paths=10000         # Number of paths
... )
>>> print(f"GBM paths shape: {gbm_paths.shape}")
(10000, 252)
>>> print(f"Final prices mean: {gbm_paths[:, -1].mean():.2f}")

>>> # Simulate Ornstein-Uhlenbeck process (mean-reverting)
>>> ou_paths = mc.simulate_ou(
...     x0=0.05,            # Initial value
...     theta=0.5,          # Mean reversion speed
...     mu=0.03,            # Long-term mean
...     sigma=0.01,         # Volatility
...     T=1.0,
...     steps=252,
...     paths=10000
... )
>>> print(f"OU final values mean: {ou_paths[:, -1].mean():.4f}")

>>> # Simulate jump-diffusion process (Merton model)
>>> jump_paths = mc.simulate_jump_diffusion(
...     s0=100.0,
...     mu=0.05,
...     sigma=0.2,
...     lambda_=10.0,       # Jump intensity (10 jumps per year)
...     jump_mean=-0.02,    # Average jump size
...     jump_std=0.05,      # Jump size volatility
...     T=1.0,
...     steps=252,
...     paths=10000
... )
>>> print(f"Jump-diffusion final prices mean: {jump_paths[:, -1].mean():.2f}")

>>> # Simulate Heston stochastic volatility model
>>> heston_paths = mc.simulate_heston(
...     s0=100.0,
...     v0=0.04,            # Initial variance
...     kappa=2.0,          # Variance mean reversion speed
...     theta=0.04,         # Long-term variance
...     sigma_v=0.3,        # Volatility of variance
...     rho=-0.7,           # Correlation between price and variance
...     mu=0.05,
...     T=1.0,
...     steps=252,
...     paths=10000
... )
>>> print(f"Heston paths shape: {heston_paths.shape}")

>>> # Simulate correlated multi-asset paths
>>> correlation = np.array([
...     [1.0, 0.5, 0.3],
...     [0.5, 1.0, 0.4],
...     [0.3, 0.4, 1.0]
... ])
>>> processes = [
...     {'type': 'gbm', 's0': 100.0, 'mu': 0.05, 'sigma': 0.2},
...     {'type': 'gbm', 's0': 50.0, 'mu': 0.06, 'sigma': 0.25},
...     {'type': 'gbm', 's0': 150.0, 'mu': 0.04, 'sigma': 0.15},
... ]
>>> correlated_paths = mc.simulate_correlated(
...     processes=processes,
...     correlation=correlation,
...     T=1.0,
...     steps=252,
...     paths=10000
... )
>>> print(f"Number of assets: {len(correlated_paths)}")
>>> print(f"Asset 1 paths shape: {correlated_paths[0].shape}")

>>> # Use quasi-random sequences for variance reduction
>>> sobol_paths = mc.simulate_gbm(
...     s0=100.0,
...     mu=0.05,
...     sigma=0.2,
...     T=1.0,
...     steps=252,
...     paths=10000,
...     quasi_random='sobol'
... )

>>> # Use antithetic variates for variance reduction
>>> antithetic_paths = mc.simulate_gbm(
...     s0=100.0,
...     mu=0.05,
...     sigma=0.2,
...     T=1.0,
...     steps=252,
...     paths=10000,
...     antithetic=True
... )
>>> print(f"Antithetic paths shape: {antithetic_paths.shape}")
(20000, 252)  # Double the paths due to antithetic variates
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dervflow._dervflow import MonteCarloEngine
else:
    from dervflow._dervflow import MonteCarloEngine

__all__ = [
    'MonteCarloEngine',
]
