# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical methods module.

This module exposes the suite of numerical algorithms implemented in Rust.
The foundational statistical helpers live in :mod:`dervflow.core`.

Integration
-----------
AdaptiveSimpsonsIntegrator
    Adaptive Simpson's rule integration with configurable tolerances.
GaussLegendreIntegrator
    Fixed-order Gauss-Legendre quadrature (2–20 points).
AdaptiveGaussLegendreIntegrator
    Adaptive Gauss-Legendre quadrature with automatic subdivision.

Root finding
------------
NewtonRaphsonSolver
    Quadratic-convergent Newton-Raphson solver using analytical derivatives.
BrentSolver
    Robust Brent-Dekker solver combining bisection, secant and IQI steps.
BisectionSolver
    Classic bracketing-based solver with guaranteed convergence.
SecantSolver
    Derivative-free solver using successive secant approximations.

Optimization
------------
GradientDescentOptimizer
    Gradient descent with backtracking line search and convergence diagnostics.
BFGSOptimizer
    Quasi-Newton BFGS optimizer with dynamic Hessian approximation.
NelderMeadOptimizer
    Derivative-free Nelder-Mead simplex optimizer.

Linear algebra
--------------
LinearAlgebra
    Wrapper exposing matrix factorizations (Cholesky, LU, QR, SVD), decompositions,
    matrix functions (powers, exponentials), norms, condition numbers, positive-
    definite adjustments and linear system solvers.

Random number generation
------------------------
RandomGenerator
    Seedable pseudo-random generator with normal and uniform draws.
ThreadLocalRandom
    Thread-local RNG convenience wrapper.
SobolSequence
    Sobol low-discrepancy sequence generator.
HaltonSequence
    Halton low-discrepancy sequence generator.

These classes are thin wrappers over the corresponding Rust implementations
and provide ergonomic, NumPy-friendly APIs for Python users.
"""

from dervflow._dervflow import (
    AdaptiveGaussLegendreIntegrator,
    AdaptiveSimpsonsIntegrator,
    BFGSOptimizer,
    BisectionSolver,
    BrentSolver,
    GaussLegendreIntegrator,
    GradientDescentOptimizer,
    HaltonSequence,
    IntegrationResult,
    LinearAlgebra,
    NelderMeadOptimizer,
    NewtonRaphsonSolver,
    OptimizationResult,
    RandomGenerator,
    RootFindingResult,
    SecantSolver,
    SobolSequence,
    ThreadLocalRandom,
)

__all__ = [
    "AdaptiveSimpsonsIntegrator",
    "GaussLegendreIntegrator",
    "AdaptiveGaussLegendreIntegrator",
    "IntegrationResult",
    "NewtonRaphsonSolver",
    "BrentSolver",
    "BisectionSolver",
    "SecantSolver",
    "RootFindingResult",
    "GradientDescentOptimizer",
    "BFGSOptimizer",
    "NelderMeadOptimizer",
    "OptimizationResult",
    "LinearAlgebra",
    "RandomGenerator",
    "ThreadLocalRandom",
    "SobolSequence",
    "HaltonSequence",
]
