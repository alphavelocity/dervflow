# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
dervflow - High-performance quantitative finance library

A production-grade quantitative finance library built with Rust and exposed to Python.
Provides fast implementations of options pricing, risk analytics, portfolio optimization,
yield curve construction, time series analysis, and Monte Carlo simulation.

Modules
-------
core
    Rust-backed statistical, vector, series, and combinatoric helpers exposed as
    direct Python functions for quantitative workflows.
options
    Options pricing models including Black-Scholes, binomial trees, Monte Carlo,
    and exotic options (Asian, barrier, lookback, digital)
risk
    Risk analytics including Greeks calculation, Value at Risk (VaR),
    and portfolio risk metrics
portfolio
    Portfolio optimization including mean-variance optimization, risk parity,
    and efficient frontier calculation
yield_curve
    Yield curve construction, interpolation, and bond analytics
timeseries
    Time series analysis including returns calculation, statistical measures,
    GARCH modeling, and stationarity tests
monte_carlo
    Monte Carlo simulation engine for stochastic processes including GBM,
    jump-diffusion, and stochastic volatility models
utils
    Utility functions and helpers

Examples
--------
>>> import dervflow
>>> print(dervflow.__version__)
0.1.0

>>> # Price a European call option using Black-Scholes
>>> from dervflow import BlackScholesModel
>>> bs = BlackScholesModel()
>>> price = bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
>>> print(f"Option price: {price:.2f}")

>>> # Optimize a portfolio
>>> import numpy as np
>>> from dervflow import PortfolioOptimizer
>>> returns = np.array([0.10, 0.12, 0.08])
>>> cov = np.array([[0.04, 0.01, 0.005],
...                 [0.01, 0.09, 0.01],
...                 [0.005, 0.01, 0.0225]])
>>> optimizer = PortfolioOptimizer(returns, cov)
>>> result = optimizer.optimize(target_return=0.10)
>>> print(f"Optimal weights: {result['weights']}")

>>> # Analyze time series
>>> from dervflow import TimeSeriesAnalyzer
>>> prices = np.array([100.0, 102.0, 101.5, 103.0, 104.5])
>>> analyzer = TimeSeriesAnalyzer(prices)
>>> returns = analyzer.returns(method='log')
>>> stats = analyzer.stat()
"""

__version__ = "0.1.0"

# Import core Rust classes from the compiled extension
from dervflow._dervflow import (
    AsianOption,
    BarrierOption,
    BinomialTreeModel,
    BlackScholesModel,
    BondAnalytics,
    DigitalOption,
    GreeksCalculator,
    LookbackOption,
    MonteCarloEngine,
    MonteCarloOptionPricer,
    MultiCurve,
    RiskMetrics,
    SABRModel,
    SwapPeriod,
    TimeSeriesAnalyzer,
    VolatilitySurface,
    YieldCurve,
    YieldCurveBuilder,
)
from dervflow.numerical import (
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

# Import Python wrapper classes
from dervflow.portfolio import (
    BlackLittermanModel,
    FactorModel,
    InvestorViews,
    PortfolioOptimizer,
    RiskParityOptimizer,
)

from . import core as core
from . import utils as utils

# Define public API
__all__ = [
    # Version
    "__version__",
    # Options pricing
    "BlackScholesModel",
    "BinomialTreeModel",
    "MonteCarloOptionPricer",
    # Exotic options
    "AsianOption",
    "BarrierOption",
    "LookbackOption",
    "DigitalOption",
    "VolatilitySurface",
    "SABRModel",
    # Risk analytics
    "GreeksCalculator",
    "RiskMetrics",
    # Portfolio optimization
    "PortfolioOptimizer",
    "RiskParityOptimizer",
    "BlackLittermanModel",
    "InvestorViews",
    "FactorModel",
    # Time series analysis
    "TimeSeriesAnalyzer",
    # Yield curves
    "YieldCurve",
    "YieldCurveBuilder",
    "BondAnalytics",
    "MultiCurve",
    "SwapPeriod",
    # Monte Carlo simulation
    "MonteCarloEngine",
    # Numerical integration
    "AdaptiveSimpsonsIntegrator",
    "GaussLegendreIntegrator",
    "AdaptiveGaussLegendreIntegrator",
    "IntegrationResult",
    # Root finding
    "NewtonRaphsonSolver",
    "BrentSolver",
    "BisectionSolver",
    "SecantSolver",
    "RootFindingResult",
    # Optimization
    "GradientDescentOptimizer",
    "BFGSOptimizer",
    "NelderMeadOptimizer",
    "OptimizationResult",
    # Linear algebra utilities
    "LinearAlgebra",
    # Core mathematics
    "core",
    "utils",
    # Random number generation
    "RandomGenerator",
    "ThreadLocalRandom",
    "SobolSequence",
    "HaltonSequence",
]
