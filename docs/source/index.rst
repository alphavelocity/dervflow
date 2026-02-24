DervFlow Documentation
======================

Welcome to the DervFlow documentation portal. DervFlow is a quantitative finance
library implemented in Rust with first-class Python bindings. It provides
high-performance primitives for option pricing, risk analytics, portfolio
optimisation, stochastic processes, and time-series tooling that are designed to
be production ready.

.. image:: https://badge.fury.io/py/dervflow.svg
   :target: https://badge.fury.io/py/dervflow
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/dervflow.svg
   :target: https://pypi.org/project/dervflow/
   :alt: Supported Python versions

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

Overview
--------

**Core capabilities**

* **Options pricing** - Black-Scholes-Merton analytics, binomial trees for
  European and American exercise, Monte Carlo pricing, and exotic payoffs
  including Asian, barrier, lookback, and digital contracts.
* **Risk analytics** - first and second order Greeks with extended
  sensitivities, portfolio Greek aggregation, Value at Risk (historical,
  variance-covariance, Cornish-Fisher, Monte Carlo) and Conditional VaR.
* **Portfolio optimisation** - mean-variance optimisation with multiple
  objectives, efficient frontier generation, and a risk parity optimiser with
  configurable risk contributions.
* **Yield curves** - bootstrapping from bond and swap quotes, interpolation via
  linear and spline methods as well as Nelson-Siegel(-Svensson), plus bond
  analytics (pricing, duration, convexity, DV01).
* **Time series** - return calculations, rolling and exponentially weighted
  stat metrics, correlation analysis, GARCH-family volatility models, and
  stationarity/normality tests.
* **Monte Carlo simulation** - high-performance path generators for GBM,
  Ornstein-Uhlenbeck, CIR, and Vasicek processes, correlated multi-asset
  simulation, and a dedicated Monte Carlo option pricer.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install dervflow

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import dervflow

   # Black-Scholes pricing
   bs_model = dervflow.BlackScholesModel()
   price = bs_model.price(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.2,
       time=1.0,
       option_type="call",
   )
   print(f"Option price: ${price:.2f}")

   # Calculate Greeks
   greeks = bs_model.greeks(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.02,
       volatility=0.2,
       time=1.0,
       option_type="call",
   )
   print(f"Delta: {greeks['delta']:.4f}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/options_pricing
   user_guide/risk_analytics
   user_guide/portfolio_optimization
   user_guide/yield_curves
   user_guide/time_series
   user_guide/monte_carlo

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/options
   api/risk
   api/portfolio
   api/yield_curve
   api/timeseries
   api/monte_carlo
   api/numerical
   api/functions
   api/parameters
   api/errors
   api/examples

.. toctree::
   :maxdepth: 2
   :caption: Mathematical Background

   theory/black_scholes
   theory/greeks
   theory/var
   theory/portfolio_theory
   theory/yield_curves
   theory/stochastic_processes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
