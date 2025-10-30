# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Options pricing module

This module provides comprehensive option pricing models and analytics:

Classes
-------
BlackScholesModel
    Black-Scholes-Merton model for European options with analytical Greeks
BinomialTreeModel
    Binomial tree models (CRR, JR) for American and European options
MonteCarloOptionPricer
    Monte Carlo simulation for option pricing with variance reduction
AsianOption
    Asian option pricing (arithmetic and geometric averaging)
BarrierOption
    Barrier option pricing (up-and-out, down-and-out, up-and-in, down-and-in)
LookbackOption
    Lookback option pricing (fixed and floating strike)
DigitalOption
    Digital/binary option pricing

Features
--------
- Analytical pricing for European options using Black-Scholes-Merton
- Numerical pricing for American options using binomial trees
- Monte Carlo simulation with variance reduction techniques
- Exotic option pricing (Asian, barrier, lookback, digital)
- Implied volatility calculation using Newton-Raphson and Brent's method
- Volatility surface construction and interpolation
- SABR model calibration
- Batch pricing for multiple options
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga)

Examples
--------
>>> from dervflow.options import BlackScholesModel
>>>
>>> # Price a European call option
>>> bs = BlackScholesModel()
>>> price = bs.price(
...     spot=100.0,
...     strike=100.0,
...     rate=0.05,
...     dividend=0.0,
...     volatility=0.2,
...     time=1.0,
...     option_type='call'
... )
>>> print(f"Option price: {price:.2f}")
10.45

>>> # Calculate Greeks
>>> greeks = bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
>>> print(f"Delta: {greeks['delta']:.4f}")
>>> print(f"Gamma: {greeks['gamma']:.4f}")
>>> print(f"Vega: {greeks['vega']:.4f}")

>>> # Calculate implied volatility
>>> market_price = 10.45
>>> iv = bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 1.0, 'call')
>>> print(f"Implied volatility: {iv:.4f}")
0.2000

>>> # Price American option using binomial tree
>>> from dervflow.options import BinomialTreeModel
>>> tree = BinomialTreeModel()
>>> american_price = tree.price(
...     spot=100.0,
...     strike=100.0,
...     rate=0.05,
...     dividend=0.0,
...     volatility=0.2,
...     time=1.0,
...     steps=100,
...     style='american',
...     option_type='put'
... )
>>> print(f"American put price: {american_price:.2f}")

>>> # Price Asian option
>>> from dervflow.options import AsianOption
>>> asian = AsianOption()
>>> asian_price = asian.price(
...     spot=100.0,
...     strike=100.0,
...     rate=0.05,
...     dividend=0.0,
...     volatility=0.2,
...     time=1.0,
...     averaging_type='arithmetic',
...     option_type='call'
... )
>>> print(f"Asian option price: {asian_price:.2f}")

>>> # Price barrier option
>>> from dervflow.options import BarrierOption
>>> barrier = BarrierOption()
>>> barrier_price = barrier.price(
...     spot=100.0,
...     strike=100.0,
...     barrier=110.0,
...     rate=0.05,
...     dividend=0.0,
...     volatility=0.2,
...     time=1.0,
...     barrier_type='up-and-out',
...     option_type='call'
... )
>>> print(f"Barrier option price: {barrier_price:.2f}")
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dervflow._dervflow import (
        AsianOption,
        BarrierOption,
        BinomialTreeModel,
        BlackScholesModel,
        DigitalOption,
        LookbackOption,
        MonteCarloOptionPricer,
        SABRModel,
        VolatilitySurface,
    )
else:
    from dervflow._dervflow import (
        AsianOption,
        BarrierOption,
        BinomialTreeModel,
        BlackScholesModel,
        DigitalOption,
        LookbackOption,
        MonteCarloOptionPricer,
        SABRModel,
        VolatilitySurface,
    )

__all__ = [
    "BlackScholesModel",
    "BinomialTreeModel",
    "MonteCarloOptionPricer",
    "AsianOption",
    "BarrierOption",
    "LookbackOption",
    "DigitalOption",
    "VolatilitySurface",
    "SABRModel",
]
