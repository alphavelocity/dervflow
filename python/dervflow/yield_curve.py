# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Yield curve construction and analysis module

This module provides tools for constructing, interpolating, and analyzing yield curves:

Classes
-------
YieldCurve
    Represents a yield curve with interpolation methods for zero rates,
    forward rates, and discount factors
MultiCurve
    Multi-curve container supporting separate discounting and forwarding curves
SwapPeriod
    Represents an accrual period used when constructing swap schedules
YieldCurveBuilder
    Constructs yield curves from market data using bootstrapping methods
BondAnalytics
    Provides bond analytics including yield to maturity, duration, convexity, and DV01

Features
--------
- Yield curve bootstrapping from bond prices and swap rates
- Multiple interpolation methods: linear, cubic spline, Nelson-Siegel, Nelson-Siegel-Svensson
- Zero rate and forward rate calculations
- Discount factor computation
- Bond pricing from yield curves
- Yield to maturity calculation
- Duration measures: Macaulay duration, Modified duration
- Convexity calculation
- DV01 (dollar value of a basis point)
- Multi-curve framework support (OIS, LIBOR curves)

Examples
--------
>>> import numpy as np
>>> from dervflow.yield_curve import YieldCurve, YieldCurveBuilder, BondAnalytics
>>>
>>> # Create a yield curve from rates
>>> maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
>>> rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.042])
>>> curve = YieldCurve(maturities, rates, method='cubic_spline')
>>>
>>> # Get zero rate at specific maturity
>>> rate_3y = curve.zero_rate(3.0)
>>> print(f"3-year zero rate: {rate_3y:.4f}")
0.0375

>>> # Calculate forward rate
>>> forward_1y2y = curve.forward_rate(1.0, 2.0)
>>> print(f"1y-2y forward rate: {forward_1y2y:.4f}")

>>> # Get discount factor
>>> df_5y = curve.discount_factor(5.0)
>>> print(f"5-year discount factor: {df_5y:.6f}")

>>> # Bootstrap yield curve from bond data
>>> bond_data = [
...     {'maturity': 0.5, 'coupon': 0.02, 'price': 100.5, 'frequency': 2},
...     {'maturity': 1.0, 'coupon': 0.025, 'price': 101.2, 'frequency': 2},
...     {'maturity': 2.0, 'coupon': 0.03, 'price': 102.0, 'frequency': 2},
... ]
>>> builder = YieldCurveBuilder()
>>> bootstrapped_curve = builder.bootstrap(bond_data)
>>>
>>> # Calculate bond analytics
>>> bond_analytics = BondAnalytics()
>>>
>>> # Calculate yield to maturity
>>> ytm = bond_analytics.yield_to_maturity(
...     price=102.0,
...     face_value=100.0,
...     coupon_rate=0.03,
...     years_to_maturity=2.0,
...     frequency=2
... )
>>> print(f"Yield to maturity: {ytm:.4f}")

>>> # Calculate duration
>>> duration = bond_analytics.duration(
...     yield_rate=0.03,
...     coupon_rate=0.03,
...     years_to_maturity=5.0,
...     frequency=2,
...     duration_type='modified'
... )
>>> print(f"Modified duration: {duration:.4f}")

>>> # Calculate convexity
>>> convexity = bond_analytics.convexity(
...     yield_rate=0.03,
...     coupon_rate=0.03,
...     years_to_maturity=5.0,
...     frequency=2
... )
>>> print(f"Convexity: {convexity:.4f}")

>>> # Calculate DV01
>>> dv01 = bond_analytics.dv01(
...     price=102.0,
...     yield_rate=0.03,
...     coupon_rate=0.03,
...     years_to_maturity=5.0,
...     frequency=2
... )
>>> print(f"DV01: {dv01:.6f}")
"""

from dervflow._dervflow import (
    BondAnalytics,
    MultiCurve,
    SwapPeriod,
    YieldCurve,
    YieldCurveBuilder,
)

__all__ = [
    "YieldCurve",
    "MultiCurve",
    "SwapPeriod",
    "YieldCurveBuilder",
    "BondAnalytics",
]
