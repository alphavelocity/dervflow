# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Time series analysis module

Provides comprehensive statistical analysis tools for financial time series including:
- Return calculations (simple, log, continuous)
- Statistical measures (mean, variance, skewness, kurtosis)
- Correlation analysis (ACF, PACF, cross-correlation)
- GARCH modeling (standard, EGARCH, GJR-GARCH)
- Statistical tests (ADF, KPSS, Ljung-Box, Jarque-Bera)
- Rolling stat metrics and exponentially weighted moving averages

Examples
--------
>>> import numpy as np
>>> from dervflow.timeseries import TimeSeriesAnalyzer
>>>
>>> # Create sample price data
>>> prices = np.array([100.0, 102.0, 101.5, 103.0, 104.5, 103.8, 105.2])
>>>
>>> # Initialize analyzer
>>> analyzer = TimeSeriesAnalyzer(prices)
>>>
>>> # Calculate returns
>>> returns = analyzer.returns(method='log')
>>>
>>> # Get statistical measures
>>> stats = analyzer.stat()
>>> print(f"Mean: {stats['mean']:.4f}, Std Dev: {stats['std_dev']:.4f}")
>>>
>>> # Calculate autocorrelation
>>> acf = analyzer.autocorrelation(max_lag=10)
>>>
>>> # Fit GARCH model
>>> garch_result = analyzer.fit_garch(variant='standard')
>>> print(f"Alpha: {garch_result['alpha']:.4f}, Beta: {garch_result['beta']:.4f}")
>>>
>>> # Test for stationarity
>>> adf_result = analyzer.stationarity_test(test='adf')
>>> print(f"ADF Statistic: {adf_result['statistic']:.4f}")
>>> print(f"P-value: {adf_result['p_value']:.4f}")
"""

from dervflow._dervflow import TimeSeriesAnalyzer

__all__ = ["TimeSeriesAnalyzer"]
