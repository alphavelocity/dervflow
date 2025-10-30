Time Series API
===============

.. currentmodule:: dervflow

:class:`TimeSeriesAnalyzer` bundles descriptive statistics, correlation
analysis, GARCH modelling and diagnostic testing into a single Rust-backed
object. Instantiate it with a NumPy array of prices or returns and call the
methods documented below.

TimeSeriesAnalyzer
------------------

.. autoclass:: TimeSeriesAnalyzer
   :members: returns, stat, statistics, autocorrelation, partial_autocorrelation, correlation, fit_garch, stationarity_test, jarque_bera_test, ljung_box_test, quantile, ewma, rolling
   :show-inheritance:

Method overview
~~~~~~~~~~~~~~~

``returns``
    Compute simple or log returns from price data.
``stat`` / ``statistics``
    Summary statistics including mean, variance, skewness and kurtosis.
``autocorrelation`` / ``partial_autocorrelation``
    Autocorrelation functions for diagnostic analysis.
``correlation``
    Pairwise correlation matrix for multivariate inputs.
``fit_garch``
    Estimate GARCH(p, q) parameters and return forecast helpers.
``stationarity_test``
    Run ADF or KPSS stationarity tests.
``jarque_bera_test`` / ``ljung_box_test``
    Normality and serial-correlation hypothesis tests.
``quantile``
    Empirical quantiles from the sample distribution.
``ewma``
    Exponentially weighted moving averages and volatility estimates.
``rolling``
    Rolling window statistics (mean, std, min, max, etc.).

.. code-block:: python

   import numpy as np
   import dervflow

   prices = np.array([100, 102, 101, 104, 106, 105], dtype=float)
   analyzer = dervflow.TimeSeriesAnalyzer(prices)

   log_returns = analyzer.returns(method="log")
   stats = analyzer.statistics()
   acf = analyzer.autocorrelation(max_lag=20)
   garch = analyzer.fit_garch(p=1, q=1)
   jb = analyzer.jarque_bera_test()
   rolling = analyzer.rolling(window=20, statistic="mean")

See Also
--------

* :doc:`../user_guide/time_series` – workflow tutorials
* :doc:`../theory/stochastic_processes` – theoretical background on stochastic
  processes
