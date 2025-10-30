Time Series Analysis Guide
==========================

This guide covers statistical analysis of financial time series data in dervflow.

Overview
--------

Time series analysis is essential for understanding financial data patterns, testing trading strategies, and modeling volatility. dervflow provides:

* Return calculations (simple, log, continuously compounded)
* Statistical measures (moments, quantiles, correlations)
* Volatility modeling (GARCH family)
* Statistical tests (stationarity, normality, autocorrelation)

Return Calculations
-------------------

Simple Returns
~~~~~~~~~~~~~~

Calculate simple returns from price data:

.. code-block:: python

   import dervflow
   import numpy as np

   # Price data
   prices = np.array([100, 102, 101, 105, 103, 107])

   # Create analyzer
   analyzer = dervflow.TimeSeriesAnalyzer(prices)

   # Calculate simple returns
   simple_returns = analyzer.returns(method='simple')
   print(f"Simple returns: {simple_returns}")

Log Returns
~~~~~~~~~~~

Log returns are preferred for statistical analysis:

.. code-block:: python

   # Calculate log returns
   log_returns = analyzer.returns(method='log')
   print(f"Log returns: {log_returns}")

   # Log returns are additive
   total_return = np.sum(log_returns)
   print(f"Total log return: {total_return:.4f}")

Continuously Compounded Returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Continuously compounded returns
   cc_returns = analyzer.returns(method='continuous')

Rolling Returns
~~~~~~~~~~~~~~~

Calculate returns over rolling windows:

.. code-block:: python

   # 20-day rolling returns
   rolling_returns = analyzer.rolling_returns(window=20, method='log')

Statistical Measures
--------------------

Basic Statistics
~~~~~~~~~~~~~~~~

Calculate mean, variance, and standard deviation:

.. code-block:: python

   # Get basic stat metrics
   stats = analyzer.stat()

   print(f"Mean return: {stats['mean']:.6f}")
   print(f"Std deviation: {stats['std']:.6f}")
   print(f"Variance: {stats['variance']:.6f}")
   print(f"Min: {stats['min']:.6f}")
   print(f"Max: {stats['max']:.6f}")

Higher Moments
~~~~~~~~~~~~~~

Calculate skewness and kurtosis:

.. code-block:: python

   # Skewness (asymmetry)
   skewness = analyzer.skewness()
   print(f"Skewness: {skewness:.4f}")

   # Kurtosis (tail heaviness)
   kurtosis = analyzer.kurtosis()
   print(f"Excess kurtosis: {kurtosis:.4f}")

   # Interpretation:
   # Skewness > 0: Right-skewed (positive tail)
   # Skewness < 0: Left-skewed (negative tail)
   # Kurtosis > 0: Heavy tails (leptokurtic)
   # Kurtosis < 0: Light tails (platykurtic)

Quantiles and Percentiles
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate quantiles
   q25 = analyzer.quantile(0.25)
   q50 = analyzer.quantile(0.50)  # Median
   q75 = analyzer.quantile(0.75)

   print(f"25th percentile: {q25:.6f}")
   print(f"Median: {q50:.6f}")
   print(f"75th percentile: {q75:.6f}")

Rolling Statistics
~~~~~~~~~~~~~~~~~~

Calculate stat metrics over rolling windows:

.. code-block:: python

   # 20-day rolling mean
   rolling_mean = analyzer.rolling_mean(window=20)

   # 20-day rolling std (volatility)
   rolling_vol = analyzer.rolling_std(window=20)

   # Annualize volatility (assuming daily data)
   annual_vol = rolling_vol * np.sqrt(252)

Exponentially Weighted Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Give more weight to recent observations:

.. code-block:: python

   # EWMA with half-life of 20 days
   ewma = analyzer.ewma(halflife=20)

   # Exponentially weighted volatility
   ewm_vol = analyzer.ewm_std(halflife=20)

Correlation Analysis
--------------------

Autocorrelation
~~~~~~~~~~~~~~~

Measure correlation of a series with its own lagged values:

.. code-block:: python

   # Autocorrelation function (ACF)
   acf = analyzer.autocorrelation(max_lag=20)

   # Plot ACF
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 4))
   plt.stem(range(len(acf)), acf)
   plt.xlabel('Lag')
   plt.ylabel('Autocorrelation')
   plt.title('Autocorrelation Function')
   plt.axhline(y=0, color='k', linestyle='--')
   plt.show()

Partial Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Partial autocorrelation function (PACF)
   pacf = analyzer.partial_autocorrelation(max_lag=20)

   # PACF helps identify AR order in ARMA models

Cross-Correlation
~~~~~~~~~~~~~~~~~

Measure correlation between two time series:

.. code-block:: python

   # Two asset returns
   returns1 = analyzer1.returns(method='log')
   returns2 = analyzer2.returns(method='log')

   # Cross-correlation
   cross_corr = dervflow.cross_correlation(returns1, returns2, max_lag=10)

Correlation Measures
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Pearson correlation (linear)
   pearson = dervflow.pearson_correlation(returns1, returns2)

   # Spearman rank correlation (monotonic)
   spearman = dervflow.spearman_correlation(returns1, returns2)

   # Kendall tau (ordinal)
   kendall = dervflow.kendall_correlation(returns1, returns2)

   print(f"Pearson: {pearson:.4f}")
   print(f"Spearman: {spearman:.4f}")
   print(f"Kendall: {kendall:.4f}")

Rolling Correlation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 60-day rolling correlation
   rolling_corr = dervflow.rolling_correlation(
       returns1, returns2, window=60
   )

Volatility Modeling
-------------------

GARCH Models
~~~~~~~~~~~~

Generalized AutoRegressive Conditional Heteroskedasticity (GARCH) models capture volatility clustering:

.. code-block:: python

   # Fit GARCH(1,1) model
   garch_model = analyzer.fit_garch(p=1, q=1)

   print(f"Omega: {garch_model['omega']:.6f}")
   print(f"Alpha: {garch_model['alpha']:.6f}")
   print(f"Beta: {garch_model['beta']:.6f}")

   # Persistence
   persistence = garch_model['alpha'] + garch_model['beta']
   print(f"Persistence: {persistence:.4f}")

Volatility Forecasting
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Forecast volatility
   forecast_horizon = 10
   vol_forecast = garch_model.forecast(horizon=forecast_horizon)

   print(f"Volatility forecast:")
   for i, vol in enumerate(vol_forecast, 1):
       print(f"  Day {i}: {vol:.4%}")

EGARCH Model
~~~~~~~~~~~~

Exponential GARCH captures asymmetric volatility response:

.. code-block:: python

   # Fit EGARCH(1,1) model
   egarch_model = analyzer.fit_egarch(p=1, q=1)

   # Negative shocks have larger impact on volatility
   print(f"Leverage effect: {egarch_model['gamma']:.4f}")

GJR-GARCH Model
~~~~~~~~~~~~~~~

GJR-GARCH (Glosten-Jagannathan-Runkle) model:

.. code-block:: python

   # Fit GJR-GARCH(1,1) model
   gjr_model = analyzer.fit_gjr_garch(p=1, q=1)

   # Asymmetric response to shocks
   print(f"Asymmetry parameter: {gjr_model['gamma']:.4f}")

Statistical Tests
-----------------

Stationarity Tests
~~~~~~~~~~~~~~~~~~

Test if a time series is stationary:

.. code-block:: python

   # Augmented Dickey-Fuller test
   adf_result = analyzer.adf_test()

   print(f"ADF statistic: {adf_result['statistic']:.4f}")
   print(f"p-value: {adf_result['p_value']:.4f}")
   print(f"Critical values: {adf_result['critical_values']}")

   if adf_result['p_value'] < 0.05:
       print("Series is stationary (reject null hypothesis)")
   else:
       print("Series is non-stationary (fail to reject null)")

   # KPSS test (null hypothesis: stationary)
   kpss_result = analyzer.kpss_test()

   print(f"KPSS statistic: {kpss_result['statistic']:.4f}")
   print(f"p-value: {kpss_result['p_value']:.4f}")

Autocorrelation Tests
~~~~~~~~~~~~~~~~~~~~~

Test for autocorrelation in residuals:

.. code-block:: python

   # Ljung-Box test
   lb_result = analyzer.ljung_box_test(lags=10)

   print(f"Ljung-Box statistic: {lb_result['statistic']:.4f}")
   print(f"p-value: {lb_result['p_value']:.4f}")

   if lb_result['p_value'] < 0.05:
       print("Significant autocorrelation detected")

Normality Tests
~~~~~~~~~~~~~~~

Test if returns are normally distributed:

.. code-block:: python

   # Jarque-Bera test
   jb_result = analyzer.jarque_bera_test()

   print(f"JB statistic: {jb_result['statistic']:.4f}")
   print(f"p-value: {jb_result['p_value']:.4f}")

   if jb_result['p_value'] < 0.05:
       print("Returns are not normally distributed")

   # Kolmogorov-Smirnov test
   ks_result = analyzer.ks_test()

   print(f"KS statistic: {ks_result['statistic']:.4f}")
   print(f"p-value: {ks_result['p_value']:.4f}")

Practical Examples
------------------

Complete Volatility Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import dervflow
   import numpy as np
   import matplotlib.pyplot as plt

   # Load price data
   prices = np.loadtxt('stock_prices.csv')

   # Create analyzer
   analyzer = dervflow.TimeSeriesAnalyzer(prices)

   # Calculate returns
   returns = analyzer.returns(method='log')

   # Basic stat metrics
   stats = analyzer.stat()
   print(f"Mean: {stats['mean']:.6f}")
   print(f"Volatility: {stats['std']:.6f}")
   print(f"Skewness: {analyzer.skewness():.4f}")
   print(f"Kurtosis: {analyzer.kurtosis():.4f}")

   # Test for stationarity
   adf = analyzer.adf_test()
   print(f"ADF p-value: {adf['p_value']:.4f}")

   # Fit GARCH model
   garch = analyzer.fit_garch(p=1, q=1)
   print(f"GARCH parameters: {garch}")

   # Plot results
   fig, axes = plt.subplots(3, 1, figsize=(12, 10))

   # Returns
   axes[0].plot(returns)
   axes[0].set_title('Log Returns')
   axes[0].set_ylabel('Return')

   # Rolling volatility
   rolling_vol = analyzer.rolling_std(window=20) * np.sqrt(252)
   axes[1].plot(rolling_vol)
   axes[1].set_title('20-Day Rolling Volatility (Annualized)')
   axes[1].set_ylabel('Volatility')

   # ACF
   acf = analyzer.autocorrelation(max_lag=20)
   axes[2].stem(range(len(acf)), acf)
   axes[2].set_title('Autocorrelation Function')
   axes[2].set_xlabel('Lag')
   axes[2].set_ylabel('ACF')

   plt.tight_layout()
   plt.show()

Performance Tips
----------------

* Use log returns for stat analysis (additive property)
* Choose appropriate window sizes for rolling stat metrics
* Test for stationarity before applying time series models
* Use GARCH models when volatility clustering is present
* Consider exponentially weighted stat metrics for recent data emphasis

Next Steps
----------

* See :doc:`../api/timeseries` for detailed API documentation
* Check out the time series notebook: ``examples/notebooks/06_time_series.ipynb``
* Learn about :doc:`risk_analytics` for risk metric calculations
