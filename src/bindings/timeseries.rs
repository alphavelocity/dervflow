// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for time series module

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::common::error::DervflowError;
use crate::timeseries::correlation::{
    autocorrelation, kendall_correlation, partial_autocorrelation, pearson_correlation,
    spearman_correlation,
};
use crate::timeseries::models::{GarchModel, GarchVariant};
use crate::timeseries::returns::{ReturnType, calculate_returns, calculate_rolling_returns};
use crate::timeseries::stat::{
    calculate_stat, ewma, quantile, quantiles, rolling_mean, rolling_std,
};
use crate::timeseries::tests::{adf_test, jarque_bera_test, kpss_test, ljung_box_test};

/// Convert DervflowError to Python exception
fn to_py_err(err: DervflowError) -> PyErr {
    PyValueError::new_err(format!("{}", err))
}

/// Parse return type string
fn parse_return_type(return_type: &str) -> PyResult<ReturnType> {
    match return_type.to_lowercase().as_str() {
        "simple" => Ok(ReturnType::Simple),
        "log" => Ok(ReturnType::Log),
        "continuous" => Ok(ReturnType::Continuous),
        _ => Err(PyValueError::new_err(format!(
            "Invalid return type '{}'. Must be 'simple', 'log', or 'continuous'",
            return_type
        ))),
    }
}

/// Parse GARCH variant string
fn parse_garch_variant(variant: &str) -> PyResult<GarchVariant> {
    match variant.to_lowercase().as_str() {
        "standard" | "garch" => Ok(GarchVariant::Standard),
        "egarch" => Ok(GarchVariant::EGARCH),
        "gjr" | "gjrgarch" | "gjr-garch" => Ok(GarchVariant::GJRGARCH),
        _ => Err(PyValueError::new_err(format!(
            "Invalid GARCH variant '{}'. Must be 'standard', 'egarch', or 'gjr'",
            variant
        ))),
    }
}

/// Time series analysis toolkit
///
/// This class provides comprehensive statistical analysis tools for financial time series,
/// including return calculations, statistical measures, correlation analysis, GARCH modeling,
/// and statistical tests.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from dervflow.timeseries import TimeSeriesAnalyzer
/// >>> prices = np.array([100.0, 102.0, 101.5, 103.0, 104.5])
/// >>> analyzer = TimeSeriesAnalyzer(prices)
/// >>> returns = analyzer.returns(method='log')
/// >>> stats = analyzer.stat()
/// >>> print(f"Mean return: {stats['mean']:.4f}")
#[pyclass(name = "TimeSeriesAnalyzer")]
pub struct PyTimeSeriesAnalyzer {
    data: Vec<f64>,
}

#[pymethods]
impl PyTimeSeriesAnalyzer {
    /// Create a new TimeSeriesAnalyzer instance
    ///
    /// Parameters
    /// ----------
    /// data : array_like
    ///     Time series data (prices or returns)
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> prices = np.array([100.0, 102.0, 101.5, 103.0])
    /// >>> analyzer = TimeSeriesAnalyzer(prices)
    #[new]
    fn new(data: PyReadonlyArray1<f64>) -> PyResult<Self> {
        Ok(PyTimeSeriesAnalyzer {
            data: data.as_slice()?.to_vec(),
        })
    }

    /// Calculate returns from price series
    ///
    /// Parameters
    /// ----------
    /// method : str, optional
    ///     Type of return calculation: 'simple', 'log', or 'continuous' (default: 'log')
    /// window : int, optional
    ///     If specified, calculate rolling returns over this window
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of returns
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If data is insufficient or contains invalid values
    ///
    /// Examples
    /// --------
    /// >>> prices = np.array([100.0, 105.0, 103.0, 108.0])
    /// >>> analyzer = TimeSeriesAnalyzer(prices)
    /// >>> returns = analyzer.returns(method='log')
    #[pyo3(signature = (method="log", window=None))]
    fn returns<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        window: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let return_type = parse_return_type(method)?;

        let returns = if let Some(w) = window {
            calculate_rolling_returns(&self.data, w, return_type).map_err(to_py_err)?
        } else {
            calculate_returns(&self.data, return_type).map_err(to_py_err)?
        };

        Ok(PyArray1::from_vec(py, returns))
    }

    /// Calculate statistical measures
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - mean: arithmetic mean
    ///     - sum: total of all observations
    ///     - variance: sample variance
    ///     - std_dev / std: standard deviation (with alias `std`)
    ///     - std_error: standard error of the mean
    ///     - skewness: skewness (asymmetry measure)
    ///     - kurtosis: excess kurtosis (tail measure)
    ///     - min / max: extrema of the series
    ///     - range: max - min
    ///     - median: 50th percentile
    ///     - q1 / q3: 25th and 75th percentiles
    ///     - iqr: interquartile range (q3 - q1)
    ///     - mean_abs_dev: mean absolute deviation around the mean
    ///     - median_abs_dev: unscaled median absolute deviation
    ///     - root_mean_square: quadratic mean of the series
    ///     - count: number of observations
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If data is insufficient (need at least 4 observations)
    ///
    /// Examples
    /// --------
    /// >>> returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> stats = analyzer.stat()
    /// >>> print(f"Mean: {stats['mean']:.4f}, Std: {stats['std_dev']:.4f}")
    fn stat(&self) -> PyResult<HashMap<String, f64>> {
        let stats = calculate_stat(&self.data).map_err(to_py_err)?;

        let mut result = HashMap::new();
        result.insert("count".to_string(), stats.count as f64);
        result.insert("sum".to_string(), stats.sum);
        result.insert("mean".to_string(), stats.mean);
        result.insert("variance".to_string(), stats.variance);
        result.insert("std_dev".to_string(), stats.std_dev);
        result.insert("std".to_string(), stats.std_dev);
        result.insert("std_error".to_string(), stats.std_error);
        result.insert("skewness".to_string(), stats.skewness);
        result.insert("kurtosis".to_string(), stats.kurtosis);
        result.insert("min".to_string(), stats.min);
        result.insert("max".to_string(), stats.max);
        result.insert("range".to_string(), stats.range);
        result.insert("median".to_string(), stats.median);
        result.insert("q1".to_string(), stats.q1);
        result.insert("q3".to_string(), stats.q3);
        result.insert("iqr".to_string(), stats.iqr);
        result.insert("mean_abs_dev".to_string(), stats.mean_abs_dev);
        result.insert("median_abs_dev".to_string(), stats.median_abs_dev);
        result.insert("root_mean_square".to_string(), stats.root_mean_square);

        Ok(result)
    }

    #[pyo3(name = "statistics")]
    fn statistics_alias(&self) -> PyResult<HashMap<String, f64>> {
        self.stat()
    }

    /// Calculate autocorrelation function (ACF)
    ///
    /// Parameters
    /// ----------
    /// max_lag : int
    ///     Maximum lag to calculate
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of autocorrelation values from lag 0 to max_lag
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If max_lag is too large for the data
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> acf = analyzer.autocorrelation(max_lag=20)
    #[pyo3(signature = (max_lag))]
    fn autocorrelation<'py>(
        &self,
        py: Python<'py>,
        max_lag: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let acf = autocorrelation(&self.data, max_lag).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, acf))
    }

    /// Calculate partial autocorrelation function (PACF)
    ///
    /// Parameters
    /// ----------
    /// max_lag : int
    ///     Maximum lag to calculate
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of partial autocorrelation values from lag 0 to max_lag
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If max_lag is too large for the data
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> pacf = analyzer.partial_autocorrelation(max_lag=20)
    #[pyo3(signature = (max_lag))]
    fn partial_autocorrelation<'py>(
        &self,
        py: Python<'py>,
        max_lag: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pacf = partial_autocorrelation(&self.data, max_lag).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, pacf))
    }

    /// Calculate correlation with another series
    ///
    /// Parameters
    /// ----------
    /// other : array_like
    ///     Other time series to correlate with
    /// method : str, optional
    ///     Correlation method: 'pearson', 'spearman', or 'kendall' (default: 'pearson')
    ///
    /// Returns
    /// -------
    /// float
    ///     Correlation coefficient
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If series have different lengths or method is invalid
    ///
    /// Examples
    /// --------
    /// >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    /// >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    /// >>> analyzer = TimeSeriesAnalyzer(x)
    /// >>> corr = analyzer.correlation(y, method='pearson')
    #[pyo3(signature = (other, method="pearson"))]
    fn correlation(&self, other: PyReadonlyArray1<f64>, method: &str) -> PyResult<f64> {
        let other_data = other.as_slice()?.to_vec();

        let corr = match method.to_lowercase().as_str() {
            "pearson" => pearson_correlation(&self.data, &other_data),
            "spearman" => spearman_correlation(&self.data, &other_data),
            "kendall" => kendall_correlation(&self.data, &other_data),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid correlation method '{}'. Must be 'pearson', 'spearman', or 'kendall'",
                    method
                )));
            }
        };

        corr.map_err(to_py_err)
    }

    /// Fit GARCH model to the data
    ///
    /// Parameters
    /// ----------
    /// variant : str, optional
    ///     GARCH variant: 'standard', 'egarch', or 'gjr' (default: 'standard')
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - omega: constant term
    ///     - alpha: ARCH coefficient
    ///     - beta: GARCH coefficient
    ///     - gamma: asymmetry parameter (for EGARCH and GJR-GARCH)
    ///     - log_likelihood: model log-likelihood
    ///     - conditional_variances: fitted conditional variances
    ///     - aic: Akaike Information Criterion
    ///     - bic: Bayesian Information Criterion
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If data is insufficient or model fails to converge
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100) * 0.01
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> garch_result = analyzer.fit_garch(variant='standard')
    /// >>> print(f"Alpha: {garch_result['alpha']:.4f}")
    /// >>> print(f"Beta: {garch_result['beta']:.4f}")
    #[pyo3(signature = (variant="standard"))]
    fn fit_garch<'py>(&self, py: Python<'py>, variant: &str) -> PyResult<Py<PyAny>> {
        let garch_variant = parse_garch_variant(variant)?;
        let model = GarchModel::fit(&self.data, garch_variant).map_err(to_py_err)?;

        let (aic, bic) = model.information_criteria(self.data.len());

        let dict = PyDict::new(py);
        dict.set_item("omega", model.params.omega)?;
        dict.set_item("alpha", model.params.alpha)?;
        dict.set_item("beta", model.params.beta)?;
        dict.set_item("gamma", model.params.gamma)?;
        dict.set_item("log_likelihood", model.log_likelihood)?;
        dict.set_item("aic", aic)?;
        dict.set_item("bic", bic)?;
        dict.set_item(
            "conditional_variances",
            PyArray1::from_vec(py, model.conditional_variances),
        )?;

        Ok(dict.into())
    }

    /// Perform stationarity test
    ///
    /// Parameters
    /// ----------
    /// test : str
    ///     Test type: 'adf' (Augmented Dickey-Fuller) or 'kpss'
    /// regression : str, optional
    ///     Regression type: 'c' (constant), 'ct' (constant+trend), or 'nc' (no constant)
    ///     (default: 'c')
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - statistic: test statistic value
    ///     - p_value: p-value
    ///     - critical_values: dict of critical values at different significance levels
    ///     - reject_null: whether to reject null hypothesis at 5% significance
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If data is insufficient or test type is invalid
    ///
    /// Notes
    /// -----
    /// ADF test:
    ///     - Null hypothesis: series has a unit root (non-stationary)
    ///     - Alternative: series is stationary
    ///
    /// KPSS test:
    ///     - Null hypothesis: series is stationary
    ///     - Alternative: series has a unit root (non-stationary)
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> adf_result = analyzer.stationarity_test(test='adf')
    /// >>> print(f"ADF statistic: {adf_result['statistic']:.4f}")
    /// >>> print(f"P-value: {adf_result['p_value']:.4f}")
    #[pyo3(signature = (test, regression="c"))]
    fn stationarity_test<'py>(
        &self,
        py: Python<'py>,
        test: &str,
        regression: &str,
    ) -> PyResult<Py<PyAny>> {
        let result = match test.to_lowercase().as_str() {
            "adf" => adf_test(&self.data, None, regression),
            "kpss" => kpss_test(&self.data, regression, None),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid test type '{}'. Must be 'adf' or 'kpss'",
                    test
                )));
            }
        };

        let test_result = result.map_err(to_py_err)?;

        let critical_dict = PyDict::new(py);
        for (sig_level, crit_val) in test_result.critical_values {
            critical_dict.set_item(sig_level.to_string(), crit_val)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("statistic", test_result.statistic)?;
        dict.set_item("p_value", test_result.p_value)?;
        dict.set_item("critical_values", critical_dict)?;
        dict.set_item("reject_null", test_result.reject_null)?;

        Ok(dict.into())
    }

    /// Perform Ljung-Box test for autocorrelation
    ///
    /// Parameters
    /// ----------
    /// lags : int
    ///     Number of lags to test
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - statistic: test statistic value
    ///     - p_value: p-value
    ///     - reject_null: whether to reject null hypothesis at 5% significance
    ///
    /// Notes
    /// -----
    /// Null hypothesis: No autocorrelation up to lag h
    /// Alternative: At least one autocorrelation is non-zero
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> lb_result = analyzer.ljung_box_test(lags=10)
    /// >>> print(f"Ljung-Box statistic: {lb_result['statistic']:.4f}")
    #[pyo3(signature = (lags))]
    fn ljung_box_test<'py>(&self, py: Python<'py>, lags: usize) -> PyResult<Py<PyAny>> {
        let result = ljung_box_test(&self.data, lags).map_err(to_py_err)?;

        let dict = PyDict::new(py);
        dict.set_item("statistic", result.statistic)?;
        dict.set_item("p_value", result.p_value)?;
        dict.set_item("reject_null", result.reject_null)?;

        Ok(dict.into())
    }

    /// Perform Jarque-Bera test for normality
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - statistic: test statistic value
    ///     - p_value: p-value
    ///     - reject_null: whether to reject null hypothesis at 5% significance
    ///
    /// Notes
    /// -----
    /// Null hypothesis: Data is normally distributed
    /// Alternative: Data is not normally distributed
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> jb_result = analyzer.jarque_bera_test()
    /// >>> print(f"Jarque-Bera statistic: {jb_result['statistic']:.4f}")
    fn jarque_bera_test<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let result = jarque_bera_test(&self.data).map_err(to_py_err)?;

        let dict = PyDict::new(py);
        dict.set_item("statistic", result.statistic)?;
        dict.set_item("p_value", result.p_value)?;
        dict.set_item("reject_null", result.reject_null)?;

        Ok(dict.into())
    }

    /// Calculate rolling statistics
    ///
    /// Parameters
    /// ----------
    /// window : int
    ///     Window size for rolling calculation
    /// statistic : str, optional
    ///     Statistic to calculate: 'mean' or 'std' (default: 'mean')
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of rolling statistics
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> rolling_mean = analyzer.rolling(window=20, statistic='mean')
    /// >>> rolling_std = analyzer.rolling(window=20, statistic='std')
    #[pyo3(signature = (window, statistic="mean"))]
    fn rolling<'py>(
        &self,
        py: Python<'py>,
        window: usize,
        statistic: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = match statistic.to_lowercase().as_str() {
            "mean" => rolling_mean(&self.data, window),
            "std" => rolling_std(&self.data, window),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid statistic '{}'. Must be 'mean' or 'std'",
                    statistic
                )));
            }
        };

        let values = result.map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    /// Calculate exponentially weighted moving average (EWMA)
    ///
    /// Parameters
    /// ----------
    /// alpha : float
    ///     Smoothing factor (0 < alpha <= 1), higher values give more weight to recent observations
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of EWMA values
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> ema = analyzer.ewma(alpha=0.1)
    #[pyo3(signature = (alpha))]
    fn ewma<'py>(&self, py: Python<'py>, alpha: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = ewma(&self.data, alpha).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    /// Calculate quantiles
    ///
    /// Parameters
    /// ----------
    /// q : float or array_like
    ///     Quantile(s) to calculate (between 0 and 1)
    ///
    /// Returns
    /// -------
    /// float or ndarray
    ///     Quantile value(s)
    ///
    /// Examples
    /// --------
    /// >>> returns = np.random.randn(100)
    /// >>> analyzer = TimeSeriesAnalyzer(returns)
    /// >>> median = analyzer.quantile(0.5)
    /// >>> quartiles = analyzer.quantile([0.25, 0.5, 0.75])
    #[pyo3(signature = (q))]
    fn quantile<'py>(&self, py: Python<'py>, q: Py<PyAny>) -> PyResult<Py<PyAny>> {
        // Try to extract as single float
        if let Ok(q_val) = q.extract::<f64>(py) {
            let result = quantile(&self.data, q_val).map_err(to_py_err)?;
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as array
        if let Ok(q_array) = q.extract::<Vec<f64>>(py) {
            let results = quantiles(&self.data, &q_array).map_err(to_py_err)?;
            return Ok(PyArray1::from_vec(py, results).into());
        }

        Err(PyValueError::new_err(
            "q must be a float or array of floats",
        ))
    }
}

/// Register time series module with Python
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyTimeSeriesAnalyzer>()?;
    Ok(())
}
