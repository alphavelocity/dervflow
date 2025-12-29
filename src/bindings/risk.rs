// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for risk module

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common::types::{Greeks, OptionParams, OptionType};
use crate::options::analytical::black_scholes_price;
use crate::risk::greeks::{
    ExtendedGreeks, FiniteDifferenceConfig, calculate_extended_greeks, calculate_numerical_greeks,
    calculate_portfolio_extended_greeks, calculate_portfolio_greeks,
};
use crate::risk::portfolio_risk::{
    ActivePortfolioMetrics, CapmMetrics, PortfolioSummary, portfolio_parametric_cvar,
    portfolio_parametric_cvar_contributions, portfolio_parametric_var,
    portfolio_parametric_var_contributions, portfolio_summary,
};
use crate::risk::var::{
    cornish_fisher_var, historical_cvar, historical_var, monte_carlo_cvar, monte_carlo_var,
    parametric_var, riskmetrics_cvar, riskmetrics_var,
};
use nalgebra::DMatrix;

/// Python wrapper for Greeks calculator
///
/// This class provides methods to calculate option Greeks (sensitivities)
/// using finite difference methods.
#[pyclass(name = "GreeksCalculator")]
pub struct PyGreeksCalculator {
    config: FiniteDifferenceConfig,
}

#[pymethods]
impl PyGreeksCalculator {
    /// Create a new GreeksCalculator
    ///
    /// Parameters
    /// ----------
    /// spot_bump : float, optional
    ///     Relative bump size for spot price (default: 0.01 = 1%)
    /// vol_bump : float, optional
    ///     Absolute bump size for volatility (default: 0.01 = 1% vol)
    /// time_bump : float, optional
    ///     Bump size for time in years (default: 1/365 = 1 day)
    /// rate_bump : float, optional
    ///     Absolute bump size for interest rate (default: 0.0001 = 1 bp)
    ///
    /// Returns
    /// -------
    /// GreeksCalculator
    ///     A new Greeks calculator instance
    #[new]
    #[pyo3(signature = (spot_bump=0.01, vol_bump=0.01, time_bump=1.0/365.0, rate_bump=0.0001))]
    fn new(spot_bump: f64, vol_bump: f64, time_bump: f64, rate_bump: f64) -> Self {
        Self {
            config: FiniteDifferenceConfig {
                spot_bump,
                vol_bump,
                time_bump,
                rate_bump,
            },
        }
    }

    /// Calculate Greeks for a single option
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying (annualized)
    /// time_to_maturity : float
    ///     Time to maturity in years
    /// option_type : str
    ///     Option type: 'call' or 'put'
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing Greeks: delta, gamma, vega, theta, rho
    ///
    /// Examples
    /// --------
    /// >>> calc = GreeksCalculator()
    /// >>> greeks = calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
    /// >>> print(greeks['delta'])
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time_to_maturity, option_type))]
    fn calculate(
        &self,
        py: Python<'_>,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time_to_maturity: f64,
        option_type: &str,
    ) -> PyResult<Py<PyAny>> {
        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time_to_maturity,
            opt_type,
        );

        let greeks = calculate_numerical_greeks(&black_scholes_price, &params, Some(self.config))
            .map_err(PyErr::from)?;

        greeks_to_dict(py, &greeks)
    }

    /// Calculate extended Greeks including second and third order sensitivities
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying (annualized)
    /// time_to_maturity : float
    ///     Time to maturity in years
    /// option_type : str
    ///     Option type: 'call' or 'put'
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing all Greeks including vanna, volga, speed, zomma, color, ultima
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time_to_maturity, option_type))]
    fn calculate_extended(
        &self,
        py: Python<'_>,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time_to_maturity: f64,
        option_type: &str,
    ) -> PyResult<Py<PyAny>> {
        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(
            spot,
            strike,
            rate,
            dividend,
            volatility,
            time_to_maturity,
            opt_type,
        );

        let extended = calculate_extended_greeks(&black_scholes_price, &params, Some(self.config))
            .map_err(PyErr::from)?;

        extended_greeks_to_dict(py, &extended)
    }

    /// Calculate portfolio Greeks from multiple positions
    ///
    /// Parameters
    /// ----------
    /// spots : array_like
    ///     Array of spot prices
    /// strikes : array_like
    ///     Array of strike prices
    /// rates : array_like
    ///     Array of interest rates
    /// dividends : array_like
    ///     Array of dividend yields
    /// volatilities : array_like
    ///     Array of volatilities
    /// times_to_maturity : array_like
    ///     Array of times to maturity
    /// option_types : list of str
    ///     List of option types ('call' or 'put')
    /// quantities : array_like
    ///     Array of position quantities (positive for long, negative for short)
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing aggregated portfolio Greeks
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spots, strikes, rates, dividends, volatilities, times_to_maturity, option_types, quantities))]
    fn portfolio_greeks(
        &self,
        py: Python<'_>,
        spots: PyReadonlyArray1<f64>,
        strikes: PyReadonlyArray1<f64>,
        rates: PyReadonlyArray1<f64>,
        dividends: PyReadonlyArray1<f64>,
        volatilities: PyReadonlyArray1<f64>,
        times_to_maturity: PyReadonlyArray1<f64>,
        option_types: Vec<String>,
        quantities: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyAny>> {
        let spots = spots.as_slice()?;
        let strikes = strikes.as_slice()?;
        let rates = rates.as_slice()?;
        let dividends = dividends.as_slice()?;
        let volatilities = volatilities.as_slice()?;
        let times = times_to_maturity.as_slice()?;
        let quantities = quantities.as_slice()?;

        // Validate array lengths
        let n = spots.len();
        if strikes.len() != n
            || rates.len() != n
            || dividends.len() != n
            || volatilities.len() != n
            || times.len() != n
            || option_types.len() != n
            || quantities.len() != n
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All input arrays must have the same length",
            ));
        }

        // Build positions
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let opt_type = parse_option_type(&option_types[i])?;
            let params = OptionParams::new(
                spots[i],
                strikes[i],
                rates[i],
                dividends[i],
                volatilities[i],
                times[i],
                opt_type,
            );
            positions.push((params, quantities[i]));
        }

        let portfolio =
            calculate_portfolio_greeks(&black_scholes_price, &positions, Some(self.config))
                .map_err(PyErr::from)?;

        greeks_to_dict(py, &portfolio)
    }

    /// Calculate portfolio extended Greeks from multiple positions
    ///
    /// Parameters
    /// ----------
    /// spots : array_like
    ///     Array of spot prices
    /// strikes : array_like
    ///     Array of strike prices
    /// rates : array_like
    ///     Array of interest rates
    /// dividends : array_like
    ///     Array of dividend yields
    /// volatilities : array_like
    ///     Array of volatilities
    /// times_to_maturity : array_like
    ///     Array of times to maturity
    /// option_types : list of str
    ///     List of option types ('call' or 'put')
    /// quantities : array_like
    ///     Array of position quantities
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing aggregated portfolio extended Greeks
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spots, strikes, rates, dividends, volatilities, times_to_maturity, option_types, quantities))]
    fn portfolio_extended_greeks(
        &self,
        py: Python<'_>,
        spots: PyReadonlyArray1<f64>,
        strikes: PyReadonlyArray1<f64>,
        rates: PyReadonlyArray1<f64>,
        dividends: PyReadonlyArray1<f64>,
        volatilities: PyReadonlyArray1<f64>,
        times_to_maturity: PyReadonlyArray1<f64>,
        option_types: Vec<String>,
        quantities: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyAny>> {
        let spots = spots.as_slice()?;
        let strikes = strikes.as_slice()?;
        let rates = rates.as_slice()?;
        let dividends = dividends.as_slice()?;
        let volatilities = volatilities.as_slice()?;
        let times = times_to_maturity.as_slice()?;
        let quantities = quantities.as_slice()?;

        // Validate array lengths
        let n = spots.len();
        if strikes.len() != n
            || rates.len() != n
            || dividends.len() != n
            || volatilities.len() != n
            || times.len() != n
            || option_types.len() != n
            || quantities.len() != n
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All input arrays must have the same length",
            ));
        }

        // Build positions
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let opt_type = parse_option_type(&option_types[i])?;
            let params = OptionParams::new(
                spots[i],
                strikes[i],
                rates[i],
                dividends[i],
                volatilities[i],
                times[i],
                opt_type,
            );
            positions.push((params, quantities[i]));
        }

        let portfolio = calculate_portfolio_extended_greeks(
            &black_scholes_price,
            &positions,
            Some(self.config),
        )
        .map_err(PyErr::from)?;

        extended_greeks_to_dict(py, &portfolio)
    }
}

/// Parse option type string to OptionType enum
fn parse_option_type(s: &str) -> PyResult<OptionType> {
    match s.to_lowercase().as_str() {
        "call" => Ok(OptionType::Call),
        "put" => Ok(OptionType::Put),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid option type: '{}'. Must be 'call' or 'put'",
            s
        ))),
    }
}

/// Convert Greeks struct to Python dictionary
fn greeks_to_dict(py: Python, greeks: &Greeks) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("delta", greeks.delta)?;
    dict.set_item("gamma", greeks.gamma)?;
    dict.set_item("vega", greeks.vega)?;
    dict.set_item("theta", greeks.theta)?;
    dict.set_item("rho", greeks.rho)?;
    Ok(dict.into())
}

/// Convert ExtendedGreeks struct to Python dictionary
fn extended_greeks_to_dict(py: Python, extended: &ExtendedGreeks) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    // First-order Greeks
    dict.set_item("delta", extended.greeks.delta)?;
    dict.set_item("gamma", extended.greeks.gamma)?;
    dict.set_item("vega", extended.greeks.vega)?;
    dict.set_item("theta", extended.greeks.theta)?;
    dict.set_item("rho", extended.greeks.rho)?;

    // Second-order Greeks
    dict.set_item("vanna", extended.vanna)?;
    dict.set_item("volga", extended.volga)?;

    // Third-order Greeks
    dict.set_item("speed", extended.speed)?;
    dict.set_item("zomma", extended.zomma)?;
    dict.set_item("color", extended.color)?;
    dict.set_item("ultima", extended.ultima)?;

    Ok(dict.into())
}

fn portfolio_summary_to_dict(py: Python, summary: PortfolioSummary) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("expected_return", summary.expected_return)?;
    dict.set_item("variance", summary.variance)?;
    dict.set_item("volatility", summary.volatility)?;
    dict.set_item("sharpe_ratio", summary.sharpe_ratio)?;
    dict.set_item("diversification_ratio", summary.diversification_ratio)?;
    dict.set_item("weight_concentration", summary.weight_concentration)?;
    dict.set_item("risk_concentration", summary.risk_concentration)?;

    let contributions = PyDict::new(py);
    contributions.set_item(
        "marginal",
        PyArray1::from_vec(py, summary.marginal_risk).unbind(),
    )?;
    contributions.set_item(
        "component",
        PyArray1::from_vec(py, summary.component_risk).unbind(),
    )?;
    contributions.set_item(
        "percentage",
        PyArray1::from_vec(py, summary.percentage_risk).unbind(),
    )?;
    dict.set_item("risk_contributions", contributions)?;

    Ok(dict.into())
}

fn active_metrics_to_dict(py: Python, metrics: ActivePortfolioMetrics) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "active_weights",
        PyArray1::from_vec(py, metrics.active_weights).unbind(),
    )?;
    dict.set_item("active_return", metrics.active_return)?;
    dict.set_item("portfolio_return", metrics.portfolio_return)?;
    dict.set_item("benchmark_return", metrics.benchmark_return)?;
    dict.set_item("tracking_error", metrics.tracking_error)?;
    dict.set_item("information_ratio", metrics.information_ratio)?;
    dict.set_item("active_share", metrics.active_share)?;

    let contributions = PyDict::new(py);
    contributions.set_item(
        "marginal",
        PyArray1::from_vec(py, metrics.marginal_tracking_error).unbind(),
    )?;
    contributions.set_item(
        "component",
        PyArray1::from_vec(py, metrics.component_tracking_error).unbind(),
    )?;
    contributions.set_item(
        "percentage",
        PyArray1::from_vec(py, metrics.percentage_tracking_error).unbind(),
    )?;
    dict.set_item("tracking_error_contributions", contributions)?;

    if let Some((marginal, component, percentage)) = metrics.active_return_contributions {
        let active_contributions = PyDict::new(py);
        active_contributions.set_item("marginal", PyArray1::from_vec(py, marginal).unbind())?;
        active_contributions.set_item("component", PyArray1::from_vec(py, component).unbind())?;
        active_contributions.set_item("percentage", PyArray1::from_vec(py, percentage).unbind())?;
        dict.set_item("active_return_contributions", active_contributions)?;
    } else {
        dict.set_item("active_return_contributions", py.None())?;
    }

    Ok(dict.into())
}

fn capm_metrics_to_dict(py: Python, metrics: CapmMetrics) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("portfolio_return", metrics.portfolio_return)?;
    dict.set_item("portfolio_excess_return", metrics.portfolio_excess_return)?;
    dict.set_item("benchmark_return", metrics.benchmark_return)?;
    dict.set_item("benchmark_excess_return", metrics.benchmark_excess_return)?;
    dict.set_item("beta", metrics.beta)?;
    dict.set_item("alpha", metrics.alpha)?;
    Ok(dict.into())
}

fn contributions_to_dict(
    py: Python,
    contributions: (Vec<f64>, Vec<f64>, Vec<f64>),
) -> PyResult<Py<PyAny>> {
    let (marginal, component, percentage) = contributions;
    let dict = PyDict::new(py);
    dict.set_item("marginal", PyArray1::from_vec(py, marginal).unbind())?;
    dict.set_item("component", PyArray1::from_vec(py, component).unbind())?;
    dict.set_item("percentage", PyArray1::from_vec(py, percentage).unbind())?;
    Ok(dict.into())
}

fn array2_to_dmatrix(array: PyReadonlyArray2<f64>) -> PyResult<DMatrix<f64>> {
    let array_view = array.as_array();
    let (rows, cols) = (array_view.shape()[0], array_view.shape()[1]);
    let data: Vec<f64> = array_view.iter().copied().collect();
    Ok(DMatrix::from_row_slice(rows, cols, &data))
}

/// Python wrapper for risk metrics including VaR calculations
///
/// This class provides methods to calculate Value at Risk (VaR) and other risk metrics
/// using multiple methodologies.
#[pyclass(name = "RiskMetrics")]
pub struct PyRiskMetrics;

#[pymethods]
impl PyRiskMetrics {
    /// Create a new RiskMetrics instance
    #[new]
    fn new() -> Self {
        Self
    }

    /// Calculate Value at Risk using specified method
    ///
    /// Parameters
    /// ----------
    /// returns : array_like
    ///     Historical returns data (for historical and parametric methods)
    /// confidence_level : float
    ///     Confidence level (e.g., 0.95 for 95%)
    /// method : str
    ///     VaR calculation method: 'historical', 'parametric', 'cornish_fisher', 'monte_carlo', or 'ewma'
    /// mean : float, optional
    ///     Expected return (required for monte_carlo method)
    /// std_dev : float, optional
    ///     Standard deviation (required for monte_carlo method)
    /// num_simulations : int, optional
    ///     Number of simulations (for monte_carlo method, default: 10000)
    /// seed : int, optional
    ///     Random seed for reproducibility (for monte_carlo method)
    /// decay : float, optional
    ///     EWMA decay factor when ``method='ewma'`` (default: 0.94)
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing 'var', 'confidence_level', and 'method'
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> rm = RiskMetrics()
    /// >>> returns = np.random.normal(0, 0.02, 1000)
    /// >>> result = rm.var(returns, 0.95, 'historical')
    /// >>> print(result['var'])
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (returns=None, confidence_level=0.95, method="historical", mean=None, std_dev=None, num_simulations=10000, seed=None, decay=None))]
    fn var(
        &self,
        py: Python<'_>,
        returns: Option<PyReadonlyArray1<f64>>,
        confidence_level: f64,
        method: &str,
        mean: Option<f64>,
        std_dev: Option<f64>,
        num_simulations: usize,
        seed: Option<u64>,
        decay: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let mut method_key = method.trim().to_lowercase();
        method_key = method_key.replace('-', "_");
        method_key = method_key.replace(' ', "_");

        let var_value = match method_key.as_str() {
            "historical" => {
                let returns = returns.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "returns array is required for historical method",
                    )
                })?;
                let returns_slice = returns.as_slice()?;
                historical_var(returns_slice, confidence_level).map_err(PyErr::from)?
            }
            "parametric" => {
                let returns = returns.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "returns array is required for parametric method",
                    )
                })?;
                let returns_slice = returns.as_slice()?;
                parametric_var(returns_slice, confidence_level).map_err(PyErr::from)?
            }
            "cornish_fisher" | "cornish-fisher" => {
                let returns = returns.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "returns array is required for cornish_fisher method",
                    )
                })?;
                let returns_slice = returns.as_slice()?;
                cornish_fisher_var(returns_slice, confidence_level).map_err(PyErr::from)?
            }
            "monte_carlo" | "montecarlo" | "mc" => {
                let mean = mean.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mean is required for monte_carlo method",
                    )
                })?;
                let std_dev = std_dev.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "std_dev is required for monte_carlo method",
                    )
                })?;
                monte_carlo_var(mean, std_dev, num_simulations, confidence_level, seed)
                    .map_err(PyErr::from)?
            }
            "ewma" | "riskmetrics" | "risk_metrics" | "riskmetric" | "risk_metric"
            | "risk metrics" => {
                let returns = returns.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "returns array is required for ewma method",
                    )
                })?;
                let returns_slice = returns.as_slice()?;
                let decay_value = decay.unwrap_or(0.94);
                riskmetrics_var(returns_slice, confidence_level, decay_value)
                    .map_err(PyErr::from)?
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid method: '{}'. Must be 'historical', 'parametric', 'cornish_fisher', 'monte_carlo', or 'ewma'",
                    method
                )));
            }
        };

        let dict = PyDict::new(py);
        dict.set_item("var", var_value)?;
        dict.set_item("confidence_level", confidence_level)?;
        dict.set_item("method", method)?;
        Ok(dict.into())
    }

    /// Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
    ///
    /// CVaR is the expected loss given that the loss exceeds VaR
    ///
    /// Parameters
    /// ----------
    /// returns : array_like
    ///     Historical returns data (for historical method)
    /// confidence_level : float
    ///     Confidence level (e.g., 0.95 for 95%)
    /// method : str, optional
    ///     CVaR calculation method: 'historical' or 'monte_carlo' (default: 'historical')
    /// mean : float, optional
    ///     Expected return (required for monte_carlo method)
    /// std_dev : float, optional
    ///     Standard deviation (required for monte_carlo method)
    /// num_simulations : int, optional
    ///     Number of simulations (for monte_carlo method, default: 10000)
    /// seed : int, optional
    ///     Random seed for reproducibility (for monte_carlo method)
    /// decay : float, optional
    ///     EWMA decay factor when ``method='ewma'`` (default: 0.94)
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing 'cvar', 'confidence_level', and 'method'
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> rm = RiskMetrics()
    /// >>> returns = np.random.normal(0, 0.02, 1000)
    /// >>> result = rm.cvar(returns, 0.95)
    /// >>> print(result['cvar'])
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (returns=None, confidence_level=0.95, method="historical", mean=None, std_dev=None, num_simulations=10000, seed=None, decay=None))]
    fn cvar(
        &self,
        py: Python<'_>,
        returns: Option<PyReadonlyArray1<f64>>,
        confidence_level: f64,
        method: &str,
        mean: Option<f64>,
        std_dev: Option<f64>,
        num_simulations: usize,
        seed: Option<u64>,
        decay: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let mut method_key = method.trim().to_lowercase();
        method_key = method_key.replace('-', "_");
        method_key = method_key.replace(' ', "_");

        let cvar_value = match method_key.as_str() {
            "historical" => {
                let returns = returns.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "returns array is required for historical method",
                    )
                })?;
                let returns_slice = returns.as_slice()?;
                historical_cvar(returns_slice, confidence_level).map_err(PyErr::from)?
            }
            "monte_carlo" | "montecarlo" | "mc" => {
                let mean = mean.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mean is required for monte_carlo method",
                    )
                })?;
                let std_dev = std_dev.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "std_dev is required for monte_carlo method",
                    )
                })?;
                monte_carlo_cvar(mean, std_dev, num_simulations, confidence_level, seed)
                    .map_err(PyErr::from)?
            }
            "ewma" | "riskmetrics" | "risk_metrics" | "riskmetric" | "risk_metric" => {
                let returns = returns.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "returns array is required for ewma method",
                    )
                })?;
                let returns_slice = returns.as_slice()?;
                let decay_value = decay.unwrap_or(0.94);
                riskmetrics_cvar(returns_slice, confidence_level, decay_value)
                    .map_err(PyErr::from)?
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid method: '{}'. Must be 'historical', 'monte_carlo', or 'ewma'",
                    method
                )));
            }
        };

        let dict = PyDict::new(py);
        dict.set_item("cvar", cvar_value)?;
        dict.set_item("confidence_level", confidence_level)?;
        dict.set_item("method", method)?;
        Ok(dict.into())
    }

    /// Calculate maximum drawdown from returns series
    ///
    /// Maximum drawdown is the largest peak-to-trough decline in cumulative returns
    ///
    /// Parameters
    /// ----------
    /// returns : array_like
    ///     Returns data
    ///
    /// Returns
    /// -------
    /// float
    ///     Maximum drawdown (positive number representing the decline)
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> rm = RiskMetrics()
    /// >>> returns = np.array([0.01, -0.02, 0.03, -0.05, 0.02])
    /// >>> mdd = rm.max_drawdown(returns)
    #[pyo3(signature = (returns))]
    fn max_drawdown(&self, returns: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let returns_slice = returns.as_slice()?;

        if returns_slice.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Returns array cannot be empty",
            ));
        }

        // Calculate cumulative returns
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &ret in returns_slice {
            cumulative *= 1.0 + ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let drawdown = (peak - cumulative) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        Ok(max_dd)
    }

    /// Compute portfolio-level risk metrics from weights and covariance matrix.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, covariance, expected_returns=None, risk_free_rate=None))]
    fn portfolio_metrics(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        expected_returns: Option<PyReadonlyArray1<f64>>,
        risk_free_rate: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let weights_vec = weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;
        let expected_returns_vec = match expected_returns {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };

        let summary = portfolio_summary(
            &weights_vec,
            &covariance_matrix,
            expected_returns_vec.as_deref(),
            risk_free_rate,
        )
        .map_err(PyErr::from)?;

        portfolio_summary_to_dict(py, summary)
    }

    /// Compute portfolio tracking error relative to a benchmark weight vector.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, benchmark_weights, covariance))]
    fn portfolio_tracking_error(
        &self,
        weights: PyReadonlyArray1<f64>,
        benchmark_weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
    ) -> PyResult<f64> {
        let weights_vec = weights.as_slice()?.to_vec();
        let benchmark_vec = benchmark_weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;

        crate::risk::portfolio_risk::portfolio_tracking_error(
            &weights_vec,
            &benchmark_vec,
            &covariance_matrix,
        )
        .map_err(PyErr::from)
    }

    /// Compute active risk metrics relative to a benchmark portfolio.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, benchmark_weights, covariance, expected_returns=None))]
    fn active_portfolio_metrics(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        benchmark_weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        expected_returns: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        let weights_vec = weights.as_slice()?.to_vec();
        let benchmark_vec = benchmark_weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;
        let expected_returns_vec = match expected_returns {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };

        let metrics = crate::risk::portfolio_risk::active_portfolio_metrics(
            &weights_vec,
            &benchmark_vec,
            &covariance_matrix,
            expected_returns_vec.as_deref(),
        )
        .map_err(PyErr::from)?;

        active_metrics_to_dict(py, metrics)
    }

    /// Calculate the active share of the portfolio relative to the benchmark.
    #[pyo3(signature = (weights, benchmark_weights))]
    fn portfolio_active_share(
        &self,
        weights: PyReadonlyArray1<f64>,
        benchmark_weights: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        let weights_vec = weights.as_slice()?.to_vec();
        let benchmark_vec = benchmark_weights.as_slice()?.to_vec();
        crate::risk::portfolio_risk::portfolio_active_share(&weights_vec, &benchmark_vec)
            .map_err(PyErr::from)
    }

    /// Compute portfolio beta relative to a benchmark given asset covariances.
    #[pyo3(signature = (weights, asset_benchmark_covariances, benchmark_variance))]
    fn portfolio_beta(
        &self,
        weights: PyReadonlyArray1<f64>,
        asset_benchmark_covariances: PyReadonlyArray1<f64>,
        benchmark_variance: f64,
    ) -> PyResult<f64> {
        let weights_vec = weights.as_slice()?.to_vec();
        let cov_vec = asset_benchmark_covariances.as_slice()?.to_vec();
        crate::risk::portfolio_risk::portfolio_beta(&weights_vec, &cov_vec, benchmark_variance)
            .map_err(PyErr::from)
    }

    /// Compute CAPM alpha, beta and excess return metrics for the portfolio.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, expected_returns, benchmark_return, risk_free_rate, asset_benchmark_covariances, benchmark_variance))]
    fn capm_metrics(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        expected_returns: PyReadonlyArray1<f64>,
        benchmark_return: f64,
        risk_free_rate: f64,
        asset_benchmark_covariances: PyReadonlyArray1<f64>,
        benchmark_variance: f64,
    ) -> PyResult<Py<PyAny>> {
        let weights_vec = weights.as_slice()?.to_vec();
        let expected_returns_vec = expected_returns.as_slice()?.to_vec();
        let cov_vec = asset_benchmark_covariances.as_slice()?.to_vec();

        let metrics = crate::risk::portfolio_risk::capm_metrics(
            &weights_vec,
            &expected_returns_vec,
            benchmark_return,
            risk_free_rate,
            &cov_vec,
            benchmark_variance,
        )
        .map_err(PyErr::from)?;

        capm_metrics_to_dict(py, metrics)
    }

    /// Portfolio Value at Risk using the variance-covariance method.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, covariance, confidence_level=0.95, expected_returns=None))]
    fn portfolio_var_parametric(
        &self,
        weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        confidence_level: f64,
        expected_returns: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<f64> {
        let weights_vec = weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;
        let expected_returns_vec = match expected_returns {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };

        portfolio_parametric_var(
            &weights_vec,
            &covariance_matrix,
            expected_returns_vec.as_deref(),
            confidence_level,
        )
        .map_err(PyErr::from)
    }

    /// Portfolio Value at Risk contributions using the variance-covariance method.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, covariance, confidence_level=0.95, expected_returns=None))]
    fn portfolio_var_contributions_parametric(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        confidence_level: f64,
        expected_returns: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        let weights_vec = weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;
        let expected_returns_vec = match expected_returns {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };

        let contributions = portfolio_parametric_var_contributions(
            &weights_vec,
            &covariance_matrix,
            expected_returns_vec.as_deref(),
            confidence_level,
        )
        .map_err(PyErr::from)?;

        contributions_to_dict(py, contributions)
    }

    /// Portfolio Conditional Value at Risk using the variance-covariance method.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, covariance, confidence_level=0.95, expected_returns=None))]
    fn portfolio_cvar_parametric(
        &self,
        weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        confidence_level: f64,
        expected_returns: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<f64> {
        let weights_vec = weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;
        let expected_returns_vec = match expected_returns {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };

        portfolio_parametric_cvar(
            &weights_vec,
            &covariance_matrix,
            expected_returns_vec.as_deref(),
            confidence_level,
        )
        .map_err(PyErr::from)
    }

    /// Portfolio CVaR contributions using the variance-covariance method.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weights, covariance, confidence_level=0.95, expected_returns=None))]
    fn portfolio_cvar_contributions_parametric(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        confidence_level: f64,
        expected_returns: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        let weights_vec = weights.as_slice()?.to_vec();
        let covariance_matrix = array2_to_dmatrix(covariance)?;
        let expected_returns_vec = match expected_returns {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };

        let contributions = portfolio_parametric_cvar_contributions(
            &weights_vec,
            &covariance_matrix,
            expected_returns_vec.as_deref(),
            confidence_level,
        )
        .map_err(PyErr::from)?;

        contributions_to_dict(py, contributions)
    }

    /// Calculate Sortino ratio
    ///
    /// Sortino ratio is similar to Sharpe ratio but only penalizes downside volatility
    ///
    /// Parameters
    /// ----------
    /// returns : array_like
    ///     Returns data
    /// risk_free_rate : float, optional
    ///     Risk-free rate (default: 0.0)
    /// target_return : float, optional
    ///     Target return for downside calculation (default: 0.0)
    ///
    /// Returns
    /// -------
    /// float
    ///     Sortino ratio
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> rm = RiskMetrics()
    /// >>> returns = np.random.normal(0.001, 0.02, 252)
    /// >>> sortino = rm.sortino_ratio(returns, risk_free_rate=0.0)
    #[pyo3(signature = (returns, risk_free_rate=0.0, target_return=0.0))]
    fn sortino_ratio(
        &self,
        returns: PyReadonlyArray1<f64>,
        risk_free_rate: f64,
        target_return: f64,
    ) -> PyResult<f64> {
        let returns_slice = returns.as_slice()?;

        if returns_slice.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Returns array cannot be empty",
            ));
        }

        // Calculate mean return
        let n = returns_slice.len() as f64;
        let mean_return: f64 = returns_slice.iter().sum::<f64>() / n;

        // Calculate downside deviation
        let downside_variance: f64 = returns_slice
            .iter()
            .map(|&r| {
                let diff = r - target_return;
                if diff < 0.0 { diff * diff } else { 0.0 }
            })
            .sum::<f64>()
            / n;

        let downside_deviation = downside_variance.sqrt();

        if downside_deviation == 0.0 {
            return Ok(f64::INFINITY);
        }

        let sortino = (mean_return - risk_free_rate) / downside_deviation;
        Ok(sortino)
    }

    /// Calculate Calmar ratio
    ///
    /// Calmar ratio is the annualized return divided by maximum drawdown
    ///
    /// Parameters
    /// ----------
    /// returns : array_like
    ///     Returns data
    /// periods_per_year : int, optional
    ///     Number of periods per year for annualization (default: 252 for daily)
    ///
    /// Returns
    /// -------
    /// float
    ///     Calmar ratio
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> rm = RiskMetrics()
    /// >>> returns = np.random.normal(0.001, 0.02, 252)
    /// >>> calmar = rm.calmar_ratio(returns)
    #[pyo3(signature = (returns, periods_per_year=252))]
    fn calmar_ratio(
        &self,
        returns: PyReadonlyArray1<f64>,
        periods_per_year: usize,
    ) -> PyResult<f64> {
        let returns_slice = returns.as_slice()?;

        if returns_slice.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Returns array cannot be empty",
            ));
        }

        // Calculate annualized return
        let n = returns_slice.len() as f64;
        let mean_return: f64 = returns_slice.iter().sum::<f64>() / n;
        let annualized_return = mean_return * periods_per_year as f64;

        // Calculate maximum drawdown
        let max_dd = self.max_drawdown(returns)?;

        if max_dd == 0.0 {
            return Ok(f64::INFINITY);
        }

        let calmar = annualized_return / max_dd;
        Ok(calmar)
    }
}

/// Register risk module with Python
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyGreeksCalculator>()?;
    parent.add_class::<PyRiskMetrics>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_option_type() {
        assert!(matches!(
            parse_option_type("call").unwrap(),
            OptionType::Call
        ));
        assert!(matches!(
            parse_option_type("Call").unwrap(),
            OptionType::Call
        ));
        assert!(matches!(
            parse_option_type("CALL").unwrap(),
            OptionType::Call
        ));
        assert!(matches!(parse_option_type("put").unwrap(), OptionType::Put));
        assert!(matches!(parse_option_type("Put").unwrap(), OptionType::Put));
        assert!(matches!(parse_option_type("PUT").unwrap(), OptionType::Put));
        assert!(parse_option_type("invalid").is_err());
    }
}
