// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for options module

use crate::common::error::DervflowError;
use crate::common::types::{ExerciseStyle, OptionParams, OptionType};
use crate::options::analytical::{black_scholes_greeks, black_scholes_price};
use crate::options::tree::binomial::{BinomialTreeType, binomial_tree_price};
use crate::options::volatility::implied_volatility;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Convert DervflowError to Python exception
fn to_py_err(err: DervflowError) -> PyErr {
    PyValueError::new_err(format!("{}", err))
}

/// Black-Scholes-Merton option pricing model
///
/// This class provides methods for pricing European options and calculating Greeks
/// using the Black-Scholes-Merton analytical formula.
///
/// The model assumes:
/// - The underlying asset follows geometric Brownian motion
/// - No dividends during the option's life (or continuous dividend yield)
/// - European exercise (can only be exercised at maturity)
/// - No transaction costs or taxes
/// - Risk-free rate and volatility are constant
///
/// Examples
/// --------
/// >>> from dervflow.options import BlackScholesModel
/// >>> bs = BlackScholesModel()
/// >>> price = bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
/// >>> print(f"Option price: {price:.2f}")
/// >>> greeks = bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
/// >>> print(f"Delta: {greeks['delta']:.4f}")
#[pyclass(name = "BlackScholesModel")]
pub struct PyBlackScholesModel;

#[pymethods]
impl PyBlackScholesModel {
    /// Create a new BlackScholesModel instance
    #[new]
    fn new() -> Self {
        PyBlackScholesModel
    }

    /// Calculate European option price using Black-Scholes-Merton formula
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    ///
    /// Returns
    /// -------
    /// float
    ///     The theoretical option price
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If input parameters are invalid
    ///
    /// Examples
    /// --------
    /// >>> bs = BlackScholesModel()
    /// >>> price = bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
    /// >>> print(f"Call price: {price:.2f}")
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type))]
    fn price(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
    ) -> PyResult<f64> {
        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);
        black_scholes_price(&params).map_err(to_py_err)
    }

    /// Calculate option Greeks using analytical formulas
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing all Greeks:
    ///     - delta: sensitivity to underlying price (∂V/∂S)
    ///     - gamma: rate of change of delta (∂²V/∂S²)
    ///     - vega: sensitivity to volatility (∂V/∂σ), per 1% change
    ///     - theta: time decay (∂V/∂t), per day
    ///     - rho: sensitivity to interest rate (∂V/∂r), per 1% change
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If input parameters are invalid
    ///
    /// Examples
    /// --------
    /// >>> bs = BlackScholesModel()
    /// >>> greeks = bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
    /// >>> print(f"Delta: {greeks['delta']:.4f}")
    /// >>> print(f"Gamma: {greeks['gamma']:.4f}")
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type))]
    fn greeks(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
    ) -> PyResult<HashMap<String, f64>> {
        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);
        let greeks = black_scholes_greeks(&params).map_err(to_py_err)?;

        let mut result = HashMap::new();
        result.insert("delta".to_string(), greeks.delta);
        result.insert("gamma".to_string(), greeks.gamma);
        result.insert("vega".to_string(), greeks.vega);
        result.insert("theta".to_string(), greeks.theta);
        result.insert("rho".to_string(), greeks.rho);

        Ok(result)
    }

    /// Calculate option prices for multiple options (batch pricing)
    ///
    /// Uses parallel processing with Rayon for improved performance on large batches.
    ///
    /// Parameters
    /// ----------
    /// spots : array_like
    ///     Array of spot prices
    /// strikes : array_like
    ///     Array of strike prices
    /// rates : array_like
    ///     Array of risk-free rates
    /// dividends : array_like
    ///     Array of dividend yields
    /// volatilities : array_like
    ///     Array of volatilities
    /// times : array_like
    ///     Array of times to maturity
    /// option_types : list of str
    ///     List of option types ('call' or 'put')
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     NumPy array of option prices
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If arrays have different lengths or invalid parameters
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> bs = BlackScholesModel()
    /// >>> spots = np.array([100.0, 105.0, 110.0])
    /// >>> strikes = np.array([100.0, 100.0, 100.0])
    /// >>> rates = np.array([0.05, 0.05, 0.05])
    /// >>> dividends = np.array([0.0, 0.0, 0.0])
    /// >>> volatilities = np.array([0.2, 0.2, 0.2])
    /// >>> times = np.array([1.0, 1.0, 1.0])
    /// >>> option_types = ['call', 'call', 'call']
    /// >>> prices = bs.price_batch(spots, strikes, rates, dividends, volatilities, times, option_types)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spots, strikes, rates, dividends, volatilities, times, option_types))]
    fn price_batch<'py>(
        &self,
        py: Python<'py>,
        spots: PyReadonlyArray1<f64>,
        strikes: PyReadonlyArray1<f64>,
        rates: PyReadonlyArray1<f64>,
        dividends: PyReadonlyArray1<f64>,
        volatilities: PyReadonlyArray1<f64>,
        times: PyReadonlyArray1<f64>,
        option_types: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let spots = spots.as_slice()?;
        let strikes = strikes.as_slice()?;
        let rates = rates.as_slice()?;
        let dividends = dividends.as_slice()?;
        let volatilities = volatilities.as_slice()?;
        let times = times.as_slice()?;

        // Check that all arrays have the same length
        let n = spots.len();
        if strikes.len() != n
            || rates.len() != n
            || dividends.len() != n
            || volatilities.len() != n
            || times.len() != n
            || option_types.len() != n
        {
            return Err(PyValueError::new_err(
                "All input arrays must have the same length",
            ));
        }

        // Parse all option types first to catch errors early
        let parsed_types: Result<Vec<OptionType>, PyErr> =
            option_types.iter().map(|s| parse_option_type(s)).collect();
        let parsed_types = parsed_types?;

        // Release GIL for computation
        let prices = py.detach(move || {
            const PARALLEL_THRESHOLD: usize = 4_096;

            if n < PARALLEL_THRESHOLD {
                let mut results = Vec::with_capacity(n);
                for i in 0..n {
                    let params = OptionParams::new(
                        spots[i],
                        strikes[i],
                        rates[i],
                        dividends[i],
                        volatilities[i],
                        times[i],
                        parsed_types[i],
                    );
                    results.push(black_scholes_price(&params)?);
                }
                Ok(results)
            } else {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let params = OptionParams::new(
                            spots[i],
                            strikes[i],
                            rates[i],
                            dividends[i],
                            volatilities[i],
                            times[i],
                            parsed_types[i],
                        );
                        black_scholes_price(&params)
                    })
                    .collect::<Result<Vec<f64>, _>>()
            }
        });

        let prices = prices.map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, prices))
    }

    /// Calculate implied volatility from market price
    ///
    /// Uses Newton-Raphson method with automatic fallback to Brent's method
    /// if convergence fails. This provides a good balance between speed and robustness.
    ///
    /// Parameters
    /// ----------
    /// market_price : float
    ///     Observed market price of the option
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    ///
    /// Returns
    /// -------
    /// float
    ///     The implied volatility
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If input parameters are invalid
    /// RuntimeError
    ///     If convergence fails after 100 iterations
    ///
    /// Examples
    /// --------
    /// >>> bs = BlackScholesModel()
    /// >>> # Calculate implied volatility from market price
    /// >>> iv = bs.implied_vol(10.45, 100.0, 100.0, 0.05, 0.0, 1.0, 'call')
    /// >>> print(f"Implied volatility: {iv:.4f}")
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (market_price, spot, strike, rate, dividend, time, option_type))]
    fn implied_vol(
        &self,
        market_price: f64,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        time: f64,
        option_type: &str,
    ) -> PyResult<f64> {
        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(spot, strike, rate, dividend, 0.0, time, opt_type);
        implied_volatility(market_price, &params).map_err(|e| match e {
            DervflowError::ConvergenceFailure { iterations, error } => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Implied volatility calculation failed to converge after {} iterations (final error: {:.6e}). \
                     This may indicate the market price is inconsistent with the model or the option is too far from at-the-money.",
                    iterations, error
                ))
            }
            _ => to_py_err(e),
        })
    }

    /// Calculate implied volatilities for multiple options (batch calculation)
    ///
    /// Uses parallel processing with Rayon for improved performance on large batches.
    ///
    /// Parameters
    /// ----------
    /// market_prices : array_like
    ///     Array of observed market prices
    /// spots : array_like
    ///     Array of spot prices
    /// strikes : array_like
    ///     Array of strike prices
    /// rates : array_like
    ///     Array of risk-free rates
    /// dividends : array_like
    ///     Array of dividend yields
    /// times : array_like
    ///     Array of times to maturity
    /// option_types : list of str
    ///     List of option types ('call' or 'put')
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     NumPy array of implied volatilities
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If arrays have different lengths or invalid parameters
    /// RuntimeError
    ///     If any implied volatility calculation fails to converge
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> bs = BlackScholesModel()
    /// >>> market_prices = np.array([10.45, 15.23, 20.12])
    /// >>> spots = np.array([100.0, 105.0, 110.0])
    /// >>> strikes = np.array([100.0, 100.0, 100.0])
    /// >>> rates = np.array([0.05, 0.05, 0.05])
    /// >>> dividends = np.array([0.0, 0.0, 0.0])
    /// >>> times = np.array([1.0, 1.0, 1.0])
    /// >>> option_types = ['call', 'call', 'call']
    /// >>> ivs = bs.implied_vol_batch(market_prices, spots, strikes, rates, dividends, times, option_types)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (market_prices, spots, strikes, rates, dividends, times, option_types))]
    fn implied_vol_batch<'py>(
        &self,
        py: Python<'py>,
        market_prices: PyReadonlyArray1<f64>,
        spots: PyReadonlyArray1<f64>,
        strikes: PyReadonlyArray1<f64>,
        rates: PyReadonlyArray1<f64>,
        dividends: PyReadonlyArray1<f64>,
        times: PyReadonlyArray1<f64>,
        option_types: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let market_prices = market_prices.as_array();
        let spots = spots.as_array();
        let strikes = strikes.as_array();
        let rates = rates.as_array();
        let dividends = dividends.as_array();
        let times = times.as_array();

        // Check that all arrays have the same length
        let n = market_prices.len();
        if spots.len() != n
            || strikes.len() != n
            || rates.len() != n
            || dividends.len() != n
            || times.len() != n
            || option_types.len() != n
        {
            return Err(PyValueError::new_err(
                "All input arrays must have the same length",
            ));
        }

        // Parse all option types first to catch errors early
        let parsed_types: Result<Vec<OptionType>, PyErr> =
            option_types.iter().map(|s| parse_option_type(s)).collect();
        let parsed_types = parsed_types?;

        // Convert arrays to owned vectors for use in closure
        let market_prices_vec: Vec<f64> = market_prices.to_vec();
        let spots_vec: Vec<f64> = spots.to_vec();
        let strikes_vec: Vec<f64> = strikes.to_vec();
        let rates_vec: Vec<f64> = rates.to_vec();
        let dividends_vec: Vec<f64> = dividends.to_vec();
        let times_vec: Vec<f64> = times.to_vec();

        // Release GIL for parallel computation
        let ivs = py.detach(move || {
            use rayon::prelude::*;

            // Use parallel iterator for batch IV calculation
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let params = OptionParams::new(
                        spots_vec[i],
                        strikes_vec[i],
                        rates_vec[i],
                        dividends_vec[i],
                        0.0, // volatility will be calculated
                        times_vec[i],
                        parsed_types[i],
                    );
                    implied_volatility(market_prices_vec[i], &params)
                })
                .collect::<Result<Vec<f64>, _>>()
        });

        let ivs = ivs.map_err(|e| match e {
            DervflowError::ConvergenceFailure { iterations, error } => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Batch implied volatility calculation failed: at least one option failed to converge after {} iterations (final error: {:.6e})",
                    iterations, error
                ))
            }
            _ => to_py_err(e),
        })?;

        Ok(PyArray1::from_vec(py, ivs))
    }

    fn __repr__(&self) -> String {
        "BlackScholesModel()".to_string()
    }

    fn __str__(&self) -> String {
        "Black-Scholes-Merton option pricing model".to_string()
    }
}

/// Parse option type string to OptionType enum
fn parse_option_type(s: &str) -> PyResult<OptionType> {
    match s.to_lowercase().as_str() {
        "call" => Ok(OptionType::Call),
        "put" => Ok(OptionType::Put),
        _ => Err(PyValueError::new_err(format!(
            "Invalid option type '{}'. Must be 'call' or 'put'",
            s
        ))),
    }
}

/// Parse exercise style string to ExerciseStyle enum
fn parse_exercise_style(s: &str) -> PyResult<ExerciseStyle> {
    match s.to_lowercase().as_str() {
        "european" => Ok(ExerciseStyle::European),
        "american" => Ok(ExerciseStyle::American),
        _ => Err(PyValueError::new_err(format!(
            "Invalid exercise style '{}'. Must be 'european' or 'american'",
            s
        ))),
    }
}

/// Parse tree type string to BinomialTreeType enum
fn parse_tree_type(s: &str) -> PyResult<BinomialTreeType> {
    match s.to_lowercase().as_str() {
        "crr" | "cox-ross-rubinstein" => Ok(BinomialTreeType::CoxRossRubinstein),
        "jr" | "jarrow-rudd" => Ok(BinomialTreeType::JarrowRudd),
        _ => Err(PyValueError::new_err(format!(
            "Invalid tree type '{}'. Must be 'crr', 'cox-ross-rubinstein', 'jr', or 'jarrow-rudd'",
            s
        ))),
    }
}

/// Binomial tree option pricing model
///
/// This class provides methods for pricing European and American options using
/// binomial tree methods. Two parameterizations are supported:
/// - Cox-Ross-Rubinstein (CRR): Standard recombining tree
/// - Jarrow-Rudd (JR): Matches the drift of the underlying process
///
/// Binomial trees are particularly useful for:
/// - Pricing American options (which can be exercised early)
/// - Understanding option pricing mechanics
/// - Validating other pricing models (converges to Black-Scholes)
///
/// Examples
/// --------
/// >>> from dervflow.options import BinomialTreeModel
/// >>> tree = BinomialTreeModel()
/// >>> # Price European call option
/// >>> price = tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, 'european', 'call', 'crr')
/// >>> print(f"European call price: {price:.2f}")
/// >>> # Price American put option
/// >>> price = tree.price(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 100, 'american', 'put', 'crr')
/// >>> print(f"American put price: {price:.2f}")
#[pyclass(name = "BinomialTreeModel")]
pub struct PyBinomialTreeModel;

#[pymethods]
impl PyBinomialTreeModel {
    /// Create a new BinomialTreeModel instance
    #[new]
    fn new() -> Self {
        PyBinomialTreeModel
    }

    /// Calculate option price using binomial tree method
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// steps : int
    ///     Number of time steps in the binomial tree (more steps = more accurate)
    /// style : str
    ///     Exercise style: 'european' or 'american'
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// tree_type : str, optional
    ///     Tree parameterization: 'crr' (Cox-Ross-Rubinstein) or 'jr' (Jarrow-Rudd)
    ///     Default is 'crr'
    ///
    /// Returns
    /// -------
    /// float
    ///     The option price
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If input parameters are invalid or steps is zero
    ///
    /// Examples
    /// --------
    /// >>> tree = BinomialTreeModel()
    /// >>> # European call with CRR tree
    /// >>> price = tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, 'european', 'call', 'crr')
    /// >>> print(f"Price: {price:.2f}")
    /// >>> # American put with JR tree
    /// >>> price = tree.price(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 100, 'american', 'put', 'jr')
    /// >>> print(f"Price: {price:.2f}")
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, steps, style, option_type, tree_type="crr"))]
    fn price(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        steps: usize,
        style: &str,
        option_type: &str,
        tree_type: &str,
    ) -> PyResult<f64> {
        let opt_type = parse_option_type(option_type)?;
        let exercise_style = parse_exercise_style(style)?;
        let tree_type_enum = parse_tree_type(tree_type)?;

        let params = OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);
        binomial_tree_price(&params, steps, exercise_style, tree_type_enum).map_err(to_py_err)
    }

    /// Calculate option prices for multiple options using binomial trees (batch pricing)
    ///
    /// Uses parallel processing with Rayon for improved performance on large batches.
    ///
    /// Parameters
    /// ----------
    /// spots : array_like
    ///     Array of spot prices
    /// strikes : array_like
    ///     Array of strike prices
    /// rates : array_like
    ///     Array of risk-free rates
    /// dividends : array_like
    ///     Array of dividend yields
    /// volatilities : array_like
    ///     Array of volatilities
    /// times : array_like
    ///     Array of times to maturity
    /// steps : int
    ///     Number of time steps in the binomial tree (same for all options)
    /// styles : list of str
    ///     List of exercise styles ('european' or 'american')
    /// option_types : list of str
    ///     List of option types ('call' or 'put')
    /// tree_type : str, optional
    ///     Tree parameterization: 'crr' or 'jr'. Default is 'crr'
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     NumPy array of option prices
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If arrays have different lengths or invalid parameters
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> tree = BinomialTreeModel()
    /// >>> spots = np.array([100.0, 105.0, 110.0])
    /// >>> strikes = np.array([100.0, 100.0, 100.0])
    /// >>> rates = np.array([0.05, 0.05, 0.05])
    /// >>> dividends = np.array([0.0, 0.0, 0.0])
    /// >>> volatilities = np.array([0.2, 0.2, 0.2])
    /// >>> times = np.array([1.0, 1.0, 1.0])
    /// >>> styles = ['american', 'american', 'american']
    /// >>> option_types = ['put', 'put', 'put']
    /// >>> prices = tree.price_batch(spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spots, strikes, rates, dividends, volatilities, times, steps, styles, option_types, tree_type="crr"))]
    fn price_batch<'py>(
        &self,
        py: Python<'py>,
        spots: PyReadonlyArray1<f64>,
        strikes: PyReadonlyArray1<f64>,
        rates: PyReadonlyArray1<f64>,
        dividends: PyReadonlyArray1<f64>,
        volatilities: PyReadonlyArray1<f64>,
        times: PyReadonlyArray1<f64>,
        steps: usize,
        styles: Vec<String>,
        option_types: Vec<String>,
        tree_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let spots = spots.as_array();
        let strikes = strikes.as_array();
        let rates = rates.as_array();
        let dividends = dividends.as_array();
        let volatilities = volatilities.as_array();
        let times = times.as_array();

        // Check that all arrays have the same length
        let n = spots.len();
        if strikes.len() != n
            || rates.len() != n
            || dividends.len() != n
            || volatilities.len() != n
            || times.len() != n
            || styles.len() != n
            || option_types.len() != n
        {
            return Err(PyValueError::new_err(
                "All input arrays must have the same length",
            ));
        }

        // Parse tree type once
        let tree_type_enum = parse_tree_type(tree_type)?;

        // Parse all option types and exercise styles first to catch errors early
        let parsed_types: Result<Vec<OptionType>, PyErr> =
            option_types.iter().map(|s| parse_option_type(s)).collect();
        let parsed_types = parsed_types?;

        let parsed_styles: Result<Vec<ExerciseStyle>, PyErr> =
            styles.iter().map(|s| parse_exercise_style(s)).collect();
        let parsed_styles = parsed_styles?;

        // Convert arrays to owned vectors for use in closure
        let spots_vec: Vec<f64> = spots.to_vec();
        let strikes_vec: Vec<f64> = strikes.to_vec();
        let rates_vec: Vec<f64> = rates.to_vec();
        let dividends_vec: Vec<f64> = dividends.to_vec();
        let volatilities_vec: Vec<f64> = volatilities.to_vec();
        let times_vec: Vec<f64> = times.to_vec();

        // Release GIL for parallel computation
        let prices = py.detach(move || {
            use rayon::prelude::*;

            // Use parallel iterator for batch pricing
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let params = OptionParams::new(
                        spots_vec[i],
                        strikes_vec[i],
                        rates_vec[i],
                        dividends_vec[i],
                        volatilities_vec[i],
                        times_vec[i],
                        parsed_types[i],
                    );
                    binomial_tree_price(&params, steps, parsed_styles[i], tree_type_enum)
                })
                .collect::<Result<Vec<f64>, _>>()
        });

        let prices = prices.map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, prices))
    }

    fn __repr__(&self) -> String {
        "BinomialTreeModel()".to_string()
    }

    fn __str__(&self) -> String {
        "Binomial tree option pricing model (CRR and JR)".to_string()
    }
}

/// Register options module with Python
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyBlackScholesModel>()?;
    parent.add_class::<PyBinomialTreeModel>()?;
    parent.add_class::<PyMonteCarloOptionPricer>()?;
    parent.add_class::<PyAsianOption>()?;
    parent.add_class::<PyBarrierOption>()?;
    parent.add_class::<PyLookbackOption>()?;
    parent.add_class::<PyDigitalOption>()?;
    parent.add_class::<PyVolatilitySurface>()?;
    parent.add_class::<PySABRModel>()?;
    Ok(())
}

/// Monte Carlo option pricing model
///
/// This class provides methods for pricing European and American options using
/// Monte Carlo simulation. Supports:
/// - European options with antithetic variates variance reduction
/// - American options using Longstaff-Schwartz algorithm
/// - Parallel path generation for improved performance
///
/// Monte Carlo methods are particularly useful for:
/// - Pricing path-dependent options
/// - Handling complex payoff structures
/// - American options (using Longstaff-Schwartz)
/// - Options where analytical solutions don't exist
///
/// Examples
/// --------
/// >>> from dervflow.options import MonteCarloOptionPricer
/// >>> mc = MonteCarloOptionPricer()
/// >>> # Price European call option
/// >>> result = mc.price_european(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call',
/// ...                            num_paths=10000, use_antithetic=True, seed=42)
/// >>> print(f"Price: {result['price']:.2f} ± {result['std_error']:.4f}")
/// >>> # Price American put option
/// >>> result = mc.price_american(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 'put',
/// ...                            num_paths=10000, num_steps=50, seed=42)
/// >>> print(f"Price: {result['price']:.2f} ± {result['std_error']:.4f}")
#[pyclass(name = "MonteCarloOptionPricer")]
pub struct PyMonteCarloOptionPricer;

#[pymethods]
impl PyMonteCarloOptionPricer {
    /// Create a new MonteCarloOptionPricer instance
    #[new]
    fn new() -> Self {
        PyMonteCarloOptionPricer
    }

    /// Price European option using Monte Carlo simulation
    ///
    /// Uses geometric Brownian motion to simulate price paths and calculates
    /// the discounted expected payoff. Supports antithetic variates for variance reduction.
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// num_paths : int, optional
    ///     Number of simulation paths. Default is 10000
    /// use_antithetic : bool, optional
    ///     Whether to use antithetic variates variance reduction. Default is True
    /// seed : int, optional
    ///     Random seed for reproducibility. Default is None (random seed)
    /// parallel : bool, optional
    ///     Whether to use parallel processing. Default is True
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - price: Estimated option price
    ///     - std_error: Standard error of the estimate
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If input parameters are invalid
    ///
    /// Examples
    /// --------
    /// >>> mc = MonteCarloOptionPricer()
    /// >>> result = mc.price_european(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call',
    /// ...                            num_paths=50000, use_antithetic=True, seed=42)
    /// >>> print(f"Call price: {result['price']:.2f}")
    /// >>> print(f"Standard error: {result['std_error']:.4f}")
    /// >>> print(f"95% CI: [{result['price'] - 1.96*result['std_error']:.2f}, "
    /// ...       f"{result['price'] + 1.96*result['std_error']:.2f}]")
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, num_paths=10000, use_antithetic=true, seed=None, parallel=true))]
    fn price_european(
        &self,
        py: Python<'_>,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        num_paths: usize,
        use_antithetic: bool,
        seed: Option<u64>,
        parallel: bool,
    ) -> PyResult<HashMap<String, f64>> {
        use crate::options::monte_carlo::{
            price_european_monte_carlo, price_european_monte_carlo_parallel,
        };

        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);

        // Release GIL for computation
        let result = py.detach(|| {
            if parallel {
                price_european_monte_carlo_parallel(&params, num_paths, use_antithetic, seed)
            } else {
                price_european_monte_carlo(&params, num_paths, use_antithetic, seed)
            }
        });

        let result = result.map_err(to_py_err)?;

        let mut output = HashMap::new();
        output.insert("price".to_string(), result.price);
        output.insert("std_error".to_string(), result.standard_error);

        Ok(output)
    }

    /// Price American option using Longstaff-Schwartz Monte Carlo algorithm
    ///
    /// Implements the Longstaff-Schwartz least squares Monte Carlo method for
    /// American option pricing. Uses backward induction with regression to
    /// estimate continuation values and determine optimal exercise strategy.
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// num_paths : int, optional
    ///     Number of simulation paths. Default is 10000
    /// num_steps : int, optional
    ///     Number of time steps per path. Default is 50
    /// seed : int, optional
    ///     Random seed for reproducibility. Default is None (random seed)
    /// parallel : bool, optional
    ///     Whether to use parallel processing. Default is True
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - price: Estimated option price
    ///     - std_error: Standard error of the estimate
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If input parameters are invalid
    ///
    /// Examples
    /// --------
    /// >>> mc = MonteCarloOptionPricer()
    /// >>> # American put option (early exercise premium)
    /// >>> result = mc.price_american(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 'put',
    /// ...                            num_paths=10000, num_steps=50, seed=42)
    /// >>> print(f"American put price: {result['price']:.2f}")
    /// >>> # Compare with European put
    /// >>> euro_result = mc.price_european(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 'put', seed=42)
    /// >>> print(f"Early exercise premium: {result['price'] - euro_result['price']:.2f}")
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, num_paths=10000, num_steps=50, seed=None, parallel=true))]
    fn price_american(
        &self,
        py: Python<'_>,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        num_paths: usize,
        num_steps: usize,
        seed: Option<u64>,
        parallel: bool,
    ) -> PyResult<HashMap<String, f64>> {
        use crate::options::monte_carlo::{
            price_american_monte_carlo, price_american_monte_carlo_parallel,
        };

        let opt_type = parse_option_type(option_type)?;
        let params = OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);

        // Release GIL for computation
        let result = py.detach(|| {
            if parallel {
                price_american_monte_carlo_parallel(&params, num_paths, num_steps, seed)
            } else {
                price_american_monte_carlo(&params, num_paths, num_steps, seed)
            }
        });

        let result = result.map_err(to_py_err)?;

        let mut output = HashMap::new();
        output.insert("price".to_string(), result.price);
        output.insert("std_error".to_string(), result.standard_error);

        Ok(output)
    }

    fn __repr__(&self) -> String {
        "MonteCarloOptionPricer()".to_string()
    }

    fn __str__(&self) -> String {
        "Monte Carlo option pricing with variance reduction and parallel processing".to_string()
    }
}

// ============================================================================
// Exotic Options Bindings
// ============================================================================

/// Asian option pricing model
///
/// This class provides methods for pricing Asian options, which are path-dependent
/// options where the payoff depends on the average price of the underlying asset
/// over a specified period.
///
/// Supports:
/// - Arithmetic average Asian options (Monte Carlo)
/// - Geometric average Asian options (analytical approximation)
/// - Fixed strike and floating strike variants
///
/// Examples
/// --------
/// >>> from dervflow.options import AsianOption
/// >>> asian = AsianOption()
/// >>> # Price arithmetic average Asian call with fixed strike
/// >>> price = asian.price_arithmetic(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call',
/// ...                                num_observations=12, fixed_strike=True,
/// ...                                num_paths=10000, seed=42)
/// >>> print(f"Asian call price: {price:.2f}")
#[pyclass(name = "AsianOption")]
pub struct PyAsianOption;

#[pymethods]
impl PyAsianOption {
    /// Create a new AsianOption instance
    #[new]
    fn new() -> Self {
        PyAsianOption
    }

    /// Price arithmetic average Asian option using Monte Carlo simulation
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// num_observations : int
    ///     Number of averaging observations
    /// fixed_strike : bool
    ///     Whether to use fixed strike (True) or floating strike (False)
    /// num_paths : int, optional
    ///     Number of simulation paths. Default is 10000
    /// seed : int, optional
    ///     Random seed for reproducibility. Default is None
    /// use_antithetic : bool, optional
    ///     Whether to use antithetic variates for variance reduction. Default is False
    /// use_control_variate : bool, optional
    ///     Apply the geometric control variate (fixed-strike options only). Default is False
    ///
    /// Returns
    /// -------
    /// float
    ///     The option price
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, num_observations, fixed_strike, num_paths=10000, seed=None, use_antithetic=false, use_control_variate=false))]
    fn price_arithmetic(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        num_observations: usize,
        fixed_strike: bool,
        num_paths: usize,
        seed: Option<u64>,
        use_antithetic: bool,
        use_control_variate: bool,
    ) -> PyResult<f64> {
        use crate::options::exotic::{
            AsianMonteCarloConfig, AsianOptionParams, price_asian_arithmetic_mc_stats,
        };

        let opt_type = parse_option_type(option_type)?;
        let base_params =
            OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);
        let params = AsianOptionParams::new(base_params, num_observations, fixed_strike);

        let config = AsianMonteCarloConfig {
            num_paths,
            seed,
            use_antithetic,
            use_control_variate,
        };

        price_asian_arithmetic_mc_stats(&params, &config)
            .map(|res| res.price)
            .map_err(to_py_err)
    }

    /// Detailed Monte Carlo pricing for arithmetic Asian options.
    ///
    /// Returns both the price estimate and the Monte Carlo standard error,
    /// exposing the same variance reduction controls as :meth:`price_arithmetic`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, num_observations, fixed_strike, num_paths=10000, seed=None, use_antithetic=true, use_control_variate=false))]
    fn price_arithmetic_stats(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        num_observations: usize,
        fixed_strike: bool,
        num_paths: usize,
        seed: Option<u64>,
        use_antithetic: bool,
        use_control_variate: bool,
    ) -> PyResult<HashMap<String, f64>> {
        use crate::options::exotic::{
            AsianMonteCarloConfig, AsianOptionParams, price_asian_arithmetic_mc_stats,
        };

        let opt_type = parse_option_type(option_type)?;
        let base_params =
            OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);
        let params = AsianOptionParams::new(base_params, num_observations, fixed_strike);

        let config = AsianMonteCarloConfig {
            num_paths,
            seed,
            use_antithetic,
            use_control_variate,
        };

        let result = price_asian_arithmetic_mc_stats(&params, &config).map_err(to_py_err)?;

        let mut output = HashMap::new();
        output.insert("price".to_string(), result.price);
        output.insert("std_error".to_string(), result.standard_error);

        Ok(output)
    }

    /// Price geometric average Asian option using analytical approximation
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// num_observations : int
    ///     Number of averaging observations
    /// fixed_strike : bool
    ///     Whether to use fixed strike (True) or floating strike (False)
    ///
    /// Returns
    /// -------
    /// float
    ///     The option price
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, num_observations, fixed_strike))]
    fn price_geometric(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        num_observations: usize,
        fixed_strike: bool,
    ) -> PyResult<f64> {
        use crate::options::exotic::{AsianOptionParams, price_asian_geometric};

        let opt_type = parse_option_type(option_type)?;
        let base_params =
            OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);
        let params = AsianOptionParams::new(base_params, num_observations, fixed_strike);

        price_asian_geometric(&params).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        "AsianOption()".to_string()
    }

    fn __str__(&self) -> String {
        "Asian option pricing (arithmetic and geometric average)".to_string()
    }
}

/// Barrier option pricing model
///
/// This class provides methods for pricing barrier options, which are options
/// that are activated or deactivated when the underlying asset price crosses
/// a specified barrier level.
///
/// Supports:
/// - Up-and-out, down-and-out (knock-out options)
/// - Up-and-in, down-and-in (knock-in options)
/// - Rebate payments
///
/// Examples
/// --------
/// >>> from dervflow.options import BarrierOption
/// >>> barrier = BarrierOption()
/// >>> # Price down-and-out call option
/// >>> price = barrier.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call',
/// ...                       barrier=90.0, barrier_type='down-and-out', rebate=0.0)
/// >>> print(f"Down-and-out call price: {price:.2f}")
#[pyclass(name = "BarrierOption")]
pub struct PyBarrierOption;

#[pymethods]
impl PyBarrierOption {
    /// Create a new BarrierOption instance
    #[new]
    fn new() -> Self {
        PyBarrierOption
    }

    /// Price barrier option using analytical formulas
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// barrier : float
    ///     Barrier level
    /// barrier_type : str
    ///     Barrier type: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
    /// rebate : float, optional
    ///     Rebate paid if barrier is hit (for knock-out) or not hit (for knock-in). Default is 0.0
    ///
    /// Returns
    /// -------
    /// float
    ///     The option price
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, barrier, barrier_type, rebate=0.0))]
    fn price(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        barrier: f64,
        barrier_type: &str,
        rebate: f64,
    ) -> PyResult<f64> {
        use crate::options::exotic::{BarrierOptionParams, BarrierType, price_barrier};

        let opt_type = parse_option_type(option_type)?;
        let base_params =
            OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);

        let barrier_type_enum = match barrier_type.to_lowercase().as_str() {
            "up-and-out" | "upandout" => BarrierType::UpAndOut,
            "down-and-out" | "downandout" => BarrierType::DownAndOut,
            "up-and-in" | "upandin" => BarrierType::UpAndIn,
            "down-and-in" | "downandin" => BarrierType::DownAndIn,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid barrier type '{}'. Must be 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'",
                    barrier_type
                )));
            }
        };

        let params = BarrierOptionParams::new(base_params, barrier, barrier_type_enum, rebate);
        price_barrier(&params).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        "BarrierOption()".to_string()
    }

    fn __str__(&self) -> String {
        "Barrier option pricing (knock-in and knock-out)".to_string()
    }
}

/// Lookback option pricing model
///
/// This class provides methods for pricing lookback options, which are path-dependent
/// options where the payoff depends on the maximum or minimum price of the underlying
/// asset over the option's life.
///
/// Supports:
/// - Fixed strike lookback options
/// - Floating strike lookback options
///
/// Examples
/// --------
/// >>> from dervflow.options import LookbackOption
/// >>> lookback = LookbackOption()
/// >>> # Price floating strike lookback call
/// >>> price = lookback.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call',
/// ...                        lookback_type='floating', current_extremum=95.0)
/// >>> print(f"Lookback call price: {price:.2f}")
#[pyclass(name = "LookbackOption")]
pub struct PyLookbackOption;

#[pymethods]
impl PyLookbackOption {
    /// Create a new LookbackOption instance
    #[new]
    fn new() -> Self {
        PyLookbackOption
    }

    /// Price lookback option using analytical formulas
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option (for fixed strike lookback)
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// lookback_type : str
    ///     Lookback type: 'fixed' or 'floating'
    /// current_extremum : float, optional
    ///     Current minimum (for floating strike calls) or maximum (for floating strike puts).
    ///     Default is None (uses spot price)
    ///
    /// Returns
    /// -------
    /// float
    ///     The option price
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, lookback_type, current_extremum=None))]
    fn price(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        lookback_type: &str,
        current_extremum: Option<f64>,
    ) -> PyResult<f64> {
        use crate::options::exotic::{LookbackOptionParams, LookbackType, price_lookback};

        let opt_type = parse_option_type(option_type)?;
        let base_params =
            OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);

        let lookback_type_enum = match lookback_type.to_lowercase().as_str() {
            "fixed" | "fixed-strike" => LookbackType::FixedStrike,
            "floating" | "floating-strike" => LookbackType::FloatingStrike,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid lookback type '{}'. Must be 'fixed' or 'floating'",
                    lookback_type
                )));
            }
        };

        let params = LookbackOptionParams::new(base_params, lookback_type_enum, current_extremum);
        price_lookback(&params).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        "LookbackOption()".to_string()
    }

    fn __str__(&self) -> String {
        "Lookback option pricing (fixed and floating strike)".to_string()
    }
}

/// Digital/Binary option pricing model
///
/// This class provides methods for pricing digital (binary) options, which pay
/// a fixed amount if the option ends in the money, or nothing otherwise.
///
/// Supports:
/// - Cash-or-nothing options (pays fixed cash amount)
/// - Asset-or-nothing options (pays asset value)
///
/// Examples
/// --------
/// >>> from dervflow.options import DigitalOption
/// >>> digital = DigitalOption()
/// >>> # Price cash-or-nothing call option
/// >>> price = digital.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call',
/// ...                       digital_type='cash-or-nothing', cash_payout=10.0)
/// >>> print(f"Digital call price: {price:.2f}")
#[pyclass(name = "DigitalOption")]
pub struct PyDigitalOption;

#[pymethods]
impl PyDigitalOption {
    /// Create a new DigitalOption instance
    #[new]
    fn new() -> Self {
        PyDigitalOption
    }

    /// Price digital/binary option using analytical formulas
    ///
    /// Parameters
    /// ----------
    /// spot : float
    ///     Current spot price of the underlying asset
    /// strike : float
    ///     Strike price of the option
    /// rate : float
    ///     Risk-free interest rate (annualized)
    /// dividend : float
    ///     Dividend yield (annualized, continuous)
    /// volatility : float
    ///     Volatility of the underlying asset (annualized)
    /// time : float
    ///     Time to maturity (in years)
    /// option_type : str
    ///     Option type: 'call' or 'put'
    /// digital_type : str
    ///     Digital type: 'cash-or-nothing' or 'asset-or-nothing'
    /// cash_payout : float, optional
    ///     Cash payout for cash-or-nothing options. Default is 1.0
    ///
    /// Returns
    /// -------
    /// float
    ///     The option price
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, rate, dividend, volatility, time, option_type, digital_type, cash_payout=1.0))]
    fn price(
        &self,
        spot: f64,
        strike: f64,
        rate: f64,
        dividend: f64,
        volatility: f64,
        time: f64,
        option_type: &str,
        digital_type: &str,
        cash_payout: f64,
    ) -> PyResult<f64> {
        use crate::options::exotic::{DigitalOptionParams, DigitalType, price_digital};

        let opt_type = parse_option_type(option_type)?;
        let base_params =
            OptionParams::new(spot, strike, rate, dividend, volatility, time, opt_type);

        let digital_type_enum = match digital_type.to_lowercase().as_str() {
            "cash-or-nothing" | "cashornothing" | "cash" => DigitalType::CashOrNothing,
            "asset-or-nothing" | "assetornothing" | "asset" => DigitalType::AssetOrNothing,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid digital type '{}'. Must be 'cash-or-nothing' or 'asset-or-nothing'",
                    digital_type
                )));
            }
        };

        let params = DigitalOptionParams::new(base_params, digital_type_enum, cash_payout);
        price_digital(&params).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        "DigitalOption()".to_string()
    }

    fn __str__(&self) -> String {
        "Digital/Binary option pricing (cash-or-nothing and asset-or-nothing)".to_string()
    }
}

// ============================================================================
// Volatility Surface Bindings
// ============================================================================

/// Volatility surface for storing and interpolating implied volatilities
///
/// The surface stores implied volatilities as a function of strike and maturity.
/// It supports multiple interpolation methods for querying volatilities at
/// arbitrary (strike, maturity) points.
///
/// Examples
/// --------
/// >>> from dervflow.options import VolatilitySurface
/// >>> import numpy as np
/// >>> strikes = [90.0, 100.0, 110.0]
/// >>> maturities = [0.25, 0.5, 1.0]
/// >>> volatilities = [[0.25, 0.23, 0.21],
/// ...                 [0.20, 0.19, 0.18],
/// ...                 [0.22, 0.21, 0.20]]
/// >>> surface = VolatilitySurface(strikes, maturities, volatilities,
/// ...                             method='bilinear', spot=100.0, rate=0.05)
/// >>> vol = surface.implied_volatility(95.0, 0.375)
/// >>> print(f"Implied volatility: {vol:.4f}")
#[pyclass(name = "VolatilitySurface")]
pub struct PyVolatilitySurface {
    inner: crate::options::volatility::VolatilitySurface,
}

#[pymethods]
impl PyVolatilitySurface {
    /// Create a new volatility surface
    ///
    /// Parameters
    /// ----------
    /// strikes : list of float
    ///     Strike prices (will be sorted automatically)
    /// maturities : list of float
    ///     Times to maturity in years (will be sorted automatically)
    /// volatilities : list of list of float
    ///     2D grid of implied volatilities [strike_idx][maturity_idx]
    /// method : str, optional
    ///     Interpolation method: 'bilinear' or 'cubic_spline'. Default is 'bilinear'
    /// spot : float
    ///     Current spot price of the underlying
    /// rate : float
    ///     Risk-free interest rate
    ///
    /// Returns
    /// -------
    /// VolatilitySurface
    ///     The constructed volatility surface
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If inputs are invalid or dimensions don't match
    #[new]
    #[pyo3(signature = (strikes, maturities, volatilities, method="bilinear", spot=100.0, rate=0.05))]
    fn new(
        strikes: Vec<f64>,
        maturities: Vec<f64>,
        volatilities: Vec<Vec<f64>>,
        method: &str,
        spot: f64,
        rate: f64,
    ) -> PyResult<Self> {
        use crate::options::volatility::{InterpolationMethod, VolatilitySurface};

        let interpolation_method = match method.to_lowercase().as_str() {
            "bilinear" => InterpolationMethod::Bilinear,
            "cubic_spline" | "cubicspline" => InterpolationMethod::CubicSpline,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid interpolation method '{}'. Must be 'bilinear' or 'cubic_spline'",
                    method
                )));
            }
        };

        let surface = VolatilitySurface::new(
            strikes,
            maturities,
            volatilities,
            interpolation_method,
            spot,
            rate,
        )
        .map_err(to_py_err)?;

        Ok(Self { inner: surface })
    }

    /// Get the implied volatility at a specific strike and maturity
    ///
    /// Parameters
    /// ----------
    /// strike : float
    ///     Strike price
    /// maturity : float
    ///     Time to maturity in years
    ///
    /// Returns
    /// -------
    /// float
    ///     The interpolated implied volatility
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If point is outside the surface bounds
    fn implied_volatility(&self, strike: f64, maturity: f64) -> PyResult<f64> {
        self.inner
            .implied_volatility(strike, maturity)
            .map_err(to_py_err)
    }

    /// Get the strikes in the surface
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Sorted strike prices
    fn strikes(&self) -> Vec<f64> {
        self.inner.strikes().to_vec()
    }

    /// Get the maturities in the surface
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Sorted maturities in years
    fn maturities(&self) -> Vec<f64> {
        self.inner.maturities().to_vec()
    }

    /// Get the volatility grid
    ///
    /// Returns
    /// -------
    /// list of list of float
    ///     2D grid of implied volatilities
    fn volatilities(&self) -> Vec<Vec<f64>> {
        self.inner.volatilities().to_vec()
    }

    /// Get the spot price
    ///
    /// Returns
    /// -------
    /// float
    ///     Spot price of the underlying
    fn spot(&self) -> f64 {
        self.inner.spot()
    }

    /// Get the risk-free rate
    ///
    /// Returns
    /// -------
    /// float
    ///     Risk-free interest rate
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    fn __repr__(&self) -> String {
        format!(
            "VolatilitySurface(strikes={}, maturities={}, method={:?})",
            self.inner.strikes().len(),
            self.inner.maturities().len(),
            self.inner.interpolation_method()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Volatility surface with {} strikes and {} maturities",
            self.inner.strikes().len(),
            self.inner.maturities().len()
        )
    }
}

/// SABR (Stochastic Alpha Beta Rho) volatility model
///
/// The SABR model is a stochastic volatility model used to model the evolution
/// of forward rates and their implied volatilities. It's widely used in interest
/// rate derivatives markets.
///
/// Examples
/// --------
/// >>> from dervflow.options import SABRModel
/// >>> sabr = SABRModel(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
/// >>> vol = sabr.implied_volatility(forward=100.0, strike=105.0, maturity=1.0)
/// >>> print(f"SABR implied volatility: {vol:.4f}")
#[pyclass(name = "SABRModel")]
pub struct PySABRModel {
    inner: crate::options::volatility::SABRParams,
}

#[pymethods]
impl PySABRModel {
    /// Create a new SABR model with specified parameters
    ///
    /// Parameters
    /// ----------
    /// alpha : float
    ///     Initial volatility (must be positive)
    /// beta : float
    ///     Elasticity parameter (0 for normal, 1 for lognormal, typically 0-1)
    /// rho : float
    ///     Correlation between forward and volatility (-1 to 1)
    /// nu : float
    ///     Volatility of volatility (must be non-negative)
    ///
    /// Returns
    /// -------
    /// SABRModel
    ///     The SABR model instance
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If parameters are out of valid ranges
    #[new]
    fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> PyResult<Self> {
        use crate::options::volatility::SABRParams;

        let params = SABRParams::new(alpha, beta, rho, nu).map_err(to_py_err)?;
        Ok(Self { inner: params })
    }

    /// Calculate implied volatility using SABR formula
    ///
    /// Parameters
    /// ----------
    /// forward : float
    ///     Forward price
    /// strike : float
    ///     Strike price
    /// maturity : float
    ///     Time to maturity in years
    ///
    /// Returns
    /// -------
    /// float
    ///     Implied volatility
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If calculation fails or inputs are invalid
    fn implied_volatility(&self, forward: f64, strike: f64, maturity: f64) -> PyResult<f64> {
        self.inner
            .implied_volatility(forward, strike, maturity)
            .map_err(to_py_err)
    }

    /// Calibrate SABR model to market volatility data
    ///
    /// Calibrates the SABR parameters (alpha, rho, nu) to match market implied volatilities.
    /// Beta is kept fixed at the value specified in the model.
    ///
    /// Parameters
    /// ----------
    /// forward : float
    ///     Forward price
    /// maturity : float
    ///     Time to maturity
    /// strikes : list of float
    ///     Strike prices
    /// market_vols : list of float
    ///     Market implied volatilities
    ///
    /// Returns
    /// -------
    /// SABRModel
    ///     New calibrated SABR model
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If calibration fails or inputs are invalid
    #[staticmethod]
    fn calibrate(
        py: Python<'_>,
        forward: f64,
        maturity: f64,
        strikes: Vec<f64>,
        market_vols: Vec<f64>,
        beta: f64,
    ) -> PyResult<Self> {
        use crate::options::volatility::calibrate_sabr;

        // Release GIL for optimization
        let params = py.detach(|| calibrate_sabr(forward, maturity, &strikes, &market_vols, beta));

        let params = params.map_err(to_py_err)?;
        Ok(Self { inner: params })
    }

    /// Get alpha parameter
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Get beta parameter
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    /// Get rho parameter
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    /// Get nu parameter
    #[getter]
    fn nu(&self) -> f64 {
        self.inner.nu
    }

    fn __repr__(&self) -> String {
        format!(
            "SABRModel(alpha={}, beta={}, rho={}, nu={})",
            self.inner.alpha, self.inner.beta, self.inner.rho, self.inner.nu
        )
    }

    fn __str__(&self) -> String {
        format!(
            "SABR model: alpha={}, beta={}, rho={}, nu={}",
            self.inner.alpha, self.inner.beta, self.inner.rho, self.inner.nu
        )
    }
}
