// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for yield curve module

use crate::common::error::DervflowError;
use crate::common::types::BondQuote;
use crate::yield_curve::{
    YieldCurve,
    bond_analytics::{
        Cashflow, bond_price, convexity, dv01, generate_bond_cashflows, macaulay_duration,
        modified_duration, yield_to_maturity,
    },
    bootstrap::{SwapQuote, bootstrap_from_bonds, bootstrap_from_swaps},
    interpolation::{InterpolationMethod, NelsonSiegelParams, NelsonSiegelSvenssonParams},
    multi_curve::{MultiCurve as CoreMultiCurve, SwapPeriod as CoreSwapPeriod},
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::PyRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyIterator;

/// Convert DervflowError to Python exception
fn to_py_err(err: DervflowError) -> PyErr {
    PyErr::from(err)
}

/// Parse interpolation method string
fn parse_interpolation_method(method: &str) -> PyResult<InterpolationMethod> {
    match method.to_lowercase().as_str() {
        "linear" => Ok(InterpolationMethod::Linear),
        "cubic_spline_natural" | "natural" => Ok(InterpolationMethod::CubicSplineNatural),
        "cubic_spline_clamped" | "clamped" => Ok(InterpolationMethod::CubicSplineClamped),
        "nelson_siegel" => Ok(InterpolationMethod::NelsonSiegel),
        "nelson_siegel_svensson" => Ok(InterpolationMethod::NelsonSiegelSvensson),
        _ => Err(PyValueError::new_err(format!(
            "Invalid interpolation method: {}. Valid options: 'linear', 'cubic_spline_natural', 'cubic_spline_clamped', 'nelson_siegel', 'nelson_siegel_svensson'",
            method
        ))),
    }
}

fn parse_swap_periods(periods: &Bound<'_, PyAny>) -> PyResult<Vec<CoreSwapPeriod>> {
    let iterator = PyIterator::from_object(periods).map_err(|_| {
        PyValueError::new_err(
            "Swap periods must be a sequence of SwapPeriod instances or (start, end, year_fraction) tuples",
        )
    })?;

    let mut result = Vec::new();
    for item in iterator {
        let obj = item?;

        if let Ok(period) = obj.extract::<PyRef<PySwapPeriod>>() {
            result.push(period.period);
            continue;
        }

        if let Ok((start, end, year_fraction)) = obj.extract::<(f64, f64, f64)>() {
            let period = CoreSwapPeriod::new(start, end, year_fraction).map_err(to_py_err)?;
            result.push(period);
            continue;
        }

        return Err(PyValueError::new_err(
            "Each swap period must be a SwapPeriod or tuple (start, end, year_fraction)",
        ));
    }

    Ok(result)
}

/// Yield curve for interest rate modeling
///
/// Represents a yield curve with various interpolation methods for calculating
/// zero rates, forward rates, and discount factors at any maturity.
///
/// Examples
/// --------
/// >>> from dervflow.yield_curve import YieldCurve
/// >>> times = [1.0, 2.0, 5.0, 10.0]
/// >>> rates = [0.03, 0.035, 0.04, 0.045]
/// >>> curve = YieldCurve(times, rates, method='linear')
/// >>> rate = curve.zero_rate(3.0)
/// >>> df = curve.discount_factor(3.0)
#[pyclass(name = "YieldCurve")]
pub struct PyYieldCurve {
    curve: YieldCurve,
}

#[pymethods]
impl PyYieldCurve {
    /// Create a new yield curve
    ///
    /// Parameters
    /// ----------
    /// times : array-like
    ///     Time points (maturities in years), must be sorted
    /// rates : array-like
    ///     Zero rates at each time point (annualized)
    /// method : str, optional
    ///     Interpolation method: 'linear', 'cubic_spline_natural', 'cubic_spline_clamped',
    ///     'nelson_siegel', 'nelson_siegel_svensson'. Default is 'linear'.
    ///
    /// Returns
    /// -------
    /// YieldCurve
    ///     A new yield curve instance
    #[new]
    #[pyo3(signature = (times, rates, method="linear"))]
    fn new(
        times: PyReadonlyArray1<f64>,
        rates: PyReadonlyArray1<f64>,
        method: &str,
    ) -> PyResult<Self> {
        let times_vec = times.as_slice()?.to_vec();
        let rates_vec = rates.as_slice()?.to_vec();
        let interp_method = parse_interpolation_method(method)?;

        let curve = YieldCurve::new(times_vec, rates_vec, interp_method).map_err(to_py_err)?;

        Ok(Self { curve })
    }

    /// Get zero rate at a specific maturity
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Time (maturity in years)
    ///
    /// Returns
    /// -------
    /// float
    ///     Zero rate at the specified maturity
    fn zero_rate(&self, time: f64) -> PyResult<f64> {
        self.curve.zero_rate(time).map_err(to_py_err)
    }

    /// Calculate forward rate between two times
    ///
    /// Parameters
    /// ----------
    /// t1 : float
    ///     Start time (years)
    /// t2 : float
    ///     End time (years)
    ///
    /// Returns
    /// -------
    /// float
    ///     Forward rate from t1 to t2
    fn forward_rate(&self, t1: f64, t2: f64) -> PyResult<f64> {
        self.curve.forward_rate(t1, t2).map_err(to_py_err)
    }

    /// Calculate discount factor at a specific maturity
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Time (maturity in years)
    ///
    /// Returns
    /// -------
    /// float
    ///     Discount factor at the specified maturity
    fn discount_factor(&self, time: f64) -> PyResult<f64> {
        self.curve.discount_factor(time).map_err(to_py_err)
    }

    /// Price a bond using the yield curve
    ///
    /// Parameters
    /// ----------
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples representing bond cashflows
    ///
    /// Returns
    /// -------
    /// float
    ///     Present value of the bond
    fn price_bond(&self, cashflows: Vec<(f64, f64)>) -> PyResult<f64> {
        self.curve.price_bond(&cashflows).map_err(to_py_err)
    }

    /// Get the underlying time points
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of time points
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.curve.times())
    }

    /// Get the underlying rates
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Array of zero rates
    fn rates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.curve.rates())
    }

    fn __repr__(&self) -> String {
        format!(
            "YieldCurve(points={}, method={:?})",
            self.curve.times().len(),
            self.curve.interpolation_method()
        )
    }
}

/// Swap period helper for Python interoperability
#[pyclass(name = "SwapPeriod")]
#[derive(Clone, Copy)]
pub struct PySwapPeriod {
    period: CoreSwapPeriod,
}

#[pymethods]
impl PySwapPeriod {
    /// Create a new swap accrual period
    #[new]
    fn new(start: f64, end: f64, year_fraction: f64) -> PyResult<Self> {
        let period = CoreSwapPeriod::new(start, end, year_fraction).map_err(to_py_err)?;
        Ok(Self { period })
    }

    /// Start time of the accrual period in years
    #[getter]
    fn start(&self) -> f64 {
        self.period.start
    }

    /// End time of the accrual period in years
    #[getter]
    fn end(&self) -> f64 {
        self.period.end
    }

    /// Year fraction associated with the accrual period
    #[getter]
    fn year_fraction(&self) -> f64 {
        self.period.year_fraction
    }

    /// Return the period as a tuple ``(start, end, year_fraction)``
    fn as_tuple(&self) -> (f64, f64, f64) {
        (
            self.period.start,
            self.period.end,
            self.period.year_fraction,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "SwapPeriod(start={:.6}, end={:.6}, year_fraction={:.6})",
            self.period.start, self.period.end, self.period.year_fraction
        )
    }
}

/// Multi-curve container exposed to Python
#[pyclass(name = "MultiCurve")]
pub struct PyMultiCurve {
    inner: CoreMultiCurve,
}

#[pymethods]
impl PyMultiCurve {
    /// Create a multi-curve instance with the provided discount curve
    #[new]
    fn new(discount_curve: PyRef<PyYieldCurve>) -> Self {
        Self {
            inner: CoreMultiCurve::new(discount_curve.curve.clone()),
        }
    }

    /// Replace (or insert) a forward curve under the supplied name
    fn set_forward_curve(&mut self, name: &str, curve: PyRef<PyYieldCurve>) {
        self.inner.set_forward_curve(name, curve.curve.clone());
    }

    /// Add a forward curve ensuring the name is unique
    fn add_forward_curve(&mut self, name: &str, curve: PyRef<PyYieldCurve>) -> PyResult<()> {
        self.inner
            .add_forward_curve(name, curve.curve.clone())
            .map_err(to_py_err)
    }

    /// Retrieve a registered forward curve by name
    fn forward_curve(&self, name: &str) -> PyResult<PyYieldCurve> {
        let curve = self.inner.forward_curve(name).map_err(to_py_err)?;
        Ok(PyYieldCurve {
            curve: curve.clone(),
        })
    }

    /// List registered forward curve names
    fn forward_curve_names(&self) -> Vec<String> {
        self.inner.forward_curve_names()
    }

    /// Access the discount curve used by the multi-curve
    fn discount_curve(&self) -> PyYieldCurve {
        PyYieldCurve {
            curve: self.inner.discount_curve().clone(),
        }
    }

    /// Discount factor for a future time using the discount curve
    fn discount_factor(&self, t: f64) -> PyResult<f64> {
        self.inner.discount_factor(t).map_err(to_py_err)
    }

    /// Forward rate between ``start`` and ``end`` using the named forward curve
    fn forward_rate(&self, curve: &str, start: f64, end: f64) -> PyResult<f64> {
        self.inner
            .forward_rate(curve, start, end)
            .map_err(to_py_err)
    }

    /// Present value of cashflows discounted by the discount curve
    fn present_value(&self, cashflows: Vec<(f64, f64)>) -> PyResult<f64> {
        self.inner.present_value(&cashflows).map_err(to_py_err)
    }

    /// Compute the par swap rate for the supplied schedule
    #[pyo3(signature = (curve, periods))]
    fn par_swap_rate(&self, curve: &str, periods: Bound<'_, PyAny>) -> PyResult<f64> {
        let schedule = parse_swap_periods(&periods)?;
        self.inner
            .par_swap_rate(curve, &schedule)
            .map_err(to_py_err)
    }

    /// Price a fixed-for-floating payer swap
    #[pyo3(signature = (curve, periods, fixed_rate, notional))]
    fn price_payer_swap(
        &self,
        curve: &str,
        periods: Bound<'_, PyAny>,
        fixed_rate: f64,
        notional: f64,
    ) -> PyResult<f64> {
        let schedule = parse_swap_periods(&periods)?;
        self.inner
            .price_payer_swap(curve, &schedule, fixed_rate, notional)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiCurve(discount_points={}, forward_curves={})",
            self.inner.discount_curve().times().len(),
            self.inner.forward_curve_names().len()
        )
    }
}

/// Yield curve builder with bootstrapping methods
///
/// Provides static methods to construct yield curves from market data
/// such as bond prices and swap rates.
///
/// Examples
/// --------
/// >>> from dervflow.yield_curve import YieldCurveBuilder
/// >>> bonds = [(1.0, 0.03, 99.5, 2), (2.0, 0.04, 99.0, 2)]
/// >>> curve = YieldCurveBuilder.bootstrap_from_bonds(bonds)
#[pyclass(name = "YieldCurveBuilder")]
pub struct PyYieldCurveBuilder;

#[pymethods]
impl PyYieldCurveBuilder {
    #[new]
    fn new() -> Self {
        PyYieldCurveBuilder
    }

    /// Bootstrap yield curve from bond prices
    ///
    /// Parameters
    /// ----------
    /// bonds : list of tuples
    ///     List of (maturity, coupon, price, frequency) tuples
    ///     - maturity: time to maturity in years
    ///     - coupon: annual coupon rate
    ///     - price: market price
    ///     - frequency: payment frequency per year
    ///
    /// Returns
    /// -------
    /// YieldCurve
    ///     Bootstrapped yield curve with linear interpolation
    #[staticmethod]
    fn bootstrap_from_bonds(bonds: Vec<(f64, f64, f64, u32)>) -> PyResult<PyYieldCurve> {
        let bond_quotes: Vec<BondQuote> = bonds
            .into_iter()
            .map(|(maturity, coupon, price, frequency)| {
                BondQuote::new(maturity, coupon, price, frequency)
            })
            .collect();

        let (times, rates) = bootstrap_from_bonds(&bond_quotes).map_err(to_py_err)?;
        let curve =
            YieldCurve::new(times, rates, InterpolationMethod::Linear).map_err(to_py_err)?;

        Ok(PyYieldCurve { curve })
    }

    /// Bootstrap yield curve from swap rates
    ///
    /// Parameters
    /// ----------
    /// swaps : list of tuples
    ///     List of (maturity, rate, frequency) tuples
    ///     - maturity: time to maturity in years
    ///     - rate: fixed swap rate
    ///     - frequency: payment frequency per year
    ///
    /// Returns
    /// -------
    /// YieldCurve
    ///     Bootstrapped yield curve with linear interpolation
    #[staticmethod]
    fn bootstrap_from_swaps(swaps: Vec<(f64, f64, u32)>) -> PyResult<PyYieldCurve> {
        let swap_quotes: Vec<SwapQuote> = swaps
            .into_iter()
            .map(|(maturity, rate, frequency)| SwapQuote::new(maturity, rate, frequency))
            .collect();

        let (times, rates) = bootstrap_from_swaps(&swap_quotes).map_err(to_py_err)?;
        let curve =
            YieldCurve::new(times, rates, InterpolationMethod::Linear).map_err(to_py_err)?;

        Ok(PyYieldCurve { curve })
    }

    /// Create yield curve from Nelson-Siegel parameters
    ///
    /// Parameters
    /// ----------
    /// beta0 : float
    ///     Level parameter (long-term rate)
    /// beta1 : float
    ///     Slope parameter
    /// beta2 : float
    ///     Curvature parameter
    /// lambda_ : float
    ///     Decay parameter
    /// times : array-like
    ///     Time points to evaluate
    ///
    /// Returns
    /// -------
    /// YieldCurve
    ///     Yield curve based on Nelson-Siegel model
    #[staticmethod]
    fn from_nelson_siegel(
        beta0: f64,
        beta1: f64,
        beta2: f64,
        lambda_: f64,
        times: PyReadonlyArray1<f64>,
    ) -> PyResult<PyYieldCurve> {
        let params = NelsonSiegelParams::new(beta0, beta1, beta2, lambda_);
        let times_vec = times.as_slice()?.to_vec();
        let curve = YieldCurve::from_nelson_siegel(params, times_vec).map_err(to_py_err)?;

        Ok(PyYieldCurve { curve })
    }

    /// Create yield curve from Nelson-Siegel-Svensson parameters
    ///
    /// Parameters
    /// ----------
    /// beta0 : float
    ///     Level parameter
    /// beta1 : float
    ///     Slope parameter
    /// beta2 : float
    ///     Curvature parameter 1
    /// beta3 : float
    ///     Curvature parameter 2
    /// lambda1 : float
    ///     Decay parameter 1
    /// lambda2 : float
    ///     Decay parameter 2
    /// times : array-like
    ///     Time points to evaluate
    ///
    /// Returns
    /// -------
    /// YieldCurve
    ///     Yield curve based on Nelson-Siegel-Svensson model
    #[staticmethod]
    fn from_nelson_siegel_svensson(
        beta0: f64,
        beta1: f64,
        beta2: f64,
        beta3: f64,
        lambda1: f64,
        lambda2: f64,
        times: PyReadonlyArray1<f64>,
    ) -> PyResult<PyYieldCurve> {
        let params = NelsonSiegelSvenssonParams::new(beta0, beta1, beta2, beta3, lambda1, lambda2);
        let times_vec = times.as_slice()?.to_vec();
        let curve =
            YieldCurve::from_nelson_siegel_svensson(params, times_vec).map_err(to_py_err)?;

        Ok(PyYieldCurve { curve })
    }
}

/// Bond analytics calculator
///
/// Provides methods for calculating bond metrics such as yield to maturity,
/// duration, convexity, and DV01.
///
/// Examples
/// --------
/// >>> from dervflow.yield_curve import BondAnalytics
/// >>> ba = BondAnalytics()
/// >>> cashflows = [(1.0, 5.0), (2.0, 5.0), (2.0, 100.0)]
/// >>> ytm = ba.yield_to_maturity(98.0, cashflows)
/// >>> duration = ba.macaulay_duration(ytm, cashflows)
#[pyclass(name = "BondAnalytics")]
pub struct PyBondAnalytics;

#[pymethods]
impl PyBondAnalytics {
    #[new]
    fn new() -> Self {
        PyBondAnalytics
    }

    /// Calculate yield to maturity
    ///
    /// Parameters
    /// ----------
    /// price : float
    ///     Current market price of the bond
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples
    /// initial_guess : float, optional
    ///     Initial guess for YTM (default: 0.05)
    ///
    /// Returns
    /// -------
    /// float
    ///     Yield to maturity
    #[pyo3(signature = (price, cashflows, initial_guess=None))]
    fn yield_to_maturity(
        &self,
        price: f64,
        cashflows: Vec<(f64, f64)>,
        initial_guess: Option<f64>,
    ) -> PyResult<f64> {
        let cfs: Vec<Cashflow> = cashflows
            .into_iter()
            .map(|(t, a)| Cashflow::new(t, a))
            .collect();

        yield_to_maturity(price, &cfs, initial_guess).map_err(to_py_err)
    }

    /// Calculate Macaulay duration
    ///
    /// Parameters
    /// ----------
    /// yield_rate : float
    ///     Yield rate (typically YTM)
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples
    ///
    /// Returns
    /// -------
    /// float
    ///     Macaulay duration in years
    fn macaulay_duration(&self, yield_rate: f64, cashflows: Vec<(f64, f64)>) -> PyResult<f64> {
        let cfs: Vec<Cashflow> = cashflows
            .into_iter()
            .map(|(t, a)| Cashflow::new(t, a))
            .collect();

        macaulay_duration(yield_rate, &cfs).map_err(to_py_err)
    }

    /// Calculate Modified duration
    ///
    /// Parameters
    /// ----------
    /// yield_rate : float
    ///     Yield rate (typically YTM)
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples
    /// frequency : int, optional
    ///     Compounding frequency per year (0 for continuous, default: 0)
    ///
    /// Returns
    /// -------
    /// float
    ///     Modified duration
    #[pyo3(signature = (yield_rate, cashflows, frequency=0))]
    fn modified_duration(
        &self,
        yield_rate: f64,
        cashflows: Vec<(f64, f64)>,
        frequency: u32,
    ) -> PyResult<f64> {
        let cfs: Vec<Cashflow> = cashflows
            .into_iter()
            .map(|(t, a)| Cashflow::new(t, a))
            .collect();

        modified_duration(yield_rate, &cfs, frequency).map_err(to_py_err)
    }

    /// Calculate convexity
    ///
    /// Parameters
    /// ----------
    /// yield_rate : float
    ///     Yield rate (typically YTM)
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples
    ///
    /// Returns
    /// -------
    /// float
    ///     Convexity
    fn convexity(&self, yield_rate: f64, cashflows: Vec<(f64, f64)>) -> PyResult<f64> {
        let cfs: Vec<Cashflow> = cashflows
            .into_iter()
            .map(|(t, a)| Cashflow::new(t, a))
            .collect();

        convexity(yield_rate, &cfs).map_err(to_py_err)
    }

    /// Calculate DV01 (Dollar Value of 01 basis point)
    ///
    /// Parameters
    /// ----------
    /// yield_rate : float
    ///     Yield rate (typically YTM)
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples
    ///
    /// Returns
    /// -------
    /// float
    ///     DV01
    fn dv01(&self, yield_rate: f64, cashflows: Vec<(f64, f64)>) -> PyResult<f64> {
        let cfs: Vec<Cashflow> = cashflows
            .into_iter()
            .map(|(t, a)| Cashflow::new(t, a))
            .collect();

        dv01(yield_rate, &cfs).map_err(to_py_err)
    }

    /// Calculate bond price from yield
    ///
    /// Parameters
    /// ----------
    /// yield_rate : float
    ///     Yield rate
    /// cashflows : list of tuples
    ///     List of (time, amount) tuples
    ///
    /// Returns
    /// -------
    /// float
    ///     Bond price
    fn bond_price(&self, yield_rate: f64, cashflows: Vec<(f64, f64)>) -> PyResult<f64> {
        let cfs: Vec<Cashflow> = cashflows
            .into_iter()
            .map(|(t, a)| Cashflow::new(t, a))
            .collect();

        bond_price(yield_rate, &cfs).map_err(to_py_err)
    }

    /// Generate cashflows for a standard coupon bond
    ///
    /// Parameters
    /// ----------
    /// maturity : float
    ///     Time to maturity in years
    /// coupon_rate : float
    ///     Annual coupon rate
    /// face_value : float
    ///     Face value of the bond
    /// frequency : int
    ///     Payment frequency per year
    ///
    /// Returns
    /// -------
    /// list of tuples
    ///     List of (time, amount) tuples
    #[staticmethod]
    fn generate_cashflows(
        maturity: f64,
        coupon_rate: f64,
        face_value: f64,
        frequency: u32,
    ) -> Vec<(f64, f64)> {
        let cashflows = generate_bond_cashflows(maturity, coupon_rate, face_value, frequency);
        cashflows
            .into_iter()
            .map(|cf| (cf.time, cf.amount))
            .collect()
    }
}

/// Register yield curve module with Python
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyYieldCurve>()?;
    parent.add_class::<PyYieldCurveBuilder>()?;
    parent.add_class::<PyBondAnalytics>()?;
    parent.add_class::<PySwapPeriod>()?;
    parent.add_class::<PyMultiCurve>()?;
    Ok(())
}
