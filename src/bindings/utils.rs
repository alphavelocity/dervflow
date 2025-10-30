// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for analytics utilities.
//!
//! The functions exposed here back the :mod:`dervflow.utils` module.  They
//! implement portfolio and risk analytics entirely in Rust while presenting a
//! Python friendly API through PyO3.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::risk::metrics;
use crate::risk::var::{monte_carlo_cvar, monte_carlo_var, parametric_var};

fn flatten_any(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(value) = obj.extract::<f64>() {
        if value.is_finite() {
            return Ok(vec![value]);
        }
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{name} must contain at least one finite value"
        )));
    }

    if let Ok(array) = obj.extract::<Vec<f64>>() {
        return Ok(array);
    }

    if let Ok(nested) = obj.extract::<Vec<Vec<f64>>>() {
        return Ok(nested.into_iter().flatten().collect());
    }

    Err(PyErr::new::<PyValueError, _>(format!(
        "{name} must be array-like"
    )))
}

fn to_1d_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let values = flatten_any(obj, name)?;
    let filtered: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();

    if filtered.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{name} must contain at least one finite value"
        )));
    }

    Ok(filtered)
}

fn aligned_series(series: Vec<(Vec<f64>, &'static str)>) -> PyResult<Vec<Vec<f64>>> {
    if series.is_empty() {
        return Ok(vec![]);
    }

    let base_length = series[0].0.len();
    let base_name = series[0].1;
    let mut combined_mask = vec![true; base_length];

    for (values, name) in &series {
        if values.len() != base_length {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{name} must match the length of {base_name}"
            )));
        }

        let mut has_finite = false;
        for (idx, value) in values.iter().enumerate() {
            let finite = value.is_finite();
            if finite {
                has_finite = true;
            }
            combined_mask[idx] &= finite;
        }

        if !has_finite {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{name} must contain at least one finite value"
            )));
        }
    }

    if !combined_mask.iter().any(|flag| *flag) {
        let names = series
            .iter()
            .map(|(_, name)| *name)
            .collect::<Vec<_>>()
            .join(" and ");
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{names} must share at least one finite observation"
        )));
    }

    let mut aligned = Vec::with_capacity(series.len());
    for (values, _) in series {
        let filtered: Vec<f64> = values
            .into_iter()
            .zip(&combined_mask)
            .filter_map(|(value, keep)| if *keep { Some(value) } else { None })
            .collect();
        aligned.push(filtered);
    }

    Ok(aligned)
}

fn validate_periods_per_year(periods_per_year: usize) -> PyResult<usize> {
    if periods_per_year == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "periods_per_year must be positive",
        ));
    }
    Ok(periods_per_year)
}

fn array_mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn std(data: &[f64], ddof: usize) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let mean_value = array_mean(data);
    let denom = (data.len().saturating_sub(ddof)).max(1) as f64;
    let variance = data
        .iter()
        .map(|value| {
            let diff = value - mean_value;
            diff * diff
        })
        .sum::<f64>()
        / denom;
    variance.sqrt()
}

fn canonical_method(method: &str) -> PyResult<&'static str> {
    let key = method.trim().to_lowercase().replace(' ', "_");
    match key.as_str() {
        "historical" | "empirical" => Ok("historical"),
        "parametric" | "gaussian" | "normal" | "variance_covariance" | "variance-covariance" => {
            Ok("parametric")
        }
        "cornish_fisher" | "cornish-fisher" => Ok("cornish_fisher"),
        "monte_carlo" | "montecarlo" | "mc" => Ok("monte_carlo"),
        _ => Err(PyErr::new::<PyValueError, _>(
            "method must be 'historical', 'parametric', 'cornish_fisher', or 'monte_carlo'",
        )),
    }
}

fn parse_confidence_level(obj: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    let Some(value_obj) = obj else {
        return Ok(0.95);
    };

    if let Ok(value) = value_obj.extract::<f64>() {
        return parse_confidence_from_float(value);
    }

    let raw = value_obj
        .extract::<String>()
        .map_err(|_| PyErr::new::<PyValueError, _>("confidence_level must be between 0 and 1"))?;

    let cleaned = raw.trim().trim_end_matches('%');
    if cleaned.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(
            "confidence_level must be between 0 and 1",
        ));
    }

    let value: f64 = cleaned
        .parse()
        .map_err(|_| PyErr::new::<PyValueError, _>("confidence_level must be numeric"))?;
    parse_confidence_from_float(value)
}

fn parse_confidence_from_float(value: f64) -> PyResult<f64> {
    let mut confidence = value;
    if confidence > 1.0 {
        confidence /= 100.0;
    }

    if !(0.0..1.0).contains(&confidence) {
        return Err(PyErr::new::<PyValueError, _>(
            "confidence_level must be between 0 and 1",
        ));
    }

    Ok(confidence)
}

fn linear_quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    if data.len() == 1 {
        return data[0];
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return *sorted.last().unwrap();
    }

    let position = (sorted.len() - 1) as f64 * q;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;

    if lower_index == upper_index {
        return sorted[lower_index];
    }

    let weight = position - lower_index as f64;
    sorted[lower_index] * (1.0 - weight) + sorted[upper_index] * weight
}

fn prepare_return_and_risk_free(
    returns: &Bound<'_, PyAny>,
    risk_free_rate: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if let Some(risk_free_obj) = risk_free_rate {
        if let Ok(risk_free_scalar) = risk_free_obj.extract::<f64>() {
            let return_array = to_1d_array(returns, "returns")?;
            let risk_free_array = vec![risk_free_scalar; return_array.len()];
            return Ok((return_array, risk_free_array));
        }

        let returns_vec = flatten_any(returns, "returns")?;
        let risk_free_vec = flatten_any(risk_free_obj, "risk_free_rate")?;
        let aligned = aligned_series(vec![
            (returns_vec, "returns"),
            (risk_free_vec, "risk_free_rate"),
        ])?;
        Ok((aligned[0].clone(), aligned[1].clone()))
    } else {
        let return_array = to_1d_array(returns, "returns")?;
        let risk_free_array = vec![0.0; return_array.len()];
        Ok((return_array, risk_free_array))
    }
}

fn prepare_returns_benchmark_risk_free(
    returns: &Bound<'_, PyAny>,
    benchmark: &Bound<'_, PyAny>,
    risk_free_rate: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if let Some(risk_free_obj) = risk_free_rate {
        if let Ok(risk_free_scalar) = risk_free_obj.extract::<f64>() {
            let returns_vec = flatten_any(returns, "returns")?;
            let benchmark_vec = flatten_any(benchmark, "benchmark_returns")?;
            let aligned = aligned_series(vec![
                (returns_vec, "returns"),
                (benchmark_vec, "benchmark_returns"),
            ])?;
            let risk_free_array = vec![risk_free_scalar; aligned[0].len()];
            return Ok((aligned[0].clone(), aligned[1].clone(), risk_free_array));
        }

        let returns_vec = flatten_any(returns, "returns")?;
        let benchmark_vec = flatten_any(benchmark, "benchmark_returns")?;
        let risk_free_vec = flatten_any(risk_free_obj, "risk_free_rate")?;
        let aligned = aligned_series(vec![
            (returns_vec, "returns"),
            (benchmark_vec, "benchmark_returns"),
            (risk_free_vec, "risk_free_rate"),
        ])?;
        Ok((aligned[0].clone(), aligned[1].clone(), aligned[2].clone()))
    } else {
        let returns_vec = flatten_any(returns, "returns")?;
        let benchmark_vec = flatten_any(benchmark, "benchmark_returns")?;
        let aligned = aligned_series(vec![
            (returns_vec, "returns"),
            (benchmark_vec, "benchmark_returns"),
        ])?;
        let risk_free_array = vec![0.0; aligned[0].len()];
        Ok((aligned[0].clone(), aligned[1].clone(), risk_free_array))
    }
}

#[pyfunction]
#[pyo3(signature = (spot, strike, rate, dividend, volatility, time))]
fn validate_option_params(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time: f64,
) -> (bool, Option<String>) {
    let values = [
        ("Spot", spot),
        ("Strike", strike),
        ("Rate", rate),
        ("Dividend", dividend),
        ("Volatility", volatility),
        ("Time", time),
    ];

    for (name, value) in values {
        if !value.is_finite() {
            return (false, Some(format!("{name} must be a finite number")));
        }
    }

    if spot <= 0.0 {
        return (false, Some("Spot price must be positive".to_string()));
    }
    if strike <= 0.0 {
        return (false, Some("Strike price must be positive".to_string()));
    }
    if volatility < 0.0 {
        return (false, Some("Volatility must be non-negative".to_string()));
    }
    if time <= 0.0 {
        return (false, Some("Time to maturity must be positive".to_string()));
    }

    (true, None)
}

#[pyfunction]
#[pyo3(signature = (weights, tolerance=1e-6))]
fn validate_portfolio_weights(
    weights: &Bound<'_, PyAny>,
    tolerance: f64,
) -> PyResult<(bool, Option<String>)> {
    let values = match to_1d_array(weights, "weights") {
        Ok(data) => data,
        Err(_) => {
            return Ok((
                false,
                Some("Weights must contain at least one finite value".to_string()),
            ));
        }
    };

    if values.is_empty() {
        return Ok((false, Some("Weights array cannot be empty".to_string())));
    }

    if values.iter().any(|&w| w < -tolerance) {
        return Ok((false, Some("All weights must be non-negative".to_string())));
    }

    let sum_weights: f64 = values.iter().sum();
    if !sum_weights.is_finite() {
        return Ok((
            false,
            Some("Weights must sum to a finite value".to_string()),
        ));
    }

    if (sum_weights - 1.0).abs() > tolerance {
        return Ok((
            false,
            Some(format!(
                "Weights must sum to 1.0 (current sum: {:.6})",
                sum_weights
            )),
        ));
    }

    Ok((true, None))
}

#[pyfunction]
#[pyo3(signature = (returns, periods_per_year=252))]
fn annualize_returns(returns: &Bound<'_, PyAny>, periods_per_year: usize) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let data = to_1d_array(returns, "returns")?;
    metrics::annualize_returns(&data, periods_per_year).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (volatility, periods_per_year=252))]
fn annualize_volatility(volatility: &Bound<'_, PyAny>, periods_per_year: usize) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;

    if let Ok(scalar) = volatility.extract::<f64>() {
        return metrics::annualize_volatility_scalar(scalar, periods_per_year).map_err(PyErr::from);
    }

    let data = to_1d_array(volatility, "volatility")?;
    metrics::annualize_volatility(&data, periods_per_year).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, risk_free_rate=None, periods_per_year=252))]
fn sharpe_ratio(
    returns: &Bound<'_, PyAny>,
    risk_free_rate: Option<&Bound<'_, PyAny>>,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let (return_array, risk_free_array) = prepare_return_and_risk_free(returns, risk_free_rate)?;
    metrics::sharpe_ratio(&return_array, &risk_free_array, periods_per_year).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns, periods_per_year=252))]
fn tracking_error(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let returns_vec = flatten_any(returns, "returns")?;
    let benchmark_vec = flatten_any(benchmark_returns, "benchmark_returns")?;
    let aligned = aligned_series(vec![
        (returns_vec, "returns"),
        (benchmark_vec, "benchmark_returns"),
    ])?;

    metrics::tracking_error(&aligned[0], &aligned[1], periods_per_year).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns, periods_per_year=252))]
fn information_ratio(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let returns_vec = flatten_any(returns, "returns")?;
    let benchmark_vec = flatten_any(benchmark_returns, "benchmark_returns")?;
    let aligned = aligned_series(vec![
        (returns_vec, "returns"),
        (benchmark_vec, "benchmark_returns"),
    ])?;

    metrics::information_ratio(&aligned[0], &aligned[1], periods_per_year).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns))]
fn beta(returns: &Bound<'_, PyAny>, benchmark_returns: &Bound<'_, PyAny>) -> PyResult<f64> {
    let returns_vec = flatten_any(returns, "returns")?;
    let benchmark_vec = flatten_any(benchmark_returns, "benchmark_returns")?;
    let aligned = aligned_series(vec![
        (returns_vec, "returns"),
        (benchmark_vec, "benchmark_returns"),
    ])?;

    metrics::beta(&aligned[0], &aligned[1]).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns, risk_free_rate=None, periods_per_year=252))]
fn alpha(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    risk_free_rate: Option<&Bound<'_, PyAny>>,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let (return_array, benchmark_array, risk_free_array) =
        prepare_returns_benchmark_risk_free(returns, benchmark_returns, risk_free_rate)?;

    metrics::alpha(
        &return_array,
        &benchmark_array,
        &risk_free_array,
        periods_per_year,
    )
    .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, target_return=0.0, periods_per_year=252))]
fn downside_deviation(
    returns: &Bound<'_, PyAny>,
    target_return: f64,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let return_array = to_1d_array(returns, "returns")?;
    metrics::downside_deviation(&return_array, target_return, periods_per_year).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, risk_free_rate=None, target_return=0.0, periods_per_year=252))]
fn sortino_ratio(
    returns: &Bound<'_, PyAny>,
    risk_free_rate: Option<&Bound<'_, PyAny>>,
    target_return: f64,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let (return_array, risk_free_array) = prepare_return_and_risk_free(returns, risk_free_rate)?;
    metrics::sortino_ratio(
        &return_array,
        &risk_free_array,
        target_return,
        periods_per_year,
    )
    .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns, risk_free_rate=None, periods_per_year=252))]
fn treynor_ratio(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    risk_free_rate: Option<&Bound<'_, PyAny>>,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let (return_array, benchmark_array, risk_free_array) =
        prepare_returns_benchmark_risk_free(returns, benchmark_returns, risk_free_rate)?;

    metrics::treynor_ratio(
        &return_array,
        &benchmark_array,
        &risk_free_array,
        periods_per_year,
    )
    .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, threshold=0.0))]
fn omega_ratio(returns: &Bound<'_, PyAny>, threshold: f64) -> PyResult<f64> {
    if !threshold.is_finite() {
        return Err(PyErr::new::<PyValueError, _>("threshold must be finite"));
    }

    let return_array = to_1d_array(returns, "returns")?;
    metrics::omega_ratio(&return_array, threshold).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns))]
fn skewness(returns: &Bound<'_, PyAny>) -> PyResult<f64> {
    let return_array = to_1d_array(returns, "returns")?;
    metrics::skewness(&return_array).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns))]
fn excess_kurtosis(returns: &Bound<'_, PyAny>) -> PyResult<f64> {
    let return_array = to_1d_array(returns, "returns")?;
    metrics::excess_kurtosis(&return_array).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns))]
fn gain_loss_ratio(returns: &Bound<'_, PyAny>) -> PyResult<f64> {
    let return_array = to_1d_array(returns, "returns")?;
    metrics::gain_loss_ratio(&return_array).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, percentile=0.95))]
fn tail_ratio(returns: &Bound<'_, PyAny>, percentile: f64) -> PyResult<f64> {
    let return_array = to_1d_array(returns, "returns")?;
    metrics::tail_ratio(&return_array, percentile).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, threshold=0.0))]
fn upside_potential_ratio(returns: &Bound<'_, PyAny>, threshold: f64) -> PyResult<f64> {
    if !threshold.is_finite() {
        return Err(PyErr::new::<PyValueError, _>("threshold must be finite"));
    }

    let return_array = to_1d_array(returns, "returns")?;
    metrics::upside_potential_ratio(&return_array, threshold).map_err(PyErr::from)
}

fn capture_ratio(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    condition: fn(f64) -> bool,
    condition_name: &str,
    periods_per_year: usize,
) -> PyResult<f64> {
    validate_periods_per_year(periods_per_year)?;
    let returns_vec = flatten_any(returns, "returns")?;
    let benchmark_vec = flatten_any(benchmark_returns, "benchmark_returns")?;
    let aligned = aligned_series(vec![
        (returns_vec, "returns"),
        (benchmark_vec, "benchmark_returns"),
    ])?;

    metrics::capture_ratio(
        &aligned[0],
        &aligned[1],
        condition,
        condition_name,
        periods_per_year,
    )
    .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns, periods_per_year=252))]
fn upside_capture_ratio(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    periods_per_year: usize,
) -> PyResult<f64> {
    capture_ratio(
        returns,
        benchmark_returns,
        |v| v > 0.0,
        "positive",
        periods_per_year,
    )
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_returns, periods_per_year=252))]
fn downside_capture_ratio(
    returns: &Bound<'_, PyAny>,
    benchmark_returns: &Bound<'_, PyAny>,
    periods_per_year: usize,
) -> PyResult<f64> {
    capture_ratio(
        returns,
        benchmark_returns,
        |v| v < 0.0,
        "negative",
        periods_per_year,
    )
}

#[pyfunction]
#[pyo3(signature = (returns=None, confidence_level=None, method="historical", mean=None, std_dev=None, num_simulations=10000, seed=None))]
fn value_at_risk(
    returns: Option<&Bound<'_, PyAny>>,
    confidence_level: Option<&Bound<'_, PyAny>>,
    method: &str,
    mean: Option<f64>,
    std_dev: Option<f64>,
    num_simulations: usize,
    seed: Option<u64>,
) -> PyResult<f64> {
    let confidence = parse_confidence_level(confidence_level)?;
    let method_key = canonical_method(method)?;

    match method_key {
        "historical" => {
            let returns = returns.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("returns are required for the historical method")
            })?;
            let return_array = to_1d_array(returns, "returns")?;
            let losses: Vec<f64> = return_array.iter().map(|value| -value).collect();
            let quantile = linear_quantile(&losses, confidence);
            Ok(quantile.max(0.0))
        }
        "parametric" => {
            let returns = returns.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("returns are required for the parametric method")
            })?;
            let return_array = to_1d_array(returns, "returns")?;

            if return_array.len() == 1 {
                let mean = return_array[0];
                return Ok((-mean).max(0.0));
            }

            let var = parametric_var(&return_array, confidence).map_err(PyErr::from)?;
            Ok(var.max(0.0))
        }
        "cornish_fisher" => {
            let returns = returns.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("returns are required for the cornish_fisher method")
            })?;
            let return_array = to_1d_array(returns, "returns")?;

            if return_array.len() == 1 {
                let mean = return_array[0];
                return Ok((-mean).max(0.0));
            }

            let mean_value = array_mean(&return_array);
            let std_dev = std(&return_array, 1);
            if std_dev == 0.0 {
                return Ok((-mean_value).max(0.0));
            }

            let standardized: Vec<f64> = return_array
                .iter()
                .map(|value| (value - mean_value) / std_dev)
                .collect();

            let normal = Normal::new(0.0, 1.0)
                .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
            let alpha = 1.0 - confidence;
            let mut z = normal.inverse_cdf(alpha);

            let skewness =
                standardized.iter().map(|z| z.powi(3)).sum::<f64>() / standardized.len() as f64;
            let kurtosis = standardized.iter().map(|z| z.powi(4)).sum::<f64>()
                / standardized.len() as f64
                - 3.0;

            z += (z.powi(2) - 1.0) * skewness / 6.0;
            z += (z.powi(3) - 3.0 * z) * kurtosis / 24.0;
            z -= (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;

            Ok((-(mean_value + std_dev * z)).max(0.0))
        }
        "monte_carlo" => {
            let mean_value = mean.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("mean must be provided for the monte_carlo method")
            })?;
            let std_dev_value = std_dev.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("std_dev must be provided for the monte_carlo method")
            })?;

            let var = monte_carlo_var(mean_value, std_dev_value, num_simulations, confidence, seed)
                .map_err(PyErr::from)?;
            Ok(var.max(0.0))
        }
        _ => unreachable!(),
    }
}

#[pyfunction]
#[pyo3(signature = (returns=None, confidence_level=None, method="historical", mean=None, std_dev=None, num_simulations=10000, seed=None))]
fn conditional_value_at_risk(
    returns: Option<&Bound<'_, PyAny>>,
    confidence_level: Option<&Bound<'_, PyAny>>,
    method: &str,
    mean: Option<f64>,
    std_dev: Option<f64>,
    num_simulations: usize,
    seed: Option<u64>,
) -> PyResult<f64> {
    let confidence = parse_confidence_level(confidence_level)?;
    let method_key = canonical_method(method)?;

    match method_key {
        "historical" => {
            let returns = returns.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("returns are required for the historical method")
            })?;
            let return_array = to_1d_array(returns, "returns")?;
            let losses: Vec<f64> = return_array.iter().map(|value| -value).collect();
            let var = linear_quantile(&losses, confidence).max(0.0);
            let tail_losses: Vec<f64> = losses
                .into_iter()
                .filter(|loss| *loss >= var - 1e-12)
                .collect();
            if tail_losses.is_empty() {
                return Ok(var);
            }
            Ok(tail_losses.iter().sum::<f64>() / tail_losses.len() as f64)
        }
        "parametric" => {
            let returns = returns.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("returns are required for the parametric method")
            })?;
            let return_array = to_1d_array(returns, "returns")?;

            let var = if return_array.len() == 1 {
                (-return_array[0]).max(0.0)
            } else {
                parametric_var(&return_array, confidence)
                    .map_err(PyErr::from)?
                    .max(0.0)
            };

            if return_array.len() <= 1 {
                return Ok(var);
            }

            let mean_value = array_mean(&return_array);
            let std_dev = std(&return_array, 1);
            if std_dev == 0.0 {
                return Ok(var);
            }

            let normal = Normal::new(0.0, 1.0)
                .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
            let alpha = 1.0 - confidence;
            let z = normal.inverse_cdf(alpha);
            let pdf = normal.pdf(z);
            let cvar = -mean_value + std_dev * (pdf / alpha);
            Ok(cvar.max(var))
        }
        "cornish_fisher" => {
            let returns = returns.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("returns are required for the cornish_fisher method")
            })?;
            let return_array = to_1d_array(returns, "returns")?;

            let var = if return_array.len() == 1 {
                (-return_array[0]).max(0.0)
            } else {
                let mean_value = array_mean(&return_array);
                let std_dev = std(&return_array, 1);
                if std_dev == 0.0 {
                    return Ok((-mean_value).max(0.0));
                }

                let standardized: Vec<f64> = return_array
                    .iter()
                    .map(|value| (value - mean_value) / std_dev)
                    .collect();

                let normal = Normal::new(0.0, 1.0)
                    .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
                let alpha = 1.0 - confidence;
                let mut z = normal.inverse_cdf(alpha);

                let skewness =
                    standardized.iter().map(|z| z.powi(3)).sum::<f64>() / standardized.len() as f64;
                let kurtosis = standardized.iter().map(|z| z.powi(4)).sum::<f64>()
                    / standardized.len() as f64
                    - 3.0;

                z += (z.powi(2) - 1.0) * skewness / 6.0;
                z += (z.powi(3) - 3.0 * z) * kurtosis / 24.0;
                z -= (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;

                (-(mean_value + std_dev * z)).max(0.0)
            };

            if return_array.len() <= 1 {
                return Ok(var);
            }

            let mean_value = array_mean(&return_array);
            let std_dev = std(&return_array, 1);
            if std_dev == 0.0 {
                return Ok(var);
            }

            let normal = Normal::new(0.0, 1.0)
                .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
            let alpha = 1.0 - confidence;
            let mut z = normal.inverse_cdf(alpha);

            let standardized: Vec<f64> = return_array
                .iter()
                .map(|value| (value - mean_value) / std_dev)
                .collect();

            let skewness =
                standardized.iter().map(|z| z.powi(3)).sum::<f64>() / standardized.len() as f64;
            let kurtosis = standardized.iter().map(|z| z.powi(4)).sum::<f64>()
                / standardized.len() as f64
                - 3.0;

            z += (z.powi(2) - 1.0) * skewness / 6.0;
            z += (z.powi(3) - 3.0 * z) * kurtosis / 24.0;
            z -= (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;

            let pdf = normal.pdf(z);
            let cvar = -mean_value + std_dev * (pdf / alpha);
            Ok(cvar.max(var))
        }
        "monte_carlo" => {
            let mean_value = mean.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("mean must be provided for the monte_carlo method")
            })?;
            let std_dev_value = std_dev.ok_or_else(|| {
                PyErr::new::<PyValueError, _>("std_dev must be provided for the monte_carlo method")
            })?;

            let var = monte_carlo_var(mean_value, std_dev_value, num_simulations, confidence, seed)
                .map_err(PyErr::from)?
                .max(0.0);
            let cvar =
                monte_carlo_cvar(mean_value, std_dev_value, num_simulations, confidence, seed)
                    .map_err(PyErr::from)?;
            Ok(cvar.max(var))
        }
        _ => unreachable!(),
    }
}

#[pyfunction]
#[pyo3(signature = (prices))]
fn drawdown_series(py: Python<'_>, prices: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
    let price_array = to_1d_array(prices, "prices")?;
    let drawdowns = metrics::drawdown_series(&price_array).map_err(PyErr::from)?;
    let array = drawdowns.into_pyarray(py);
    Ok(array.to_owned().into())
}

#[pyfunction]
#[pyo3(signature = (prices))]
fn max_drawdown(prices: &Bound<'_, PyAny>) -> PyResult<f64> {
    let price_array = to_1d_array(prices, "prices")?;
    metrics::max_drawdown(&price_array).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (prices))]
fn pain_index(prices: &Bound<'_, PyAny>) -> PyResult<f64> {
    let price_array = to_1d_array(prices, "prices")?;
    metrics::pain_index(&price_array).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (prices))]
fn ulcer_index(prices: &Bound<'_, PyAny>) -> PyResult<f64> {
    let price_array = to_1d_array(prices, "prices")?;
    metrics::ulcer_index(&price_array).map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (annual_return, max_drawdown))]
fn calmar_ratio(annual_return: f64, max_drawdown: f64) -> PyResult<f64> {
    metrics::calmar_ratio(annual_return, max_drawdown).map_err(PyErr::from)
}

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "utils")?;
    module.add_function(wrap_pyfunction!(validate_option_params, &module)?)?;
    module.add_function(wrap_pyfunction!(validate_portfolio_weights, &module)?)?;
    module.add_function(wrap_pyfunction!(annualize_returns, &module)?)?;
    module.add_function(wrap_pyfunction!(annualize_volatility, &module)?)?;
    module.add_function(wrap_pyfunction!(sharpe_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(tracking_error, &module)?)?;
    module.add_function(wrap_pyfunction!(information_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(beta, &module)?)?;
    module.add_function(wrap_pyfunction!(alpha, &module)?)?;
    module.add_function(wrap_pyfunction!(downside_deviation, &module)?)?;
    module.add_function(wrap_pyfunction!(sortino_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(treynor_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(omega_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(skewness, &module)?)?;
    module.add_function(wrap_pyfunction!(excess_kurtosis, &module)?)?;
    module.add_function(wrap_pyfunction!(gain_loss_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(tail_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(upside_potential_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(upside_capture_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(downside_capture_ratio, &module)?)?;
    module.add_function(wrap_pyfunction!(value_at_risk, &module)?)?;
    module.add_function(wrap_pyfunction!(conditional_value_at_risk, &module)?)?;
    module.add_function(wrap_pyfunction!(drawdown_series, &module)?)?;
    module.add_function(wrap_pyfunction!(max_drawdown, &module)?)?;
    module.add_function(wrap_pyfunction!(pain_index, &module)?)?;
    module.add_function(wrap_pyfunction!(ulcer_index, &module)?)?;
    module.add_function(wrap_pyfunction!(calmar_ratio, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
