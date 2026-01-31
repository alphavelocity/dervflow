// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Portfolio-level risk metrics utilities
//!
//! This module provides helper functions to analyse the risk of a
//! multi-asset portfolio.  The implementations mirror the routines used by
//! the portfolio optimisation module but expose them in a standalone form so
//! they can be reused from the risk analytics APIs as well as from Python.
//!
//! The provided functionality includes:
//!
//! - Portfolio variance and volatility calculations
//! - Marginal and component risk contributions
//! - Diversification and concentration statistics
//! - Parametric (variance-covariance) portfolio VaR and CVaR
//! - Marginal/component contributions to parametric VaR and CVaR
//! - Active risk analytics (tracking error, information ratio, active share) and CAPM metrics
//!
//! These helpers operate on plain weight vectors and covariance matrices and
//! perform extensive input validation to make them convenient and safe to use
//! in higher-level APIs.

use crate::common::error::{DervflowError, Result};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// Summary statistics for a portfolio.
#[derive(Debug, Clone)]
pub struct PortfolioSummary {
    /// Optional expected portfolio return.
    pub expected_return: Option<f64>,
    /// Portfolio variance (σ²).
    pub variance: f64,
    /// Portfolio volatility (σ).
    pub volatility: f64,
    /// Optional Sharpe ratio relative to the supplied risk-free rate.
    pub sharpe_ratio: Option<f64>,
    /// Diversification ratio = (Σ wᵢσᵢ) / σₚ.
    pub diversification_ratio: f64,
    /// Herfindahl-Hirschman index of the weight distribution.
    pub weight_concentration: f64,
    /// Herfindahl-Hirschman index of risk contributions.
    pub risk_concentration: f64,
    /// Marginal contribution of each asset to portfolio volatility (∂σ/∂wᵢ).
    pub marginal_risk: Vec<f64>,
    /// Component risk contribution of each asset (wᵢ · ∂σ/∂wᵢ).
    pub component_risk: Vec<f64>,
    /// Percentage contribution of each asset (component / σₚ).
    pub percentage_risk: Vec<f64>,
}

/// Active risk and performance statistics relative to a benchmark portfolio.
#[derive(Debug, Clone)]
pub struct ActivePortfolioMetrics {
    /// Active weight vector (w - wᴮ).
    pub active_weights: Vec<f64>,
    /// Difference between portfolio and benchmark expected returns.
    pub active_return: Option<f64>,
    /// Expected portfolio return Σ wᵢμᵢ when expected returns are provided.
    pub portfolio_return: Option<f64>,
    /// Expected benchmark return Σ wᵢᴮμᵢ when expected returns are provided.
    pub benchmark_return: Option<f64>,
    /// Tracking error (standard deviation of active returns).
    pub tracking_error: f64,
    /// Information ratio = active return / tracking error.
    pub information_ratio: Option<f64>,
    /// Active share = ½ Σ |wᵢ - wᵢᴮ|.
    pub active_share: f64,
    /// Marginal contribution of each asset to tracking error.
    pub marginal_tracking_error: Vec<f64>,
    /// Component contribution of each asset to tracking error.
    pub component_tracking_error: Vec<f64>,
    /// Percentage contribution of each asset to tracking error.
    pub percentage_tracking_error: Vec<f64>,
    /// Optional marginal/component/percentage contributions to active return.
    pub active_return_contributions: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
}

/// Capital Asset Pricing Model (CAPM) metrics for a portfolio relative to a benchmark.
#[derive(Debug, Clone)]
pub struct CapmMetrics {
    /// Expected portfolio return Σ wᵢμᵢ.
    pub portfolio_return: f64,
    /// Expected excess portfolio return above risk-free rate.
    pub portfolio_excess_return: f64,
    /// Expected benchmark return.
    pub benchmark_return: f64,
    /// Expected benchmark excess return above risk-free rate.
    pub benchmark_excess_return: f64,
    /// Portfolio beta with respect to the benchmark.
    pub beta: f64,
    /// CAPM alpha = portfolio excess return − β · benchmark excess return.
    pub alpha: f64,
}

/// Calculate the expected portfolio return given weights and expected asset returns.
pub fn portfolio_return(weights: &[f64], expected_returns: &[f64]) -> Result<f64> {
    if weights.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Weights vector cannot be empty".to_string(),
        ));
    }

    if weights.len() != expected_returns.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Weights length ({}) does not match expected returns length ({})",
            weights.len(),
            expected_returns.len()
        )));
    }

    Ok(weights
        .iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum())
}

/// Calculate the portfolio variance wᵀΣw.
pub fn portfolio_variance(weights: &[f64], covariance: &DMatrix<f64>) -> Result<f64> {
    validate_inputs(weights, covariance)?;

    let w = DVector::from_column_slice(weights);
    let variance_matrix = w.transpose() * covariance * &w;
    let variance = variance_matrix[(0, 0)];

    if variance.is_sign_negative() && variance.abs() > 1e-12 {
        return Err(DervflowError::NumericalError(
            "Covariance matrix produced a negative variance".to_string(),
        ));
    }

    Ok(variance.max(0.0))
}

/// Calculate the portfolio volatility (standard deviation).
pub fn portfolio_volatility(weights: &[f64], covariance: &DMatrix<f64>) -> Result<f64> {
    let variance = portfolio_variance(weights, covariance)?;
    Ok(variance.sqrt())
}

/// Compute marginal, component and percentage risk contributions.
pub fn risk_contributions(
    weights: &[f64],
    covariance: &DMatrix<f64>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_inputs(weights, covariance)?;

    let n = weights.len();
    let w = DVector::from_column_slice(weights);

    let variance = (w.transpose() * covariance * &w)[(0, 0)].max(0.0);
    if variance < 1e-24 {
        return Ok((vec![0.0; n], vec![0.0; n], vec![0.0; n]));
    }

    let volatility = variance.sqrt();
    let marginal = covariance * &w;

    let mut marginal_vol = Vec::with_capacity(n);
    let mut component = Vec::with_capacity(n);
    for i in 0..n {
        let m = marginal[i] / volatility;
        marginal_vol.push(m);
        component.push(weights[i] * m);
    }

    let percentage = if volatility > 0.0 {
        component.iter().map(|c| c / volatility).collect()
    } else {
        vec![0.0; n]
    };

    Ok((marginal_vol, component, percentage))
}

/// Create a portfolio risk summary for the supplied weights and covariance matrix.
pub fn portfolio_summary(
    weights: &[f64],
    covariance: &DMatrix<f64>,
    expected_returns: Option<&[f64]>,
    risk_free_rate: Option<f64>,
) -> Result<PortfolioSummary> {
    validate_inputs(weights, covariance)?;

    if let Some(er) = expected_returns
        && er.len() != weights.len()
    {
        return Err(DervflowError::InvalidInput(format!(
            "Expected returns length ({}) does not match number of assets ({})",
            er.len(),
            weights.len()
        )));
    }

    let variance = portfolio_variance(weights, covariance)?;
    let volatility = variance.sqrt();

    let (marginal, component, percentage) = risk_contributions(weights, covariance)?;

    let expected_return = expected_returns
        .map(|er| portfolio_return(weights, er))
        .transpose()?;

    let sharpe_ratio = expected_return.and_then(|er| {
        if volatility > 1e-12 {
            Some((er - risk_free_rate.unwrap_or(0.0)) / volatility)
        } else {
            None
        }
    });

    let diversification_ratio = diversification_ratio(weights, covariance, volatility);
    let weight_concentration = herfindahl_index(weights);
    let risk_concentration = herfindahl_index(&percentage);

    Ok(PortfolioSummary {
        expected_return,
        variance,
        volatility,
        sharpe_ratio,
        diversification_ratio,
        weight_concentration,
        risk_concentration,
        marginal_risk: marginal,
        component_risk: component,
        percentage_risk: percentage,
    })
}

/// Parametric (variance-covariance) Value at Risk for the portfolio.
pub fn portfolio_parametric_var(
    weights: &[f64],
    covariance: &DMatrix<f64>,
    expected_returns: Option<&[f64]>,
    confidence_level: f64,
) -> Result<f64> {
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    let volatility = portfolio_volatility(weights, covariance)?;
    if volatility <= 0.0 {
        return Ok(0.0);
    }

    let mean = expected_returns
        .map(|er| portfolio_return(weights, er))
        .transpose()?;

    let normal = Normal::new(0.0, 1.0).map_err(|err| {
        DervflowError::NumericalError(format!("Failed to construct normal distribution: {}", err))
    })?;
    let z = normal.inverse_cdf(1.0 - confidence_level);

    let mean_value = mean.unwrap_or(0.0);
    let var = -mean_value - z * volatility;
    Ok(var.max(0.0))
}

/// Parametric (variance-covariance) Conditional Value at Risk (Expected Shortfall).
pub fn portfolio_parametric_cvar(
    weights: &[f64],
    covariance: &DMatrix<f64>,
    expected_returns: Option<&[f64]>,
    confidence_level: f64,
) -> Result<f64> {
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    let volatility = portfolio_volatility(weights, covariance)?;
    if volatility <= 0.0 {
        return Ok(0.0);
    }

    let mean = expected_returns
        .map(|er| portfolio_return(weights, er))
        .transpose()?;

    let normal = Normal::new(0.0, 1.0).map_err(|err| {
        DervflowError::NumericalError(format!("Failed to construct normal distribution: {}", err))
    })?;
    let alpha = 1.0 - confidence_level;
    let z = normal.inverse_cdf(alpha);
    let pdf = normal.pdf(z);

    let mean_value = mean.unwrap_or(0.0);
    let cvar = -mean_value + volatility * (pdf / alpha);
    Ok(cvar.max(0.0))
}

/// Parametric VaR contributions using the variance-covariance method.
pub fn portfolio_parametric_var_contributions(
    weights: &[f64],
    covariance: &DMatrix<f64>,
    expected_returns: Option<&[f64]>,
    confidence_level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    validate_inputs(weights, covariance)?;

    let n = weights.len();
    if let Some(er) = expected_returns
        && er.len() != n
    {
        return Err(DervflowError::InvalidInput(format!(
            "Expected returns length ({}) does not match number of assets ({})",
            er.len(),
            n
        )));
    }

    let volatility = portfolio_volatility(weights, covariance)?;
    if volatility <= 0.0 {
        return Ok((vec![0.0; n], vec![0.0; n], vec![0.0; n]));
    }

    let mean_vec: Vec<f64> = expected_returns
        .map(|er| er.to_vec())
        .unwrap_or_else(|| vec![0.0; n]);

    let w = DVector::from_column_slice(weights);
    let covariance_w = covariance * &w;

    let normal = Normal::new(0.0, 1.0).map_err(|err| {
        DervflowError::NumericalError(format!("Failed to construct normal distribution: {}", err))
    })?;
    let z = normal.inverse_cdf(1.0 - confidence_level);

    let mean_portfolio = expected_returns
        .map(|er| portfolio_return(weights, er))
        .transpose()?;
    let var = (-mean_portfolio.unwrap_or(0.0) - z * volatility).max(0.0);

    if var <= 1e-24 {
        return Ok((vec![0.0; n], vec![0.0; n], vec![0.0; n]));
    }

    let mut marginal = Vec::with_capacity(n);
    let mut component = Vec::with_capacity(n);

    for i in 0..n {
        let grad = -mean_vec[i] - z * (covariance_w[i] / volatility);
        marginal.push(grad);
        component.push(weights[i] * grad);
    }

    let percentage = component.iter().map(|c| c / var).collect();

    Ok((marginal, component, percentage))
}

/// Parametric CVaR contributions using the variance-covariance method.
pub fn portfolio_parametric_cvar_contributions(
    weights: &[f64],
    covariance: &DMatrix<f64>,
    expected_returns: Option<&[f64]>,
    confidence_level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    validate_inputs(weights, covariance)?;

    let n = weights.len();
    if let Some(er) = expected_returns
        && er.len() != n
    {
        return Err(DervflowError::InvalidInput(format!(
            "Expected returns length ({}) does not match number of assets ({})",
            er.len(),
            n
        )));
    }

    let volatility = portfolio_volatility(weights, covariance)?;
    if volatility <= 0.0 {
        return Ok((vec![0.0; n], vec![0.0; n], vec![0.0; n]));
    }

    let mean_vec: Vec<f64> = expected_returns
        .map(|er| er.to_vec())
        .unwrap_or_else(|| vec![0.0; n]);

    let w = DVector::from_column_slice(weights);
    let covariance_w = covariance * &w;

    let normal = Normal::new(0.0, 1.0).map_err(|err| {
        DervflowError::NumericalError(format!("Failed to construct normal distribution: {}", err))
    })?;
    let alpha = 1.0 - confidence_level;
    let z = normal.inverse_cdf(alpha);
    let pdf = normal.pdf(z);

    let mean_portfolio = expected_returns
        .map(|er| portfolio_return(weights, er))
        .transpose()?;
    let cvar = (-mean_portfolio.unwrap_or(0.0) + volatility * (pdf / alpha)).max(0.0);

    if cvar <= 1e-24 {
        return Ok((vec![0.0; n], vec![0.0; n], vec![0.0; n]));
    }

    let mut marginal = Vec::with_capacity(n);
    let mut component = Vec::with_capacity(n);

    for i in 0..n {
        let grad = -mean_vec[i] + (pdf / alpha) * (covariance_w[i] / volatility);
        marginal.push(grad);
        component.push(weights[i] * grad);
    }

    let percentage = component.iter().map(|c| c / cvar).collect();

    Ok((marginal, component, percentage))
}

/// Compute the tracking error (active risk) between a portfolio and benchmark weights.
pub fn portfolio_tracking_error(
    weights: &[f64],
    benchmark_weights: &[f64],
    covariance: &DMatrix<f64>,
) -> Result<f64> {
    validate_inputs(weights, covariance)?;

    if weights.len() != benchmark_weights.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Benchmark weights length ({}) does not match number of assets ({})",
            benchmark_weights.len(),
            weights.len()
        )));
    }

    let active_weights: Vec<f64> = weights
        .iter()
        .zip(benchmark_weights.iter())
        .map(|(w, b)| w - b)
        .collect();

    if active_weights
        .iter()
        .all(|aw| aw.abs() <= f64::EPSILON * 10.0)
    {
        return Ok(0.0);
    }

    let active = DVector::from_column_slice(&active_weights);
    let variance = (active.transpose() * covariance * &active)[(0, 0)].max(0.0);
    Ok(variance.sqrt())
}

/// Calculate the portfolio active share relative to a benchmark.
pub fn portfolio_active_share(weights: &[f64], benchmark_weights: &[f64]) -> Result<f64> {
    if weights.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Weights vector cannot be empty".to_string(),
        ));
    }

    if weights.len() != benchmark_weights.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Benchmark weights length ({}) does not match number of assets ({})",
            benchmark_weights.len(),
            weights.len()
        )));
    }

    let sum_abs: f64 = weights
        .iter()
        .zip(benchmark_weights.iter())
        .map(|(w, b)| (w - b).abs())
        .sum();

    Ok(0.5 * sum_abs)
}

/// Compute detailed active risk metrics relative to a benchmark.
#[allow(clippy::too_many_arguments)]
pub fn active_portfolio_metrics(
    weights: &[f64],
    benchmark_weights: &[f64],
    covariance: &DMatrix<f64>,
    expected_returns: Option<&[f64]>,
) -> Result<ActivePortfolioMetrics> {
    validate_inputs(weights, covariance)?;

    if weights.len() != benchmark_weights.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Benchmark weights length ({}) does not match number of assets ({})",
            benchmark_weights.len(),
            weights.len()
        )));
    }

    if let Some(er) = expected_returns
        && er.len() != weights.len()
    {
        return Err(DervflowError::InvalidInput(format!(
            "Expected returns length ({}) does not match number of assets ({})",
            er.len(),
            weights.len()
        )));
    }

    let active_weights: Vec<f64> = weights
        .iter()
        .zip(benchmark_weights.iter())
        .map(|(w, b)| w - b)
        .collect();

    let tracking_error = portfolio_tracking_error(weights, benchmark_weights, covariance)?;

    let (
        portfolio_return_value,
        benchmark_return_value,
        active_return,
        active_return_contributions,
    ) = if let Some(er) = expected_returns {
        let portfolio_return_value = portfolio_return(weights, er)?;
        let benchmark_return_value = portfolio_return(benchmark_weights, er)?;
        let active_return_value = portfolio_return_value - benchmark_return_value;

        let marginal = er.to_vec();
        let component: Vec<f64> = active_weights
            .iter()
            .zip(er.iter())
            .map(|(aw, mu)| aw * mu)
            .collect();
        let percentage = if active_return_value.abs() > 1e-12 {
            component.iter().map(|c| c / active_return_value).collect()
        } else {
            vec![0.0; component.len()]
        };

        (
            Some(portfolio_return_value),
            Some(benchmark_return_value),
            Some(active_return_value),
            Some((marginal, component, percentage)),
        )
    } else {
        (None, None, None, None)
    };

    let information_ratio = active_return.and_then(|ar| {
        if tracking_error > 1e-12 {
            Some(ar / tracking_error)
        } else {
            None
        }
    });

    if tracking_error <= 1e-24 {
        let zeros = vec![0.0; active_weights.len()];
        return Ok(ActivePortfolioMetrics {
            active_weights,
            active_return,
            portfolio_return: portfolio_return_value,
            benchmark_return: benchmark_return_value,
            tracking_error,
            information_ratio,
            active_share: portfolio_active_share(weights, benchmark_weights)?,
            marginal_tracking_error: zeros.clone(),
            component_tracking_error: zeros.clone(),
            percentage_tracking_error: zeros,
            active_return_contributions,
        });
    }

    let active_vector = DVector::from_column_slice(&active_weights);
    let covariance_active = covariance * &active_vector;

    let mut marginal = Vec::with_capacity(active_weights.len());
    let mut component = Vec::with_capacity(active_weights.len());
    for i in 0..active_weights.len() {
        let grad = covariance_active[i] / tracking_error;
        marginal.push(grad);
        component.push(active_weights[i] * grad);
    }

    let percentage = component
        .iter()
        .map(|c| c / tracking_error)
        .collect::<Vec<_>>();

    Ok(ActivePortfolioMetrics {
        active_weights,
        active_return,
        portfolio_return: portfolio_return_value,
        benchmark_return: benchmark_return_value,
        tracking_error,
        information_ratio,
        active_share: portfolio_active_share(weights, benchmark_weights)?,
        marginal_tracking_error: marginal,
        component_tracking_error: component,
        percentage_tracking_error: percentage,
        active_return_contributions,
    })
}

/// Compute the portfolio beta relative to a benchmark using asset covariances.
pub fn portfolio_beta(
    weights: &[f64],
    asset_benchmark_covariances: &[f64],
    benchmark_variance: f64,
) -> Result<f64> {
    if weights.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Weights vector cannot be empty".to_string(),
        ));
    }

    if benchmark_variance <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "Benchmark variance must be positive".to_string(),
        ));
    }

    if weights.len() != asset_benchmark_covariances.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Asset-benchmark covariance length ({}) does not match number of assets ({})",
            asset_benchmark_covariances.len(),
            weights.len()
        )));
    }

    let covariance_with_benchmark: f64 = weights
        .iter()
        .zip(asset_benchmark_covariances.iter())
        .map(|(w, cov)| w * cov)
        .sum();

    Ok(covariance_with_benchmark / benchmark_variance)
}

/// Compute CAPM alpha, beta and excess returns for a portfolio relative to a benchmark.
pub fn capm_metrics(
    weights: &[f64],
    expected_returns: &[f64],
    benchmark_return: f64,
    risk_free_rate: f64,
    asset_benchmark_covariances: &[f64],
    benchmark_variance: f64,
) -> Result<CapmMetrics> {
    if weights.len() != expected_returns.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Expected returns length ({}) does not match number of assets ({})",
            expected_returns.len(),
            weights.len()
        )));
    }

    let portfolio_return = portfolio_return(weights, expected_returns)?;
    let portfolio_excess_return = portfolio_return - risk_free_rate;
    let benchmark_excess_return = benchmark_return - risk_free_rate;
    let beta = portfolio_beta(weights, asset_benchmark_covariances, benchmark_variance)?;
    let alpha = portfolio_excess_return - beta * benchmark_excess_return;

    Ok(CapmMetrics {
        portfolio_return,
        portfolio_excess_return,
        benchmark_return,
        benchmark_excess_return,
        beta,
        alpha,
    })
}

fn validate_inputs(weights: &[f64], covariance: &DMatrix<f64>) -> Result<()> {
    if weights.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Weights vector cannot be empty".to_string(),
        ));
    }

    let n = weights.len();
    if covariance.nrows() != n || covariance.ncols() != n {
        return Err(DervflowError::InvalidInput(format!(
            "Covariance matrix dimensions ({} x {}) do not match number of assets ({})",
            covariance.nrows(),
            covariance.ncols(),
            n
        )));
    }

    // Basic symmetry check to guard against invalid inputs.
    for i in 0..n {
        for j in i + 1..n {
            if (covariance[(i, j)] - covariance[(j, i)]).abs() > 1e-10 {
                return Err(DervflowError::InvalidInput(
                    "Covariance matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    Ok(())
}

fn diversification_ratio(
    weights: &[f64],
    covariance: &DMatrix<f64>,
    portfolio_volatility: f64,
) -> f64 {
    if portfolio_volatility <= 1e-12 {
        return 0.0;
    }

    let mut numerator = 0.0;
    for (i, &weight) in weights.iter().enumerate() {
        let asset_var = covariance[(i, i)].max(0.0);
        numerator += weight.abs() * asset_var.sqrt();
    }

    if numerator <= 0.0 {
        0.0
    } else {
        numerator / portfolio_volatility
    }
}

fn herfindahl_index(values: &[f64]) -> f64 {
    let sum_abs: f64 = values.iter().map(|v| v.abs()).sum();
    if sum_abs <= 1e-12 {
        return 0.0;
    }

    values
        .iter()
        .map(|v| {
            let proportion = v.abs() / sum_abs;
            proportion * proportion
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_covariance() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.01, 0.005, 0.01, 0.09, 0.01, 0.005, 0.01, 0.0225],
        )
    }

    fn sample_covariance_two_assets() -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09])
    }

    #[test]
    fn test_portfolio_return() {
        let weights = vec![0.4, 0.3, 0.3];
        let expected_returns = vec![0.1, 0.12, 0.08];
        let result = portfolio_return(&weights, &expected_returns).unwrap();
        let manual: f64 = weights
            .iter()
            .zip(expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum();
        assert!((result - manual).abs() < 1e-12);
    }

    #[test]
    fn test_variance_and_volatility() {
        let weights = vec![0.5, 0.3, 0.2];
        let covariance = sample_covariance();
        let variance = portfolio_variance(&weights, &covariance).unwrap();
        let volatility = portfolio_volatility(&weights, &covariance).unwrap();

        assert!(variance > 0.0);
        assert!((volatility.powi(2) - variance).abs() < 1e-12);
    }

    #[test]
    fn test_risk_contributions_sum_to_one() {
        let weights = vec![0.4, 0.4, 0.2];
        let covariance = sample_covariance();
        let (marginal, component, percentage) = risk_contributions(&weights, &covariance).unwrap();

        assert_eq!(marginal.len(), 3);
        assert_eq!(component.len(), 3);
        assert_eq!(percentage.len(), 3);

        let summary = portfolio_summary(&weights, &covariance, None, None).unwrap();
        let total_component: f64 = summary.component_risk.iter().sum();
        assert!((total_component - summary.volatility).abs() < 1e-10);
        let pct_sum: f64 = percentage.iter().sum();
        assert!((pct_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parametric_var_and_cvar() {
        let weights = vec![0.6, 0.4];
        let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
        let expected_returns = vec![0.1, 0.12];

        let var =
            portfolio_parametric_var(&weights, &covariance, Some(&expected_returns), 0.95).unwrap();
        let cvar = portfolio_parametric_cvar(&weights, &covariance, Some(&expected_returns), 0.95)
            .unwrap();

        assert!(var >= 0.0);
        assert!(cvar >= var);
    }

    #[test]
    fn test_tracking_error_and_active_metrics() {
        let weights = vec![0.6, 0.4];
        let benchmark_weights = vec![0.5, 0.5];
        let covariance = sample_covariance_two_assets();
        let expected_returns = vec![0.08, 0.12];

        let tracking_error =
            portfolio_tracking_error(&weights, &benchmark_weights, &covariance).unwrap();
        assert!(tracking_error > 0.0);

        let metrics = active_portfolio_metrics(
            &weights,
            &benchmark_weights,
            &covariance,
            Some(&expected_returns),
        )
        .unwrap();

        let expected_active = [0.1, -0.1];
        for (value, expected) in metrics.active_weights.iter().zip(expected_active.iter()) {
            assert!((value - expected).abs() < 1e-12);
        }
        assert!(metrics.active_share > 0.09 && metrics.active_share < 0.11);
        assert!((metrics.tracking_error - tracking_error).abs() < 1e-12);
        assert!(metrics.active_return.unwrap() < 0.0);
        assert!(
            (metrics.portfolio_return.unwrap()
                - portfolio_return(&weights, &expected_returns).unwrap())
            .abs()
                < 1e-12
        );
        assert!(
            (metrics.benchmark_return.unwrap()
                - portfolio_return(&benchmark_weights, &expected_returns).unwrap())
            .abs()
                < 1e-12
        );
        let component_sum: f64 = metrics.component_tracking_error.iter().sum();
        assert!((component_sum - metrics.tracking_error).abs() < 1e-10);
        let pct_sum: f64 = metrics.percentage_tracking_error.iter().sum();
        assert!((pct_sum - 1.0).abs() < 1e-10);
        let (marginal_active, component_active, percentage_active) =
            metrics.active_return_contributions.as_ref().unwrap();
        for (m, expected) in marginal_active.iter().zip(expected_returns.iter()) {
            assert!((m - expected).abs() < 1e-12);
        }
        let expected_active_component: Vec<f64> = expected_active
            .iter()
            .zip(expected_returns.iter())
            .map(|(aw, mu)| aw * mu)
            .collect();
        for (value, expected) in component_active
            .iter()
            .zip(expected_active_component.iter())
        {
            assert!((value - expected).abs() < 1e-12);
        }
        let component_total: f64 = component_active.iter().sum();
        assert!((component_total - metrics.active_return.unwrap()).abs() < 1e-12);
        if metrics.active_return.unwrap().abs() > 1e-12 {
            let percentage_total: f64 = percentage_active.iter().sum();
            assert!((percentage_total - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_portfolio_beta_and_capm_metrics() {
        let weights = vec![0.6, 0.4];
        let asset_benchmark_covariances = vec![0.03, 0.05];
        let benchmark_variance = 0.04;
        let beta =
            portfolio_beta(&weights, &asset_benchmark_covariances, benchmark_variance).unwrap();
        assert!((beta - 0.95).abs() < 1e-12);

        let expected_returns = vec![0.08, 0.12];
        let metrics = capm_metrics(
            &weights,
            &expected_returns,
            0.09,
            0.02,
            &asset_benchmark_covariances,
            benchmark_variance,
        )
        .unwrap();

        assert!((metrics.portfolio_return - 0.096).abs() < 1e-12);
        assert!((metrics.beta - beta).abs() < 1e-12);
        assert!((metrics.alpha - 0.0095).abs() < 1e-6);
        assert!((metrics.portfolio_excess_return - 0.076).abs() < 1e-12);
        assert!((metrics.benchmark_excess_return - 0.07).abs() < 1e-12);
    }

    #[test]
    fn test_parametric_var_contributions_sum_to_var() {
        let weights = vec![0.5, 0.3, 0.2];
        let covariance = sample_covariance();
        let expected_returns = vec![0.08, 0.12, 0.06];

        let (marginal, component, percentage) = portfolio_parametric_var_contributions(
            &weights,
            &covariance,
            Some(&expected_returns),
            0.975,
        )
        .unwrap();

        assert_eq!(marginal.len(), weights.len());
        assert_eq!(component.len(), weights.len());
        assert_eq!(percentage.len(), weights.len());

        let var = portfolio_parametric_var(&weights, &covariance, Some(&expected_returns), 0.975)
            .unwrap();
        let component_sum: f64 = component.iter().sum();
        let percentage_sum: f64 = percentage.iter().sum();

        assert!((component_sum - var).abs() < 1e-10);
        assert!((percentage_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parametric_cvar_contributions_sum_to_cvar() {
        let weights = vec![0.4, 0.4, 0.2];
        let covariance = sample_covariance();
        let expected_returns = vec![0.07, 0.11, 0.05];

        let (marginal, component, percentage) = portfolio_parametric_cvar_contributions(
            &weights,
            &covariance,
            Some(&expected_returns),
            0.99,
        )
        .unwrap();

        assert_eq!(marginal.len(), weights.len());
        assert_eq!(component.len(), weights.len());
        assert_eq!(percentage.len(), weights.len());

        let cvar = portfolio_parametric_cvar(&weights, &covariance, Some(&expected_returns), 0.99)
            .unwrap();
        let component_sum: f64 = component.iter().sum();
        let percentage_sum: f64 = percentage.iter().sum();

        assert!((component_sum - cvar).abs() < 1e-10);
        assert!((percentage_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parametric_contributions_handle_zero_volatility() {
        let weights = vec![0.5, 0.5];
        let covariance = DMatrix::zeros(2, 2);

        let var_contrib =
            portfolio_parametric_var_contributions(&weights, &covariance, None, 0.95).unwrap();
        let cvar_contrib =
            portfolio_parametric_cvar_contributions(&weights, &covariance, None, 0.95).unwrap();

        for collection in [var_contrib, cvar_contrib] {
            assert!(collection.0.iter().all(|v| *v == 0.0));
            assert!(collection.1.iter().all(|v| *v == 0.0));
            assert!(collection.2.iter().all(|v| *v == 0.0));
        }
    }
}
