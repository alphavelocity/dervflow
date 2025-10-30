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

    if let Some(er) = expected_returns {
        if er.len() != weights.len() {
            return Err(DervflowError::InvalidInput(format!(
                "Expected returns length ({}) does not match number of assets ({})",
                er.len(),
                weights.len()
            )));
        }
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
}
