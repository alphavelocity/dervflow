// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Value at Risk (VaR) calculations
//!
//! Provides multiple methods for calculating Value at Risk:
//! - Historical simulation
//! - Parametric (variance-covariance)
//! - Monte Carlo simulation
//!
//! Also includes Conditional VaR (CVaR/Expected Shortfall) calculations.

use crate::common::error::{DervflowError, Result};
use crate::numerical::random::RandomGenerator;

/// VaR calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaRMethod {
    /// Historical simulation using empirical distribution
    Historical,
    /// Parametric variance-covariance method (assumes normal distribution)
    VarianceCovariance,
    /// Parametric with Cornish-Fisher expansion (accounts for skewness and kurtosis)
    CornishFisher,
    /// Monte Carlo simulation
    MonteCarlo,
}

/// Result of a VaR calculation
#[derive(Debug, Clone, Copy)]
pub struct VaRResult {
    /// Value at Risk (loss amount at specified confidence level)
    pub value_at_risk: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Method used for calculation
    pub method: VaRMethod,
}

impl VaRResult {
    /// Create a new VaRResult
    pub fn new(value_at_risk: f64, confidence_level: f64, method: VaRMethod) -> Self {
        Self {
            value_at_risk,
            confidence_level,
            method,
        }
    }
}

/// Calculate Value at Risk using historical simulation
///
/// # Arguments
/// * `returns` - Historical returns data (negative values represent losses, positive values represent gains)
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
/// VaR value (positive number representing potential loss at the given confidence level)
pub fn historical_var(returns: &[f64], confidence_level: f64) -> Result<f64> {
    if returns.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    // Sort returns in ascending order (worst losses first)
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate the index for the confidence level
    // For 95% confidence, we want the 5th percentile (worst 5% of outcomes)
    let alpha = 1.0 - confidence_level;
    let index = (alpha * sorted_returns.len() as f64).ceil() as usize;
    let index = index.min(sorted_returns.len()).saturating_sub(1);

    // VaR is the negative of the return at this percentile (to express as a positive loss)
    let var = -sorted_returns[index];

    Ok(var)
}

/// Calculate Conditional Value at Risk (CVaR/Expected Shortfall) using historical simulation
///
/// CVaR is the expected loss given that the loss exceeds VaR
///
/// # Arguments
/// * `returns` - Historical returns data (negative values represent losses, positive values represent gains)
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
/// CVaR value (positive number representing expected loss in the tail beyond VaR)
pub fn historical_cvar(returns: &[f64], confidence_level: f64) -> Result<f64> {
    if returns.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    // Sort returns in ascending order (worst losses first)
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate the index for the confidence level
    let alpha = 1.0 - confidence_level;
    let index = (alpha * sorted_returns.len() as f64).ceil() as usize;
    let index = index.min(sorted_returns.len());

    // CVaR is the average of all returns worse than VaR
    if index == 0 {
        return Ok(-sorted_returns[0]);
    }

    let tail_sum: f64 = sorted_returns[..index].iter().sum();
    let cvar = -tail_sum / index as f64;

    Ok(cvar)
}

/// Calculate Value at Risk using parametric variance-covariance method
///
/// Assumes returns are normally distributed
///
/// # Arguments
/// * `returns` - Historical returns data
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
/// VaR value (positive number representing potential loss)
pub fn parametric_var(returns: &[f64], confidence_level: f64) -> Result<f64> {
    if returns.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    // Calculate mean and standard deviation
    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    if std_dev.is_nan() || std_dev.is_infinite() {
        return Err(DervflowError::NumericalError(
            "Invalid standard deviation calculated".to_string(),
        ));
    }

    // Get the z-score for the confidence level
    let alpha = 1.0 - confidence_level;
    let z_score = inverse_normal_cdf(alpha)?;

    // VaR = -(mean + z_score * std_dev)
    // Since z_score is negative for left tail, this gives a positive VaR
    let var = -(mean + z_score * std_dev);

    Ok(var)
}

/// Calculate Value at Risk using Cornish-Fisher expansion
///
/// Accounts for skewness and kurtosis in the return distribution
///
/// # Arguments
/// * `returns` - Historical returns data
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
/// VaR value (positive number representing potential loss)
pub fn cornish_fisher_var(returns: &[f64], confidence_level: f64) -> Result<f64> {
    if returns.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    // Calculate moments
    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    // Calculate skewness and excess kurtosis
    let skewness: f64 = returns
        .iter()
        .map(|r| ((r - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n;

    let kurtosis: f64 = returns
        .iter()
        .map(|r| ((r - mean) / std_dev).powi(4))
        .sum::<f64>()
        / n;
    let excess_kurtosis = kurtosis - 3.0;

    if std_dev.is_nan() || std_dev.is_infinite() {
        return Err(DervflowError::NumericalError(
            "Invalid standard deviation calculated".to_string(),
        ));
    }

    // Get the z-score for the confidence level
    let alpha = 1.0 - confidence_level;
    let z = inverse_normal_cdf(alpha)?;

    // Cornish-Fisher expansion
    let z_cf =
        z + (z.powi(2) - 1.0) * skewness / 6.0 + (z.powi(3) - 3.0 * z) * excess_kurtosis / 24.0
            - (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;

    // VaR with Cornish-Fisher adjustment
    let var = -(mean + z_cf * std_dev);

    Ok(var)
}

/// Calculate Value at Risk using Monte Carlo simulation
///
/// # Arguments
/// * `mean` - Expected return
/// * `std_dev` - Standard deviation of returns
/// * `num_simulations` - Number of Monte Carlo paths to simulate
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// VaR value (positive number representing potential loss)
pub fn monte_carlo_var(
    mean: f64,
    std_dev: f64,
    num_simulations: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> Result<f64> {
    if num_simulations == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of simulations must be positive".to_string(),
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    if std_dev < 0.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Standard deviation must be non-negative, got {}",
            std_dev
        )));
    }

    // Generate simulated returns
    let mut rng = if let Some(s) = seed {
        RandomGenerator::new(s)
    } else {
        RandomGenerator::from_entropy()
    };

    let mut simulated_returns = Vec::with_capacity(num_simulations);
    for _ in 0..num_simulations {
        let z = rng.standard_normal();
        let return_val = mean + std_dev * z;
        simulated_returns.push(return_val);
    }

    // Use historical VaR on simulated returns
    historical_var(&simulated_returns, confidence_level)
}

/// Calculate CVaR using Monte Carlo simulation
///
/// # Arguments
/// * `mean` - Expected return
/// * `std_dev` - Standard deviation of returns
/// * `num_simulations` - Number of Monte Carlo paths to simulate
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// CVaR value (positive number representing expected loss in tail)
pub fn monte_carlo_cvar(
    mean: f64,
    std_dev: f64,
    num_simulations: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> Result<f64> {
    if num_simulations == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of simulations must be positive".to_string(),
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    if std_dev < 0.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Standard deviation must be non-negative, got {}",
            std_dev
        )));
    }

    // Generate simulated returns
    let mut rng = if let Some(s) = seed {
        RandomGenerator::new(s)
    } else {
        RandomGenerator::from_entropy()
    };

    let mut simulated_returns = Vec::with_capacity(num_simulations);
    for _ in 0..num_simulations {
        let z = rng.standard_normal();
        let return_val = mean + std_dev * z;
        simulated_returns.push(return_val);
    }

    // Use historical CVaR on simulated returns
    historical_cvar(&simulated_returns, confidence_level)
}

/// Inverse normal cumulative distribution function (quantile function)
///
/// Approximation using Beasley-Springer-Moro algorithm
fn inverse_normal_cdf(p: f64) -> Result<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Probability must be between 0 and 1, got {}",
            p
        )));
    }

    // Coefficients for the approximation
    #[allow(clippy::excessive_precision)]
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    #[allow(clippy::excessive_precision)]
    let b = [
        -5.447609879822406e+01,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];

    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];

    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    // Define break-points
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let x: f64;

    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_var() {
        let returns = vec![-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06];
        let var = historical_var(&returns, 0.95).unwrap();
        // At 95% confidence, we expect the 5th percentile (worst 5%)
        // With 10 observations, 5% is 0.5, so we take the 1st worst observation
        assert!(var > 0.0);
        assert!(var <= 0.05); // Should be around the worst return
    }

    #[test]
    fn test_historical_cvar() {
        let returns = vec![-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06];
        let cvar = historical_cvar(&returns, 0.95).unwrap();
        // CVaR should be at least as large as VaR
        let var = historical_var(&returns, 0.95).unwrap();
        assert!(cvar >= var);
    }

    #[test]
    fn test_parametric_var() {
        // Create normally distributed returns
        let returns = vec![
            0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.015, -0.015, 0.005,
        ];
        let var = parametric_var(&returns, 0.95).unwrap();
        assert!(var > 0.0);
    }

    #[test]
    fn test_cornish_fisher_var() {
        let returns = vec![
            0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.015, -0.015, 0.005,
        ];
        let var = cornish_fisher_var(&returns, 0.95).unwrap();
        assert!(var > 0.0);
    }

    #[test]
    fn test_monte_carlo_var() {
        let var = monte_carlo_var(0.0, 0.02, 10000, 0.95, Some(42)).unwrap();
        assert!(var > 0.0);
        // For normal distribution with mean 0 and std 0.02, 95% VaR should be around 1.645 * 0.02
        assert!((var - 0.0329).abs() < 0.005); // Allow some tolerance
    }

    #[test]
    fn test_monte_carlo_cvar() {
        let cvar = monte_carlo_cvar(0.0, 0.02, 10000, 0.95, Some(42)).unwrap();
        let var = monte_carlo_var(0.0, 0.02, 10000, 0.95, Some(42)).unwrap();
        assert!(cvar >= var);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Test some known values
        let z_005 = inverse_normal_cdf(0.05).unwrap();
        assert!((z_005 + 1.645).abs() < 0.01); // Should be approximately -1.645

        let z_05 = inverse_normal_cdf(0.5).unwrap();
        assert!(z_05.abs() < 0.001); // Should be approximately 0

        let z_095 = inverse_normal_cdf(0.95).unwrap();
        assert!((z_095 - 1.645).abs() < 0.01); // Should be approximately 1.645
    }

    #[test]
    fn test_var_invalid_inputs() {
        let returns = vec![0.01, -0.01, 0.02];

        // Empty returns
        assert!(historical_var(&[], 0.95).is_err());

        // Invalid confidence level
        assert!(historical_var(&returns, 0.0).is_err());
        assert!(historical_var(&returns, 1.0).is_err());
        assert!(historical_var(&returns, 1.5).is_err());
    }

    #[test]
    fn test_var_result_creation() {
        let result = VaRResult::new(0.05, 0.95, VaRMethod::Historical);
        assert_eq!(result.value_at_risk, 0.05);
        assert_eq!(result.confidence_level, 0.95);
        assert_eq!(result.method, VaRMethod::Historical);
    }
}
