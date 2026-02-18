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
use std::f64::consts::PI;

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
    /// Exponentially weighted moving average
    Ewma,
    /// Extreme Value Theory using Peaks-Over-Threshold and GPD tail fit
    ExtremeValueTheory,
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
    let mut values = returns.to_vec();
    historical_var_in_place(&mut values, confidence_level)
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
    let mut values = returns.to_vec();
    historical_cvar_in_place(&mut values, confidence_level)
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
    validate_confidence_level(confidence_level)?;
    let (mean, std_dev) = mean_std_dev(returns)?;

    // Get the z-score for the confidence level
    let alpha = 1.0 - confidence_level;
    let z_score = inverse_normal_cdf(alpha)?;

    // VaR = -(mean + z_score * std_dev)
    // Since z_score is negative for left tail, this gives a positive VaR
    let var = -(mean + z_score * std_dev);

    Ok(var.max(0.0))
}

/// Calculate Conditional Value at Risk (CVaR) using parametric variance-covariance method
///
/// Assumes returns are normally distributed and uses the closed-form expected shortfall.
///
/// # Arguments
/// * `returns` - Historical returns data
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
/// CVaR value (positive number representing expected loss in the tail)
pub fn parametric_cvar(returns: &[f64], confidence_level: f64) -> Result<f64> {
    validate_confidence_level(confidence_level)?;
    let (mean, std_dev) = mean_std_dev(returns)?;

    let alpha = 1.0 - confidence_level;
    let z = inverse_normal_cdf(alpha)?;
    let pdf = (-0.5 * z * z).exp() / (2.0 * PI).sqrt();

    let cvar = -(mean - std_dev * (pdf / alpha));
    let var_floor = -(mean + z * std_dev);

    Ok(cvar.max(var_floor).max(0.0))
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
    if returns.len() < 4 {
        return Err(DervflowError::InvalidInput(
            "At least four observations are required for Cornish-Fisher VaR".to_string(),
        ));
    }

    validate_confidence_level(confidence_level)?;

    // Calculate moments
    let n = returns.len() as f64;
    let (mean, std_dev) = mean_std_dev(returns)?;

    if std_dev == 0.0 {
        return Ok((-mean).max(0.0));
    }

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

    // Get the z-score for the confidence level
    let alpha = 1.0 - confidence_level;
    let z = inverse_normal_cdf(alpha)?;

    // Cornish-Fisher expansion
    let z_cf =
        z + (z.powi(2) - 1.0) * skewness / 6.0 + (z.powi(3) - 3.0 * z) * excess_kurtosis / 24.0
            - (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;

    // VaR with Cornish-Fisher adjustment
    let var = -(mean + z_cf * std_dev);

    Ok(var.max(0.0))
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
    validate_monte_carlo_inputs(mean, std_dev, num_simulations, confidence_level)?;
    let mut simulated_returns = generate_monte_carlo_returns(mean, std_dev, num_simulations, seed);

    // Use historical VaR on simulated returns without additional allocations.
    historical_var_in_place(&mut simulated_returns, confidence_level)
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
    validate_monte_carlo_inputs(mean, std_dev, num_simulations, confidence_level)?;
    let mut simulated_returns = generate_monte_carlo_returns(mean, std_dev, num_simulations, seed);

    // Use historical CVaR on simulated returns without additional allocations.
    historical_cvar_in_place(&mut simulated_returns, confidence_level)
}

/// Calculate Value at Risk using the RiskMetrics 1996 EWMA volatility model
///
/// The RiskMetrics approach estimates the conditional volatility using an
/// exponentially weighted moving average (EWMA) with decay factor ``lambda``.
/// The VaR forecast assumes zero mean returns and a normal distribution.
///
/// # Arguments
/// * `returns` - Historical return observations (negative values are losses)
/// * `confidence_level` - Tail confidence level (e.g., 0.95 for 95%)
/// * `decay` - EWMA decay factor in the half-open interval [0, 1)
///
/// # Returns
/// VaR estimate expressed as a positive loss amount
pub fn riskmetrics_var(returns: &[f64], confidence_level: f64, decay: f64) -> Result<f64> {
    validate_confidence_level(confidence_level)?;
    let sigma = compute_ewma_sigma(returns, decay)?;
    let alpha = 1.0 - confidence_level;
    let z = inverse_normal_cdf(alpha)?;
    let var = -z * sigma;

    Ok(var.max(0.0))
}

/// Calculate Conditional VaR using the RiskMetrics 1996 EWMA volatility model
///
/// The RiskMetrics CVaR assumes normally distributed, zero-mean returns with
/// volatility estimated via the EWMA filter. The closed-form expression for the
/// expected shortfall of a normal distribution is utilised to avoid numerical
/// integration.
pub fn riskmetrics_cvar(returns: &[f64], confidence_level: f64, decay: f64) -> Result<f64> {
    validate_confidence_level(confidence_level)?;
    let sigma = compute_ewma_sigma(returns, decay)?;
    let alpha = 1.0 - confidence_level;
    let z = inverse_normal_cdf(alpha)?;
    let pdf = (-0.5 * z * z).exp() / (2.0 * PI).sqrt();
    let var = -z * sigma;
    let cvar = sigma * (pdf / alpha);

    Ok(cvar.max(var).max(0.0))
}

#[derive(Debug, Clone, Copy)]
struct PreparedEvt {
    fit: EvtTailFit,
    alpha: f64,
}

/// Calculate Value at Risk using Extreme Value Theory (Peaks-Over-Threshold).
///
/// This estimator focuses on tail losses above a high loss threshold and fits a
/// Generalized Pareto Distribution (GPD) to the exceedances.
///
/// # Arguments
/// * `returns` - Historical returns data (negative values are losses)
/// * `confidence_level` - Tail confidence level (e.g., 0.99 for 99%)
/// * `threshold_quantile` - Loss threshold quantile used for POT in (0, 1)
pub fn evt_var(returns: &[f64], confidence_level: f64, threshold_quantile: f64) -> Result<f64> {
    let prepared = prepare_evt_tail(returns, confidence_level, threshold_quantile)?;

    if let Some(prepared) = prepared {
        let var = evt_var_from_fit(&prepared.fit, prepared.alpha)?;

        Ok(var.max(0.0))
    } else {
        historical_var(returns, confidence_level)
    }
}

/// Calculate Conditional VaR using Extreme Value Theory (Peaks-Over-Threshold).
///
/// CVaR is calculated from the fitted GPD tail. For shape parameters near one,
/// the estimate can become unstable/infinite and an error is returned.
pub fn evt_cvar(returns: &[f64], confidence_level: f64, threshold_quantile: f64) -> Result<f64> {
    let prepared = prepare_evt_tail(returns, confidence_level, threshold_quantile)?;

    if let Some(prepared) = prepared {
        if prepared.fit.shape >= 1.0 {
            return Err(DervflowError::NumericalError(
                "EVT CVaR undefined for shape parameter >= 1".to_string(),
            ));
        }

        let var = evt_var_from_fit(&prepared.fit, prepared.alpha)?;
        let conditional_excess = (prepared.fit.scale
            + prepared.fit.shape * (var - prepared.fit.threshold_loss))
            / (1.0 - prepared.fit.shape);
        let cvar = var + conditional_excess;

        if !cvar.is_finite() {
            return Err(DervflowError::NumericalError(
                "EVT CVaR computation produced non-finite value".to_string(),
            ));
        }

        Ok(cvar.max(var).max(0.0))
    } else {
        historical_cvar(returns, confidence_level)
    }
}

fn prepare_evt_tail(
    returns: &[f64],
    confidence_level: f64,
    threshold_quantile: f64,
) -> Result<Option<PreparedEvt>> {
    validate_confidence_level(confidence_level)?;

    let fit = fit_evt_tail(returns, threshold_quantile)?;
    let alpha = 1.0 - confidence_level;

    if alpha >= fit.tail_probability {
        return Ok(None);
    }

    Ok(Some(PreparedEvt { fit, alpha }))
}

fn evt_var_from_fit(fit: &EvtTailFit, alpha: f64) -> Result<f64> {
    if alpha <= 0.0 || !alpha.is_finite() {
        return Err(DervflowError::NumericalError(
            "Invalid EVT alpha state".to_string(),
        ));
    }

    if fit.tail_probability <= 0.0
        || !fit.tail_probability.is_finite()
        || !fit.threshold_loss.is_finite()
        || !fit.shape.is_finite()
        || !fit.scale.is_finite()
        || fit.scale < 0.0
    {
        return Err(DervflowError::NumericalError(
            "Invalid EVT tail fit state".to_string(),
        ));
    }

    let tail_ratio = fit.tail_probability / alpha;
    if tail_ratio <= 0.0 || !tail_ratio.is_finite() {
        return Err(DervflowError::NumericalError(
            "Invalid EVT tail ratio state".to_string(),
        ));
    }

    let var = if fit.shape.abs() < 1e-10 {
        fit.threshold_loss + fit.scale * tail_ratio.ln()
    } else {
        fit.threshold_loss + (fit.scale / fit.shape) * (tail_ratio.powf(fit.shape) - 1.0)
    };

    if !var.is_finite() {
        return Err(DervflowError::NumericalError(
            "EVT VaR extrapolation produced non-finite value".to_string(),
        ));
    }

    Ok(var)
}

fn validate_confidence_level(confidence_level: f64) -> Result<()> {
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }
    Ok(())
}

#[inline]
fn validate_monte_carlo_inputs(
    mean: f64,
    std_dev: f64,
    num_simulations: usize,
    confidence_level: f64,
) -> Result<()> {
    if num_simulations == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of simulations must be positive".to_string(),
        ));
    }

    validate_confidence_level(confidence_level)?;

    if !mean.is_finite() {
        return Err(DervflowError::InvalidInput(format!(
            "Mean must be finite, got {}",
            mean
        )));
    }

    if !std_dev.is_finite() {
        return Err(DervflowError::InvalidInput(format!(
            "Standard deviation must be finite, got {}",
            std_dev
        )));
    }

    if std_dev < 0.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Standard deviation must be non-negative, got {}",
            std_dev
        )));
    }

    Ok(())
}

#[inline]
fn generate_monte_carlo_returns(
    mean: f64,
    std_dev: f64,
    num_simulations: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = if let Some(s) = seed {
        RandomGenerator::new(s)
    } else {
        RandomGenerator::from_entropy()
    };

    let mut simulated_returns = Vec::with_capacity(num_simulations);
    for _ in 0..num_simulations {
        simulated_returns.push(mean + std_dev * rng.standard_normal());
    }

    simulated_returns
}

fn validate_historical_inputs(returns: &[f64], confidence_level: f64) -> Result<()> {
    if returns.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    if !returns.iter().all(|value| value.is_finite()) {
        return Err(DervflowError::InvalidInput(
            "Returns must contain only finite values".to_string(),
        ));
    }

    validate_confidence_level(confidence_level)
}

fn historical_tail_count(length: usize, confidence_level: f64) -> usize {
    let alpha = 1.0 - confidence_level;
    let tail_count = (alpha * length as f64).ceil() as usize;
    tail_count.max(1).min(length)
}

fn prepare_historical_tail_in_place(
    values: &mut [f64],
    confidence_level: f64,
) -> Result<(usize, usize)> {
    validate_historical_inputs(values, confidence_level)?;
    let tail_count = historical_tail_count(values.len(), confidence_level);
    let target_index = tail_count.saturating_sub(1);

    values.select_nth_unstable_by(target_index, |a, b| a.total_cmp(b));

    Ok((tail_count, target_index))
}

#[inline]
fn historical_var_in_place(values: &mut [f64], confidence_level: f64) -> Result<f64> {
    let (_tail_count, target_index) = prepare_historical_tail_in_place(values, confidence_level)?;

    // VaR is the negative of the return at this percentile (to express as a positive loss)
    let var = -values[target_index];

    Ok(var.max(0.0))
}

#[inline]
fn historical_cvar_in_place(values: &mut [f64], confidence_level: f64) -> Result<f64> {
    let (tail_count, target_index) = prepare_historical_tail_in_place(values, confidence_level)?;

    // CVaR is the average of all returns worse than VaR
    let tail_sum: f64 = values[..tail_count].iter().sum();
    let cvar = -tail_sum / tail_count as f64;
    let var = -values[target_index];

    let var_clamped = var.max(0.0);
    Ok(cvar.max(var_clamped).max(0.0))
}

fn mean_std_dev(returns: &[f64]) -> Result<(f64, f64)> {
    if returns.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "At least two observations are required".to_string(),
        ));
    }

    if !returns.iter().all(|value| value.is_finite()) {
        return Err(DervflowError::InvalidInput(
            "Returns must contain only finite values".to_string(),
        ));
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut count = 0.0;

    for &value in returns {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    if count < 2.0 {
        return Err(DervflowError::InvalidInput(
            "At least two observations are required".to_string(),
        ));
    }

    let variance = m2 / (count - 1.0);
    if !variance.is_finite() || variance < 0.0 {
        return Err(DervflowError::NumericalError(
            "Invalid standard deviation calculated".to_string(),
        ));
    }

    Ok((mean, variance.sqrt()))
}

fn compute_ewma_sigma(returns: &[f64], decay: f64) -> Result<f64> {
    if returns.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    if !(0.0..1.0).contains(&decay) {
        return Err(DervflowError::InvalidInput(format!(
            "Decay factor must be in [0, 1), got {}",
            decay
        )));
    }

    if !returns.iter().all(|value| value.is_finite()) {
        return Err(DervflowError::InvalidInput(
            "Returns must contain only finite values".to_string(),
        ));
    }

    let mut variance = returns[0] * returns[0];
    let weight = 1.0 - decay;

    for &ret in returns.iter().skip(1) {
        variance = decay * variance + weight * ret * ret;
    }

    if !variance.is_finite() || variance < 0.0 {
        return Err(DervflowError::NumericalError(
            "Failed to compute EWMA variance".to_string(),
        ));
    }

    Ok(variance.sqrt())
}

#[derive(Debug, Clone, Copy)]
struct EvtTailFit {
    threshold_loss: f64,
    tail_probability: f64,
    shape: f64,
    scale: f64,
}

fn fit_evt_tail(returns: &[f64], threshold_quantile: f64) -> Result<EvtTailFit> {
    if returns.len() < 20 {
        return Err(DervflowError::InvalidInput(
            "At least 20 observations are required for EVT estimation".to_string(),
        ));
    }

    if !returns.iter().all(|value| value.is_finite()) {
        return Err(DervflowError::InvalidInput(
            "Returns must contain only finite values".to_string(),
        ));
    }

    if !(0.0..1.0).contains(&threshold_quantile) {
        return Err(DervflowError::InvalidInput(format!(
            "Threshold quantile must be in (0, 1), got {}",
            threshold_quantile
        )));
    }

    let mut losses: Vec<f64> = returns.iter().map(|&r| (-r).max(0.0)).collect();
    let threshold_index = ((threshold_quantile * losses.len() as f64).floor() as usize)
        .min(losses.len().saturating_sub(1));
    losses.select_nth_unstable_by(threshold_index, |a, b| a.total_cmp(b));
    let threshold_loss = losses[threshold_index];

    let mut count = 0usize;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for &loss in &losses {
        let excess = loss - threshold_loss;
        if excess > 0.0 {
            count += 1;
            sum += excess;
            sum_sq += excess * excess;
        }
    }

    if count == 0 {
        return Ok(EvtTailFit {
            threshold_loss,
            tail_probability: 0.0,
            shape: 0.0,
            scale: 0.0,
        });
    }

    if count < 10 {
        return Err(DervflowError::InvalidInput(
            "Not enough tail exceedances for EVT fit; lower threshold quantile".to_string(),
        ));
    }

    let count_f64 = count as f64;
    let m1 = sum / count_f64;
    let m2 = (sum_sq - (sum * sum) / count_f64) / (count_f64 - 1.0);

    if m1 <= 0.0 || !m1.is_finite() || !m2.is_finite() {
        return Err(DervflowError::NumericalError(
            "Failed to compute EVT exceedance moments".to_string(),
        ));
    }

    let tail_probability = count_f64 / losses.len() as f64;

    if m2 <= f64::EPSILON {
        return Ok(EvtTailFit {
            threshold_loss,
            tail_probability,
            shape: 0.0,
            scale: m1.max(1e-12),
        });
    }

    let moment_ratio = m1 * m1 / m2;
    let shape = ((1.0 - moment_ratio) / 2.0).clamp(-0.45, 0.45);
    let scale = (m1 * (1.0 - shape)).max(1e-12);

    Ok(EvtTailFit {
        threshold_loss,
        tail_probability,
        shape,
        scale,
    })
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
    fn test_parametric_cvar_matches_closed_form() {
        let returns = vec![
            0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.015, -0.015, 0.005,
        ];
        let confidence = 0.95;
        let cvar = parametric_cvar(&returns, confidence).unwrap();
        let var = parametric_var(&returns, confidence).unwrap();

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let alpha = 1.0 - confidence;
        let z = inverse_normal_cdf(alpha).unwrap();
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let expected = -(mean - std_dev * (pdf / alpha));

        assert!((cvar - expected).abs() < 1e-10);
        assert!(cvar >= var);
    }

    #[test]
    fn test_parametric_methods_clamp_non_loss_tail_to_zero() {
        let returns = vec![0.12, 0.11, 0.1, 0.13, 0.09, 0.14, 0.08, 0.15];

        let var = parametric_var(&returns, 0.95).unwrap();
        let cvar = parametric_cvar(&returns, 0.95).unwrap();

        assert_eq!(var, 0.0);
        assert_eq!(cvar, 0.0);
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
    fn test_cornish_fisher_var_clamps_to_zero_for_zero_volatility_positive_mean() {
        let returns = vec![0.02; 12];
        let var = cornish_fisher_var(&returns, 0.95).unwrap();
        assert_eq!(var, 0.0);
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
    fn test_monte_carlo_methods_reject_non_finite_inputs() {
        assert!(monte_carlo_var(f64::NAN, 0.02, 1000, 0.95, Some(42)).is_err());
        assert!(monte_carlo_var(0.0, f64::INFINITY, 1000, 0.95, Some(42)).is_err());
        assert!(monte_carlo_cvar(f64::NEG_INFINITY, 0.02, 1000, 0.95, Some(42)).is_err());
        assert!(monte_carlo_cvar(0.0, f64::NAN, 1000, 0.95, Some(42)).is_err());
    }

    #[test]
    fn test_riskmetrics_var() {
        let returns = [0.01, -0.015, 0.02, -0.005, 0.012];
        let var = riskmetrics_var(&returns, 0.95, 0.94).unwrap();
        assert!((var - 0.018_059_277_868).abs() < 1e-9);
    }

    #[test]
    fn test_riskmetrics_cvar_matches_closed_form() {
        let returns = [0.01, -0.015, 0.02, -0.005, 0.012];
        let confidence = 0.975;
        let decay = 0.93;

        let var = riskmetrics_var(&returns, confidence, decay).unwrap();
        let cvar = riskmetrics_cvar(&returns, confidence, decay).unwrap();

        assert!(cvar >= var);

        let mut variance = returns[0] * returns[0];
        for &value in returns.iter().skip(1) {
            variance = decay * variance + (1.0 - decay) * value * value;
        }
        let sigma = variance.sqrt();
        let alpha = 1.0 - confidence;
        let z = inverse_normal_cdf(alpha).unwrap();
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let expected_cvar = sigma * (pdf / alpha);

        assert!((cvar - expected_cvar).abs() < 1e-9);
    }

    #[test]
    fn test_riskmetrics_var_invalid_decay() {
        let returns = [0.01, -0.02, 0.015];
        assert!(riskmetrics_var(&returns, 0.95, 1.0).is_err());
        assert!(riskmetrics_var(&returns, 0.95, -0.1).is_err());
    }

    #[test]
    fn test_riskmetrics_cvar_invalid_inputs() {
        let returns = [0.01, -0.02, 0.015];
        assert!(riskmetrics_cvar(&returns, 0.95, 1.0).is_err());
        assert!(riskmetrics_cvar(&returns, 0.0, 0.94).is_err());
        assert!(riskmetrics_cvar(&[], 0.95, 0.94).is_err());
    }

    #[test]
    fn test_evt_var_and_cvar() {
        let returns: Vec<f64> = (0..4000)
            .map(|i| {
                let x = (i as f64 + 1.0) / 4001.0;
                let tail = ((1.0 - x).max(1e-6)).powf(-0.3) - 1.0;
                let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
                sign * 0.004 * tail
            })
            .collect();

        let var = evt_var(&returns, 0.99, 0.9).unwrap();
        let cvar = evt_cvar(&returns, 0.99, 0.9).unwrap();

        assert!(var > 0.0);
        assert!(cvar >= var);
    }

    #[test]
    fn test_evt_invalid_threshold_quantile() {
        let returns = vec![0.01; 100];
        assert!(evt_var(&returns, 0.99, 1.0).is_err());
        assert!(evt_var(&returns, 0.99, -0.2).is_err());
    }

    #[test]
    fn test_evt_var_rejects_non_finite_inputs() {
        let returns = vec![
            0.01,
            -0.02,
            f64::NAN,
            0.015,
            -0.01,
            0.005,
            -0.005,
            0.01,
            -0.008,
            0.003,
            -0.002,
            0.004,
            -0.006,
            0.007,
            -0.009,
            0.011,
            -0.012,
            0.013,
            -0.014,
            0.015,
        ];
        assert!(evt_var(&returns, 0.99, 0.9).is_err());
        assert!(evt_cvar(&returns, 0.99, 0.9).is_err());
    }

    #[test]
    fn test_evt_handles_low_tail_variance_exceedances() {
        let mut returns = vec![0.001; 200];
        returns.extend(vec![-0.01; 20]);

        let var = evt_var(&returns, 0.99, 0.9).unwrap();
        let cvar = evt_cvar(&returns, 0.99, 0.9).unwrap();

        assert!(var >= 0.0);
        assert!(cvar >= var);
    }

    #[test]
    fn test_evt_all_positive_returns_collapses_to_zero_tail_risk() {
        let returns = vec![
            0.001, 0.002, 0.0005, 0.0015, 0.0008, 0.0012, 0.0011, 0.0007, 0.0013, 0.0009, 0.0014,
            0.0010, 0.0006, 0.0016, 0.0004, 0.0017, 0.0003, 0.0018, 0.0002, 0.0019, 0.0001, 0.0020,
        ];

        let var = evt_var(&returns, 0.99, 0.9).unwrap();
        let cvar = evt_cvar(&returns, 0.99, 0.9).unwrap();

        assert_eq!(var, 0.0);
        assert_eq!(cvar, 0.0);
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
        let returns_with_nan = vec![0.01, f64::NAN, -0.02];
        let returns_with_inf = vec![0.01, f64::INFINITY, -0.02];

        // Empty returns
        assert!(historical_var(&[], 0.95).is_err());
        assert!(historical_cvar(&[], 0.95).is_err());

        // Invalid confidence level
        assert!(historical_var(&returns, 0.0).is_err());
        assert!(historical_var(&returns, 1.0).is_err());
        assert!(historical_var(&returns, 1.5).is_err());
        assert!(historical_cvar(&returns, 0.0).is_err());
        assert!(historical_cvar(&returns, 1.0).is_err());

        // Parametric methods require at least two observations
        assert!(parametric_var(&[0.01], 0.95).is_err());
        assert!(parametric_cvar(&[0.01], 0.95).is_err());

        // Cornish-Fisher requires enough observations for higher moments
        assert!(cornish_fisher_var(&[0.01, -0.01, 0.02], 0.95).is_err());

        // Historical methods reject non-finite values
        assert!(historical_var(&returns_with_nan, 0.95).is_err());
        assert!(historical_cvar(&returns_with_nan, 0.95).is_err());
        assert!(historical_var(&returns_with_inf, 0.95).is_err());
        assert!(historical_cvar(&returns_with_inf, 0.95).is_err());
    }

    #[test]
    fn test_historical_var_cvar_non_negative_for_positive_returns() {
        let returns = vec![0.01, 0.02, 0.03, 0.015, 0.025];
        let var = historical_var(&returns, 0.95).unwrap();
        let cvar = historical_cvar(&returns, 0.95).unwrap();
        assert!(var >= 0.0);
        assert!(cvar >= var);
    }

    #[test]
    fn test_var_result_creation() {
        let result = VaRResult::new(0.05, 0.95, VaRMethod::Historical);
        assert_eq!(result.value_at_risk, 0.05);
        assert_eq!(result.confidence_level, 0.95);
        assert_eq!(result.method, VaRMethod::Historical);
    }
}
