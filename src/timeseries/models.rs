// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Time series models (ARMA, GARCH)
//!
//! This module provides implementations of time series models including
//! GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
//! for volatility forecasting.

use crate::common::error::{DervflowError, Result};
use std::f64::consts::PI;

/// GARCH model variant types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GarchVariant {
    /// Standard GARCH(1,1) model
    Standard,
    /// Exponential GARCH (EGARCH) model
    EGARCH,
    /// GJR-GARCH model (asymmetric GARCH)
    GJRGARCH,
}

/// GARCH model parameters
#[derive(Debug, Clone, Copy)]
pub struct GarchParams {
    /// Constant term (omega)
    pub omega: f64,
    /// ARCH coefficient (alpha)
    pub alpha: f64,
    /// GARCH coefficient (beta)
    pub beta: f64,
    /// Asymmetry parameter (gamma) - used in EGARCH and GJR-GARCH
    pub gamma: f64,
}

impl GarchParams {
    /// Create new GARCH parameters
    pub fn new(omega: f64, alpha: f64, beta: f64, gamma: f64) -> Self {
        Self {
            omega,
            alpha,
            beta,
            gamma,
        }
    }

    /// Validate GARCH parameters for standard GARCH(1,1)
    pub fn validate_standard(&self) -> Result<()> {
        if self.omega <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "omega must be positive".to_string(),
            ));
        }
        if self.alpha < 0.0 {
            return Err(DervflowError::InvalidInput(
                "alpha must be non-negative".to_string(),
            ));
        }
        if self.beta < 0.0 {
            return Err(DervflowError::InvalidInput(
                "beta must be non-negative".to_string(),
            ));
        }
        if self.alpha + self.beta >= 1.0 {
            return Err(DervflowError::InvalidInput(
                "alpha + beta must be less than 1 for stationarity".to_string(),
            ));
        }
        Ok(())
    }
}

/// GARCH model for volatility forecasting
#[derive(Debug, Clone)]
pub struct GarchModel {
    /// Model parameters
    pub params: GarchParams,
    /// Model variant
    pub variant: GarchVariant,
    /// Fitted conditional variances
    pub conditional_variances: Vec<f64>,
    /// Log-likelihood of the fitted model
    pub log_likelihood: f64,
}

impl GarchModel {
    /// Create a new GARCH model with given parameters
    pub fn new(params: GarchParams, variant: GarchVariant) -> Self {
        Self {
            params,
            variant,
            conditional_variances: Vec::new(),
            log_likelihood: 0.0,
        }
    }

    /// Fit GARCH model to return series using maximum likelihood estimation
    pub fn fit(returns: &[f64], variant: GarchVariant) -> Result<Self> {
        if returns.len() < 10 {
            return Err(DervflowError::DataError(
                "need at least 10 observations to fit GARCH model".to_string(),
            ));
        }

        // Check for NaN or infinite values
        if returns.iter().any(|&r| !r.is_finite()) {
            return Err(DervflowError::DataError(
                "returns contain NaN or infinite values".to_string(),
            ));
        }

        // Initial parameter estimates
        let initial_params = Self::initial_estimates(returns);

        // Optimize parameters using maximum likelihood
        let optimized_params = Self::optimize_parameters(returns, initial_params, variant)?;

        // Calculate conditional variances with optimized parameters
        let conditional_variances =
            Self::calculate_conditional_variances(returns, &optimized_params, variant);

        // Calculate log-likelihood
        let log_likelihood = Self::log_likelihood(returns, &conditional_variances);

        Ok(Self {
            params: optimized_params,
            variant,
            conditional_variances,
            log_likelihood,
        })
    }

    /// Get initial parameter estimates
    fn initial_estimates(returns: &[f64]) -> GarchParams {
        // Calculate sample variance
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        // Use typical starting values
        GarchParams::new(
            variance * 0.1, // omega
            0.1,            // alpha
            0.8,            // beta
            0.0,            // gamma
        )
    }

    /// Optimize GARCH parameters using maximum likelihood estimation
    fn optimize_parameters(
        returns: &[f64],
        initial: GarchParams,
        variant: GarchVariant,
    ) -> Result<GarchParams> {
        // Simple grid search with refinement for MLE
        // In production, would use proper optimization (L-BFGS-B)

        let mut best_params = initial;
        let mut best_ll = f64::NEG_INFINITY;

        // Grid search over parameter space
        let omega_range = [0.00001, 0.0001, 0.001, 0.01];
        let alpha_range = [0.05, 0.1, 0.15, 0.2];
        let beta_range = [0.7, 0.8, 0.85, 0.9];
        let gamma_range = match variant {
            GarchVariant::Standard => vec![0.0],
            _ => vec![-0.1, 0.0, 0.1, 0.2],
        };

        for &omega in &omega_range {
            for &alpha in &alpha_range {
                for &beta in &beta_range {
                    if alpha + beta >= 0.999 {
                        continue; // Skip non-stationary combinations
                    }

                    for &gamma in &gamma_range {
                        let params = GarchParams::new(omega, alpha, beta, gamma);

                        // Calculate conditional variances
                        let cond_var =
                            Self::calculate_conditional_variances(returns, &params, variant);

                        // Calculate log-likelihood
                        let ll = Self::log_likelihood(returns, &cond_var);

                        if ll.is_finite() && ll > best_ll {
                            best_ll = ll;
                            best_params = params;
                        }
                    }
                }
            }
        }

        if best_ll.is_finite() {
            Ok(best_params)
        } else {
            Err(DervflowError::ConvergenceFailure {
                iterations: 0,
                error: 0.0,
            })
        }
    }

    /// Calculate conditional variances for the return series
    fn calculate_conditional_variances(
        returns: &[f64],
        params: &GarchParams,
        variant: GarchVariant,
    ) -> Vec<f64> {
        let n = returns.len();
        let mut variances = Vec::with_capacity(n);

        // Initialize with unconditional variance estimate
        let mean = returns.iter().sum::<f64>() / n as f64;
        let initial_var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;

        variances.push(initial_var);

        // Calculate conditional variances recursively
        for t in 1..n {
            let prev_return = returns[t - 1];
            let prev_variance = variances[t - 1];

            let variance = match variant {
                GarchVariant::Standard => {
                    // Standard GARCH(1,1): σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
                    params.omega + params.alpha * prev_return.powi(2) + params.beta * prev_variance
                }
                GarchVariant::EGARCH => {
                    // EGARCH: log(σ²_t) = ω + α*|z_{t-1}| + γ*z_{t-1} + β*log(σ²_{t-1})
                    let z = prev_return / prev_variance.sqrt();
                    let log_var = params.omega
                        + params.alpha * z.abs()
                        + params.gamma * z
                        + params.beta * prev_variance.ln();
                    log_var.exp()
                }
                GarchVariant::GJRGARCH => {
                    // GJR-GARCH: σ²_t = ω + α*ε²_{t-1} + γ*ε²_{t-1}*I_{t-1} + β*σ²_{t-1}
                    // where I_{t-1} = 1 if ε_{t-1} < 0, else 0
                    let indicator = if prev_return < 0.0 { 1.0 } else { 0.0 };
                    params.omega
                        + params.alpha * prev_return.powi(2)
                        + params.gamma * prev_return.powi(2) * indicator
                        + params.beta * prev_variance
                }
            };

            // Ensure variance is positive and finite
            variances.push(variance.clamp(1e-10, 1e10));
        }

        variances
    }

    /// Calculate log-likelihood for GARCH model
    fn log_likelihood(returns: &[f64], conditional_variances: &[f64]) -> f64 {
        let n = returns.len();
        let mut ll = 0.0;

        for t in 0..n {
            let var = conditional_variances[t];
            let ret = returns[t];

            // Gaussian log-likelihood: -0.5 * (log(2π) + log(σ²) + ε²/σ²)
            ll += -0.5 * ((2.0 * PI).ln() + var.ln() + ret.powi(2) / var);
        }

        ll
    }

    /// Forecast volatility for h steps ahead
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.conditional_variances.is_empty() {
            return Err(DervflowError::DataError(
                "model must be fitted before forecasting".to_string(),
            ));
        }

        let mut forecasts = Vec::with_capacity(horizon);
        let last_variance = *self.conditional_variances.last().unwrap();
        let last_return_sq = 0.0; // Assume zero expected return

        match self.variant {
            GarchVariant::Standard => {
                // For standard GARCH(1,1):
                // h=1: σ²_{t+1} = ω + α*ε²_t + β*σ²_t
                // h>1: σ²_{t+h} = ω + (α+β)*σ²_{t+h-1}
                let mut prev_var = last_variance;

                for h in 1..=horizon {
                    let forecast = if h == 1 {
                        self.params.omega
                            + self.params.alpha * last_return_sq
                            + self.params.beta * prev_var
                    } else {
                        self.params.omega + (self.params.alpha + self.params.beta) * prev_var
                    };
                    forecasts.push(forecast.sqrt()); // Return volatility (std dev)
                    prev_var = forecast;
                }
            }
            GarchVariant::EGARCH => {
                // EGARCH multi-step forecast
                let mut prev_log_var = last_variance.ln();

                for _h in 1..=horizon {
                    let log_var = self.params.omega + self.params.beta * prev_log_var;
                    let forecast = log_var.exp();
                    forecasts.push(forecast.sqrt());
                    prev_log_var = log_var;
                }
            }
            GarchVariant::GJRGARCH => {
                // GJR-GARCH multi-step forecast
                let mut prev_var = last_variance;

                for h in 1..=horizon {
                    let forecast = if h == 1 {
                        self.params.omega
                            + self.params.alpha * last_return_sq
                            + self.params.beta * prev_var
                    } else {
                        self.params.omega + (self.params.alpha + self.params.beta) * prev_var
                    };
                    forecasts.push(forecast.sqrt());
                    prev_var = forecast;
                }
            }
        }

        Ok(forecasts)
    }

    /// Get the unconditional variance (long-run variance)
    pub fn unconditional_variance(&self) -> Result<f64> {
        match self.variant {
            GarchVariant::Standard | GarchVariant::GJRGARCH => {
                let persistence = self.params.alpha + self.params.beta;
                if persistence >= 1.0 {
                    return Err(DervflowError::NumericalError(
                        "model is not stationary (alpha + beta >= 1)".to_string(),
                    ));
                }
                Ok(self.params.omega / (1.0 - persistence))
            }
            GarchVariant::EGARCH => {
                // EGARCH unconditional variance is more complex
                // Return approximate value
                Ok(self.params.omega.exp())
            }
        }
    }

    /// Get model information criteria (AIC, BIC)
    pub fn information_criteria(&self, n_obs: usize) -> (f64, f64) {
        let n_params = match self.variant {
            GarchVariant::Standard => 3.0, // omega, alpha, beta
            _ => 4.0,                      // omega, alpha, beta, gamma
        };

        let aic = -2.0 * self.log_likelihood + 2.0 * n_params;
        let bic = -2.0 * self.log_likelihood + n_params * (n_obs as f64).ln();

        (aic, bic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_garch_params_validation() {
        let valid = GarchParams::new(0.01, 0.1, 0.8, 0.0);
        assert!(valid.validate_standard().is_ok());

        let invalid_omega = GarchParams::new(-0.01, 0.1, 0.8, 0.0);
        assert!(invalid_omega.validate_standard().is_err());

        let invalid_alpha = GarchParams::new(0.01, -0.1, 0.8, 0.0);
        assert!(invalid_alpha.validate_standard().is_err());

        let non_stationary = GarchParams::new(0.01, 0.5, 0.6, 0.0);
        assert!(non_stationary.validate_standard().is_err());
    }

    #[test]
    fn test_garch_conditional_variances() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let params = GarchParams::new(0.0001, 0.1, 0.85, 0.0);

        let variances =
            GarchModel::calculate_conditional_variances(&returns, &params, GarchVariant::Standard);

        assert_eq!(variances.len(), returns.len());
        assert!(variances.iter().all(|&v| v > 0.0 && v.is_finite()));
    }

    #[test]
    fn test_garch_fit_standard() {
        // Generate synthetic GARCH data
        let mut returns = Vec::new();
        let true_params = GarchParams::new(0.0001, 0.1, 0.85, 0.0);
        let mut variance: f64 = 0.0004;

        for _ in 0..100 {
            let z = rand::random::<f64>() - 0.5;
            let ret = variance.sqrt() * z;
            returns.push(ret);
            variance =
                true_params.omega + true_params.alpha * ret.powi(2) + true_params.beta * variance;
        }

        let model = GarchModel::fit(&returns, GarchVariant::Standard);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.conditional_variances.len(), returns.len());
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_garch_forecast() {
        let returns = vec![
            0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.008, -0.012, 0.018, -0.005,
        ];
        let model = GarchModel::fit(&returns, GarchVariant::Standard).unwrap();

        let forecasts = model.forecast(5);
        assert!(forecasts.is_ok());

        let forecasts = forecasts.unwrap();
        assert_eq!(forecasts.len(), 5);
        assert!(forecasts.iter().all(|&f| f > 0.0 && f.is_finite()));
    }

    #[test]
    fn test_garch_unconditional_variance() {
        let params = GarchParams::new(0.0001, 0.1, 0.85, 0.0);
        let model = GarchModel::new(params, GarchVariant::Standard);

        let uncond_var = model.unconditional_variance();
        assert!(uncond_var.is_ok());

        let uncond_var = uncond_var.unwrap();
        let expected = 0.0001 / (1.0 - 0.1 - 0.85);
        assert_relative_eq!(uncond_var, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_gjr_garch_asymmetry() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let params = GarchParams::new(0.0001, 0.1, 0.8, 0.05);

        let variances =
            GarchModel::calculate_conditional_variances(&returns, &params, GarchVariant::GJRGARCH);

        assert_eq!(variances.len(), returns.len());
        assert!(variances.iter().all(|&v| v > 0.0 && v.is_finite()));
    }

    #[test]
    fn test_egarch() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let params = GarchParams::new(0.0001, 0.1, 0.9, -0.05);

        let variances =
            GarchModel::calculate_conditional_variances(&returns, &params, GarchVariant::EGARCH);

        assert_eq!(variances.len(), returns.len());
        assert!(variances.iter().all(|&v| v > 0.0 && v.is_finite()));
    }

    #[test]
    fn test_information_criteria() {
        let returns = vec![
            0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.008, -0.012, 0.018, -0.005,
        ];
        let model = GarchModel::fit(&returns, GarchVariant::Standard).unwrap();

        let (aic, bic) = model.information_criteria(returns.len());
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(bic > aic); // BIC penalizes complexity more
    }
}
