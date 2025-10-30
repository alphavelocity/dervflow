// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Risk parity allocation
//!
//! Implements risk parity portfolio allocation where each asset contributes
//! equally to the total portfolio risk.
//!
//! Risk contribution of asset i: RC_i = w_i * (Σw)_i / sqrt(w^T * Σ * w)
//! Risk parity condition: RC_i = RC_j for all i, j

use crate::common::error::{DervflowError, Result};
use nalgebra::{DMatrix, DVector};

/// Risk parity optimizer
pub struct RiskParityOptimizer {
    /// Covariance matrix
    covariance: DMatrix<f64>,
    /// Number of assets
    n_assets: usize,
}

impl RiskParityOptimizer {
    /// Create a new risk parity optimizer
    ///
    /// # Arguments
    /// * `covariance` - Covariance matrix (n_assets x n_assets)
    ///
    /// # Returns
    /// Result containing the optimizer or an error
    pub fn new(covariance: DMatrix<f64>) -> Result<Self> {
        let n_assets = covariance.nrows();

        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "Covariance matrix cannot be empty".to_string(),
            ));
        }

        if covariance.ncols() != n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Covariance matrix must be square, got {} x {}",
                n_assets,
                covariance.ncols()
            )));
        }

        // Check if covariance matrix is symmetric
        for i in 0..n_assets {
            for j in i + 1..n_assets {
                if (covariance[(i, j)] - covariance[(j, i)]).abs() > 1e-10 {
                    return Err(DervflowError::InvalidInput(
                        "Covariance matrix must be symmetric".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            covariance,
            n_assets,
        })
    }

    /// Optimize portfolio using risk parity approach
    ///
    /// Uses iterative algorithm to find weights where each asset contributes
    /// equally to portfolio risk.
    ///
    /// # Arguments
    /// * `target_risk_contributions` - Optional target risk contributions (default: equal)
    /// * `max_iterations` - Maximum number of iterations (default: 1000)
    /// * `tolerance` - Convergence tolerance (default: 1e-8)
    ///
    /// # Returns
    /// Result containing the optimal weights or an error
    pub fn optimize(
        &self,
        target_risk_contributions: Option<&[f64]>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> Result<Vec<f64>> {
        let max_iter = max_iterations.unwrap_or(1000);
        let tol = tolerance.unwrap_or(1e-8);

        // Set target risk contributions (default: equal)
        let target_rc = if let Some(rc) = target_risk_contributions {
            if rc.len() != self.n_assets {
                return Err(DervflowError::InvalidInput(format!(
                    "Target risk contributions length ({}) does not match number of assets ({})",
                    rc.len(),
                    self.n_assets
                )));
            }

            let sum: f64 = rc.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(DervflowError::InvalidInput(format!(
                    "Target risk contributions must sum to 1, got {}",
                    sum
                )));
            }

            rc.to_vec()
        } else {
            vec![1.0 / self.n_assets as f64; self.n_assets]
        };

        // Initialize weights (equal weights)
        let mut weights = vec![1.0 / self.n_assets as f64; self.n_assets];

        // Iterative algorithm to find risk parity weights
        for iteration in 0..max_iter {
            let risk_contributions = self.calculate_risk_contributions(&weights)?;

            // Calculate adjustment factors
            let mut new_weights = Vec::with_capacity(self.n_assets);
            for i in 0..self.n_assets {
                // Adjust weight based on ratio of target to actual risk contribution
                let adjustment = if risk_contributions[i] > 1e-10 {
                    (target_rc[i] / risk_contributions[i]).sqrt()
                } else {
                    1.0
                };
                new_weights.push(weights[i] * adjustment);
            }

            // Normalize weights to sum to 1
            let sum: f64 = new_weights.iter().sum();
            for w in &mut new_weights {
                *w /= sum;
            }

            // Check convergence
            let max_change = weights
                .iter()
                .zip(new_weights.iter())
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f64::max);

            weights = new_weights;

            if max_change < tol {
                return Ok(weights);
            }

            // Additional convergence check: risk contributions close to target
            let risk_contributions = self.calculate_risk_contributions(&weights)?;
            let max_rc_diff = risk_contributions
                .iter()
                .zip(target_rc.iter())
                .map(|(rc, target)| (rc - target).abs())
                .fold(0.0, f64::max);

            if max_rc_diff < tol {
                return Ok(weights);
            }

            if iteration == max_iter - 1 {
                return Err(DervflowError::ConvergenceFailure {
                    iterations: max_iter,
                    error: max_change,
                });
            }
        }

        Ok(weights)
    }

    /// Calculate risk contributions for given weights
    ///
    /// Risk contribution of asset i: RC_i = w_i * (Σw)_i / sqrt(w^T * Σ * w)
    ///
    /// # Arguments
    /// * `weights` - Portfolio weights
    ///
    /// # Returns
    /// Vector of risk contributions (sum to 1)
    pub fn calculate_risk_contributions(&self, weights: &[f64]) -> Result<Vec<f64>> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Weights length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets
            )));
        }

        let w = DVector::from_vec(weights.to_vec());

        // Calculate portfolio variance: w^T * Σ * w
        let portfolio_variance = w.transpose() * &self.covariance * &w;
        let portfolio_variance = portfolio_variance[(0, 0)];

        if portfolio_variance < 1e-15 {
            return Err(DervflowError::NumericalError(
                "Portfolio variance is too small".to_string(),
            ));
        }

        let portfolio_volatility = portfolio_variance.sqrt();

        // Calculate marginal risk contributions: Σ * w
        let marginal_contributions = &self.covariance * &w;

        // Calculate risk contributions: w_i * (Σw)_i / σ_p
        let mut risk_contributions = Vec::with_capacity(self.n_assets);
        for i in 0..self.n_assets {
            let rc = weights[i] * marginal_contributions[i] / portfolio_volatility;
            risk_contributions.push(rc);
        }

        // Normalize to sum to 1
        let total_rc: f64 = risk_contributions.iter().sum();
        if total_rc > 1e-15 {
            for rc in &mut risk_contributions {
                *rc /= total_rc;
            }
        }

        Ok(risk_contributions)
    }

    /// Calculate portfolio volatility for given weights
    pub fn calculate_volatility(&self, weights: &[f64]) -> Result<f64> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(
                "Weights length does not match number of assets".to_string(),
            ));
        }

        let w = DVector::from_vec(weights.to_vec());
        let variance = w.transpose() * &self.covariance * &w;
        Ok(variance[(0, 0)].sqrt())
    }
}

/// Calculate equal risk contribution (ERC) portfolio
///
/// This is a convenience function that creates a RiskParityOptimizer and
/// optimizes for equal risk contributions.
///
/// # Arguments
/// * `covariance` - Covariance matrix
///
/// # Returns
/// Optimal weights for equal risk contribution portfolio
pub fn equal_risk_contribution(covariance: DMatrix<f64>) -> Result<Vec<f64>> {
    let optimizer = RiskParityOptimizer::new(covariance)?;
    optimizer.optimize(None, None, None)
}

/// Calculate risk parity portfolio with custom target risk contributions
///
/// # Arguments
/// * `covariance` - Covariance matrix
/// * `target_risk_contributions` - Target risk contributions for each asset (must sum to 1)
///
/// # Returns
/// Optimal weights for the specified risk contribution targets
pub fn risk_parity_with_targets(
    covariance: DMatrix<f64>,
    target_risk_contributions: &[f64],
) -> Result<Vec<f64>> {
    let optimizer = RiskParityOptimizer::new(covariance)?;
    optimizer.optimize(Some(target_risk_contributions), None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_covariance() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.01, 0.005, 0.01, 0.09, 0.01, 0.005, 0.01, 0.0225],
        )
    }

    #[test]
    fn test_optimizer_creation() {
        let cov = create_test_covariance();
        let optimizer = RiskParityOptimizer::new(cov);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_optimizer_invalid_covariance() {
        let cov = DMatrix::from_row_slice(2, 3, &[0.04, 0.01, 0.005, 0.01, 0.09, 0.01]);
        let optimizer = RiskParityOptimizer::new(cov);
        assert!(optimizer.is_err());
    }

    #[test]
    fn test_equal_risk_contribution() {
        let cov = create_test_covariance();
        let weights = equal_risk_contribution(cov);
        assert!(weights.is_ok());

        let weights = weights.unwrap();
        assert_eq!(weights.len(), 3);

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All weights should be positive
        for &w in &weights {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_risk_contributions_equal() {
        let cov = create_test_covariance();
        let optimizer = RiskParityOptimizer::new(cov).unwrap();
        let weights = optimizer.optimize(None, None, None).unwrap();

        let risk_contributions = optimizer.calculate_risk_contributions(&weights).unwrap();

        // Risk contributions should be approximately equal
        let target = 1.0 / 3.0;
        for &rc in &risk_contributions {
            assert_relative_eq!(rc, target, epsilon = 1e-4);
        }

        // Risk contributions should sum to 1
        let sum: f64 = risk_contributions.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_risk_parity_with_custom_targets() {
        let cov = create_test_covariance();
        let target_rc = vec![0.5, 0.3, 0.2];

        let weights = risk_parity_with_targets(cov.clone(), &target_rc);
        assert!(weights.is_ok());

        let weights = weights.unwrap();
        let optimizer = RiskParityOptimizer::new(cov).unwrap();
        let risk_contributions = optimizer.calculate_risk_contributions(&weights).unwrap();

        // Risk contributions should match targets
        for i in 0..3 {
            assert_relative_eq!(risk_contributions[i], target_rc[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_invalid_target_risk_contributions() {
        let cov = create_test_covariance();
        let optimizer = RiskParityOptimizer::new(cov).unwrap();

        // Target RC that doesn't sum to 1
        let invalid_target = vec![0.5, 0.3, 0.3];
        let result = optimizer.optimize(Some(&invalid_target), None, None);
        assert!(result.is_err());

        // Wrong length
        let invalid_target = vec![0.5, 0.5];
        let result = optimizer.optimize(Some(&invalid_target), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_volatility() {
        let cov = create_test_covariance();
        let optimizer = RiskParityOptimizer::new(cov).unwrap();

        let weights = vec![0.3, 0.4, 0.3];
        let volatility = optimizer.calculate_volatility(&weights);
        assert!(volatility.is_ok());

        let volatility = volatility.unwrap();
        assert!(volatility > 0.0);
        assert!(volatility < 0.5); // Reasonable range
    }

    #[test]
    fn test_risk_contributions_calculation() {
        let cov = create_test_covariance();
        let optimizer = RiskParityOptimizer::new(cov).unwrap();

        let weights = vec![0.3, 0.4, 0.3];
        let risk_contributions = optimizer.calculate_risk_contributions(&weights);
        assert!(risk_contributions.is_ok());

        let risk_contributions = risk_contributions.unwrap();
        assert_eq!(risk_contributions.len(), 3);

        // Risk contributions should sum to 1
        let sum: f64 = risk_contributions.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All risk contributions should be non-negative
        for &rc in &risk_contributions {
            assert!(rc >= 0.0);
        }
    }

    #[test]
    fn test_convergence_with_max_iterations() {
        let cov = create_test_covariance();
        let optimizer = RiskParityOptimizer::new(cov).unwrap();

        // Very few iterations should still converge for simple case
        let result = optimizer.optimize(None, Some(100), Some(1e-6));
        assert!(result.is_ok());
    }

    #[test]
    fn test_two_asset_portfolio() {
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
        let weights = equal_risk_contribution(cov);
        assert!(weights.is_ok());

        let weights = weights.unwrap();
        assert_eq!(weights.len(), 2);

        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }
}
