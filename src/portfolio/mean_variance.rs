// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Mean-variance optimization
//!
//! Implements Markowitz mean-variance portfolio optimization using quadratic programming.
//! Solves the optimization problem:
//!   minimize: (1/2) * w^T * Σ * w - λ * μ^T * w
//!   subject to: sum(w) = 1 (budget constraint)
//!               w_min <= w <= w_max (box constraints)
//!
//! where:
//!   w = portfolio weights
//!   Σ = covariance matrix
//!   μ = expected returns
//!   λ = risk aversion parameter

use crate::common::error::{DervflowError, Result};
use crate::risk::portfolio_risk::{
    PortfolioSummary, portfolio_parametric_cvar, portfolio_parametric_var, portfolio_summary,
    risk_contributions as portfolio_risk_contributions,
};
use nalgebra::{DMatrix, DVector};
use osqp::{CscMatrix, Problem, Settings};

/// Optimization target for mean-variance optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationTarget {
    /// Minimize variance for a given target return
    MinimizeVariance { target_return: f64 },
    /// Maximize return for a given target variance
    MaximizeReturn { target_variance: f64 },
    /// Maximize Sharpe ratio (risk-adjusted return)
    MaximizeSharpeRatio { risk_free_rate: f64 },
    /// Minimize variance without return constraint
    MinimumVariance,
}

/// Result of portfolio optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal portfolio weights
    pub weights: Vec<f64>,
    /// Expected portfolio return
    pub expected_return: f64,
    /// Portfolio volatility (standard deviation)
    pub volatility: f64,
    /// Sharpe ratio (if risk-free rate provided)
    pub sharpe_ratio: Option<f64>,
    /// Optimization status
    pub status: String,
}

impl OptimizationResult {
    /// Create a new OptimizationResult
    pub fn new(
        weights: Vec<f64>,
        expected_return: f64,
        volatility: f64,
        sharpe_ratio: Option<f64>,
        status: String,
    ) -> Self {
        Self {
            weights,
            expected_return,
            volatility,
            sharpe_ratio,
            status,
        }
    }

    /// Calculate portfolio metrics from weights
    pub fn from_weights(
        weights: Vec<f64>,
        expected_returns: &[f64],
        covariance: &DMatrix<f64>,
        risk_free_rate: Option<f64>,
        status: String,
    ) -> Self {
        let expected_return = calculate_portfolio_return(&weights, expected_returns);
        let volatility = calculate_portfolio_volatility(&weights, covariance);
        let sharpe_ratio = risk_free_rate
            .map(|rf| calculate_portfolio_sharpe_ratio(&weights, expected_returns, covariance, rf));

        Self::new(weights, expected_return, volatility, sharpe_ratio, status)
    }
}

/// Mean-variance optimizer using quadratic programming
pub struct MeanVarianceOptimizer {
    /// Expected returns for each asset
    expected_returns: Vec<f64>,
    /// Covariance matrix
    covariance: DMatrix<f64>,
    /// Number of assets
    n_assets: usize,
}

impl MeanVarianceOptimizer {
    /// Create a new mean-variance optimizer
    ///
    /// # Arguments
    /// * `expected_returns` - Expected returns for each asset
    /// * `covariance` - Covariance matrix (n_assets x n_assets)
    ///
    /// # Returns
    /// Result containing the optimizer or an error
    pub fn new(expected_returns: Vec<f64>, covariance: DMatrix<f64>) -> Result<Self> {
        let n_assets = expected_returns.len();

        // Validate inputs
        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "Expected returns cannot be empty".to_string(),
            ));
        }

        if covariance.nrows() != n_assets || covariance.ncols() != n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Covariance matrix dimensions ({} x {}) do not match number of assets ({})",
                covariance.nrows(),
                covariance.ncols(),
                n_assets
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
            expected_returns,
            covariance,
            n_assets,
        })
    }

    /// Optimize portfolio for a given target
    ///
    /// # Arguments
    /// * `target` - Optimization target (minimize variance, maximize return, etc.)
    /// * `min_weights` - Minimum weight for each asset (optional, default 0)
    /// * `max_weights` - Maximum weight for each asset (optional, default 1)
    ///
    /// # Returns
    /// Result containing the optimization result or an error
    pub fn optimize(
        &self,
        target: OptimizationTarget,
        min_weights: Option<&[f64]>,
        max_weights: Option<&[f64]>,
    ) -> Result<OptimizationResult> {
        // Set default bounds if not provided
        let default_min = vec![0.0; self.n_assets];
        let default_max = vec![1.0; self.n_assets];
        let min_w = min_weights.unwrap_or(&default_min);
        let max_w = max_weights.unwrap_or(&default_max);

        // Validate bounds
        if min_w.len() != self.n_assets || max_w.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(
                "Weight bounds must match number of assets".to_string(),
            ));
        }

        for i in 0..self.n_assets {
            if min_w[i] > max_w[i] {
                return Err(DervflowError::InvalidInput(format!(
                    "Minimum weight ({}) exceeds maximum weight ({}) for asset {}",
                    min_w[i], max_w[i], i
                )));
            }
        }

        match target {
            OptimizationTarget::MinimizeVariance { target_return } => {
                self.minimize_variance_with_target_return(target_return, min_w, max_w)
            }
            OptimizationTarget::MaximizeReturn { target_variance } => {
                self.maximize_return_with_target_variance(target_variance, min_w, max_w)
            }
            OptimizationTarget::MaximizeSharpeRatio { risk_free_rate } => {
                self.maximize_sharpe_ratio(risk_free_rate, min_w, max_w)
            }
            OptimizationTarget::MinimumVariance => {
                self.minimize_variance_unconstrained(min_w, max_w)
            }
        }
    }

    /// Minimize portfolio variance for a given target return
    fn minimize_variance_with_target_return(
        &self,
        target_return: f64,
        min_weights: &[f64],
        max_weights: &[f64],
    ) -> Result<OptimizationResult> {
        // Set up QP problem: minimize (1/2) * w^T * Σ * w
        // subject to: μ^T * w = target_return
        //             sum(w) = 1
        //             min_weights <= w <= max_weights

        let n = self.n_assets;

        // Objective: P = Σ (covariance matrix), q = 0
        let p_matrix = self.covariance_to_csc();
        let q = vec![0.0; n];

        // Constraints: A * w <= u and A * w >= l
        // Row 0: sum(w) = 1 (budget constraint)
        // Row 1: μ^T * w = target_return (return constraint)
        // Rows 2..n+2: identity matrix for variable bounds
        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        // Build constraint matrix in CSC format (column by column)
        for j in 0..n {
            // Budget constraint: coefficient is 1 for all assets
            a_data.push(1.0);
            a_indices.push(0);

            // Return constraint: coefficient is expected return
            a_data.push(self.expected_returns[j]);
            a_indices.push(1);

            // Variable bound: identity matrix entry
            a_data.push(1.0);
            a_indices.push(2 + j);

            a_indptr.push(a_data.len());
        }

        let a_matrix = CscMatrix {
            nrows: 2 + n,
            ncols: n,
            indptr: a_indptr.into(),
            indices: a_indices.into(),
            data: a_data.into(),
        };

        // Constraint bounds
        let mut l = vec![1.0, target_return]; // Equality constraints
        let mut u = vec![1.0, target_return];

        // Add variable bounds
        l.extend_from_slice(min_weights);
        u.extend_from_slice(max_weights);

        // Solve QP problem
        let settings = Settings::default()
            .verbose(false)
            .eps_abs(1e-8)
            .eps_rel(1e-8);

        let mut problem = Problem::new(p_matrix, &q, a_matrix, &l, &u, &settings).map_err(|e| {
            DervflowError::OptimizationInfeasible(format!("Failed to create problem: {}", e))
        })?;

        let result = problem.solve();

        if !matches!(result, osqp::Status::Solved(_)) {
            return Err(DervflowError::OptimizationInfeasible(format!(
                "Optimization failed: {:?}. The target return may be infeasible given the constraints.",
                result
            )));
        }

        let weights = result
            .x()
            .ok_or_else(|| DervflowError::OptimizationInfeasible("No solution found".to_string()))?
            .to_vec();
        Ok(OptimizationResult::from_weights(
            weights,
            &self.expected_returns,
            &self.covariance,
            None,
            format!("{:?}", result),
        ))
    }

    /// Maximize portfolio return for a given target variance
    fn maximize_return_with_target_variance(
        &self,
        target_variance: f64,
        min_weights: &[f64],
        max_weights: &[f64],
    ) -> Result<OptimizationResult> {
        // This is more complex as variance constraint is quadratic
        // We'll use a binary search approach to find the optimal portfolio
        // by varying the target return until we hit the target variance

        if target_variance < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Target variance must be non-negative".to_string(),
            ));
        }

        let target_volatility = target_variance.sqrt();

        // Find the range of feasible returns
        let min_return = self
            .expected_returns
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_return = self
            .expected_returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Binary search for the return that gives us the target volatility
        let mut low = min_return;
        let mut high = max_return;
        let mut best_result: Option<OptimizationResult> = None;
        let tolerance = 1e-6;
        let max_iterations = 50;

        for _ in 0..max_iterations {
            let mid = (low + high) / 2.0;

            match self.minimize_variance_with_target_return(mid, min_weights, max_weights) {
                Ok(result) => {
                    let vol_diff = result.volatility - target_volatility;

                    if vol_diff.abs() < tolerance {
                        return Ok(result);
                    }

                    best_result = Some(result.clone());

                    if vol_diff < 0.0 {
                        // Current volatility is too low, need higher return
                        low = mid;
                    } else {
                        // Current volatility is too high, need lower return
                        high = mid;
                    }
                }
                Err(_) => {
                    // Infeasible at this return level
                    high = mid;
                }
            }

            if (high - low).abs() < tolerance {
                break;
            }
        }

        best_result.ok_or_else(|| {
            DervflowError::OptimizationInfeasible(
                "Could not find portfolio with target variance".to_string(),
            )
        })
    }

    /// Maximize Sharpe ratio
    fn maximize_sharpe_ratio(
        &self,
        risk_free_rate: f64,
        min_weights: &[f64],
        max_weights: &[f64],
    ) -> Result<OptimizationResult> {
        // Maximizing Sharpe ratio is equivalent to solving:
        // maximize: (μ - rf)^T * w / sqrt(w^T * Σ * w)
        //
        // This can be reformulated as a QP problem by introducing auxiliary variables
        // We'll use a simpler approach: maximize excess return for unit variance

        let n = self.n_assets;

        // Compute excess returns
        let excess_returns: Vec<f64> = self
            .expected_returns
            .iter()
            .map(|&r| r - risk_free_rate)
            .collect();

        // Set up QP problem: minimize (1/2) * w^T * Σ * w - λ * (μ - rf)^T * w
        // We'll use a large λ to emphasize return maximization
        // Then normalize to unit variance

        // First, find the maximum return portfolio
        let p_matrix = self.covariance_to_csc();

        // Objective: minimize variance, maximize excess return
        // We use negative excess returns in q to maximize
        let lambda = 1.0; // Risk aversion parameter
        let q: Vec<f64> = excess_returns.iter().map(|&r| -lambda * r).collect();

        // Constraints: sum(w) = 1 and variable bounds
        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        for j in 0..n {
            // Budget constraint
            a_data.push(1.0);
            a_indices.push(0);

            // Variable bound: identity matrix entry
            a_data.push(1.0);
            a_indices.push(1 + j);

            a_indptr.push(a_data.len());
        }

        let a_matrix = CscMatrix {
            nrows: 1 + n,
            ncols: n,
            indptr: a_indptr.into(),
            indices: a_indices.into(),
            data: a_data.into(),
        };

        let mut l = vec![1.0];
        let mut u = vec![1.0];

        // Add variable bounds
        l.extend_from_slice(min_weights);
        u.extend_from_slice(max_weights);

        // Solve QP problem
        let settings = Settings::default()
            .verbose(false)
            .eps_abs(1e-8)
            .eps_rel(1e-8);

        let mut problem = Problem::new(p_matrix, &q, a_matrix, &l, &u, &settings).map_err(|e| {
            DervflowError::OptimizationInfeasible(format!("Failed to create problem: {}", e))
        })?;

        let result = problem.solve();

        if !matches!(result, osqp::Status::Solved(_)) {
            return Err(DervflowError::OptimizationInfeasible(format!(
                "Optimization failed: {:?}",
                result
            )));
        }

        let weights = result
            .x()
            .ok_or_else(|| DervflowError::OptimizationInfeasible("No solution found".to_string()))?
            .to_vec();
        Ok(OptimizationResult::from_weights(
            weights,
            &self.expected_returns,
            &self.covariance,
            Some(risk_free_rate),
            format!("{:?}", result),
        ))
    }

    /// Minimize portfolio variance without return constraint
    fn minimize_variance_unconstrained(
        &self,
        min_weights: &[f64],
        max_weights: &[f64],
    ) -> Result<OptimizationResult> {
        let n = self.n_assets;

        // Set up QP problem: minimize (1/2) * w^T * Σ * w
        // subject to: sum(w) = 1
        //             min_weights <= w <= max_weights

        let p_matrix = self.covariance_to_csc();
        let q = vec![0.0; n];

        // Constraint: sum(w) = 1 and variable bounds
        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        for j in 0..n {
            // Budget constraint
            a_data.push(1.0);
            a_indices.push(0);

            // Variable bound: identity matrix entry
            a_data.push(1.0);
            a_indices.push(1 + j);

            a_indptr.push(a_data.len());
        }

        let a_matrix = CscMatrix {
            nrows: 1 + n,
            ncols: n,
            indptr: a_indptr.into(),
            indices: a_indices.into(),
            data: a_data.into(),
        };

        let mut l = vec![1.0];
        let mut u = vec![1.0];

        // Add variable bounds
        l.extend_from_slice(min_weights);
        u.extend_from_slice(max_weights);

        // Solve QP problem
        let settings = Settings::default()
            .verbose(false)
            .eps_abs(1e-8)
            .eps_rel(1e-8);

        let mut problem = Problem::new(p_matrix, &q, a_matrix, &l, &u, &settings).map_err(|e| {
            DervflowError::OptimizationInfeasible(format!("Failed to create problem: {}", e))
        })?;

        let result = problem.solve();

        if !matches!(result, osqp::Status::Solved(_)) {
            return Err(DervflowError::OptimizationInfeasible(format!(
                "Optimization failed: {:?}",
                result
            )));
        }

        let weights = result
            .x()
            .ok_or_else(|| DervflowError::OptimizationInfeasible("No solution found".to_string()))?
            .to_vec();
        Ok(OptimizationResult::from_weights(
            weights,
            &self.expected_returns,
            &self.covariance,
            None,
            format!("{:?}", result),
        ))
    }

    /// Convert covariance matrix to CSC format for OSQP
    fn covariance_to_csc(&self) -> CscMatrix<'_> {
        let n = self.n_assets;
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        // Convert to CSC format (column by column)
        // OSQP expects upper triangular part only for symmetric matrices
        for j in 0..n {
            for i in 0..=j {
                let val = self.covariance[(i, j)];
                if val.abs() > 1e-15 {
                    data.push(val);
                    indices.push(i);
                }
            }
            indptr.push(data.len());
        }

        CscMatrix {
            nrows: n,
            ncols: n,
            indptr: indptr.into(),
            indices: indices.into(),
            data: data.into(),
        }
    }

    /// Compute the Sharpe ratio for a given set of weights and risk-free rate.
    pub fn sharpe_ratio(&self, weights: &[f64], risk_free_rate: f64) -> Result<f64> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Weight vector length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets
            )));
        }

        Ok(calculate_portfolio_sharpe_ratio(
            weights,
            &self.expected_returns,
            &self.covariance,
            risk_free_rate,
        ))
    }

    /// Compute percentage risk contributions for the supplied weights.
    pub fn risk_contributions(&self, weights: &[f64]) -> Result<Vec<f64>> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Weight vector length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets
            )));
        }

        let (_, _, percentage) = portfolio_risk_contributions(weights, &self.covariance)?;
        Ok(percentage)
    }

    /// Parametric (variance-covariance) Value at Risk for the portfolio.
    pub fn value_at_risk(&self, weights: &[f64], confidence_level: f64) -> Result<f64> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Weight vector length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets
            )));
        }

        portfolio_parametric_var(
            weights,
            &self.covariance,
            Some(&self.expected_returns),
            confidence_level,
        )
    }

    /// Parametric Conditional Value at Risk (Expected Shortfall) for the portfolio.
    pub fn conditional_value_at_risk(&self, weights: &[f64], confidence_level: f64) -> Result<f64> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Weight vector length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets
            )));
        }

        portfolio_parametric_cvar(
            weights,
            &self.covariance,
            Some(&self.expected_returns),
            confidence_level,
        )
    }

    /// Produce a comprehensive portfolio risk summary for the supplied weights.
    pub fn portfolio_summary(
        &self,
        weights: &[f64],
        risk_free_rate: Option<f64>,
    ) -> Result<PortfolioSummary> {
        if weights.len() != self.n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Weight vector length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets
            )));
        }

        portfolio_summary(
            weights,
            &self.covariance,
            Some(&self.expected_returns),
            risk_free_rate,
        )
    }
}

/// Calculate portfolio expected return
pub fn calculate_portfolio_return(weights: &[f64], expected_returns: &[f64]) -> f64 {
    weights
        .iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum()
}

/// Calculate portfolio volatility (standard deviation)
pub fn calculate_portfolio_volatility(weights: &[f64], covariance: &DMatrix<f64>) -> f64 {
    let w = DVector::from_vec(weights.to_vec());
    let variance = w.transpose() * covariance * &w;
    variance[(0, 0)].sqrt()
}

/// Calculate the portfolio Sharpe ratio given weights and a risk-free rate.
pub fn calculate_portfolio_sharpe_ratio(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &DMatrix<f64>,
    risk_free_rate: f64,
) -> f64 {
    let expected_return = calculate_portfolio_return(weights, expected_returns);
    let volatility = calculate_portfolio_volatility(weights, covariance);

    if volatility > 1e-10 {
        (expected_return - risk_free_rate) / volatility
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_data() -> (Vec<f64>, DMatrix<f64>) {
        let expected_returns = vec![0.10, 0.12, 0.08];
        let covariance = DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.01, 0.005, 0.01, 0.09, 0.01, 0.005, 0.01, 0.0225],
        );
        (expected_returns, covariance)
    }

    #[test]
    fn test_optimizer_creation() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_optimizer_invalid_dimensions() {
        let returns = vec![0.10, 0.12];
        let cov = DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.01, 0.005, 0.01, 0.09, 0.01, 0.005, 0.01, 0.0225],
        );
        let optimizer = MeanVarianceOptimizer::new(returns, cov);
        assert!(optimizer.is_err());
    }

    #[test]
    fn test_minimum_variance_portfolio() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();

        let result = optimizer.optimize(OptimizationTarget::MinimumVariance, None, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.weights.len(), 3);

        // Weights should sum to 1
        let sum: f64 = result.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All weights should be non-negative (default bounds)
        for &w in &result.weights {
            assert!(w >= -1e-6);
        }
    }

    #[test]
    fn test_minimize_variance_with_target_return() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();

        let target_return = 0.10;
        let result = optimizer.optimize(
            OptimizationTarget::MinimizeVariance { target_return },
            None,
            None,
        );
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_relative_eq!(result.expected_return, target_return, epsilon = 1e-4);

        let sum: f64 = result.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_maximize_sharpe_ratio() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();

        let risk_free_rate = 0.03;
        let result = optimizer.optimize(
            OptimizationTarget::MaximizeSharpeRatio { risk_free_rate },
            None,
            None,
        );
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.sharpe_ratio.is_some());
        assert!(result.sharpe_ratio.unwrap() > 0.0);

        let sum: f64 = result.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_box_constraints() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();

        let min_weights = vec![0.1, 0.1, 0.1];
        let max_weights = vec![0.5, 0.5, 0.5];

        let result = optimizer.optimize(
            OptimizationTarget::MinimumVariance,
            Some(&min_weights),
            Some(&max_weights),
        );
        assert!(result.is_ok());

        let result = result.unwrap();
        for i in 0..3 {
            assert!(result.weights[i] >= min_weights[i] - 1e-6);
            assert!(result.weights[i] <= max_weights[i] + 1e-6);
        }
    }

    #[test]
    fn test_infeasible_target_return() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();

        // Target return higher than maximum possible
        let target_return = 0.20;
        let result = optimizer.optimize(
            OptimizationTarget::MinimizeVariance { target_return },
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_portfolio_return_calculation() {
        let weights = vec![0.3, 0.4, 0.3];
        let returns = vec![0.10, 0.12, 0.08];
        let portfolio_return = calculate_portfolio_return(&weights, &returns);
        assert_relative_eq!(portfolio_return, 0.102, epsilon = 1e-6);
    }

    #[test]
    fn test_portfolio_volatility_calculation() {
        let weights = vec![0.3, 0.4, 0.3];
        let cov = DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.01, 0.005, 0.01, 0.09, 0.01, 0.005, 0.01, 0.0225],
        );
        let volatility = calculate_portfolio_volatility(&weights, &cov);
        assert!(volatility > 0.0);
        assert!(volatility < 0.3); // Reasonable range
    }

    #[test]
    fn test_risk_contributions_sum_to_one() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();
        let weights = vec![0.4, 0.3, 0.3];
        let contributions = optimizer.risk_contributions(&weights).unwrap();

        assert_eq!(contributions.len(), 3);
        let sum: f64 = contributions.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        assert!(contributions.iter().all(|value| *value >= -1e-12));
    }

    #[test]
    fn test_value_at_risk_and_cvar() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns, cov).unwrap();
        let weights = vec![0.5, 0.3, 0.2];

        let var = optimizer.value_at_risk(&weights, 0.95).unwrap();
        let cvar = optimizer.conditional_value_at_risk(&weights, 0.95).unwrap();

        assert!(var >= 0.0);
        assert!(cvar >= var);
    }

    #[test]
    fn test_portfolio_summary_matches_metrics() {
        let (returns, cov) = create_test_data();
        let optimizer = MeanVarianceOptimizer::new(returns.clone(), cov.clone()).unwrap();
        let weights = vec![0.3, 0.4, 0.3];
        let summary = optimizer.portfolio_summary(&weights, Some(0.02)).unwrap();

        let expected_return = calculate_portfolio_return(&weights, &returns);
        let volatility = calculate_portfolio_volatility(&weights, &cov);

        assert_relative_eq!(
            summary.expected_return.unwrap(),
            expected_return,
            epsilon = 1e-12
        );
        assert_relative_eq!(summary.volatility, volatility, epsilon = 1e-12);
        assert!(summary.sharpe_ratio.unwrap() > 0.0);
        assert!(summary.diversification_ratio >= 0.0);
    }
}
