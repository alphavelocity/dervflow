// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Efficient frontier calculation
//!
//! Computes the efficient frontier by generating multiple optimal portfolios
//! with varying target returns or risk levels.

use crate::common::error::{DervflowError, Result};
use crate::portfolio::mean_variance::{
    MeanVarianceOptimizer, OptimizationResult, OptimizationTarget,
};
use nalgebra::DMatrix;

/// Efficient frontier calculator
pub struct EfficientFrontier {
    /// Mean-variance optimizer
    optimizer: MeanVarianceOptimizer,
    /// Minimum feasible return
    min_return: f64,
    /// Maximum feasible return
    max_return: f64,
}

impl EfficientFrontier {
    /// Create a new efficient frontier calculator
    ///
    /// # Arguments
    /// * `expected_returns` - Expected returns for each asset
    /// * `covariance` - Covariance matrix
    ///
    /// # Returns
    /// Result containing the efficient frontier calculator or an error
    pub fn new(expected_returns: Vec<f64>, covariance: DMatrix<f64>) -> Result<Self> {
        let optimizer = MeanVarianceOptimizer::new(expected_returns.clone(), covariance)?;

        // Determine feasible return range
        let min_return = expected_returns
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_return = expected_returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(Self {
            optimizer,
            min_return,
            max_return,
        })
    }

    /// Generate efficient frontier points
    ///
    /// # Arguments
    /// * `num_points` - Number of points to generate along the frontier
    /// * `min_weights` - Minimum weight for each asset (optional)
    /// * `max_weights` - Maximum weight for each asset (optional)
    ///
    /// # Returns
    /// Vector of optimization results representing points on the efficient frontier
    pub fn generate(
        &self,
        num_points: usize,
        min_weights: Option<&[f64]>,
        max_weights: Option<&[f64]>,
    ) -> Result<Vec<OptimizationResult>> {
        if num_points == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of points must be at least 1".to_string(),
            ));
        }

        let mut frontier_points = Vec::with_capacity(num_points);

        // First, find the minimum variance portfolio
        let min_var_result = self.optimizer.optimize(
            OptimizationTarget::MinimumVariance,
            min_weights,
            max_weights,
        )?;

        // Use the minimum variance portfolio's return as the lower bound
        let min_frontier_return = min_var_result.expected_return;

        // Find a reasonable upper bound by trying to optimize for maximum return
        // We'll use the maximum expected return among assets as an upper bound
        let max_frontier_return = self.max_return;

        // Generate points along the frontier
        if num_points == 1 {
            frontier_points.push(min_var_result);
        } else {
            let return_step = (max_frontier_return - min_frontier_return) / (num_points - 1) as f64;

            for i in 0..num_points {
                let target_return = min_frontier_return + i as f64 * return_step;

                match self.optimizer.optimize(
                    OptimizationTarget::MinimizeVariance { target_return },
                    min_weights,
                    max_weights,
                ) {
                    Ok(result) => {
                        frontier_points.push(result);
                    }
                    Err(_) => {
                        // If optimization fails (infeasible), we've reached the end of the frontier
                        break;
                    }
                }
            }
        }

        if frontier_points.is_empty() {
            return Err(DervflowError::OptimizationInfeasible(
                "Could not generate any feasible frontier points".to_string(),
            ));
        }

        Ok(frontier_points)
    }

    /// Generate efficient frontier with custom return range
    ///
    /// # Arguments
    /// * `min_return` - Minimum target return
    /// * `max_return` - Maximum target return
    /// * `num_points` - Number of points to generate
    /// * `min_weights` - Minimum weight for each asset (optional)
    /// * `max_weights` - Maximum weight for each asset (optional)
    ///
    /// # Returns
    /// Vector of optimization results
    pub fn generate_with_range(
        &self,
        min_return: f64,
        max_return: f64,
        num_points: usize,
        min_weights: Option<&[f64]>,
        max_weights: Option<&[f64]>,
    ) -> Result<Vec<OptimizationResult>> {
        if num_points == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of points must be at least 1".to_string(),
            ));
        }

        if min_return >= max_return {
            return Err(DervflowError::InvalidInput(
                "min_return must be less than max_return".to_string(),
            ));
        }

        let mut frontier_points = Vec::with_capacity(num_points);
        let return_step = (max_return - min_return) / (num_points - 1) as f64;

        for i in 0..num_points {
            let target_return = min_return + i as f64 * return_step;

            match self.optimizer.optimize(
                OptimizationTarget::MinimizeVariance { target_return },
                min_weights,
                max_weights,
            ) {
                Ok(result) => {
                    frontier_points.push(result);
                }
                Err(_) => {
                    // Skip infeasible points
                    continue;
                }
            }
        }

        if frontier_points.is_empty() {
            return Err(DervflowError::OptimizationInfeasible(
                "Could not generate any feasible frontier points in the specified range"
                    .to_string(),
            ));
        }

        Ok(frontier_points)
    }

    /// Find the maximum Sharpe ratio portfolio on the efficient frontier
    ///
    /// # Arguments
    /// * `risk_free_rate` - Risk-free rate for Sharpe ratio calculation
    /// * `min_weights` - Minimum weight for each asset (optional)
    /// * `max_weights` - Maximum weight for each asset (optional)
    ///
    /// # Returns
    /// Optimization result for the maximum Sharpe ratio portfolio
    pub fn max_sharpe_portfolio(
        &self,
        risk_free_rate: f64,
        min_weights: Option<&[f64]>,
        max_weights: Option<&[f64]>,
    ) -> Result<OptimizationResult> {
        self.optimizer.optimize(
            OptimizationTarget::MaximizeSharpeRatio { risk_free_rate },
            min_weights,
            max_weights,
        )
    }

    /// Get the minimum variance portfolio
    ///
    /// # Arguments
    /// * `min_weights` - Minimum weight for each asset (optional)
    /// * `max_weights` - Maximum weight for each asset (optional)
    ///
    /// # Returns
    /// Optimization result for the minimum variance portfolio
    pub fn min_variance_portfolio(
        &self,
        min_weights: Option<&[f64]>,
        max_weights: Option<&[f64]>,
    ) -> Result<OptimizationResult> {
        self.optimizer.optimize(
            OptimizationTarget::MinimumVariance,
            min_weights,
            max_weights,
        )
    }

    /// Calculate the Capital Market Line (CML) points
    ///
    /// The CML represents portfolios that combine the risk-free asset with the market portfolio
    ///
    /// # Arguments
    /// * `risk_free_rate` - Risk-free rate
    /// * `num_points` - Number of points to generate
    /// * `min_weights` - Minimum weight for each asset (optional)
    /// * `max_weights` - Maximum weight for each asset (optional)
    ///
    /// # Returns
    /// Vector of (volatility, return) pairs representing the CML
    pub fn capital_market_line(
        &self,
        risk_free_rate: f64,
        num_points: usize,
        min_weights: Option<&[f64]>,
        max_weights: Option<&[f64]>,
    ) -> Result<Vec<(f64, f64)>> {
        // Find the tangency portfolio (max Sharpe ratio)
        let tangency = self.max_sharpe_portfolio(risk_free_rate, min_weights, max_weights)?;

        let mut cml_points = Vec::with_capacity(num_points);

        // Generate points along the CML
        // CML equation: E[R] = Rf + (E[Rm] - Rf) / σm * σ
        let market_return = tangency.expected_return;
        let market_vol = tangency.volatility;
        let sharpe_ratio = (market_return - risk_free_rate) / market_vol;

        let max_vol = market_vol * 2.0; // Extend CML beyond market portfolio
        let vol_step = max_vol / (num_points - 1) as f64;

        for i in 0..num_points {
            let vol = i as f64 * vol_step;
            let ret = risk_free_rate + sharpe_ratio * vol;
            cml_points.push((vol, ret));
        }

        Ok(cml_points)
    }

    /// Get feasible return range
    pub fn return_range(&self) -> (f64, f64) {
        (self.min_return, self.max_return)
    }
}

/// Extract return-volatility pairs from optimization results
pub fn extract_return_volatility_pairs(results: &[OptimizationResult]) -> Vec<(f64, f64)> {
    results
        .iter()
        .map(|r| (r.volatility, r.expected_return))
        .collect()
}

/// Extract Sharpe ratios from optimization results
pub fn extract_sharpe_ratios(results: &[OptimizationResult]) -> Vec<Option<f64>> {
    results.iter().map(|r| r.sharpe_ratio).collect()
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
    fn test_efficient_frontier_creation() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov);
        assert!(frontier.is_ok());
    }

    #[test]
    fn test_generate_frontier() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let results = frontier.generate(10, None, None);
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Check that returns are increasing along the frontier
        for i in 1..results.len() {
            assert!(results[i].expected_return >= results[i - 1].expected_return - 1e-6);
        }
    }

    #[test]
    fn test_generate_frontier_with_range() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let results = frontier.generate_with_range(0.08, 0.11, 5, None, None);
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.is_empty());

        // Check that all returns are within the specified range
        for result in &results {
            assert!(result.expected_return >= 0.08 - 1e-4);
            assert!(result.expected_return <= 0.11 + 1e-4);
        }
    }

    #[test]
    fn test_min_variance_portfolio() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let result = frontier.min_variance_portfolio(None, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.weights.len(), 3);

        let sum: f64 = result.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_max_sharpe_portfolio() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let risk_free_rate = 0.03;
        let result = frontier.max_sharpe_portfolio(risk_free_rate, None, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.sharpe_ratio.is_some());
        assert!(result.sharpe_ratio.unwrap() > 0.0);
    }

    #[test]
    fn test_capital_market_line() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let risk_free_rate = 0.03;
        let cml = frontier.capital_market_line(risk_free_rate, 10, None, None);
        assert!(cml.is_ok());

        let cml = cml.unwrap();
        assert_eq!(cml.len(), 10);

        // Check that CML is a straight line (constant Sharpe ratio)
        let sharpe = (cml[1].1 - cml[0].1) / (cml[1].0 - cml[0].0);
        for i in 2..cml.len() {
            let current_sharpe = (cml[i].1 - cml[i - 1].1) / (cml[i].0 - cml[i - 1].0);
            assert_relative_eq!(current_sharpe, sharpe, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_extract_return_volatility_pairs() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();
        let results = frontier.generate(5, None, None).unwrap();

        let pairs = extract_return_volatility_pairs(&results);
        assert_eq!(pairs.len(), results.len());

        for (i, pair) in pairs.iter().enumerate() {
            assert_relative_eq!(pair.0, results[i].volatility, epsilon = 1e-10);
            assert_relative_eq!(pair.1, results[i].expected_return, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extract_sharpe_ratios() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let risk_free_rate = 0.03;
        let result = frontier
            .max_sharpe_portfolio(risk_free_rate, None, None)
            .unwrap();

        let sharpe_ratios = extract_sharpe_ratios(&[result]);
        assert_eq!(sharpe_ratios.len(), 1);
        assert!(sharpe_ratios[0].is_some());
    }

    #[test]
    fn test_return_range() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let (min_ret, max_ret) = frontier.return_range();
        assert_relative_eq!(min_ret, 0.08, epsilon = 1e-10);
        assert_relative_eq!(max_ret, 0.12, epsilon = 1e-10);
    }

    #[test]
    fn test_frontier_with_constraints() {
        let (returns, cov) = create_test_data();
        let frontier = EfficientFrontier::new(returns, cov).unwrap();

        let min_weights = vec![0.1, 0.1, 0.1];
        let max_weights = vec![0.5, 0.5, 0.5];

        let results = frontier.generate(5, Some(&min_weights), Some(&max_weights));
        assert!(results.is_ok());

        let results = results.unwrap();
        for result in results {
            for i in 0..3 {
                assert!(result.weights[i] >= min_weights[i] - 1e-6);
                assert!(result.weights[i] <= max_weights[i] + 1e-6);
            }
        }
    }
}
