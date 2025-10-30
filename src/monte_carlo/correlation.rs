// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Correlated path generation
//!
//! Provides functionality for generating correlated multi-asset paths using Cholesky decomposition

use crate::common::error::{DervflowError, Result};
use crate::monte_carlo::path_generator::SimulationParams;
use crate::monte_carlo::processes::StochasticProcess;
use crate::numerical::random::RandomGenerator;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3};
use rayon::prelude::*;

/// Correlated path generator for multi-asset simulation
pub struct CorrelatedPathGenerator {
    rng: RandomGenerator,
}

impl CorrelatedPathGenerator {
    /// Create a new correlated path generator with a seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RandomGenerator::new(seed),
        }
    }

    /// Create a new correlated path generator from entropy
    pub fn from_entropy() -> Self {
        Self {
            rng: RandomGenerator::from_entropy(),
        }
    }

    /// Generate correlated paths for multiple assets
    ///
    /// # Arguments
    /// * `processes` - Vector of stochastic processes (one per asset)
    /// * `correlation` - Correlation matrix (n x n where n = number of assets)
    /// * `params` - Simulation parameters (same for all assets)
    ///
    /// # Returns
    /// 3D array of shape (num_assets, num_paths, num_steps + 1)
    /// where paths[i, j, k] is the value of asset i, path j, at time step k
    pub fn generate_correlated_paths(
        &mut self,
        processes: &[&dyn StochasticProcess],
        correlation: &Array2<f64>,
        params: &SimulationParams,
    ) -> Result<Array3<f64>> {
        let n_assets = processes.len();

        // Validate inputs
        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "At least one process is required".to_string(),
            ));
        }

        if correlation.shape() != [n_assets, n_assets] {
            return Err(DervflowError::InvalidInput(format!(
                "Correlation matrix must be {}x{}, got {}x{}",
                n_assets,
                n_assets,
                correlation.shape()[0],
                correlation.shape()[1]
            )));
        }

        // Validate correlation matrix
        self.validate_correlation_matrix(correlation)?;

        // Compute Cholesky decomposition
        let cholesky = self.cholesky_decomposition(correlation)?;

        // Initialize output array
        let mut paths = Array3::zeros((n_assets, params.num_paths, params.num_steps + 1));

        // Set initial values for all assets
        for i in 0..n_assets {
            for j in 0..params.num_paths {
                paths[[i, j, 0]] = params.initial_value;
            }
        }

        // Simulate each path
        for path_idx in 0..params.num_paths {
            // Initialize current values for all assets
            let mut current_values: Vec<f64> = vec![params.initial_value; n_assets];
            let mut t = 0.0;

            for step_idx in 1..=params.num_steps {
                // Generate independent standard normal random variables
                let independent_normals: Vec<f64> =
                    (0..n_assets).map(|_| self.rng.standard_normal()).collect();

                // Apply Cholesky decomposition to get correlated normals
                let correlated_normals = self.apply_cholesky(&cholesky, &independent_normals);

                // Simulate one step for each asset
                for asset_idx in 0..n_assets {
                    let dw = correlated_normals[asset_idx];
                    let new_value = processes[asset_idx].simulate_step(
                        t,
                        current_values[asset_idx],
                        params.dt,
                        dw,
                    );
                    current_values[asset_idx] = new_value;
                    paths[[asset_idx, path_idx, step_idx]] = new_value;
                }

                t += params.dt;
            }
        }

        Ok(paths)
    }

    /// Generate correlated paths with different initial values for each asset
    ///
    /// # Arguments
    /// * `processes` - Vector of stochastic processes (one per asset)
    /// * `initial_values` - Initial value for each asset
    /// * `correlation` - Correlation matrix (n x n where n = number of assets)
    /// * `params` - Simulation parameters (initial_value field is ignored)
    ///
    /// # Returns
    /// 3D array of shape (num_assets, num_paths, num_steps + 1)
    pub fn generate_correlated_paths_with_initials(
        &mut self,
        processes: &[&dyn StochasticProcess],
        initial_values: &[f64],
        correlation: &Array2<f64>,
        params: &SimulationParams,
    ) -> Result<Array3<f64>> {
        let n_assets = processes.len();

        // Validate inputs
        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "At least one process is required".to_string(),
            ));
        }

        if initial_values.len() != n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Number of initial values ({}) must match number of processes ({})",
                initial_values.len(),
                n_assets
            )));
        }

        if correlation.shape() != [n_assets, n_assets] {
            return Err(DervflowError::InvalidInput(format!(
                "Correlation matrix must be {}x{}, got {}x{}",
                n_assets,
                n_assets,
                correlation.shape()[0],
                correlation.shape()[1]
            )));
        }

        // Validate correlation matrix
        self.validate_correlation_matrix(correlation)?;

        // Compute Cholesky decomposition
        let cholesky = self.cholesky_decomposition(correlation)?;

        // Initialize output array
        let mut paths = Array3::zeros((n_assets, params.num_paths, params.num_steps + 1));

        // Set initial values for all assets
        for i in 0..n_assets {
            for j in 0..params.num_paths {
                paths[[i, j, 0]] = initial_values[i];
            }
        }

        // Simulate each path
        for path_idx in 0..params.num_paths {
            // Initialize current values for all assets
            let mut current_values: Vec<f64> = initial_values.to_vec();
            let mut t = 0.0;

            for step_idx in 1..=params.num_steps {
                // Generate independent standard normal random variables
                let independent_normals: Vec<f64> =
                    (0..n_assets).map(|_| self.rng.standard_normal()).collect();

                // Apply Cholesky decomposition to get correlated normals
                let correlated_normals = self.apply_cholesky(&cholesky, &independent_normals);

                // Simulate one step for each asset
                for asset_idx in 0..n_assets {
                    let dw = correlated_normals[asset_idx];
                    let new_value = processes[asset_idx].simulate_step(
                        t,
                        current_values[asset_idx],
                        params.dt,
                        dw,
                    );
                    current_values[asset_idx] = new_value;
                    paths[[asset_idx, path_idx, step_idx]] = new_value;
                }

                t += params.dt;
            }
        }

        Ok(paths)
    }

    /// Validate that the correlation matrix is valid
    fn validate_correlation_matrix(&self, correlation: &Array2<f64>) -> Result<()> {
        let n = correlation.shape()[0];

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                if (correlation[[i, j]] - correlation[[j, i]]).abs() > 1e-10 {
                    return Err(DervflowError::InvalidInput(
                        "Correlation matrix must be symmetric".to_string(),
                    ));
                }
            }
        }

        // Check diagonal elements are 1
        for i in 0..n {
            if (correlation[[i, i]] - 1.0).abs() > 1e-10 {
                return Err(DervflowError::InvalidInput(
                    "Diagonal elements of correlation matrix must be 1".to_string(),
                ));
            }
        }

        // Check off-diagonal elements are in [-1, 1]
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let corr = correlation[[i, j]];
                    if !(-1.0..=1.0).contains(&corr) {
                        return Err(DervflowError::InvalidInput(format!(
                            "Correlation coefficient at ({}, {}) = {} is outside [-1, 1]",
                            i, j, corr
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute Cholesky decomposition of correlation matrix
    fn cholesky_decomposition(&self, correlation: &Array2<f64>) -> Result<DMatrix<f64>> {
        let n = correlation.shape()[0];

        // Convert ndarray to nalgebra DMatrix
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = correlation[[i, j]];
            }
        }

        // Compute Cholesky decomposition
        match matrix.clone().cholesky() {
            Some(chol) => Ok(chol.l().clone()),
            None => Err(DervflowError::NumericalError(
                "Correlation matrix is not positive definite. Cholesky decomposition failed."
                    .to_string(),
            )),
        }
    }

    /// Apply Cholesky matrix to independent normals to get correlated normals
    fn apply_cholesky(&self, cholesky: &DMatrix<f64>, independent: &[f64]) -> Vec<f64> {
        let z = DVector::from_vec(independent.to_vec());
        let correlated = cholesky * z;
        correlated.as_slice().to_vec()
    }

    /// Generate correlated paths in parallel
    ///
    /// # Arguments
    /// * `processes` - Vector of stochastic processes (one per asset)
    /// * `correlation` - Correlation matrix (n x n where n = number of assets)
    /// * `params` - Simulation parameters (same for all assets)
    /// * `base_seed` - Base seed for random number generation
    ///
    /// # Returns
    /// 3D array of shape (num_assets, num_paths, num_steps + 1)
    pub fn generate_correlated_paths_parallel(
        &self,
        processes: &[&dyn StochasticProcess],
        correlation: &Array2<f64>,
        params: &SimulationParams,
        base_seed: u64,
    ) -> Result<Array3<f64>> {
        let n_assets = processes.len();

        // Validate inputs
        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "At least one process is required".to_string(),
            ));
        }

        if correlation.shape() != [n_assets, n_assets] {
            return Err(DervflowError::InvalidInput(format!(
                "Correlation matrix must be {}x{}, got {}x{}",
                n_assets,
                n_assets,
                correlation.shape()[0],
                correlation.shape()[1]
            )));
        }

        // Validate correlation matrix
        self.validate_correlation_matrix(correlation)?;

        // Compute Cholesky decomposition
        let cholesky = self.cholesky_decomposition(correlation)?;

        // Initialize output array
        let mut paths = Array3::zeros((n_assets, params.num_paths, params.num_steps + 1));

        // Set initial values for all assets
        for i in 0..n_assets {
            for j in 0..params.num_paths {
                paths[[i, j, 0]] = params.initial_value;
            }
        }

        // Generate paths in parallel
        let path_data: Vec<Vec<Vec<f64>>> = (0..params.num_paths)
            .into_par_iter()
            .map(|path_idx| {
                // Create thread-local RNG with unique seed
                let mut local_rng = RandomGenerator::new(base_seed.wrapping_add(path_idx as u64));

                // Initialize current values for all assets
                let mut current_values: Vec<f64> = vec![params.initial_value; n_assets];
                let mut asset_paths: Vec<Vec<f64>> = vec![vec![params.initial_value]; n_assets];
                let mut t = 0.0;

                for _ in 1..=params.num_steps {
                    // Generate independent standard normal random variables
                    let independent_normals: Vec<f64> =
                        (0..n_assets).map(|_| local_rng.standard_normal()).collect();

                    // Apply Cholesky decomposition to get correlated normals
                    let z = DVector::from_vec(independent_normals);
                    let correlated = &cholesky * z;
                    let correlated_normals: Vec<f64> = correlated.as_slice().to_vec();

                    // Simulate one step for each asset
                    for asset_idx in 0..n_assets {
                        let dw = correlated_normals[asset_idx];
                        let new_value = processes[asset_idx].simulate_step(
                            t,
                            current_values[asset_idx],
                            params.dt,
                            dw,
                        );
                        current_values[asset_idx] = new_value;
                        asset_paths[asset_idx].push(new_value);
                    }

                    t += params.dt;
                }

                asset_paths
            })
            .collect();

        // Copy results into output array
        for path_idx in 0..params.num_paths {
            for asset_idx in 0..n_assets {
                for step_idx in 0..=params.num_steps {
                    paths[[asset_idx, path_idx, step_idx]] =
                        path_data[path_idx][asset_idx][step_idx];
                }
            }
        }

        Ok(paths)
    }

    /// Generate correlated paths in parallel with different initial values
    ///
    /// # Arguments
    /// * `processes` - Vector of stochastic processes (one per asset)
    /// * `initial_values` - Initial value for each asset
    /// * `correlation` - Correlation matrix (n x n where n = number of assets)
    /// * `params` - Simulation parameters (initial_value field is ignored)
    /// * `base_seed` - Base seed for random number generation
    ///
    /// # Returns
    /// 3D array of shape (num_assets, num_paths, num_steps + 1)
    pub fn generate_correlated_paths_parallel_with_initials(
        &self,
        processes: &[&dyn StochasticProcess],
        initial_values: &[f64],
        correlation: &Array2<f64>,
        params: &SimulationParams,
        base_seed: u64,
    ) -> Result<Array3<f64>> {
        let n_assets = processes.len();

        // Validate inputs
        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "At least one process is required".to_string(),
            ));
        }

        if initial_values.len() != n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "Number of initial values ({}) must match number of processes ({})",
                initial_values.len(),
                n_assets
            )));
        }

        if correlation.shape() != [n_assets, n_assets] {
            return Err(DervflowError::InvalidInput(format!(
                "Correlation matrix must be {}x{}, got {}x{}",
                n_assets,
                n_assets,
                correlation.shape()[0],
                correlation.shape()[1]
            )));
        }

        // Validate correlation matrix
        self.validate_correlation_matrix(correlation)?;

        // Compute Cholesky decomposition
        let cholesky = self.cholesky_decomposition(correlation)?;

        // Initialize output array
        let mut paths = Array3::zeros((n_assets, params.num_paths, params.num_steps + 1));

        // Set initial values for all assets
        for i in 0..n_assets {
            for j in 0..params.num_paths {
                paths[[i, j, 0]] = initial_values[i];
            }
        }

        // Generate paths in parallel
        let initial_values_vec = initial_values.to_vec();
        let path_data: Vec<Vec<Vec<f64>>> = (0..params.num_paths)
            .into_par_iter()
            .map(|path_idx| {
                // Create thread-local RNG with unique seed
                let mut local_rng = RandomGenerator::new(base_seed.wrapping_add(path_idx as u64));

                // Initialize current values for all assets
                let mut current_values: Vec<f64> = initial_values_vec.clone();
                let mut asset_paths: Vec<Vec<f64>> =
                    initial_values_vec.iter().map(|&val| vec![val]).collect();
                let mut t = 0.0;

                for _ in 1..=params.num_steps {
                    // Generate independent standard normal random variables
                    let independent_normals: Vec<f64> =
                        (0..n_assets).map(|_| local_rng.standard_normal()).collect();

                    // Apply Cholesky decomposition to get correlated normals
                    let z = DVector::from_vec(independent_normals);
                    let correlated = &cholesky * z;
                    let correlated_normals: Vec<f64> = correlated.as_slice().to_vec();

                    // Simulate one step for each asset
                    for asset_idx in 0..n_assets {
                        let dw = correlated_normals[asset_idx];
                        let new_value = processes[asset_idx].simulate_step(
                            t,
                            current_values[asset_idx],
                            params.dt,
                            dw,
                        );
                        current_values[asset_idx] = new_value;
                        asset_paths[asset_idx].push(new_value);
                    }

                    t += params.dt;
                }

                asset_paths
            })
            .collect();

        // Copy results into output array
        for path_idx in 0..params.num_paths {
            for asset_idx in 0..n_assets {
                for step_idx in 0..=params.num_steps {
                    paths[[asset_idx, path_idx, step_idx]] =
                        path_data[path_idx][asset_idx][step_idx];
                }
            }
        }

        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::processes::GeometricBrownianMotion;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_validate_correlation_matrix_valid() {
        let generator = CorrelatedPathGenerator::new(42);

        // Valid 2x2 correlation matrix
        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
        assert!(generator.validate_correlation_matrix(&corr).is_ok());

        // Valid 3x3 correlation matrix
        let corr = arr2(&[[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]);
        assert!(generator.validate_correlation_matrix(&corr).is_ok());
    }

    #[test]
    fn test_validate_correlation_matrix_invalid_diagonal() {
        let generator = CorrelatedPathGenerator::new(42);

        // Invalid diagonal
        let corr = arr2(&[[0.9, 0.5], [0.5, 1.0]]);
        assert!(generator.validate_correlation_matrix(&corr).is_err());
    }

    #[test]
    fn test_validate_correlation_matrix_asymmetric() {
        let generator = CorrelatedPathGenerator::new(42);

        // Asymmetric matrix
        let corr = arr2(&[[1.0, 0.5], [0.6, 1.0]]);
        assert!(generator.validate_correlation_matrix(&corr).is_err());
    }

    #[test]
    fn test_validate_correlation_matrix_out_of_range() {
        let generator = CorrelatedPathGenerator::new(42);

        // Correlation > 1
        let corr = arr2(&[[1.0, 1.5], [1.5, 1.0]]);
        assert!(generator.validate_correlation_matrix(&corr).is_err());

        // Correlation < -1
        let corr = arr2(&[[1.0, -1.5], [-1.5, 1.0]]);
        assert!(generator.validate_correlation_matrix(&corr).is_err());
    }

    #[test]
    fn test_cholesky_decomposition() {
        let generator = CorrelatedPathGenerator::new(42);

        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
        let chol = generator.cholesky_decomposition(&corr).unwrap();

        // Verify L * L^T = R
        let reconstructed = &chol * chol.transpose();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[(i, j)], corr[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_decomposition_not_positive_definite() {
        let generator = CorrelatedPathGenerator::new(42);

        // This matrix is not positive definite (violates triangle inequality)
        // For a valid correlation matrix: |corr(i,j)| <= sqrt(corr(i,k) * corr(k,j))
        // Here: corr(0,1) = 1.0, corr(0,2) = 1.0, corr(1,2) = -0.5
        // This violates the positive definite requirement
        let corr = arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, -0.5], [1.0, -0.5, 1.0]]);

        // Should fail
        assert!(generator.cholesky_decomposition(&corr).is_err());
    }

    #[test]
    fn test_generate_correlated_paths_two_assets() {
        let gbm1 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let gbm2 = GeometricBrownianMotion::new(0.06, 0.25).unwrap();

        let processes: Vec<&dyn StochasticProcess> = vec![&gbm1, &gbm2];

        let corr = arr2(&[[1.0, 0.7], [0.7, 1.0]]);

        let params = SimulationParams::new(100.0, 1000, 100, 0.01).unwrap();

        let mut generator = CorrelatedPathGenerator::new(42);
        let paths = generator
            .generate_correlated_paths(&processes, &corr, &params)
            .unwrap();

        // Check shape
        assert_eq!(paths.shape(), &[2, 1000, 101]);

        // Check initial values
        for i in 0..2 {
            for j in 0..1000 {
                assert_relative_eq!(paths[[i, j, 0]], 100.0, epsilon = 1e-10);
            }
        }

        // Check that paths are positive
        for i in 0..2 {
            for j in 0..1000 {
                for k in 0..=100 {
                    assert!(paths[[i, j, k]] > 0.0);
                }
            }
        }
    }

    #[test]
    fn test_generate_correlated_paths_with_initials() {
        let gbm1 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let gbm2 = GeometricBrownianMotion::new(0.06, 0.25).unwrap();

        let processes: Vec<&dyn StochasticProcess> = vec![&gbm1, &gbm2];
        let initial_values = vec![100.0, 50.0];

        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);

        let params = SimulationParams::new(100.0, 100, 50, 0.01).unwrap();

        let mut generator = CorrelatedPathGenerator::new(42);
        let paths = generator
            .generate_correlated_paths_with_initials(&processes, &initial_values, &corr, &params)
            .unwrap();

        // Check shape
        assert_eq!(paths.shape(), &[2, 100, 51]);

        // Check initial values
        for j in 0..100 {
            assert_relative_eq!(paths[[0, j, 0]], 100.0, epsilon = 1e-10);
            assert_relative_eq!(paths[[1, j, 0]], 50.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_correlation_preservation() {
        let gbm1 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let gbm2 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();

        let processes: Vec<&dyn StochasticProcess> = vec![&gbm1, &gbm2];

        let target_corr = 0.8;
        let corr = arr2(&[[1.0, target_corr], [target_corr, 1.0]]);

        let params = SimulationParams::new(100.0, 10000, 252, 1.0 / 252.0).unwrap();

        let mut generator = CorrelatedPathGenerator::new(42);
        let paths = generator
            .generate_correlated_paths(&processes, &corr, &params)
            .unwrap();

        // Calculate log returns for both assets
        let mut returns1 = Vec::new();
        let mut returns2 = Vec::new();

        for path_idx in 0..params.num_paths {
            let final_val1 = paths[[0, path_idx, params.num_steps]];
            let initial_val1 = paths[[0, path_idx, 0]];
            returns1.push((final_val1 / initial_val1).ln());

            let final_val2 = paths[[1, path_idx, params.num_steps]];
            let initial_val2 = paths[[1, path_idx, 0]];
            returns2.push((final_val2 / initial_val2).ln());
        }

        // Calculate sample correlation
        let mean1: f64 = returns1.iter().sum::<f64>() / returns1.len() as f64;
        let mean2: f64 = returns2.iter().sum::<f64>() / returns2.len() as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..returns1.len() {
            let diff1 = returns1[i] - mean1;
            let diff2 = returns2[i] - mean2;
            cov += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }

        let sample_corr = cov / (var1 * var2).sqrt();

        // Sample correlation should be close to target (within 10% due to randomness)
        assert!((sample_corr - target_corr).abs() < 0.1);
    }

    #[test]
    fn test_invalid_inputs() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let processes: Vec<&dyn StochasticProcess> = vec![&gbm];

        let params = SimulationParams::new(100.0, 100, 50, 0.01).unwrap();
        let mut generator = CorrelatedPathGenerator::new(42);

        // Wrong correlation matrix size
        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
        assert!(
            generator
                .generate_correlated_paths(&processes, &corr, &params)
                .is_err()
        );

        // Empty processes
        let processes: Vec<&dyn StochasticProcess> = vec![];
        let corr = arr2(&[[1.0]]);
        assert!(
            generator
                .generate_correlated_paths(&processes, &corr, &params)
                .is_err()
        );
    }

    #[test]
    fn test_generate_correlated_paths_parallel() {
        let gbm1 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let gbm2 = GeometricBrownianMotion::new(0.06, 0.25).unwrap();

        let processes: Vec<&dyn StochasticProcess> = vec![&gbm1, &gbm2];

        let corr = arr2(&[[1.0, 0.7], [0.7, 1.0]]);

        let params = SimulationParams::new(100.0, 1000, 100, 0.01).unwrap();

        let generator = CorrelatedPathGenerator::new(42);
        let paths = generator
            .generate_correlated_paths_parallel(&processes, &corr, &params, 42)
            .unwrap();

        // Check shape
        assert_eq!(paths.shape(), &[2, 1000, 101]);

        // Check initial values
        for i in 0..2 {
            for j in 0..1000 {
                assert_relative_eq!(paths[[i, j, 0]], 100.0, epsilon = 1e-10);
            }
        }

        // Check that paths are positive
        for i in 0..2 {
            for j in 0..1000 {
                for k in 0..=100 {
                    assert!(paths[[i, j, k]] > 0.0);
                }
            }
        }
    }

    #[test]
    fn test_generate_correlated_paths_parallel_with_initials() {
        let gbm1 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let gbm2 = GeometricBrownianMotion::new(0.06, 0.25).unwrap();

        let processes: Vec<&dyn StochasticProcess> = vec![&gbm1, &gbm2];
        let initial_values = vec![100.0, 50.0];

        let corr = arr2(&[[1.0, 0.5], [0.5, 1.0]]);

        let params = SimulationParams::new(100.0, 100, 50, 0.01).unwrap();

        let generator = CorrelatedPathGenerator::new(42);
        let paths = generator
            .generate_correlated_paths_parallel_with_initials(
                &processes,
                &initial_values,
                &corr,
                &params,
                42,
            )
            .unwrap();

        // Check shape
        assert_eq!(paths.shape(), &[2, 100, 51]);

        // Check initial values
        for j in 0..100 {
            assert_relative_eq!(paths[[0, j, 0]], 100.0, epsilon = 1e-10);
            assert_relative_eq!(paths[[1, j, 0]], 50.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_correlation_preservation() {
        let gbm1 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let gbm2 = GeometricBrownianMotion::new(0.05, 0.2).unwrap();

        let processes: Vec<&dyn StochasticProcess> = vec![&gbm1, &gbm2];

        let target_corr = 0.8;
        let corr = arr2(&[[1.0, target_corr], [target_corr, 1.0]]);

        let params = SimulationParams::new(100.0, 10000, 252, 1.0 / 252.0).unwrap();

        let generator = CorrelatedPathGenerator::new(42);
        let paths = generator
            .generate_correlated_paths_parallel(&processes, &corr, &params, 42)
            .unwrap();

        // Calculate log returns for both assets
        let mut returns1 = Vec::new();
        let mut returns2 = Vec::new();

        for path_idx in 0..params.num_paths {
            let final_val1 = paths[[0, path_idx, params.num_steps]];
            let initial_val1 = paths[[0, path_idx, 0]];
            returns1.push((final_val1 / initial_val1).ln());

            let final_val2 = paths[[1, path_idx, params.num_steps]];
            let initial_val2 = paths[[1, path_idx, 0]];
            returns2.push((final_val2 / initial_val2).ln());
        }

        // Calculate sample correlation
        let mean1: f64 = returns1.iter().sum::<f64>() / returns1.len() as f64;
        let mean2: f64 = returns2.iter().sum::<f64>() / returns2.len() as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..returns1.len() {
            let diff1 = returns1[i] - mean1;
            let diff2 = returns2[i] - mean2;
            cov += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }

        let sample_corr = cov / (var1 * var2).sqrt();

        // Sample correlation should be close to target (within 10% due to randomness)
        assert!((sample_corr - target_corr).abs() < 0.1);
    }
}
