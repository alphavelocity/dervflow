// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Path generation engine
//!
//! Provides efficient path simulation for stochastic processes

use crate::common::error::{DervflowError, Result};
use crate::monte_carlo::processes::StochasticProcess;
use crate::numerical::random::RandomGenerator;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Parameters for path simulation
#[derive(Debug, Clone, Copy)]
pub struct SimulationParams {
    /// Initial value of the process
    pub initial_value: f64,
    /// Number of paths to simulate
    pub num_paths: usize,
    /// Number of time steps per path
    pub num_steps: usize,
    /// Time step size (dt)
    pub dt: f64,
}

impl SimulationParams {
    /// Create new simulation parameters
    pub fn new(initial_value: f64, num_paths: usize, num_steps: usize, dt: f64) -> Result<Self> {
        if initial_value <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Initial value must be positive".to_string(),
            ));
        }
        if num_paths == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of paths must be positive".to_string(),
            ));
        }
        if num_steps == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of steps must be positive".to_string(),
            ));
        }
        if dt <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Time step must be positive".to_string(),
            ));
        }

        Ok(Self {
            initial_value,
            num_paths,
            num_steps,
            dt,
        })
    }

    /// Get total time horizon
    pub fn total_time(&self) -> f64 {
        self.num_steps as f64 * self.dt
    }
}

/// Path generator for stochastic processes
pub struct PathGenerator {
    rng: RandomGenerator,
}

impl PathGenerator {
    /// Create a new path generator with a seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RandomGenerator::new(seed),
        }
    }

    /// Create a new path generator from entropy
    pub fn from_entropy() -> Self {
        Self {
            rng: RandomGenerator::from_entropy(),
        }
    }

    /// Generate paths for a single stochastic process
    ///
    /// Returns a 2D array of shape (num_paths, num_steps + 1)
    /// where each row is a path and columns are time points
    pub fn generate_paths(
        &mut self,
        process: &dyn StochasticProcess,
        params: &SimulationParams,
    ) -> Array2<f64> {
        let mut paths = Array2::zeros((params.num_paths, params.num_steps + 1));

        // Set initial values
        for i in 0..params.num_paths {
            paths[[i, 0]] = params.initial_value;
        }

        // Simulate each path
        for i in 0..params.num_paths {
            let mut x = params.initial_value;
            for j in 1..=params.num_steps {
                let t = (j - 1) as f64 * params.dt;
                let dw = self.rng.standard_normal();
                x = process.simulate_step(t, x, params.dt, dw);
                paths[[i, j]] = x;
            }
        }

        paths
    }

    /// Generate a single path for a stochastic process
    ///
    /// Returns a 1D array of length (num_steps + 1)
    pub fn generate_single_path(
        &mut self,
        process: &dyn StochasticProcess,
        params: &SimulationParams,
    ) -> Array1<f64> {
        let mut path = Array1::zeros(params.num_steps + 1);
        path[0] = params.initial_value;

        let mut x = params.initial_value;
        for j in 1..=params.num_steps {
            let t = (j - 1) as f64 * params.dt;
            let dw = self.rng.standard_normal();
            x = process.simulate_step(t, x, params.dt, dw);
            path[j] = x;
        }

        path
    }

    /// Generate paths with antithetic variates for variance reduction
    ///
    /// For each random draw dW, also simulate with -dW
    /// Returns a 2D array of shape (2 * num_paths, num_steps + 1)
    pub fn generate_paths_antithetic(
        &mut self,
        process: &dyn StochasticProcess,
        params: &SimulationParams,
    ) -> Array2<f64> {
        let mut paths = Array2::zeros((2 * params.num_paths, params.num_steps + 1));

        // Set initial values
        for i in 0..(2 * params.num_paths) {
            paths[[i, 0]] = params.initial_value;
        }

        // Simulate pairs of paths with antithetic variates
        for i in 0..params.num_paths {
            let mut x_pos = params.initial_value;
            let mut x_neg = params.initial_value;
            for j in 1..=params.num_steps {
                let t = (j - 1) as f64 * params.dt;
                let dw = self.rng.standard_normal();

                // Positive path
                x_pos = process.simulate_step(t, x_pos, params.dt, dw);
                paths[[2 * i, j]] = x_pos;

                // Antithetic path
                x_neg = process.simulate_step(t, x_neg, params.dt, -dw);
                paths[[2 * i + 1, j]] = x_neg;
            }
        }

        paths
    }

    /// Generate paths in parallel using Rayon
    ///
    /// Returns a 2D array of shape (num_paths, num_steps + 1)
    /// Each path is simulated independently in parallel
    pub fn generate_paths_parallel(
        &self,
        process: &dyn StochasticProcess,
        params: &SimulationParams,
        base_seed: u64,
    ) -> Array2<f64> {
        let row_len = params.num_steps + 1;
        let mut paths = Array2::zeros((params.num_paths, row_len));

        let data = paths
            .as_slice_mut()
            .expect("paths array should be contiguous");

        data.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(path_idx, row)| {
                let mut local_rng = RandomGenerator::new(base_seed.wrapping_add(path_idx as u64));
                let mut x = params.initial_value;
                row[0] = params.initial_value;
                for (step_idx, value) in row.iter_mut().skip(1).enumerate() {
                    let t = step_idx as f64 * params.dt;
                    let dw = local_rng.standard_normal();
                    x = process.simulate_step(t, x, params.dt, dw);
                    *value = x;
                }
            });

        paths
    }

    /// Generate paths in parallel with antithetic variates
    ///
    /// Returns a 2D array of shape (2 * num_paths, num_steps + 1)
    /// Pairs of antithetic paths are generated in parallel
    pub fn generate_paths_parallel_antithetic(
        &self,
        process: &dyn StochasticProcess,
        params: &SimulationParams,
        base_seed: u64,
    ) -> Array2<f64> {
        let row_len = params.num_steps + 1;
        let mut paths = Array2::zeros((2 * params.num_paths, row_len));

        let data = paths
            .as_slice_mut()
            .expect("paths array should be contiguous");

        data.par_chunks_mut(2 * row_len)
            .enumerate()
            .for_each(|(path_idx, rows)| {
                let (row_pos, row_neg) = rows.split_at_mut(row_len);
                let mut local_rng = RandomGenerator::new(base_seed.wrapping_add(path_idx as u64));
                let mut x_pos = params.initial_value;
                let mut x_neg = params.initial_value;
                row_pos[0] = params.initial_value;
                row_neg[0] = params.initial_value;

                for (step_idx, (value_pos, value_neg)) in row_pos
                    .iter_mut()
                    .skip(1)
                    .zip(row_neg.iter_mut().skip(1))
                    .enumerate()
                {
                    let t = step_idx as f64 * params.dt;
                    let dw = local_rng.standard_normal();
                    x_pos = process.simulate_step(t, x_pos, params.dt, dw);
                    x_neg = process.simulate_step(t, x_neg, params.dt, -dw);
                    *value_pos = x_pos;
                    *value_neg = x_neg;
                }
            });

        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::processes::GeometricBrownianMotion;
    use approx::assert_relative_eq;

    #[test]
    fn test_simulation_params_creation() {
        let params = SimulationParams::new(100.0, 1000, 252, 1.0 / 252.0).unwrap();
        assert_eq!(params.initial_value, 100.0);
        assert_eq!(params.num_paths, 1000);
        assert_eq!(params.num_steps, 252);
        assert_relative_eq!(params.total_time(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simulation_params_invalid() {
        assert!(SimulationParams::new(-100.0, 1000, 252, 1.0 / 252.0).is_err());
        assert!(SimulationParams::new(100.0, 0, 252, 1.0 / 252.0).is_err());
        assert!(SimulationParams::new(100.0, 1000, 0, 1.0 / 252.0).is_err());
        assert!(SimulationParams::new(100.0, 1000, 252, -1.0 / 252.0).is_err());
    }

    #[test]
    fn test_generate_paths() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 10, 100, 0.01).unwrap();
        let mut generator = PathGenerator::new(42);

        let paths = generator.generate_paths(&gbm, &params);

        assert_eq!(paths.shape(), &[10, 101]);

        // Check initial values
        for i in 0..10 {
            assert_relative_eq!(paths[[i, 0]], 100.0, epsilon = 1e-10);
        }

        // Check that paths are positive
        for i in 0..10 {
            for j in 0..=100 {
                assert!(paths[[i, j]] > 0.0);
            }
        }
    }

    #[test]
    fn test_generate_single_path() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 1, 100, 0.01).unwrap();
        let mut generator = PathGenerator::new(42);

        let path = generator.generate_single_path(&gbm, &params);

        assert_eq!(path.len(), 101);
        assert_relative_eq!(path[0], 100.0, epsilon = 1e-10);

        // Check that path is positive
        for &value in path.iter() {
            assert!(value > 0.0);
        }
    }

    #[test]
    fn test_generate_paths_antithetic() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 5, 100, 0.01).unwrap();
        let mut generator = PathGenerator::new(42);

        let paths = generator.generate_paths_antithetic(&gbm, &params);

        // Should have 2 * num_paths
        assert_eq!(paths.shape(), &[10, 101]);

        // Check initial values
        for i in 0..10 {
            assert_relative_eq!(paths[[i, 0]], 100.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_path_stat() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 10000, 252, 1.0 / 252.0).unwrap();
        let mut generator = PathGenerator::new(42);

        let paths = generator.generate_paths(&gbm, &params);

        // Calculate average final value
        let mut sum = 0.0;
        for i in 0..params.num_paths {
            sum += paths[[i, params.num_steps]];
        }
        let avg_final = sum / params.num_paths as f64;

        // Expected value: S0 * exp(μ * T)
        let expected = params.initial_value * (gbm.mu * params.total_time()).exp();

        // Should be close to expected value (within 5% due to randomness)
        let relative_error = (avg_final - expected).abs() / expected;
        assert!(relative_error < 0.05);
    }

    #[test]
    fn test_generate_paths_parallel() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 1000, 100, 0.01).unwrap();
        let generator = PathGenerator::new(42);

        let paths = generator.generate_paths_parallel(&gbm, &params, 42);

        assert_eq!(paths.shape(), &[1000, 101]);

        // Check initial values
        for i in 0..1000 {
            assert_relative_eq!(paths[[i, 0]], 100.0, epsilon = 1e-10);
        }

        // Check that paths are positive
        for i in 0..1000 {
            for j in 0..=100 {
                assert!(paths[[i, j]] > 0.0);
            }
        }
    }

    #[test]
    fn test_generate_paths_parallel_antithetic() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 500, 100, 0.01).unwrap();
        let generator = PathGenerator::new(42);

        let paths = generator.generate_paths_parallel_antithetic(&gbm, &params, 42);

        // Should have 2 * num_paths
        assert_eq!(paths.shape(), &[1000, 101]);

        // Check initial values
        for i in 0..1000 {
            assert_relative_eq!(paths[[i, 0]], 100.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 100, 50, 0.01).unwrap();

        // Sequential generation
        let mut generator_seq = PathGenerator::new(42);
        let paths_seq = generator_seq.generate_paths(&gbm, &params);

        // Parallel generation with same base seed
        let generator_par = PathGenerator::new(42);
        let paths_par = generator_par.generate_paths_parallel(&gbm, &params, 42);

        // Both should have same shape
        assert_eq!(paths_seq.shape(), paths_par.shape());

        // Calculate stat metrics for both
        let mut sum_seq = 0.0;
        let mut sum_par = 0.0;
        for i in 0..params.num_paths {
            sum_seq += paths_seq[[i, params.num_steps]];
            sum_par += paths_par[[i, params.num_steps]];
        }
        let avg_seq = sum_seq / params.num_paths as f64;
        let avg_par = sum_par / params.num_paths as f64;

        // Averages should be similar (within 20% due to different random sequences)
        let relative_diff = (avg_seq - avg_par).abs() / avg_seq;
        assert!(relative_diff < 0.2);
    }

    #[test]
    fn test_parallel_performance_benefit() {
        // This test verifies that parallel generation works correctly
        // Actual performance benefit depends on hardware
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let params = SimulationParams::new(100.0, 1000, 100, 0.01).unwrap();
        let generator = PathGenerator::new(42);

        let paths = generator.generate_paths_parallel(&gbm, &params, 42);

        // Verify correctness
        assert_eq!(paths.shape(), &[1000, 101]);

        // Calculate average final value
        let mut sum = 0.0;
        for i in 0..params.num_paths {
            sum += paths[[i, params.num_steps]];
        }
        let avg_final = sum / params.num_paths as f64;

        // Expected value: S0 * exp(μ * T)
        let expected = params.initial_value * (gbm.mu * params.total_time()).exp();

        // Should be close to expected value (within 10% due to randomness)
        let relative_error = (avg_final - expected).abs() / expected;
        assert!(relative_error < 0.1);
    }
}
