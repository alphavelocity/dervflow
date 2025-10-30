// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for Monte Carlo module

use ndarray::s;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::monte_carlo::correlation::CorrelatedPathGenerator;
use crate::monte_carlo::path_generator::{PathGenerator, SimulationParams};
use crate::monte_carlo::processes::*;

/// Monte Carlo simulation engine for stochastic processes
#[pyclass]
pub struct MonteCarloEngine {
    seed: Option<u64>,
}

#[pymethods]
impl MonteCarloEngine {
    /// Create a new Monte Carlo engine
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        Self { seed }
    }

    /// Simulate Geometric Brownian Motion paths
    ///
    /// Args:
    ///     s0: Initial value
    ///     mu: Drift parameter (expected return)
    ///     sigma: Volatility parameter
    ///     T: Total time horizon
    ///     steps: Number of time steps
    ///     paths: Number of paths to simulate
    ///     parallel: Whether to use parallel processing (default: False)
    ///
    /// Returns:
    ///     2D numpy array of shape (paths, steps + 1) containing simulated paths
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[pyo3(signature = (s0, mu, sigma, T, steps, paths, parallel=false))]
    fn simulate_gbm<'py>(
        &self,
        py: Python<'py>,
        s0: f64,
        mu: f64,
        sigma: f64,
        T: f64,
        steps: usize,
        paths: usize,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let gbm = GeometricBrownianMotion::new(mu, sigma)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dt = T / steps as f64;
        let params = SimulationParams::new(s0, paths, steps, dt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result = if parallel {
            let seed = self.seed.unwrap_or_else(rand::random);
            let generator = PathGenerator::new(seed);
            generator.generate_paths_parallel(&gbm, &params, seed)
        } else {
            let mut generator = match self.seed {
                Some(s) => PathGenerator::new(s),
                None => PathGenerator::from_entropy(),
            };
            generator.generate_paths(&gbm, &params)
        };

        Ok(PyArray2::from_owned_array(py, result))
    }

    /// Simulate Ornstein-Uhlenbeck process paths
    ///
    /// Args:
    ///     x0: Initial value
    ///     theta: Mean reversion speed
    ///     mu: Long-term mean
    ///     sigma: Volatility parameter
    ///     T: Total time horizon
    ///     steps: Number of time steps
    ///     paths: Number of paths to simulate
    ///     parallel: Whether to use parallel processing (default: False)
    ///
    /// Returns:
    ///     2D numpy array of shape (paths, steps + 1) containing simulated paths
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[pyo3(signature = (x0, theta, mu, sigma, T, steps, paths, parallel=false))]
    fn simulate_ou<'py>(
        &self,
        py: Python<'py>,
        x0: f64,
        theta: f64,
        mu: f64,
        sigma: f64,
        T: f64,
        steps: usize,
        paths: usize,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ou = OrnsteinUhlenbeck::new(theta, mu, sigma)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dt = T / steps as f64;
        let params = SimulationParams::new(x0, paths, steps, dt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result = if parallel {
            let seed = self.seed.unwrap_or_else(rand::random);
            let generator = PathGenerator::new(seed);
            generator.generate_paths_parallel(&ou, &params, seed)
        } else {
            let mut generator = match self.seed {
                Some(s) => PathGenerator::new(s),
                None => PathGenerator::from_entropy(),
            };
            generator.generate_paths(&ou, &params)
        };

        Ok(PyArray2::from_owned_array(py, result))
    }

    /// Simulate Cox-Ingersoll-Ross process paths
    ///
    /// Args:
    ///     x0: Initial value
    ///     kappa: Mean reversion speed
    ///     theta: Long-term mean
    ///     sigma: Volatility parameter
    ///     T: Total time horizon
    ///     steps: Number of time steps
    ///     paths: Number of paths to simulate
    ///     parallel: Whether to use parallel processing (default: False)
    ///
    /// Returns:
    ///     2D numpy array of shape (paths, steps + 1) containing simulated paths
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[pyo3(signature = (x0, kappa, theta, sigma, T, steps, paths, parallel=false))]
    fn simulate_cir<'py>(
        &self,
        py: Python<'py>,
        x0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        T: f64,
        steps: usize,
        paths: usize,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let cir = CoxIngersollRoss::new(kappa, theta, sigma)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dt = T / steps as f64;
        let params = SimulationParams::new(x0, paths, steps, dt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result = if parallel {
            let seed = self.seed.unwrap_or_else(rand::random);
            let generator = PathGenerator::new(seed);
            generator.generate_paths_parallel(&cir, &params, seed)
        } else {
            let mut generator = match self.seed {
                Some(s) => PathGenerator::new(s),
                None => PathGenerator::from_entropy(),
            };
            generator.generate_paths(&cir, &params)
        };

        Ok(PyArray2::from_owned_array(py, result))
    }

    /// Simulate Vasicek interest rate model paths
    ///
    /// Args:
    ///     r0: Initial interest rate
    ///     kappa: Mean reversion speed
    ///     theta: Long-term mean rate
    ///     sigma: Volatility parameter
    ///     T: Total time horizon
    ///     steps: Number of time steps
    ///     paths: Number of paths to simulate
    ///     parallel: Whether to use parallel processing (default: False)
    ///
    /// Returns:
    ///     2D numpy array of shape (paths, steps + 1) containing simulated paths
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[pyo3(signature = (r0, kappa, theta, sigma, T, steps, paths, parallel=false))]
    fn simulate_vasicek<'py>(
        &self,
        py: Python<'py>,
        r0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        T: f64,
        steps: usize,
        paths: usize,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vasicek =
            Vasicek::new(kappa, theta, sigma).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dt = T / steps as f64;
        let params = SimulationParams::new(r0, paths, steps, dt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result = if parallel {
            let seed = self.seed.unwrap_or_else(rand::random);
            let generator = PathGenerator::new(seed);
            generator.generate_paths_parallel(&vasicek, &params, seed)
        } else {
            let mut generator = match self.seed {
                Some(s) => PathGenerator::new(s),
                None => PathGenerator::from_entropy(),
            };
            generator.generate_paths(&vasicek, &params)
        };

        Ok(PyArray2::from_owned_array(py, result))
    }

    /// Simulate correlated multi-asset paths using GBM
    ///
    /// Args:
    ///     initial_values: List of initial values for each asset
    ///     mu_values: List of drift parameters for each asset
    ///     sigma_values: List of volatility parameters for each asset
    ///     correlation: 2D numpy array of correlation matrix (n x n)
    ///     T: Total time horizon
    ///     steps: Number of time steps
    ///     paths: Number of paths to simulate
    ///     parallel: Whether to use parallel processing (default: False)
    ///
    /// Returns:
    ///     List of 2D numpy arrays, one per asset, each of shape (paths, steps)
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[pyo3(signature = (initial_values, mu_values, sigma_values, correlation, T, steps, paths, parallel=false))]
    fn simulate_correlated<'py>(
        &self,
        py: Python<'py>,
        initial_values: Vec<f64>,
        mu_values: Vec<f64>,
        sigma_values: Vec<f64>,
        correlation: &Bound<'py, PyArray2<f64>>,
        T: f64,
        steps: usize,
        paths: usize,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let n_assets = initial_values.len();

        if mu_values.len() != n_assets || sigma_values.len() != n_assets {
            return Err(PyValueError::new_err(
                "All parameter lists must have the same length",
            ));
        }

        // Create GBM processes for each asset
        let mut gbm_processes = Vec::new();
        for i in 0..n_assets {
            let gbm = GeometricBrownianMotion::new(mu_values[i], sigma_values[i])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            gbm_processes.push(gbm);
        }

        // Convert correlation matrix to ndarray
        let corr_array = correlation.readonly().as_array().to_owned();

        // Create simulation parameters (using first initial value as placeholder)
        let dt = T / steps as f64;
        let params = SimulationParams::new(initial_values[0], paths, steps, dt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Create process references
        let process_refs: Vec<&dyn StochasticProcess> = gbm_processes
            .iter()
            .map(|p| p as &dyn StochasticProcess)
            .collect();

        // Generate correlated paths
        let result = if parallel {
            let seed = self.seed.unwrap_or_else(rand::random);
            let generator = CorrelatedPathGenerator::new(seed);
            generator.generate_correlated_paths_parallel_with_initials(
                &process_refs,
                &initial_values,
                &corr_array,
                &params,
                seed,
            )
        } else {
            let mut generator = match self.seed {
                Some(s) => CorrelatedPathGenerator::new(s),
                None => CorrelatedPathGenerator::from_entropy(),
            };
            generator.generate_correlated_paths_with_initials(
                &process_refs,
                &initial_values,
                &corr_array,
                &params,
            )
        };

        let paths_array = result.map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Convert 3D array to list of 2D arrays (one per asset)
        // Shape: (num_assets, num_paths, num_steps + 1) -> List of (num_paths, num_steps)
        // Exclude the initial value (first column)
        let list = PyList::empty(py);
        for i in 0..paths_array.shape()[0] {
            // Slice to exclude the initial value: [asset, all_paths, 1..]
            let asset_paths = paths_array.slice(s![i, .., 1..]).to_owned();
            list.append(PyArray2::from_owned_array(py, asset_paths))?;
        }

        Ok(list)
    }

    /// Simulate Merton jump-diffusion paths
    ///
    /// Args:
    ///     s0: Initial asset price
    ///     mu: Drift parameter of the diffusion component
    ///     sigma: Volatility of the diffusion component
    ///     lambda_: Jump intensity (expected number of jumps per unit time)
    ///     jump_mean: Mean of the log jump size
    ///     jump_std: Standard deviation of the log jump size
    ///     T: Total time horizon
    ///     steps: Number of time steps
    ///     paths: Number of simulated paths
    ///     parallel: Whether to simulate paths in parallel
    ///
    /// Returns:
    ///     2D numpy array of shape (paths, steps + 1) containing simulated price paths
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    #[pyo3(signature = (s0, mu, sigma, lambda_, jump_mean, jump_std, T, steps, paths, parallel=false))]
    fn simulate_jump_diffusion<'py>(
        &self,
        py: Python<'py>,
        s0: f64,
        mu: f64,
        sigma: f64,
        lambda_: f64,
        jump_mean: f64,
        jump_std: f64,
        T: f64,
        steps: usize,
        paths: usize,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let process = MertonJumpDiffusion::new(mu, sigma, lambda_, jump_mean, jump_std)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dt = T / steps as f64;
        let params = SimulationParams::new(s0, paths, steps, dt)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result = if parallel {
            let seed = self.seed.unwrap_or_else(rand::random);
            let generator = PathGenerator::new(seed);
            generator.generate_paths_parallel(&process, &params, seed)
        } else {
            let mut generator = match self.seed {
                Some(seed) => PathGenerator::new(seed),
                None => PathGenerator::from_entropy(),
            };
            generator.generate_paths(&process, &params)
        };

        Ok(PyArray2::from_owned_array(py, result))
    }
}

/// Register Monte Carlo module with Python
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<MonteCarloEngine>()?;
    Ok(())
}
