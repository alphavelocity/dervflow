// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Monte Carlo option pricing
//!
//! Provides Monte Carlo simulation-based option pricing:
//! - European option pricing with variance reduction
//! - American option pricing using Longstaff-Schwartz algorithm
//! - Parallel path generation for performance

use crate::common::error::{DervflowError, Result};
use crate::common::types::{OptionParams, OptionType};
use crate::numerical::random::RandomGenerator;
use rayon::prelude::*;

/// Result of Monte Carlo pricing including price and standard error
#[derive(Debug, Clone, Copy)]
pub struct MonteCarloResult {
    /// Estimated option price
    pub price: f64,
    /// Standard error of the estimate
    pub standard_error: f64,
}

impl MonteCarloResult {
    /// Create a new MonteCarloResult
    pub fn new(price: f64, standard_error: f64) -> Self {
        Self {
            price,
            standard_error,
        }
    }
}

/// Generate a single GBM path and return the terminal value
fn simulate_gbm_terminal(
    spot: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time: f64,
    z: f64,
) -> f64 {
    let drift = (rate - dividend - 0.5 * volatility * volatility) * time;
    let diffusion = volatility * time.sqrt() * z;
    spot * (drift + diffusion).exp()
}

/// Calculate option payoff at maturity
fn calculate_payoff(terminal_price: f64, strike: f64, option_type: OptionType) -> f64 {
    match option_type {
        OptionType::Call => (terminal_price - strike).max(0.0),
        OptionType::Put => (strike - terminal_price).max(0.0),
    }
}

/// Price European option using Monte Carlo simulation
///
/// Uses geometric Brownian motion to simulate price paths and calculates
/// the discounted expected payoff. Supports antithetic variates for variance reduction.
///
/// # Arguments
/// * `params` - Option parameters
/// * `num_paths` - Number of simulation paths
/// * `use_antithetic` - Whether to use antithetic variates variance reduction
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `Ok(MonteCarloResult)` - Price and standard error
/// * `Err(DervflowError)` - If validation fails
///
/// # Examples
/// ```
/// use dervflow::options::monte_carlo::price_european_monte_carlo;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
/// let result = price_european_monte_carlo(&params, 10000, true, Some(42)).unwrap();
/// assert!(result.price > 0.0);
/// ```
pub fn price_european_monte_carlo(
    params: &OptionParams,
    num_paths: usize,
    use_antithetic: bool,
    seed: Option<u64>,
) -> Result<MonteCarloResult> {
    // Validate parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    if num_paths == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of paths must be positive".to_string(),
        ));
    }

    // Handle edge case: option at expiry
    if params.time_to_maturity == 0.0 {
        let intrinsic = match params.option_type {
            OptionType::Call => (params.spot - params.strike).max(0.0),
            OptionType::Put => (params.strike - params.spot).max(0.0),
        };
        return Ok(MonteCarloResult::new(intrinsic, 0.0));
    }

    // Create random number generator
    let mut rng = match seed {
        Some(s) => RandomGenerator::new(s),
        None => RandomGenerator::from_entropy(),
    };

    // Determine effective number of paths
    let effective_paths = if use_antithetic {
        // With antithetic variates, we generate half the paths and double them
        num_paths.div_ceil(2)
    } else {
        num_paths
    };

    // Simulate paths and calculate payoffs
    let mut payoffs = Vec::with_capacity(if use_antithetic {
        effective_paths * 2
    } else {
        effective_paths
    });

    for _ in 0..effective_paths {
        let z = rng.standard_normal();

        // Simulate terminal price
        let terminal_price = simulate_gbm_terminal(
            params.spot,
            params.rate,
            params.dividend,
            params.volatility,
            params.time_to_maturity,
            z,
        );

        // Calculate payoff
        let payoff = calculate_payoff(terminal_price, params.strike, params.option_type);
        payoffs.push(payoff);

        // If using antithetic variates, also simulate with -z
        if use_antithetic {
            let terminal_price_anti = simulate_gbm_terminal(
                params.spot,
                params.rate,
                params.dividend,
                params.volatility,
                params.time_to_maturity,
                -z,
            );

            let payoff_anti =
                calculate_payoff(terminal_price_anti, params.strike, params.option_type);
            payoffs.push(payoff_anti);
        }
    }

    // Calculate mean and standard error
    let n = payoffs.len() as f64;
    let mean_payoff: f64 = payoffs.iter().sum::<f64>() / n;
    let variance: f64 = payoffs
        .iter()
        .map(|p| (p - mean_payoff).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let standard_error = (variance / n).sqrt();

    // Discount to present value
    let discount_factor = (-params.rate * params.time_to_maturity).exp();
    let price = mean_payoff * discount_factor;
    let price_std_error = standard_error * discount_factor;

    Ok(MonteCarloResult::new(price, price_std_error))
}

/// Price European option using parallel Monte Carlo simulation
///
/// Uses Rayon to parallelize path generation across available CPU cores.
/// Each thread uses its own random number generator for thread safety.
///
/// # Arguments
/// * `params` - Option parameters
/// * `num_paths` - Number of simulation paths
/// * `use_antithetic` - Whether to use antithetic variates variance reduction
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `Ok(MonteCarloResult)` - Price and standard error
/// * `Err(DervflowError)` - If validation fails
pub fn price_european_monte_carlo_parallel(
    params: &OptionParams,
    num_paths: usize,
    use_antithetic: bool,
    seed: Option<u64>,
) -> Result<MonteCarloResult> {
    // Validate parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    if num_paths == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of paths must be positive".to_string(),
        ));
    }

    // Handle edge case: option at expiry
    if params.time_to_maturity == 0.0 {
        let intrinsic = match params.option_type {
            OptionType::Call => (params.spot - params.strike).max(0.0),
            OptionType::Put => (params.strike - params.spot).max(0.0),
        };
        return Ok(MonteCarloResult::new(intrinsic, 0.0));
    }

    // Determine effective number of paths
    let effective_paths = if use_antithetic {
        num_paths.div_ceil(2)
    } else {
        num_paths
    };

    // Simulate paths in parallel
    let payoffs: Vec<f64> = (0..effective_paths)
        .into_par_iter()
        .flat_map(|i| {
            // Create thread-local RNG with unique seed
            let thread_seed = seed.unwrap_or(0).wrapping_add(i as u64);
            let mut rng = RandomGenerator::new(thread_seed);

            let z = rng.standard_normal();

            // Simulate terminal price
            let terminal_price = simulate_gbm_terminal(
                params.spot,
                params.rate,
                params.dividend,
                params.volatility,
                params.time_to_maturity,
                z,
            );

            // Calculate payoff
            let payoff = calculate_payoff(terminal_price, params.strike, params.option_type);

            // If using antithetic variates, also simulate with -z
            if use_antithetic {
                let terminal_price_anti = simulate_gbm_terminal(
                    params.spot,
                    params.rate,
                    params.dividend,
                    params.volatility,
                    params.time_to_maturity,
                    -z,
                );

                let payoff_anti =
                    calculate_payoff(terminal_price_anti, params.strike, params.option_type);
                vec![payoff, payoff_anti]
            } else {
                vec![payoff]
            }
        })
        .collect();

    // Calculate mean and standard error
    let n = payoffs.len() as f64;
    let mean_payoff: f64 = payoffs.iter().sum::<f64>() / n;
    let variance: f64 = payoffs
        .iter()
        .map(|p| (p - mean_payoff).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let standard_error = (variance / n).sqrt();

    // Discount to present value
    let discount_factor = (-params.rate * params.time_to_maturity).exp();
    let price = mean_payoff * discount_factor;
    let price_std_error = standard_error * discount_factor;

    Ok(MonteCarloResult::new(price, price_std_error))
}

/// Generate full GBM paths (not just terminal values)
fn simulate_gbm_path(
    spot: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time: f64,
    num_steps: usize,
    random_normals: &[f64],
) -> Vec<f64> {
    let dt = time / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let drift = (rate - dividend - 0.5 * volatility * volatility) * dt;
    let diffusion_coef = volatility * sqrt_dt;

    let mut path = Vec::with_capacity(num_steps + 1);
    path.push(spot);

    let mut current_price = spot;
    for z in random_normals.iter().take(num_steps) {
        current_price *= (drift + diffusion_coef * z).exp();
        path.push(current_price);
    }

    path
}

/// Perform polynomial regression to estimate continuation value
///
/// Uses least squares regression with polynomial basis functions:
/// 1, x, x^2, x^3 where x is the normalized stock price
fn polynomial_regression(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return vec![0.0; degree + 1];
    }

    // Normalize x values to improve numerical stability
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let x_std: f64 = (x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let x_std = if x_std < 1e-10 { 1.0 } else { x_std };

    // Build design matrix A where A[i][j] = x[i]^j (normalized)
    let mut a = vec![vec![0.0; degree + 1]; n];
    for i in 0..n {
        let x_norm = (x[i] - x_mean) / x_std;
        a[i][0] = 1.0;
        for j in 1..=degree {
            a[i][j] = a[i][j - 1] * x_norm;
        }
    }

    // Compute A^T * A
    let mut ata = vec![vec![0.0; degree + 1]; degree + 1];
    for (i, ata_i) in ata.iter_mut().enumerate().take(degree + 1) {
        for j in 0..=degree {
            for a_k in a.iter().take(n) {
                ata_i[j] += a_k[i] * a_k[j];
            }
        }
    }

    // Compute A^T * y
    let mut aty = vec![0.0; degree + 1];
    for (i, aty_i) in aty.iter_mut().enumerate().take(degree + 1) {
        for (k, a_k) in a.iter().enumerate().take(n) {
            *aty_i += a_k[i] * y[k];
        }
    }

    // Solve (A^T * A) * beta = A^T * y using Gaussian elimination
    // Add small regularization for numerical stability
    for (i, ata_i) in ata.iter_mut().enumerate().take(degree + 1) {
        ata_i[i] += 1e-10;
    }

    // Forward elimination
    for i in 0..degree {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..=degree {
            if ata[k][i].abs() > ata[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            ata.swap(i, max_row);
            aty.swap(i, max_row);
        }

        // Eliminate column
        for k in (i + 1)..=degree {
            if ata[i][i].abs() < 1e-10 {
                continue;
            }
            let factor = ata[k][i] / ata[i][i];
            let ata_i_clone = ata[i].clone();
            for (j, ata_k_j) in ata[k].iter_mut().enumerate().skip(i).take(degree + 1 - i) {
                *ata_k_j -= factor * ata_i_clone[j];
            }
            aty[k] -= factor * aty[i];
        }
    }

    // Back substitution
    let mut beta = vec![0.0; degree + 1];
    for i in (0..=degree).rev() {
        if ata[i][i].abs() < 1e-10 {
            beta[i] = 0.0;
            continue;
        }
        let mut sum = aty[i];
        for (j, beta_j) in beta.iter().enumerate().take(degree + 1).skip(i + 1) {
            sum -= ata[i][j] * beta_j;
        }
        beta[i] = sum / ata[i][i];
    }

    beta
}

/// Evaluate polynomial with given coefficients
fn evaluate_polynomial(coeffs: &[f64], x: f64, x_mean: f64, x_std: f64) -> f64 {
    let x_norm = (x - x_mean) / x_std;
    let mut result = 0.0;
    let mut x_power = 1.0;
    for &coeff in coeffs {
        result += coeff * x_power;
        x_power *= x_norm;
    }
    result
}

/// Price American option using Longstaff-Schwartz Monte Carlo algorithm
///
/// Implements the Longstaff-Schwartz least squares Monte Carlo method for
/// American option pricing. Uses backward induction with regression to
/// estimate continuation values and determine optimal exercise strategy.
///
/// # Arguments
/// * `params` - Option parameters
/// * `num_paths` - Number of simulation paths
/// * `num_steps` - Number of time steps per path
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `Ok(MonteCarloResult)` - Price and standard error
/// * `Err(DervflowError)` - If validation fails
///
/// # Examples
/// ```
/// use dervflow::options::monte_carlo::price_american_monte_carlo;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
/// let result = price_american_monte_carlo(&params, 10000, 50, Some(42)).unwrap();
/// assert!(result.price > 0.0);
/// ```
pub fn price_american_monte_carlo(
    params: &OptionParams,
    num_paths: usize,
    num_steps: usize,
    seed: Option<u64>,
) -> Result<MonteCarloResult> {
    // Validate parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

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

    // Handle edge case: option at expiry
    if params.time_to_maturity == 0.0 {
        let intrinsic = match params.option_type {
            OptionType::Call => (params.spot - params.strike).max(0.0),
            OptionType::Put => (params.strike - params.spot).max(0.0),
        };
        return Ok(MonteCarloResult::new(intrinsic, 0.0));
    }

    // Create random number generator
    let mut rng = match seed {
        Some(s) => RandomGenerator::new(s),
        None => RandomGenerator::from_entropy(),
    };

    // Generate all paths
    let mut paths = Vec::with_capacity(num_paths);
    for _ in 0..num_paths {
        let random_normals = rng.standard_normal_vec(num_steps);
        let path = simulate_gbm_path(
            params.spot,
            params.rate,
            params.dividend,
            params.volatility,
            params.time_to_maturity,
            num_steps,
            &random_normals,
        );
        paths.push(path);
    }

    // Initialize cash flows with terminal payoffs
    let mut cash_flows = Vec::with_capacity(num_paths);
    for path in &paths {
        let terminal_price = path[num_steps];
        let payoff = calculate_payoff(terminal_price, params.strike, params.option_type);
        cash_flows.push(payoff);
    }

    // Backward induction through time steps
    let dt = params.time_to_maturity / num_steps as f64;
    let discount = (-params.rate * dt).exp();

    for step in (1..num_steps).rev() {
        // Collect in-the-money paths for regression
        let mut itm_indices = Vec::new();
        let mut itm_prices = Vec::new();
        let mut itm_continuation = Vec::new();

        for (i, path) in paths.iter().enumerate() {
            let price = path[step];
            let immediate_exercise = calculate_payoff(price, params.strike, params.option_type);

            // Only consider paths that are in the money
            if immediate_exercise > 0.0 {
                itm_indices.push(i);
                itm_prices.push(price);
                itm_continuation.push(cash_flows[i] * discount);
            }
        }

        // If we have in-the-money paths, perform regression
        if itm_indices.len() >= 4 {
            // Calculate normalization parameters
            let x_mean: f64 = itm_prices.iter().sum::<f64>() / itm_prices.len() as f64;
            let x_std: f64 = (itm_prices.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>()
                / itm_prices.len() as f64)
                .sqrt();
            let x_std = if x_std < 1e-10 { 1.0 } else { x_std };

            // Perform polynomial regression (degree 3)
            let coeffs = polynomial_regression(&itm_prices, &itm_continuation, 3);

            // Update cash flows based on optimal exercise decision
            for (idx, &path_idx) in itm_indices.iter().enumerate() {
                let price = itm_prices[idx];
                let immediate_exercise = calculate_payoff(price, params.strike, params.option_type);
                let continuation_value = evaluate_polynomial(&coeffs, price, x_mean, x_std);

                // Exercise if immediate value exceeds continuation value
                if immediate_exercise > continuation_value {
                    cash_flows[path_idx] = immediate_exercise;
                } else {
                    // Keep the discounted future cash flow
                    cash_flows[path_idx] *= discount;
                }
            }

            // Discount cash flows for paths not in the money
            for (i, _) in paths.iter().enumerate() {
                if !itm_indices.contains(&i) {
                    cash_flows[i] *= discount;
                }
            }
        } else {
            // Not enough ITM paths for regression, just discount all cash flows
            for cf in &mut cash_flows {
                *cf *= discount;
            }
        }
    }

    // Discount from first time step to present
    for cf in &mut cash_flows {
        *cf *= discount;
    }

    // Calculate mean and standard error
    let n = cash_flows.len() as f64;
    let mean_price: f64 = cash_flows.iter().sum::<f64>() / n;
    let variance: f64 = cash_flows
        .iter()
        .map(|p| (p - mean_price).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let standard_error = (variance / n).sqrt();

    Ok(MonteCarloResult::new(mean_price, standard_error))
}

/// Price American option using parallel Longstaff-Schwartz Monte Carlo
///
/// Parallel version of the Longstaff-Schwartz algorithm using Rayon.
/// Path generation is parallelized, but the backward induction remains sequential
/// as it requires information from all paths at each time step.
///
/// # Arguments
/// * `params` - Option parameters
/// * `num_paths` - Number of simulation paths
/// * `num_steps` - Number of time steps per path
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `Ok(MonteCarloResult)` - Price and standard error
/// * `Err(DervflowError)` - If validation fails
pub fn price_american_monte_carlo_parallel(
    params: &OptionParams,
    num_paths: usize,
    num_steps: usize,
    seed: Option<u64>,
) -> Result<MonteCarloResult> {
    // Validate parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

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

    // Handle edge case: option at expiry
    if params.time_to_maturity == 0.0 {
        let intrinsic = match params.option_type {
            OptionType::Call => (params.spot - params.strike).max(0.0),
            OptionType::Put => (params.strike - params.spot).max(0.0),
        };
        return Ok(MonteCarloResult::new(intrinsic, 0.0));
    }

    // Generate all paths in parallel
    let paths: Vec<Vec<f64>> = (0..num_paths)
        .into_par_iter()
        .map(|i| {
            let thread_seed = seed.unwrap_or(0).wrapping_add(i as u64);
            let mut rng = RandomGenerator::new(thread_seed);
            let random_normals = rng.standard_normal_vec(num_steps);
            simulate_gbm_path(
                params.spot,
                params.rate,
                params.dividend,
                params.volatility,
                params.time_to_maturity,
                num_steps,
                &random_normals,
            )
        })
        .collect();

    // Initialize cash flows with terminal payoffs
    let mut cash_flows: Vec<f64> = paths
        .iter()
        .map(|path| {
            let terminal_price = path[num_steps];
            calculate_payoff(terminal_price, params.strike, params.option_type)
        })
        .collect();

    // Backward induction through time steps (sequential)
    let dt = params.time_to_maturity / num_steps as f64;
    let discount = (-params.rate * dt).exp();

    for step in (1..num_steps).rev() {
        // Collect in-the-money paths for regression
        let mut itm_indices = Vec::new();
        let mut itm_prices = Vec::new();
        let mut itm_continuation = Vec::new();

        for (i, path) in paths.iter().enumerate() {
            let price = path[step];
            let immediate_exercise = calculate_payoff(price, params.strike, params.option_type);

            if immediate_exercise > 0.0 {
                itm_indices.push(i);
                itm_prices.push(price);
                itm_continuation.push(cash_flows[i] * discount);
            }
        }

        // If we have in-the-money paths, perform regression
        if itm_indices.len() >= 4 {
            let x_mean: f64 = itm_prices.iter().sum::<f64>() / itm_prices.len() as f64;
            let x_std: f64 = (itm_prices.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>()
                / itm_prices.len() as f64)
                .sqrt();
            let x_std = if x_std < 1e-10 { 1.0 } else { x_std };

            let coeffs = polynomial_regression(&itm_prices, &itm_continuation, 3);

            for (idx, &path_idx) in itm_indices.iter().enumerate() {
                let price = itm_prices[idx];
                let immediate_exercise = calculate_payoff(price, params.strike, params.option_type);
                let continuation_value = evaluate_polynomial(&coeffs, price, x_mean, x_std);

                if immediate_exercise > continuation_value {
                    cash_flows[path_idx] = immediate_exercise;
                } else {
                    cash_flows[path_idx] *= discount;
                }
            }

            for (i, _) in paths.iter().enumerate() {
                if !itm_indices.contains(&i) {
                    cash_flows[i] *= discount;
                }
            }
        } else {
            for cf in &mut cash_flows {
                *cf *= discount;
            }
        }
    }

    // Discount from first time step to present
    for cf in &mut cash_flows {
        *cf *= discount;
    }

    // Calculate mean and standard error
    let n = cash_flows.len() as f64;
    let mean_price: f64 = cash_flows.iter().sum::<f64>() / n;
    let variance: f64 = cash_flows
        .iter()
        .map(|p| (p - mean_price).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let standard_error = (variance / n).sqrt();

    Ok(MonteCarloResult::new(mean_price, standard_error))
}
