// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Exotic option pricing
//!
//! Provides pricing for path-dependent and exotic options:
//! - Asian options (arithmetic and geometric average)
//! - Barrier options (knock-in and knock-out)
//! - Lookback options (fixed and floating strike)
//! - Digital/Binary options

use crate::common::error::{DervflowError, Result};
use crate::common::types::{OptionParams, OptionType};
use crate::numerical::random::RandomGenerator;
use crate::options::monte_carlo::MonteCarloResult;
use std::f64::consts::{PI, SQRT_2};

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate the cumulative distribution function (CDF) of the standard normal distribution
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Calculate the probability density function (PDF) of the standard normal distribution
#[allow(dead_code)]
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

// ============================================================================
// Asian Options
// ============================================================================

/// Parameters for Asian option pricing
#[derive(Debug, Clone, Copy)]
pub struct AsianOptionParams {
    /// Base option parameters
    pub base_params: OptionParams,
    /// Number of averaging observations
    pub num_observations: usize,
    /// Whether to use fixed strike (true) or floating strike (false)
    pub fixed_strike: bool,
}

impl AsianOptionParams {
    /// Create new Asian option parameters
    pub fn new(base_params: OptionParams, num_observations: usize, fixed_strike: bool) -> Self {
        Self {
            base_params,
            num_observations,
            fixed_strike,
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        self.base_params
            .validate()
            .map_err(DervflowError::InvalidInput)?;
        if self.num_observations == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of observations must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration for Asian arithmetic Monte Carlo pricing
#[derive(Debug, Clone, Copy)]
pub struct AsianMonteCarloConfig {
    /// Number of simulation paths (after applying variance reduction)
    pub num_paths: usize,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to use antithetic variates for variance reduction
    pub use_antithetic: bool,
    /// Whether to apply geometric control variates (fixed-strike only)
    pub use_control_variate: bool,
}

impl Default for AsianMonteCarloConfig {
    fn default() -> Self {
        Self {
            num_paths: 10_000,
            seed: None,
            use_antithetic: true,
            use_control_variate: false,
        }
    }
}

/// Detailed Monte Carlo pricer for arithmetic-average Asian options.
///
/// Supports variance reduction through antithetic variates and a geometric
/// control variate (for fixed-strike contracts). Returns both the estimated
/// option value and its Monte Carlo standard error.
pub fn price_asian_arithmetic_mc_stats(
    params: &AsianOptionParams,
    config: &AsianMonteCarloConfig,
) -> Result<MonteCarloResult> {
    params.validate()?;

    if config.num_paths == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of paths must be positive".to_string(),
        ));
    }

    if config.use_control_variate && !params.fixed_strike {
        return Err(DervflowError::InvalidInput(
            "Geometric control variate is only available for fixed-strike Asian options"
                .to_string(),
        ));
    }

    let mut rng = match config.seed {
        Some(s) => RandomGenerator::new(s),
        None => RandomGenerator::from_entropy(),
    };

    let base = &params.base_params;
    let dt = base.time_to_maturity / params.num_observations as f64;
    let sqrt_dt = dt.sqrt();
    let drift = (base.rate - base.dividend - 0.5 * base.volatility * base.volatility) * dt;
    let diffusion = base.volatility * sqrt_dt;

    let need_geom = config.use_control_variate;
    let discount_factor = (-base.rate * base.time_to_maturity).exp();
    let geom_payoff_growth = if need_geom {
        let n = params.num_observations as f64;
        let sigma_adj = base.volatility * ((2.0 * n + 1.0) / (6.0 * (n + 1.0))).sqrt();
        let mu = (base.rate - base.dividend - 0.5 * base.volatility * base.volatility) * (n + 1.0)
            / (2.0 * n);
        let r_adj = 0.5 * (base.rate + mu + sigma_adj * sigma_adj);
        Some((r_adj * base.time_to_maturity).exp())
    } else {
        None
    };

    let mut sum_payoff = 0.0;
    let mut sum_payoff_sq = 0.0;
    let mut sum_geom_payoff = 0.0;
    let mut sum_geom_payoff_sq = 0.0;
    let mut sum_cross = 0.0;
    let mut generated_paths = 0usize;

    let mut normals = vec![0.0; params.num_observations];

    while generated_paths < config.num_paths {
        for z in normals.iter_mut() {
            *z = rng.standard_normal();
        }

        let (payoff, geom) =
            simulate_asian_path(&normals, false, params, drift, diffusion, need_geom);
        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
        if let Some(geom_payoff) = geom {
            sum_geom_payoff += geom_payoff;
            sum_geom_payoff_sq += geom_payoff * geom_payoff;
            sum_cross += payoff * geom_payoff;
        }
        generated_paths += 1;

        if config.use_antithetic && generated_paths < config.num_paths {
            let (payoff, geom) =
                simulate_asian_path(&normals, true, params, drift, diffusion, need_geom);
            sum_payoff += payoff;
            sum_payoff_sq += payoff * payoff;
            if let Some(geom_payoff) = geom {
                sum_geom_payoff += geom_payoff;
                sum_geom_payoff_sq += geom_payoff * geom_payoff;
                sum_cross += payoff * geom_payoff;
            }
            generated_paths += 1;
        }
    }

    let n = generated_paths as f64;
    let mean_payoff = sum_payoff / n;

    let mut variance = if generated_paths > 1 {
        (sum_payoff_sq - n * mean_payoff * mean_payoff) / (n - 1.0)
    } else {
        0.0
    };

    let mut price = mean_payoff * discount_factor;

    if need_geom {
        let mean_geom = sum_geom_payoff / n;
        let var_geom = if generated_paths > 1 {
            (sum_geom_payoff_sq - n * mean_geom * mean_geom) / (n - 1.0)
        } else {
            0.0
        };

        if var_geom > 1e-12 {
            let cov = if generated_paths > 1 {
                (sum_cross - n * mean_payoff * mean_geom) / (n - 1.0)
            } else {
                0.0
            };

            let gamma = cov / var_geom;
            let expected_geom_price = price_asian_geometric(params)?;
            let expected_geom_payoff = expected_geom_price * geom_payoff_growth.unwrap();
            let adjusted_mean = mean_payoff - gamma * (mean_geom - expected_geom_payoff);

            price = adjusted_mean * discount_factor;
            variance = (variance - (cov * cov) / var_geom).max(0.0);
        }
    }

    let standard_error = if generated_paths > 1 {
        (variance / n).sqrt() * discount_factor
    } else {
        0.0
    };

    Ok(MonteCarloResult::new(price, standard_error))
}

fn simulate_asian_path(
    normals: &[f64],
    invert: bool,
    params: &AsianOptionParams,
    drift: f64,
    diffusion: f64,
    need_geom: bool,
) -> (f64, Option<f64>) {
    let base = &params.base_params;
    let mut price = base.spot;
    let mut sum = 0.0;
    let mut log_sum = 0.0;

    for &raw_z in normals {
        let z = if invert { -raw_z } else { raw_z };
        price *= (drift + diffusion * z).exp();
        sum += price;
        if need_geom {
            log_sum += price.ln();
        }
    }

    let average = sum / params.num_observations as f64;
    let payoff = if params.fixed_strike {
        match base.option_type {
            OptionType::Call => (average - base.strike).max(0.0),
            OptionType::Put => (base.strike - average).max(0.0),
        }
    } else {
        match base.option_type {
            OptionType::Call => (price - average).max(0.0),
            OptionType::Put => (average - price).max(0.0),
        }
    };

    if need_geom {
        let geom_avg = (log_sum / params.num_observations as f64).exp();
        let geom_payoff = match base.option_type {
            OptionType::Call => (geom_avg - base.strike).max(0.0),
            OptionType::Put => (base.strike - geom_avg).max(0.0),
        };
        (payoff, Some(geom_payoff))
    } else {
        (payoff, None)
    }
}

/// Price arithmetic average Asian option using Monte Carlo simulation.
///
/// This compatibility wrapper retains the original signature and returns only
/// the option price. Prefer [`price_asian_arithmetic_mc_stats`] for access to
/// the Monte Carlo standard error and variance reduction controls.
pub fn price_asian_arithmetic_mc(
    params: &AsianOptionParams,
    num_paths: usize,
    seed: Option<u64>,
) -> Result<f64> {
    let config = AsianMonteCarloConfig {
        num_paths,
        seed,
        use_antithetic: false,
        use_control_variate: false,
    };

    price_asian_arithmetic_mc_stats(params, &config).map(|res| res.price)
}

/// Price geometric average Asian option using analytical approximation
pub fn price_asian_geometric(params: &AsianOptionParams) -> Result<f64> {
    params.validate()?;

    let base = &params.base_params;
    let n = params.num_observations as f64;

    // Adjusted parameters for geometric average
    let sigma_adj = base.volatility * ((2.0 * n + 1.0) / (6.0 * (n + 1.0))).sqrt();
    let mu = (base.rate - base.dividend - 0.5 * base.volatility * base.volatility) * (n + 1.0)
        / (2.0 * n);
    let r_adj = 0.5 * (base.rate + mu + sigma_adj * sigma_adj);

    // Calculate d1 and d2 for geometric average
    let forward = base.spot * ((mu + 0.5 * sigma_adj * sigma_adj) * base.time_to_maturity).exp();
    let d1 = (forward / base.strike).ln() / (sigma_adj * base.time_to_maturity.sqrt())
        + 0.5 * sigma_adj * base.time_to_maturity.sqrt();
    let d2 = d1 - sigma_adj * base.time_to_maturity.sqrt();

    // Calculate price using modified Black-Scholes formula
    let discount = (-r_adj * base.time_to_maturity).exp();
    let price = match base.option_type {
        OptionType::Call => {
            forward * discount * normal_cdf(d1) - base.strike * discount * normal_cdf(d2)
        }
        OptionType::Put => {
            base.strike * discount * normal_cdf(-d2) - forward * discount * normal_cdf(-d1)
        }
    };

    Ok(price.max(0.0))
}

// ============================================================================
// Barrier Options
// ============================================================================

/// Barrier type for barrier options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierType {
    UpAndOut,
    DownAndOut,
    UpAndIn,
    DownAndIn,
}

/// Parameters for barrier option pricing
#[derive(Debug, Clone, Copy)]
pub struct BarrierOptionParams {
    /// Base option parameters
    pub base_params: OptionParams,
    /// Barrier level
    pub barrier: f64,
    /// Barrier type
    pub barrier_type: BarrierType,
    /// Rebate paid if barrier is hit (for knock-out) or not hit (for knock-in)
    pub rebate: f64,
}

impl BarrierOptionParams {
    /// Create new barrier option parameters
    pub fn new(
        base_params: OptionParams,
        barrier: f64,
        barrier_type: BarrierType,
        rebate: f64,
    ) -> Self {
        Self {
            base_params,
            barrier,
            barrier_type,
            rebate,
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        self.base_params
            .validate()
            .map_err(DervflowError::InvalidInput)?;
        if self.barrier <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Barrier must be positive".to_string(),
            ));
        }
        if self.rebate < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Rebate must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Price barrier option using analytical formulas
pub fn price_barrier(params: &BarrierOptionParams) -> Result<f64> {
    params.validate()?;

    let base = &params.base_params;
    let s = base.spot;
    let k = base.strike;
    let h = params.barrier;
    let r = base.rate;
    let q = base.dividend;
    let sigma = base.volatility;
    let t = base.time_to_maturity;

    // Check barrier validity
    match params.barrier_type {
        BarrierType::UpAndOut | BarrierType::UpAndIn => {
            if h <= s {
                return Err(DervflowError::InvalidInput(
                    "Up barrier must be above spot price".to_string(),
                ));
            }
        }
        BarrierType::DownAndOut | BarrierType::DownAndIn => {
            if h >= s {
                return Err(DervflowError::InvalidInput(
                    "Down barrier must be below spot price".to_string(),
                ));
            }
        }
    }

    let mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma);
    let _lambda = (mu * mu + 2.0 * r / (sigma * sigma)).sqrt();

    // Calculate barrier option price using Merton's formulas
    let price = match (params.barrier_type, base.option_type) {
        (BarrierType::DownAndOut, OptionType::Call) => {
            if k >= h {
                // Standard down-and-out call
                let x1 = (s / k).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let x2 = (s / h).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y1 =
                    (h * h / (s * k)).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y2 = (h / s).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();

                let a = s * (-q * t).exp() * normal_cdf(x1)
                    - k * (-r * t).exp() * normal_cdf(x1 - sigma * t.sqrt());
                let b = s * (-q * t).exp() * normal_cdf(x2)
                    - k * (-r * t).exp() * normal_cdf(x2 - sigma * t.sqrt());
                let c = s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(y1)
                    - k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(y1 - sigma * t.sqrt());
                let d = s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(y2)
                    - k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(y2 - sigma * t.sqrt());

                a - b + c - d
            } else {
                // Strike below barrier
                let x2 = (s / h).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y2 = (h / s).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();

                let b = s * (-q * t).exp() * normal_cdf(x2)
                    - k * (-r * t).exp() * normal_cdf(x2 - sigma * t.sqrt());
                let d = s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(y2)
                    - k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(y2 - sigma * t.sqrt());

                b - d
            }
        }
        (BarrierType::UpAndOut, OptionType::Call) => {
            if k >= h {
                0.0 // Already knocked out
            } else {
                let x1 = (s / k).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y1 =
                    (h * h / (s * k)).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();

                let a = s * (-q * t).exp() * normal_cdf(x1)
                    - k * (-r * t).exp() * normal_cdf(x1 - sigma * t.sqrt());
                let c = s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(y1)
                    - k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(y1 - sigma * t.sqrt());

                a - c
            }
        }
        (BarrierType::DownAndOut, OptionType::Put) => {
            if k <= h {
                0.0 // Already knocked out
            } else {
                let x1 = (s / k).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y1 =
                    (h * h / (s * k)).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();

                let a = -s * (-q * t).exp() * normal_cdf(-x1)
                    + k * (-r * t).exp() * normal_cdf(-x1 + sigma * t.sqrt());
                let c = -s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(-y1)
                    + k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(-y1 + sigma * t.sqrt());

                a - c
            }
        }
        (BarrierType::UpAndOut, OptionType::Put) => {
            if k >= h {
                let x2 = (s / h).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y2 = (h / s).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();

                let b = -s * (-q * t).exp() * normal_cdf(-x2)
                    + k * (-r * t).exp() * normal_cdf(-x2 + sigma * t.sqrt());
                let d = -s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(-y2)
                    + k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(-y2 + sigma * t.sqrt());

                b - d
            } else {
                let x1 = (s / k).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let x2 = (s / h).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y1 =
                    (h * h / (s * k)).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
                let y2 = (h / s).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();

                let a = -s * (-q * t).exp() * normal_cdf(-x1)
                    + k * (-r * t).exp() * normal_cdf(-x1 + sigma * t.sqrt());
                let b = -s * (-q * t).exp() * normal_cdf(-x2)
                    + k * (-r * t).exp() * normal_cdf(-x2 + sigma * t.sqrt());
                let c = -s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(-y1)
                    + k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(-y1 + sigma * t.sqrt());
                let d = -s * (-q * t).exp() * (h / s).powf(2.0 * (mu + 1.0)) * normal_cdf(-y2)
                    + k * (-r * t).exp()
                        * (h / s).powf(2.0 * mu)
                        * normal_cdf(-y2 + sigma * t.sqrt());

                a - b + c - d
            }
        }
        // Knock-in options: vanilla - knock-out
        (BarrierType::DownAndIn, _) | (BarrierType::UpAndIn, _) => {
            // Calculate vanilla option price
            let vanilla = calculate_vanilla_price(base)?;

            // Calculate corresponding knock-out price
            let knockout_type = match params.barrier_type {
                BarrierType::DownAndIn => BarrierType::DownAndOut,
                BarrierType::UpAndIn => BarrierType::UpAndOut,
                _ => unreachable!(),
            };
            let knockout_params = BarrierOptionParams {
                base_params: *base,
                barrier: params.barrier,
                barrier_type: knockout_type,
                rebate: 0.0,
            };
            let knockout = price_barrier(&knockout_params)?;

            vanilla - knockout
        }
    };

    // Add rebate value if applicable
    let rebate_value = if params.rebate > 0.0 {
        params.rebate * (-r * t).exp()
    } else {
        0.0
    };

    Ok((price + rebate_value).max(0.0))
}

/// Calculate vanilla option price (helper for knock-in options)
fn calculate_vanilla_price(params: &OptionParams) -> Result<f64> {
    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.dividend;
    let sigma = params.volatility;
    let t = params.time_to_maturity;

    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let price = match params.option_type {
        OptionType::Call => {
            s * (-q * t).exp() * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
        }
        OptionType::Put => {
            k * (-r * t).exp() * normal_cdf(-d2) - s * (-q * t).exp() * normal_cdf(-d1)
        }
    };

    Ok(price.max(0.0))
}

// ============================================================================
// Lookback Options
// ============================================================================

/// Lookback option type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookbackType {
    FixedStrike,
    FloatingStrike,
}

/// Parameters for lookback option pricing
#[derive(Debug, Clone, Copy)]
pub struct LookbackOptionParams {
    /// Base option parameters
    pub base_params: OptionParams,
    /// Lookback type
    pub lookback_type: LookbackType,
    /// Current minimum (for floating strike calls) or maximum (for floating strike puts)
    pub current_extremum: Option<f64>,
}

impl LookbackOptionParams {
    /// Create new lookback option parameters
    pub fn new(
        base_params: OptionParams,
        lookback_type: LookbackType,
        current_extremum: Option<f64>,
    ) -> Self {
        Self {
            base_params,
            lookback_type,
            current_extremum,
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        self.base_params
            .validate()
            .map_err(DervflowError::InvalidInput)?;
        if self.current_extremum.map(|ext| ext <= 0.0).unwrap_or(false) {
            return Err(DervflowError::InvalidInput(
                "Current extremum must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Price lookback option using analytical formulas
pub fn price_lookback(params: &LookbackOptionParams) -> Result<f64> {
    params.validate()?;

    let base = &params.base_params;
    let s = base.spot;
    let k = base.strike;
    let r = base.rate;
    let q = base.dividend;
    let sigma = base.volatility;
    let t = base.time_to_maturity;

    if t == 0.0 {
        return Ok(0.0);
    }

    let price = match params.lookback_type {
        LookbackType::FixedStrike => {
            // Fixed strike lookback
            let a1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
            let a2 = a1 - sigma * t.sqrt();
            let a3 = ((s / k).ln() + (-r + q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());

            match base.option_type {
                OptionType::Call => {
                    let m = params.current_extremum.unwrap_or(s);
                    if m > k {
                        // Already in the money
                        s * (-q * t).exp() * normal_cdf(a1) - k * (-r * t).exp() * normal_cdf(a2)
                            + s * (-r * t).exp()
                                * (sigma * sigma / (2.0 * (r - q)))
                                * (-(r - q) * t).exp()
                                * normal_cdf(-a1)
                            + s * (-r * t).exp()
                                * (sigma * sigma / (2.0 * (r - q)))
                                * (s / k).powf(-2.0 * (r - q) / (sigma * sigma))
                                * normal_cdf(a3)
                    } else {
                        s * (-q * t).exp() * normal_cdf(a1) - k * (-r * t).exp() * normal_cdf(a2)
                            + s * (-r * t).exp()
                                * (sigma * sigma / (2.0 * (r - q)))
                                * (-(r - q) * t).exp()
                                * normal_cdf(-a1)
                            + s * (-r * t).exp()
                                * (sigma * sigma / (2.0 * (r - q)))
                                * (s / k).powf(-2.0 * (r - q) / (sigma * sigma))
                                * normal_cdf(a3)
                    }
                }
                OptionType::Put => {
                    // Both branches are identical, so we don't need the condition
                    k * (-r * t).exp() * normal_cdf(-a2) - s * (-q * t).exp() * normal_cdf(-a1)
                        + s * (-r * t).exp()
                            * (sigma * sigma / (2.0 * (r - q)))
                            * (-(r - q) * t).exp()
                            * normal_cdf(a1)
                        - s * (-r * t).exp()
                            * (sigma * sigma / (2.0 * (r - q)))
                            * (s / k).powf(-2.0 * (r - q) / (sigma * sigma))
                            * normal_cdf(-a3)
                }
            }
        }
        LookbackType::FloatingStrike => {
            // Floating strike lookback
            let m = params.current_extremum.unwrap_or(s);

            match base.option_type {
                OptionType::Call => {
                    // Payoff: S_T - min(S_t)
                    let b1 =
                        ((s / m).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
                    let b2 = b1 - sigma * t.sqrt();
                    let b3 =
                        ((s / m).ln() + (-r + q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());

                    s * (-q * t).exp() * normal_cdf(b1) - m * (-q * t).exp() * normal_cdf(b2)
                        + s * (-q * t).exp()
                            * (sigma * sigma / (2.0 * (r - q)))
                            * (-(r - q) * t).exp()
                            * normal_cdf(-b1)
                        - s * (-q * t).exp()
                            * (sigma * sigma / (2.0 * (r - q)))
                            * (s / m).powf(-2.0 * (r - q) / (sigma * sigma))
                            * normal_cdf(b3)
                }
                OptionType::Put => {
                    // Payoff: max(S_t) - S_T
                    let b1 =
                        ((m / s).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
                    let b2 = b1 - sigma * t.sqrt();
                    let b3 =
                        ((m / s).ln() + (-r + q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());

                    m * (-q * t).exp() * normal_cdf(b1)
                        - s * (-q * t).exp() * normal_cdf(b2)
                        - s * (-q * t).exp()
                            * (sigma * sigma / (2.0 * (r - q)))
                            * (-(r - q) * t).exp()
                            * normal_cdf(-b1)
                        + s * (-q * t).exp()
                            * (sigma * sigma / (2.0 * (r - q)))
                            * (m / s).powf(-2.0 * (r - q) / (sigma * sigma))
                            * normal_cdf(-b3)
                }
            }
        }
    };

    Ok(price.max(0.0))
}

// ============================================================================
// Digital/Binary Options
// ============================================================================

/// Digital option type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalType {
    CashOrNothing,
    AssetOrNothing,
}

/// Parameters for digital option pricing
#[derive(Debug, Clone, Copy)]
pub struct DigitalOptionParams {
    /// Base option parameters
    pub base_params: OptionParams,
    /// Digital option type
    pub digital_type: DigitalType,
    /// Cash payout for cash-or-nothing options
    pub cash_payout: f64,
}

impl DigitalOptionParams {
    /// Create new digital option parameters
    pub fn new(base_params: OptionParams, digital_type: DigitalType, cash_payout: f64) -> Self {
        Self {
            base_params,
            digital_type,
            cash_payout,
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        self.base_params
            .validate()
            .map_err(DervflowError::InvalidInput)?;
        if self.cash_payout < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Cash payout must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Price digital/binary option using analytical formulas
pub fn price_digital(params: &DigitalOptionParams) -> Result<f64> {
    params.validate()?;

    let base = &params.base_params;
    let s = base.spot;
    let k = base.strike;
    let r = base.rate;
    let q = base.dividend;
    let sigma = base.volatility;
    let t = base.time_to_maturity;

    if t == 0.0 {
        let intrinsic = match base.option_type {
            OptionType::Call => {
                if s > k {
                    1.0
                } else {
                    0.0
                }
            }
            OptionType::Put => {
                if s < k {
                    1.0
                } else {
                    0.0
                }
            }
        };
        return Ok(intrinsic * params.cash_payout);
    }

    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let price = match params.digital_type {
        DigitalType::CashOrNothing => {
            // Pays fixed cash amount if option ends in the money
            let prob = match base.option_type {
                OptionType::Call => normal_cdf(d2),
                OptionType::Put => normal_cdf(-d2),
            };
            params.cash_payout * (-r * t).exp() * prob
        }
        DigitalType::AssetOrNothing => {
            // Pays asset value if option ends in the money
            let prob = match base.option_type {
                OptionType::Call => normal_cdf(d1),
                OptionType::Put => normal_cdf(-d1),
            };
            s * (-q * t).exp() * prob
        }
    };

    Ok(price.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asian_option_params_validation() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params = AsianOptionParams::new(base_params, 12, true);
        assert!(params.validate().is_ok());

        // Test zero observations
        let invalid_params = AsianOptionParams::new(base_params, 0, true);
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_barrier_option_params_validation() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params = BarrierOptionParams::new(base_params, 90.0, BarrierType::DownAndOut, 0.0);
        assert!(params.validate().is_ok());

        // Test negative barrier
        let invalid_params =
            BarrierOptionParams::new(base_params, -90.0, BarrierType::DownAndOut, 0.0);
        assert!(invalid_params.validate().is_err());

        // Test negative rebate
        let invalid_params =
            BarrierOptionParams::new(base_params, 90.0, BarrierType::DownAndOut, -5.0);
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_lookback_option_params_validation() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params = LookbackOptionParams::new(base_params, LookbackType::FixedStrike, None);
        assert!(params.validate().is_ok());

        let params_with_extremum =
            LookbackOptionParams::new(base_params, LookbackType::FloatingStrike, Some(95.0));
        assert!(params_with_extremum.validate().is_ok());

        // Test negative extremum
        let invalid_params =
            LookbackOptionParams::new(base_params, LookbackType::FloatingStrike, Some(-95.0));
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_digital_option_params_validation() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params = DigitalOptionParams::new(base_params, DigitalType::CashOrNothing, 10.0);
        assert!(params.validate().is_ok());

        // Test negative payout
        let invalid_params =
            DigitalOptionParams::new(base_params, DigitalType::CashOrNothing, -10.0);
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_asian_geometric_vs_arithmetic() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params = AsianOptionParams::new(base_params, 12, true);

        let geom_price = price_asian_geometric(&params).unwrap();
        let arith_price = price_asian_arithmetic_mc(&params, 10000, Some(42)).unwrap();

        // Geometric average is always <= arithmetic average, so geometric Asian should be cheaper
        assert!(geom_price <= arith_price);
    }

    #[test]
    fn test_asian_arithmetic_mc_stats_matches_price() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params = AsianOptionParams::new(base_params, 12, true);

        let config = AsianMonteCarloConfig {
            num_paths: 4096,
            seed: Some(1337),
            use_antithetic: false,
            use_control_variate: false,
        };

        let stats = price_asian_arithmetic_mc_stats(&params, &config).unwrap();
        let price = price_asian_arithmetic_mc(&params, config.num_paths, config.seed).unwrap();

        assert!((stats.price - price).abs() < 1e-10);
        assert!(stats.standard_error > 0.0);
    }

    #[test]
    fn test_asian_arithmetic_control_variate_reduces_variance() {
        let base_params = OptionParams::new(100.0, 95.0, 0.03, 0.0, 0.25, 1.0, OptionType::Call);
        let params = AsianOptionParams::new(base_params, 24, true);

        let mut base_config = AsianMonteCarloConfig {
            num_paths: 8192,
            seed: Some(7),
            use_antithetic: false,
            use_control_variate: false,
        };

        let plain = price_asian_arithmetic_mc_stats(&params, &base_config).unwrap();

        base_config.use_control_variate = true;
        let cv = price_asian_arithmetic_mc_stats(&params, &base_config).unwrap();

        assert!(cv.standard_error <= plain.standard_error);
    }

    #[test]
    fn test_asian_arithmetic_control_variate_requires_fixed_strike() {
        let base_params = OptionParams::new(100.0, 100.0, 0.03, 0.0, 0.2, 1.0, OptionType::Call);
        let params = AsianOptionParams::new(base_params, 12, false);

        let config = AsianMonteCarloConfig {
            num_paths: 128,
            seed: Some(1),
            use_antithetic: false,
            use_control_variate: true,
        };

        let err = price_asian_arithmetic_mc_stats(&params, &config).unwrap_err();
        match err {
            DervflowError::InvalidInput(msg) => {
                assert_eq!(
                    msg,
                    "Geometric control variate is only available for fixed-strike Asian options"
                );
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_barrier_knock_in_plus_knock_out() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        let knock_out_params =
            BarrierOptionParams::new(base_params, 90.0, BarrierType::DownAndOut, 0.0);
        let knock_in_params =
            BarrierOptionParams::new(base_params, 90.0, BarrierType::DownAndIn, 0.0);

        let knock_out = price_barrier(&knock_out_params).unwrap();
        let knock_in = price_barrier(&knock_in_params).unwrap();
        let vanilla = calculate_vanilla_price(&base_params).unwrap();

        // Knock-in + knock-out should equal vanilla
        assert!((knock_in + knock_out - vanilla).abs() < 0.01);
    }

    #[test]
    fn test_digital_call_plus_put() {
        let base_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let payout = 10.0;

        let call_params = DigitalOptionParams::new(base_params, DigitalType::CashOrNothing, payout);
        let put_params = DigitalOptionParams::new(
            OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put),
            DigitalType::CashOrNothing,
            payout,
        );

        let call_price = price_digital(&call_params).unwrap();
        let put_price = price_digital(&put_params).unwrap();
        let discounted_payout = payout * (-0.05_f64 * 1.0).exp();

        // Call + put should equal discounted payout
        assert!((call_price + put_price - discounted_payout).abs() < 0.01);
    }
}
