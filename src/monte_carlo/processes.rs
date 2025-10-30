// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Stochastic processes
//!
//! Implements various stochastic processes for Monte Carlo simulation:
//! - Diffusion processes (GBM, OU, CIR, Vasicek)
//! - Jump processes (Merton, Kou)
//! - Stochastic volatility models (Heston, SABR)

use crate::common::error::{DervflowError, Result};
use crate::numerical::random::RandomGenerator;

/// Trait for stochastic processes
pub trait StochasticProcess: Send + Sync {
    /// Compute the drift term μ(t, x)
    fn drift(&self, t: f64, x: f64) -> f64;

    /// Compute the diffusion term σ(t, x)
    fn diffusion(&self, t: f64, x: f64) -> f64;

    /// Simulate one time step using Euler-Maruyama scheme
    /// x(t + dt) = x(t) + μ(t, x) * dt + σ(t, x) * sqrt(dt) * dW
    fn simulate_step(&self, t: f64, x: f64, dt: f64, dw: f64) -> f64 {
        x + self.drift(t, x) * dt + self.diffusion(t, x) * dt.sqrt() * dw
    }

    /// Get the name of the process
    fn process_name(&self) -> &str;
}

/// Geometric Brownian Motion (GBM)
///
/// dS(t) = μ * S(t) * dt + σ * S(t) * dW(t)
///
/// Used for modeling stock prices under the Black-Scholes framework
#[derive(Debug, Clone, Copy)]
pub struct GeometricBrownianMotion {
    /// Drift parameter (expected return)
    pub mu: f64,
    /// Volatility parameter
    pub sigma: f64,
}

impl GeometricBrownianMotion {
    /// Create a new GBM process
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Volatility must be non-negative".to_string(),
            ));
        }
        Ok(Self { mu, sigma })
    }
}

impl StochasticProcess for GeometricBrownianMotion {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.mu * x
    }

    fn diffusion(&self, _t: f64, x: f64) -> f64 {
        self.sigma * x
    }

    fn process_name(&self) -> &str {
        "Geometric Brownian Motion"
    }
}

/// Ornstein-Uhlenbeck (OU) process
///
/// dX(t) = θ * (μ - X(t)) * dt + σ * dW(t)
///
/// Mean-reverting process used for modeling interest rates, volatility, and spreads
#[derive(Debug, Clone, Copy)]
pub struct OrnsteinUhlenbeck {
    /// Mean reversion speed
    pub theta: f64,
    /// Long-term mean
    pub mu: f64,
    /// Volatility parameter
    pub sigma: f64,
}

impl OrnsteinUhlenbeck {
    /// Create a new OU process
    pub fn new(theta: f64, mu: f64, sigma: f64) -> Result<Self> {
        if theta <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Mean reversion speed must be positive".to_string(),
            ));
        }
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Volatility must be non-negative".to_string(),
            ));
        }
        Ok(Self { theta, mu, sigma })
    }
}

impl StochasticProcess for OrnsteinUhlenbeck {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.theta * (self.mu - x)
    }

    fn diffusion(&self, _t: f64, _x: f64) -> f64 {
        self.sigma
    }

    fn process_name(&self) -> &str {
        "Ornstein-Uhlenbeck"
    }
}

/// Cox-Ingersoll-Ross (CIR) process
///
/// dX(t) = κ * (θ - X(t)) * dt + σ * sqrt(X(t)) * dW(t)
///
/// Mean-reverting process with square-root diffusion, used for interest rate modeling
/// Ensures non-negative values when 2κθ ≥ σ² (Feller condition)
#[derive(Debug, Clone, Copy)]
pub struct CoxIngersollRoss {
    /// Mean reversion speed
    pub kappa: f64,
    /// Long-term mean
    pub theta: f64,
    /// Volatility parameter
    pub sigma: f64,
}

impl CoxIngersollRoss {
    /// Create a new CIR process
    pub fn new(kappa: f64, theta: f64, sigma: f64) -> Result<Self> {
        if kappa <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Mean reversion speed must be positive".to_string(),
            ));
        }
        if theta <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Long-term mean must be positive".to_string(),
            ));
        }
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Volatility must be non-negative".to_string(),
            ));
        }

        // Check Feller condition (ensures process stays positive)
        if 2.0 * kappa * theta < sigma * sigma {
            eprintln!(
                "Warning: Feller condition not satisfied (2κθ < σ²). Process may reach zero."
            );
        }

        Ok(Self {
            kappa,
            theta,
            sigma,
        })
    }

    /// Check if Feller condition is satisfied
    pub fn satisfies_feller_condition(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.sigma * self.sigma
    }
}

impl StochasticProcess for CoxIngersollRoss {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.kappa * (self.theta - x)
    }

    fn diffusion(&self, _t: f64, x: f64) -> f64 {
        // Ensure non-negative argument for sqrt
        self.sigma * x.max(0.0).sqrt()
    }

    fn simulate_step(&self, t: f64, x: f64, dt: f64, dw: f64) -> f64 {
        // Use full truncation scheme to ensure positivity
        let x_pos = x.max(0.0);
        let new_x = x_pos + self.drift(t, x_pos) * dt + self.diffusion(t, x_pos) * dt.sqrt() * dw;
        new_x.max(0.0)
    }

    fn process_name(&self) -> &str {
        "Cox-Ingersoll-Ross"
    }
}

/// Vasicek interest rate model
///
/// dr(t) = κ * (θ - r(t)) * dt + σ * dW(t)
///
/// Mean-reverting process for interest rates (similar to OU but specifically for rates)
/// Can produce negative rates
#[derive(Debug, Clone, Copy)]
pub struct Vasicek {
    /// Mean reversion speed
    pub kappa: f64,
    /// Long-term mean rate
    pub theta: f64,
    /// Volatility parameter
    pub sigma: f64,
}

impl Vasicek {
    /// Create a new Vasicek process
    pub fn new(kappa: f64, theta: f64, sigma: f64) -> Result<Self> {
        if kappa <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Mean reversion speed must be positive".to_string(),
            ));
        }
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Volatility must be non-negative".to_string(),
            ));
        }
        Ok(Self {
            kappa,
            theta,
            sigma,
        })
    }
}

impl StochasticProcess for Vasicek {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.kappa * (self.theta - x)
    }

    fn diffusion(&self, _t: f64, _x: f64) -> f64 {
        self.sigma
    }

    fn process_name(&self) -> &str {
        "Vasicek"
    }
}

/// Merton Jump-Diffusion model
///
/// dS(t) = μ * S(t) * dt + σ * S(t) * dW(t) + S(t) * dJ(t)
///
/// where J(t) is a compound Poisson process with jump intensity λ
/// and log-normal jump sizes: log(1 + J) ~ N(μ_J, σ_J²)
///
/// Used for modeling asset prices with sudden jumps (e.g., earnings announcements, market crashes)
#[derive(Debug, Clone, Copy)]
pub struct MertonJumpDiffusion {
    /// Drift parameter (expected return)
    pub mu: f64,
    /// Diffusion volatility
    pub sigma: f64,
    /// Jump intensity (average number of jumps per unit time)
    pub lambda: f64,
    /// Mean of log jump size
    pub jump_mean: f64,
    /// Standard deviation of log jump size
    pub jump_std: f64,
}

impl MertonJumpDiffusion {
    /// Create a new Merton jump-diffusion process
    pub fn new(mu: f64, sigma: f64, lambda: f64, jump_mean: f64, jump_std: f64) -> Result<Self> {
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Diffusion volatility must be non-negative".to_string(),
            ));
        }
        if lambda < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Jump intensity must be non-negative".to_string(),
            ));
        }
        if jump_std < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Jump standard deviation must be non-negative".to_string(),
            ));
        }
        Ok(Self {
            mu,
            sigma,
            lambda,
            jump_mean,
            jump_std,
        })
    }

    /// Simulate one step with jumps
    pub fn simulate_step_with_rng(
        &self,
        t: f64,
        x: f64,
        dt: f64,
        dw: f64,
        rng: &mut RandomGenerator,
    ) -> f64 {
        // Diffusion component
        let diffusion_part = x + self.drift(t, x) * dt + self.diffusion(t, x) * dt.sqrt() * dw;

        // Jump component
        let num_jumps = self.sample_poisson(self.lambda * dt, rng);
        let mut jump_product = 1.0;

        for _ in 0..num_jumps {
            let log_jump = rng.normal(self.jump_mean, self.jump_std);
            jump_product *= log_jump.exp();
        }

        diffusion_part * jump_product
    }

    /// Sample from Poisson distribution
    fn sample_poisson(&self, lambda: f64, rng: &mut RandomGenerator) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }

        // Use Knuth's algorithm for small lambda
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= rng.uniform();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // Use normal approximation for large lambda
            let sample = rng.normal(lambda, lambda.sqrt());
            sample.max(0.0).round() as u32
        }
    }
}

impl StochasticProcess for MertonJumpDiffusion {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.mu * x
    }

    fn diffusion(&self, _t: f64, x: f64) -> f64 {
        self.sigma * x
    }

    fn process_name(&self) -> &str {
        "Merton Jump-Diffusion"
    }
}

/// Kou Double Exponential Jump-Diffusion model
///
/// dS(t) = μ * S(t) * dt + σ * S(t) * dW(t) + S(t) * dJ(t)
///
/// where J(t) is a compound Poisson process with jump intensity λ
/// and double exponential jump sizes:
/// - Upward jumps: Y ~ Exp(η₁) with probability p
/// - Downward jumps: Y ~ -Exp(η₂) with probability (1-p)
///
/// Provides better fit to empirical distributions with asymmetric jumps
#[derive(Debug, Clone, Copy)]
pub struct KouJumpDiffusion {
    /// Drift parameter (expected return)
    pub mu: f64,
    /// Diffusion volatility
    pub sigma: f64,
    /// Jump intensity (average number of jumps per unit time)
    pub lambda: f64,
    /// Probability of upward jump
    pub p_up: f64,
    /// Rate parameter for upward jumps (η₁)
    pub eta_up: f64,
    /// Rate parameter for downward jumps (η₂)
    pub eta_down: f64,
}

impl KouJumpDiffusion {
    /// Create a new Kou jump-diffusion process
    pub fn new(
        mu: f64,
        sigma: f64,
        lambda: f64,
        p_up: f64,
        eta_up: f64,
        eta_down: f64,
    ) -> Result<Self> {
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Diffusion volatility must be non-negative".to_string(),
            ));
        }
        if lambda < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Jump intensity must be non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&p_up) {
            return Err(DervflowError::InvalidInput(
                "Upward jump probability must be in [0, 1]".to_string(),
            ));
        }
        if eta_up <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Upward jump rate must be positive".to_string(),
            ));
        }
        if eta_down <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Downward jump rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            mu,
            sigma,
            lambda,
            p_up,
            eta_up,
            eta_down,
        })
    }

    /// Simulate one step with jumps
    pub fn simulate_step_with_rng(
        &self,
        t: f64,
        x: f64,
        dt: f64,
        dw: f64,
        rng: &mut RandomGenerator,
    ) -> f64 {
        // Diffusion component
        let diffusion_part = x + self.drift(t, x) * dt + self.diffusion(t, x) * dt.sqrt() * dw;

        // Jump component
        let num_jumps = self.sample_poisson(self.lambda * dt, rng);
        let mut jump_sum = 0.0;

        for _ in 0..num_jumps {
            let u = rng.uniform();
            let jump_size = if u < self.p_up {
                // Upward jump: exponential distribution
                -rng.uniform().ln() / self.eta_up
            } else {
                // Downward jump: negative exponential distribution
                rng.uniform().ln() / self.eta_down
            };
            jump_sum += jump_size;
        }

        diffusion_part * jump_sum.exp()
    }

    /// Sample from Poisson distribution
    fn sample_poisson(&self, lambda: f64, rng: &mut RandomGenerator) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }

        // Use Knuth's algorithm for small lambda
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= rng.uniform();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // Use normal approximation for large lambda
            let sample = rng.normal(lambda, lambda.sqrt());
            sample.max(0.0).round() as u32
        }
    }
}

impl StochasticProcess for KouJumpDiffusion {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.mu * x
    }

    fn diffusion(&self, _t: f64, x: f64) -> f64 {
        self.sigma * x
    }

    fn process_name(&self) -> &str {
        "Kou Jump-Diffusion"
    }
}

/// Heston Stochastic Volatility Model
///
/// Asset price: dS(t) = μ * S(t) * dt + sqrt(V(t)) * S(t) * dW₁(t)
/// Variance: dV(t) = κ * (θ - V(t)) * dt + σ * sqrt(V(t)) * dW₂(t)
///
/// where W₁ and W₂ are correlated Brownian motions with correlation ρ
///
/// Used for modeling stochastic volatility in option pricing
#[derive(Debug, Clone, Copy)]
pub struct HestonModel {
    /// Drift parameter for asset price
    pub mu: f64,
    /// Mean reversion speed for variance
    pub kappa: f64,
    /// Long-term variance mean
    pub theta: f64,
    /// Volatility of volatility
    pub sigma: f64,
    /// Correlation between asset and variance Brownian motions
    pub rho: f64,
}

impl HestonModel {
    /// Create a new Heston model
    pub fn new(mu: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Result<Self> {
        if kappa <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Mean reversion speed must be positive".to_string(),
            ));
        }
        if theta <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Long-term variance must be positive".to_string(),
            ));
        }
        if sigma < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Volatility of volatility must be non-negative".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&rho) {
            return Err(DervflowError::InvalidInput(
                "Correlation must be in [-1, 1]".to_string(),
            ));
        }

        // Check Feller condition for variance process
        if 2.0 * kappa * theta < sigma * sigma {
            eprintln!(
                "Warning: Feller condition not satisfied (2κθ < σ²). Variance may reach zero."
            );
        }

        Ok(Self {
            mu,
            kappa,
            theta,
            sigma,
            rho,
        })
    }

    /// Check if Feller condition is satisfied
    pub fn satisfies_feller_condition(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.sigma * self.sigma
    }

    /// Simulate one step for both asset price and variance
    /// Returns (new_price, new_variance)
    pub fn simulate_step_full(
        &self,
        _t: f64,
        s: f64,
        v: f64,
        dt: f64,
        dw1: f64,
        dw2: f64,
    ) -> (f64, f64) {
        // Ensure variance is non-negative
        let v_pos = v.max(0.0);

        // Simulate variance using full truncation scheme
        let v_drift = self.kappa * (self.theta - v_pos);
        let v_diffusion = self.sigma * v_pos.sqrt();
        let new_v = (v_pos + v_drift * dt + v_diffusion * dt.sqrt() * dw2).max(0.0);

        // Simulate asset price using current variance
        let s_drift = self.mu * s;
        let s_diffusion = v_pos.sqrt() * s;
        let new_s = s + s_drift * dt + s_diffusion * dt.sqrt() * dw1;

        (new_s, new_v)
    }
}

/// SABR (Stochastic Alpha Beta Rho) Model
///
/// Forward rate: dF(t) = α(t) * F(t)^β * dW₁(t)
/// Volatility: dα(t) = ν * α(t) * dW₂(t)
///
/// where W₁ and W₂ are correlated Brownian motions with correlation ρ
///
/// Widely used for interest rate derivatives and volatility surface modeling
#[derive(Debug, Clone, Copy)]
pub struct SABRModel {
    /// Initial volatility (alpha)
    pub alpha: f64,
    /// Beta parameter (elasticity)
    pub beta: f64,
    /// Volatility of volatility (nu)
    pub nu: f64,
    /// Correlation between forward and volatility
    pub rho: f64,
}

impl SABRModel {
    /// Create a new SABR model
    pub fn new(alpha: f64, beta: f64, nu: f64, rho: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Initial volatility must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(DervflowError::InvalidInput(
                "Beta must be in [0, 1]".to_string(),
            ));
        }
        if nu < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Volatility of volatility must be non-negative".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&rho) {
            return Err(DervflowError::InvalidInput(
                "Correlation must be in [-1, 1]".to_string(),
            ));
        }

        Ok(Self {
            alpha,
            beta,
            nu,
            rho,
        })
    }

    /// Simulate one step for both forward rate and volatility
    /// Returns (new_forward, new_alpha)
    pub fn simulate_step_full(
        &self,
        _t: f64,
        f: f64,
        alpha: f64,
        dt: f64,
        dw1: f64,
        dw2: f64,
    ) -> (f64, f64) {
        // Ensure positive values
        let f_pos = f.max(1e-10);
        let alpha_pos = alpha.max(1e-10);

        // Simulate volatility (log-normal process)
        let new_alpha = alpha_pos * (self.nu * dt.sqrt() * dw2).exp();

        // Simulate forward rate
        let f_diffusion = alpha_pos * f_pos.powf(self.beta);
        let new_f = f_pos + f_diffusion * dt.sqrt() * dw1;

        (new_f.max(0.0), new_alpha)
    }

    /// Calculate implied volatility using SABR formula (Hagan approximation)
    pub fn implied_volatility(&self, forward: f64, strike: f64, time: f64) -> Result<f64> {
        if forward <= 0.0 || strike <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Forward and strike must be positive".to_string(),
            ));
        }
        if time <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Time must be positive".to_string(),
            ));
        }

        let eps = 1e-7;

        // ATM case
        if (forward - strike).abs() < eps {
            let f_mid = (forward + strike) / 2.0;
            let f_beta = f_mid.powf(self.beta);

            let term1 = self.alpha / f_beta;
            let term2 = (1.0 - self.beta).powi(2) * self.alpha.powi(2) / (24.0 * f_beta.powi(2));
            let term3 = self.rho * self.beta * self.nu * self.alpha / (4.0 * f_beta);
            let term4 = (2.0 - 3.0 * self.rho.powi(2)) * self.nu.powi(2) / 24.0;

            return Ok(term1 * (1.0 + (term2 + term3 + term4) * time));
        }

        // Non-ATM case
        let log_fk = (forward / strike).ln();
        let f_mid = (forward * strike).sqrt();
        let f_beta = f_mid.powf(self.beta - 1.0);

        let z = (self.nu / self.alpha) * f_beta * log_fk;
        let x_z =
            ((1.0 - 2.0 * self.rho * z + z * z).sqrt() + z - self.rho).ln() / (1.0 - self.rho);

        let numerator = self.alpha;
        let denominator = f_beta
            * (1.0
                + (1.0 - self.beta).powi(2) * log_fk.powi(2) / 24.0
                + (1.0 - self.beta).powi(4) * log_fk.powi(4) / 1920.0);

        let term1 = numerator / denominator;
        let term2 = z / x_z;

        let term3 =
            (1.0 - self.beta).powi(2) * self.alpha.powi(2) / (24.0 * f_mid.powf(2.0 * self.beta));
        let term4 = self.rho * self.beta * self.nu * self.alpha / (4.0 * f_mid.powf(self.beta));
        let term5 = (2.0 - 3.0 * self.rho.powi(2)) * self.nu.powi(2) / 24.0;

        Ok(term1 * term2 * (1.0 + (term3 + term4 + term5) * time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gbm_creation() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        assert_eq!(gbm.mu, 0.05);
        assert_eq!(gbm.sigma, 0.2);
        assert_eq!(gbm.process_name(), "Geometric Brownian Motion");
    }

    #[test]
    fn test_gbm_invalid_volatility() {
        let result = GeometricBrownianMotion::new(0.05, -0.2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gbm_drift_diffusion() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let x = 100.0;

        assert_relative_eq!(gbm.drift(0.0, x), 0.05 * x, epsilon = 1e-10);
        assert_relative_eq!(gbm.diffusion(0.0, x), 0.2 * x, epsilon = 1e-10);
    }

    #[test]
    fn test_ou_creation() {
        let ou = OrnsteinUhlenbeck::new(0.5, 0.03, 0.01).unwrap();
        assert_eq!(ou.theta, 0.5);
        assert_eq!(ou.mu, 0.03);
        assert_eq!(ou.sigma, 0.01);
        assert_eq!(ou.process_name(), "Ornstein-Uhlenbeck");
    }

    #[test]
    fn test_ou_invalid_params() {
        assert!(OrnsteinUhlenbeck::new(-0.5, 0.03, 0.01).is_err());
        assert!(OrnsteinUhlenbeck::new(0.5, 0.03, -0.01).is_err());
    }

    #[test]
    fn test_ou_mean_reversion() {
        let ou = OrnsteinUhlenbeck::new(0.5, 0.03, 0.01).unwrap();

        // When x > mu, drift should be negative (pulling down)
        assert!(ou.drift(0.0, 0.05) < 0.0);

        // When x < mu, drift should be positive (pulling up)
        assert!(ou.drift(0.0, 0.01) > 0.0);

        // When x = mu, drift should be zero
        assert_relative_eq!(ou.drift(0.0, 0.03), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cir_creation() {
        let cir = CoxIngersollRoss::new(0.5, 0.03, 0.1).unwrap();
        assert_eq!(cir.kappa, 0.5);
        assert_eq!(cir.theta, 0.03);
        assert_eq!(cir.sigma, 0.1);
        assert_eq!(cir.process_name(), "Cox-Ingersoll-Ross");
    }

    #[test]
    fn test_cir_invalid_params() {
        assert!(CoxIngersollRoss::new(-0.5, 0.03, 0.1).is_err());
        assert!(CoxIngersollRoss::new(0.5, -0.03, 0.1).is_err());
        assert!(CoxIngersollRoss::new(0.5, 0.03, -0.1).is_err());
    }

    #[test]
    fn test_cir_feller_condition() {
        // Satisfies Feller condition: 2 * 0.5 * 0.03 = 0.03 >= 0.1^2 = 0.01
        let cir_good = CoxIngersollRoss::new(0.5, 0.03, 0.1).unwrap();
        assert!(cir_good.satisfies_feller_condition());

        // Violates Feller condition: 2 * 0.1 * 0.01 = 0.002 < 0.1^2 = 0.01
        let cir_bad = CoxIngersollRoss::new(0.1, 0.01, 0.1).unwrap();
        assert!(!cir_bad.satisfies_feller_condition());
    }

    #[test]
    fn test_cir_diffusion_non_negative() {
        let cir = CoxIngersollRoss::new(0.5, 0.03, 0.1).unwrap();

        // Positive value
        assert!(cir.diffusion(0.0, 0.04) > 0.0);

        // Zero value
        assert_relative_eq!(cir.diffusion(0.0, 0.0), 0.0, epsilon = 1e-10);

        // Negative value (should be treated as zero)
        assert_relative_eq!(cir.diffusion(0.0, -0.01), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cir_simulate_step_positivity() {
        let cir = CoxIngersollRoss::new(0.5, 0.03, 0.1).unwrap();

        // Even with large negative shock, result should be non-negative
        let x = 0.01;
        let dt = 0.01;
        let dw = -10.0; // Large negative shock

        let new_x = cir.simulate_step(0.0, x, dt, dw);
        assert!(new_x >= 0.0);
    }

    #[test]
    fn test_vasicek_creation() {
        let vasicek = Vasicek::new(0.5, 0.03, 0.01).unwrap();
        assert_eq!(vasicek.kappa, 0.5);
        assert_eq!(vasicek.theta, 0.03);
        assert_eq!(vasicek.sigma, 0.01);
        assert_eq!(vasicek.process_name(), "Vasicek");
    }

    #[test]
    fn test_vasicek_invalid_params() {
        assert!(Vasicek::new(-0.5, 0.03, 0.01).is_err());
        assert!(Vasicek::new(0.5, 0.03, -0.01).is_err());
    }

    #[test]
    fn test_vasicek_mean_reversion() {
        let vasicek = Vasicek::new(0.5, 0.03, 0.01).unwrap();

        // When r > theta, drift should be negative
        assert!(vasicek.drift(0.0, 0.05) < 0.0);

        // When r < theta, drift should be positive
        assert!(vasicek.drift(0.0, 0.01) > 0.0);

        // When r = theta, drift should be zero
        assert_relative_eq!(vasicek.drift(0.0, 0.03), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simulate_step() {
        let gbm = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
        let x0 = 100.0;
        let dt = 0.01;
        let dw = 0.5;

        let x1 = gbm.simulate_step(0.0, x0, dt, dw);

        // Check that result is reasonable
        assert!(x1 > 0.0);
        assert!((x1 - x0).abs() < 10.0); // Should not jump too much in one step
    }

    #[test]
    fn test_merton_jump_diffusion_creation() {
        let mjd = MertonJumpDiffusion::new(0.05, 0.2, 1.0, -0.1, 0.15).unwrap();
        assert_eq!(mjd.mu, 0.05);
        assert_eq!(mjd.sigma, 0.2);
        assert_eq!(mjd.lambda, 1.0);
        assert_eq!(mjd.jump_mean, -0.1);
        assert_eq!(mjd.jump_std, 0.15);
        assert_eq!(mjd.process_name(), "Merton Jump-Diffusion");
    }

    #[test]
    fn test_merton_invalid_params() {
        assert!(MertonJumpDiffusion::new(0.05, -0.2, 1.0, -0.1, 0.15).is_err());
        assert!(MertonJumpDiffusion::new(0.05, 0.2, -1.0, -0.1, 0.15).is_err());
        assert!(MertonJumpDiffusion::new(0.05, 0.2, 1.0, -0.1, -0.15).is_err());
    }

    #[test]
    fn test_merton_simulate_step_with_rng() {
        let mjd = MertonJumpDiffusion::new(0.05, 0.2, 1.0, -0.1, 0.15).unwrap();
        let mut rng = RandomGenerator::new(42);

        let x0 = 100.0;
        let dt = 0.01;
        let dw = 0.5;

        let x1 = mjd.simulate_step_with_rng(0.0, x0, dt, dw, &mut rng);

        // Check that result is reasonable
        assert!(x1 > 0.0);
    }

    #[test]
    fn test_kou_jump_diffusion_creation() {
        let kou = KouJumpDiffusion::new(0.05, 0.2, 1.0, 0.6, 10.0, 5.0).unwrap();
        assert_eq!(kou.mu, 0.05);
        assert_eq!(kou.sigma, 0.2);
        assert_eq!(kou.lambda, 1.0);
        assert_eq!(kou.p_up, 0.6);
        assert_eq!(kou.eta_up, 10.0);
        assert_eq!(kou.eta_down, 5.0);
        assert_eq!(kou.process_name(), "Kou Jump-Diffusion");
    }

    #[test]
    fn test_kou_invalid_params() {
        assert!(KouJumpDiffusion::new(0.05, -0.2, 1.0, 0.6, 10.0, 5.0).is_err());
        assert!(KouJumpDiffusion::new(0.05, 0.2, -1.0, 0.6, 10.0, 5.0).is_err());
        assert!(KouJumpDiffusion::new(0.05, 0.2, 1.0, 1.5, 10.0, 5.0).is_err());
        assert!(KouJumpDiffusion::new(0.05, 0.2, 1.0, 0.6, -10.0, 5.0).is_err());
        assert!(KouJumpDiffusion::new(0.05, 0.2, 1.0, 0.6, 10.0, -5.0).is_err());
    }

    #[test]
    fn test_kou_simulate_step_with_rng() {
        let kou = KouJumpDiffusion::new(0.05, 0.2, 1.0, 0.6, 10.0, 5.0).unwrap();
        let mut rng = RandomGenerator::new(42);

        let x0 = 100.0;
        let dt = 0.01;
        let dw = 0.5;

        let x1 = kou.simulate_step_with_rng(0.0, x0, dt, dw, &mut rng);

        // Check that result is reasonable
        assert!(x1 > 0.0);
    }

    #[test]
    fn test_heston_creation() {
        let heston = HestonModel::new(0.05, 2.0, 0.04, 0.3, -0.7).unwrap();
        assert_eq!(heston.mu, 0.05);
        assert_eq!(heston.kappa, 2.0);
        assert_eq!(heston.theta, 0.04);
        assert_eq!(heston.sigma, 0.3);
        assert_eq!(heston.rho, -0.7);
    }

    #[test]
    fn test_heston_invalid_params() {
        assert!(HestonModel::new(0.05, -2.0, 0.04, 0.3, -0.7).is_err());
        assert!(HestonModel::new(0.05, 2.0, -0.04, 0.3, -0.7).is_err());
        assert!(HestonModel::new(0.05, 2.0, 0.04, -0.3, -0.7).is_err());
        assert!(HestonModel::new(0.05, 2.0, 0.04, 0.3, -1.5).is_err());
    }

    #[test]
    fn test_heston_feller_condition() {
        // Satisfies Feller condition: 2 * 2.0 * 0.04 = 0.16 >= 0.3^2 = 0.09
        let heston_good = HestonModel::new(0.05, 2.0, 0.04, 0.3, -0.7).unwrap();
        assert!(heston_good.satisfies_feller_condition());

        // Violates Feller condition: 2 * 0.5 * 0.01 = 0.01 < 0.3^2 = 0.09
        let heston_bad = HestonModel::new(0.05, 0.5, 0.01, 0.3, -0.7).unwrap();
        assert!(!heston_bad.satisfies_feller_condition());
    }

    #[test]
    fn test_heston_simulate_step_full() {
        let heston = HestonModel::new(0.05, 2.0, 0.04, 0.3, -0.7).unwrap();

        let s0 = 100.0;
        let v0 = 0.04;
        let dt = 0.01;
        let dw1 = 0.5;
        let dw2 = 0.3;

        let (s1, v1) = heston.simulate_step_full(0.0, s0, v0, dt, dw1, dw2);

        // Check that results are reasonable
        assert!(s1 > 0.0);
        assert!(v1 >= 0.0);
    }

    #[test]
    fn test_heston_variance_positivity() {
        let heston = HestonModel::new(0.05, 2.0, 0.04, 0.3, -0.7).unwrap();

        // Even with negative variance input, should return non-negative
        let s0 = 100.0;
        let v0 = -0.01;
        let dt = 0.01;
        let dw1 = 0.0;
        let dw2 = 0.0;

        let (_s1, v1) = heston.simulate_step_full(0.0, s0, v0, dt, dw1, dw2);
        assert!(v1 >= 0.0);
    }

    #[test]
    fn test_sabr_creation() {
        let sabr = SABRModel::new(0.3, 0.5, 0.4, -0.3).unwrap();
        assert_eq!(sabr.alpha, 0.3);
        assert_eq!(sabr.beta, 0.5);
        assert_eq!(sabr.nu, 0.4);
        assert_eq!(sabr.rho, -0.3);
    }

    #[test]
    fn test_sabr_invalid_params() {
        assert!(SABRModel::new(-0.3, 0.5, 0.4, -0.3).is_err());
        assert!(SABRModel::new(0.3, 1.5, 0.4, -0.3).is_err());
        assert!(SABRModel::new(0.3, 0.5, -0.4, -0.3).is_err());
        assert!(SABRModel::new(0.3, 0.5, 0.4, -1.5).is_err());
    }

    #[test]
    fn test_sabr_simulate_step_full() {
        let sabr = SABRModel::new(0.3, 0.5, 0.4, -0.3).unwrap();

        let f0 = 0.05;
        let alpha0 = 0.3;
        let dt = 0.01;
        let dw1 = 0.5;
        let dw2 = 0.3;

        let (f1, alpha1) = sabr.simulate_step_full(0.0, f0, alpha0, dt, dw1, dw2);

        // Check that results are reasonable
        assert!(f1 >= 0.0);
        assert!(alpha1 > 0.0);
    }

    #[test]
    fn test_sabr_implied_volatility_atm() {
        let sabr = SABRModel::new(0.3, 0.5, 0.4, -0.3).unwrap();

        let forward = 0.05;
        let strike = 0.05;
        let time = 1.0;

        let iv = sabr.implied_volatility(forward, strike, time).unwrap();

        // ATM implied vol should be positive and reasonable
        assert!(iv > 0.0);
        assert!(iv < 2.0);
    }

    #[test]
    fn test_sabr_implied_volatility_otm() {
        let sabr = SABRModel::new(0.3, 0.5, 0.4, -0.3).unwrap();

        let forward = 0.05;
        let strike = 0.06;
        let time = 1.0;

        let iv = sabr.implied_volatility(forward, strike, time).unwrap();

        // OTM implied vol should be positive and reasonable
        assert!(iv > 0.0);
        assert!(iv < 2.0);
    }

    #[test]
    fn test_sabr_invalid_inputs() {
        let sabr = SABRModel::new(0.3, 0.5, 0.4, -0.3).unwrap();

        assert!(sabr.implied_volatility(-0.05, 0.05, 1.0).is_err());
        assert!(sabr.implied_volatility(0.05, -0.05, 1.0).is_err());
        assert!(sabr.implied_volatility(0.05, 0.05, -1.0).is_err());
    }
}
