// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Core traits for extensibility and plugin architecture

use crate::common::error::Result;
use crate::common::types::OptionParams;

/// Trait for option pricing models
///
/// Implement this trait to create custom pricing models that can be
/// registered and used throughout the library.
pub trait PricingModel: Send + Sync {
    /// Calculate the price of an option given parameters
    ///
    /// # Arguments
    /// * `params` - Option parameters including spot, strike, volatility, etc.
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated option price or an error
    fn price(&self, params: &OptionParams) -> Result<f64>;

    /// Get the name of this pricing model
    ///
    /// # Returns
    /// * `&str` - A string identifier for this model (e.g., "black_scholes", "binomial_tree")
    fn model_name(&self) -> &str;
}

/// Trait for stochastic processes
///
/// Implement this trait to define custom stochastic processes for
/// Monte Carlo simulation and path generation.
pub trait StochasticProcess: Send + Sync {
    /// Calculate the drift term of the process at time t with value x
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current value of the process
    ///
    /// # Returns
    /// * `f64` - The drift coefficient μ(t, x)
    fn drift(&self, t: f64, x: f64) -> f64;

    /// Calculate the diffusion term of the process at time t with value x
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current value of the process
    ///
    /// # Returns
    /// * `f64` - The diffusion coefficient σ(t, x)
    fn diffusion(&self, t: f64, x: f64) -> f64;

    /// Simulate one step of the process
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current value of the process
    /// * `dt` - Time step size
    /// * `dw` - Random increment (typically N(0, sqrt(dt)))
    ///
    /// # Returns
    /// * `f64` - The new value of the process at time t + dt
    fn simulate_step(&self, t: f64, x: f64, dt: f64, dw: f64) -> f64;

    /// Get the name of this stochastic process
    ///
    /// # Returns
    /// * `&str` - A string identifier for this process (e.g., "gbm", "ornstein_uhlenbeck")
    fn process_name(&self) -> &str;
}

/// Trait for interpolation methods
///
/// Implement this trait to create custom interpolation schemes for
/// yield curves, volatility surfaces, and other applications.
pub trait Interpolator: Send + Sync {
    /// Interpolate a value at point x
    ///
    /// # Arguments
    /// * `x` - The point at which to interpolate
    ///
    /// # Returns
    /// * `Result<f64>` - The interpolated value or an error
    fn interpolate(&self, x: f64) -> Result<f64>;

    /// Fit the interpolator to data points
    ///
    /// # Arguments
    /// * `x` - Array of x-coordinates
    /// * `y` - Array of y-coordinates (same length as x)
    ///
    /// # Returns
    /// * `Result<()>` - Success or an error if fitting fails
    fn fit(&mut self, x: &[f64], y: &[f64]) -> Result<()>;

    /// Get the name of this interpolation method
    ///
    /// # Returns
    /// * `&str` - A string identifier for this method (e.g., "linear", "cubic_spline")
    fn interpolator_name(&self) -> &str;
}

/// Trait for optimization algorithms
///
/// Implement this trait to create custom optimization methods for
/// portfolio optimization, calibration, and other applications.
pub trait Optimizer: Send + Sync {
    /// Optimize an objective function
    ///
    /// # Arguments
    /// * `objective` - The function to minimize
    /// * `initial` - Initial guess for the solution
    ///
    /// # Returns
    /// * `Result<Vec<f64>>` - The optimal solution or an error
    fn optimize(&self, objective: &dyn Fn(&[f64]) -> f64, initial: &[f64]) -> Result<Vec<f64>>;

    /// Get the name of this optimization algorithm
    ///
    /// # Returns
    /// * `&str` - A string identifier for this algorithm (e.g., "bfgs", "nelder_mead")
    fn optimizer_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::error::DervflowError;
    use crate::common::types::OptionType;

    // Mock implementations for testing

    struct MockPricingModel;

    impl PricingModel for MockPricingModel {
        fn price(&self, params: &OptionParams) -> Result<f64> {
            // Simple mock: return 10% of spot price
            Ok(params.spot * 0.1)
        }

        fn model_name(&self) -> &str {
            "mock_model"
        }
    }

    struct MockStochasticProcess;

    impl StochasticProcess for MockStochasticProcess {
        fn drift(&self, _t: f64, _x: f64) -> f64 {
            0.05 // 5% drift
        }

        fn diffusion(&self, _t: f64, _x: f64) -> f64 {
            0.2 // 20% volatility
        }

        fn simulate_step(&self, _t: f64, x: f64, dt: f64, dw: f64) -> f64 {
            // Simple Euler-Maruyama step
            let drift = self.drift(_t, x);
            let diffusion = self.diffusion(_t, x);
            x + drift * x * dt + diffusion * x * dw
        }

        fn process_name(&self) -> &str {
            "mock_process"
        }
    }

    struct MockInterpolator {
        data: Vec<(f64, f64)>,
    }

    impl MockInterpolator {
        fn new() -> Self {
            Self { data: Vec::new() }
        }
    }

    impl Interpolator for MockInterpolator {
        fn interpolate(&self, x: f64) -> Result<f64> {
            if self.data.is_empty() {
                return Err(DervflowError::DataError("No data fitted".to_string()));
            }
            // Simple linear interpolation (mock)
            Ok(x * 2.0)
        }

        fn fit(&mut self, x: &[f64], y: &[f64]) -> Result<()> {
            if x.len() != y.len() {
                return Err(DervflowError::InvalidInput(
                    "x and y must have same length".to_string(),
                ));
            }
            self.data = x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect();
            Ok(())
        }

        fn interpolator_name(&self) -> &str {
            "mock_interpolator"
        }
    }

    struct MockOptimizer;

    impl Optimizer for MockOptimizer {
        fn optimize(
            &self,
            _objective: &dyn Fn(&[f64]) -> f64,
            initial: &[f64],
        ) -> Result<Vec<f64>> {
            // Mock: just return the initial guess
            Ok(initial.to_vec())
        }

        fn optimizer_name(&self) -> &str {
            "mock_optimizer"
        }
    }

    #[test]
    fn test_pricing_model_trait() {
        let model = MockPricingModel;
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);

        let price = model.price(&params).unwrap();
        assert_eq!(price, 10.0);
        assert_eq!(model.model_name(), "mock_model");
    }

    #[test]
    fn test_stochastic_process_trait() {
        let process = MockStochasticProcess;

        assert_eq!(process.drift(0.0, 100.0), 0.05);
        assert_eq!(process.diffusion(0.0, 100.0), 0.2);
        assert_eq!(process.process_name(), "mock_process");

        let new_value = process.simulate_step(0.0, 100.0, 0.01, 0.1);
        assert!(new_value > 0.0);
    }

    #[test]
    fn test_interpolator_trait() {
        let mut interpolator = MockInterpolator::new();

        // Test without fitting
        assert!(interpolator.interpolate(1.0).is_err());

        // Fit data
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        assert!(interpolator.fit(&x, &y).is_ok());

        // Test interpolation after fitting
        let result = interpolator.interpolate(2.5).unwrap();
        assert_eq!(result, 5.0);
        assert_eq!(interpolator.interpolator_name(), "mock_interpolator");
    }

    #[test]
    fn test_interpolator_validation() {
        let mut interpolator = MockInterpolator::new();

        let x = vec![1.0, 2.0];
        let y = vec![2.0, 4.0, 6.0]; // Different length

        assert!(interpolator.fit(&x, &y).is_err());
    }

    #[test]
    fn test_optimizer_trait() {
        let optimizer = MockOptimizer;

        let objective = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let initial = vec![1.0, 2.0];

        let result = optimizer.optimize(&objective, &initial).unwrap();
        assert_eq!(result, vec![1.0, 2.0]);
        assert_eq!(optimizer.optimizer_name(), "mock_optimizer");
    }

    #[test]
    fn test_traits_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn PricingModel>>();
        assert_send_sync::<Box<dyn StochasticProcess>>();
        assert_send_sync::<Box<dyn Interpolator>>();
        assert_send_sync::<Box<dyn Optimizer>>();
    }
}
