// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Analytical option pricing models
//!
//! This module provides analytical formulas for option pricing, including:
//! - Black-Scholes-Merton model for European options
//! - Analytical Greeks calculations

use crate::common::error::{DervflowError, Result};
use crate::common::types::{Greeks, OptionParams, OptionType};
use std::f64::consts::{PI, SQRT_2};

/// Calculate the cumulative distribution function (CDF) of the standard normal distribution
///
/// Uses the error function approximation for numerical stability and accuracy.
///
/// # Arguments
/// * `x` - The value at which to evaluate the CDF
///
/// # Returns
/// The probability that a standard normal random variable is less than or equal to x
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Error function approximation using Abramowitz and Stegun formula
///
/// Provides accuracy to about 1.5e-7
fn erf(x: f64) -> f64 {
    // Constants for the approximation
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
///
/// # Arguments
/// * `x` - The value at which to evaluate the PDF
///
/// # Returns
/// The probability density at x
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Calculate European option price using the Black-Scholes-Merton formula
///
/// The Black-Scholes-Merton model assumes:
/// - The underlying asset follows geometric Brownian motion
/// - No dividends during the option's life (or continuous dividend yield)
/// - European exercise (can only be exercised at maturity)
/// - No transaction costs or taxes
/// - Risk-free rate and volatility are constant
///
/// # Arguments
/// * `params` - Option parameters including spot, strike, rate, dividend, volatility, time, and option type
///
/// # Returns
/// * `Ok(f64)` - The theoretical option price
/// * `Err(DervflowError)` - If input validation fails or numerical issues occur
///
/// # Examples
/// ```
/// use dervflow::options::analytical::black_scholes_price;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
/// let price = black_scholes_price(&params).unwrap();
/// assert!(price > 0.0);
/// ```
pub fn black_scholes_price(params: &OptionParams) -> Result<f64> {
    // Validate input parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    // Handle edge case: option at expiry
    if params.time_to_maturity == 0.0 {
        return Ok(intrinsic_value(params));
    }

    // Handle edge case: zero volatility
    if params.volatility == 0.0 {
        return Ok(intrinsic_value_at_expiry(params));
    }

    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.dividend;
    let sigma = params.volatility;
    let t = params.time_to_maturity;

    // Calculate d1 and d2
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    // Check for numerical issues
    if !d1.is_finite() || !d2.is_finite() {
        return Err(DervflowError::NumericalError(
            "Numerical overflow in Black-Scholes calculation".to_string(),
        ));
    }

    // Calculate option price based on type
    let price = match params.option_type {
        OptionType::Call => {
            s * (-q * t).exp() * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
        }
        OptionType::Put => {
            k * (-r * t).exp() * normal_cdf(-d2) - s * (-q * t).exp() * normal_cdf(-d1)
        }
    };

    // Ensure price is non-negative (can be slightly negative due to numerical errors)
    Ok(price.max(0.0))
}

/// Calculate the intrinsic value of an option
///
/// The intrinsic value is the value if exercised immediately
fn intrinsic_value(params: &OptionParams) -> f64 {
    match params.option_type {
        OptionType::Call => (params.spot - params.strike).max(0.0),
        OptionType::Put => (params.strike - params.spot).max(0.0),
    }
}

/// Calculate the intrinsic value at expiry with zero volatility
///
/// With zero volatility, the option value is the discounted intrinsic value
fn intrinsic_value_at_expiry(params: &OptionParams) -> f64 {
    let forward = params.spot * ((params.rate - params.dividend) * params.time_to_maturity).exp();
    let intrinsic = match params.option_type {
        OptionType::Call => (forward - params.strike).max(0.0),
        OptionType::Put => (params.strike - forward).max(0.0),
    };
    intrinsic * (-params.rate * params.time_to_maturity).exp()
}

/// Calculate analytical Greeks for Black-Scholes-Merton model
///
/// Computes all first-order Greeks using analytical formulas:
/// - Delta: ∂V/∂S (sensitivity to underlying price)
/// - Gamma: ∂²V/∂S² (rate of change of delta)
/// - Vega: ∂V/∂σ (sensitivity to volatility)
/// - Theta: ∂V/∂t (time decay)
/// - Rho: ∂V/∂r (sensitivity to interest rate)
///
/// # Arguments
/// * `params` - Option parameters including spot, strike, rate, dividend, volatility, time, and option type
///
/// # Returns
/// * `Ok(Greeks)` - Struct containing all Greek values
/// * `Err(DervflowError)` - If input validation fails or numerical issues occur
///
/// # Examples
/// ```
/// use dervflow::options::analytical::black_scholes_greeks;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
/// let greeks = black_scholes_greeks(&params).unwrap();
/// assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
/// ```
pub fn black_scholes_greeks(params: &OptionParams) -> Result<Greeks> {
    // Validate input parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    // Handle edge case: option at expiry
    if params.time_to_maturity == 0.0 {
        return Ok(Greeks::zero());
    }

    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.dividend;
    let sigma = params.volatility;
    let t = params.time_to_maturity;

    // Handle edge case: zero volatility
    if sigma == 0.0 {
        return Ok(Greeks::zero());
    }

    // Calculate d1 and d2
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    // Check for numerical issues
    if !d1.is_finite() || !d2.is_finite() {
        return Err(DervflowError::NumericalError(
            "Numerical overflow in Greeks calculation".to_string(),
        ));
    }

    // Common terms
    let exp_qt = (-q * t).exp();
    let exp_rt = (-r * t).exp();
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let nprime_d1 = normal_pdf(d1);

    // Calculate Delta (∂V/∂S)
    let delta = match params.option_type {
        OptionType::Call => exp_qt * nd1,
        OptionType::Put => exp_qt * (nd1 - 1.0),
    };

    // Calculate Gamma (∂²V/∂S²)
    // Gamma is the same for calls and puts
    let gamma = (exp_qt * nprime_d1) / (s * sigma * sqrt_t);

    // Calculate Vega (∂V/∂σ)
    // Vega is the same for calls and puts
    // Note: Vega is typically expressed per 1% change in volatility
    let vega = s * exp_qt * nprime_d1 * sqrt_t / 100.0;

    // Calculate Theta (∂V/∂t)
    // Note: Theta is typically expressed per day, so we divide by 365
    let theta = match params.option_type {
        OptionType::Call => {
            let term1 = -(s * exp_qt * nprime_d1 * sigma) / (2.0 * sqrt_t);
            let term2 = q * s * exp_qt * nd1;
            let term3 = r * k * exp_rt * nd2;
            (term1 - term2 + term3) / 365.0
        }
        OptionType::Put => {
            let term1 = -(s * exp_qt * nprime_d1 * sigma) / (2.0 * sqrt_t);
            let term2 = q * s * exp_qt * (nd1 - 1.0);
            let term3 = r * k * exp_rt * (nd2 - 1.0);
            (term1 - term2 + term3) / 365.0
        }
    };

    // Calculate Rho (∂V/∂r)
    // Note: Rho is typically expressed per 1% change in interest rate
    let rho = match params.option_type {
        OptionType::Call => k * t * exp_rt * nd2 / 100.0,
        OptionType::Put => -k * t * exp_rt * (1.0 - nd2) / 100.0,
    };

    Ok(Greeks::new(delta, gamma, vega, theta, rho))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf() {
        // Test standard normal CDF values
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.0) - 0.8413447).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.1586553).abs() < 1e-6);
        assert!((normal_cdf(2.0) - 0.9772499).abs() < 1e-6);
        assert!((normal_cdf(-2.0) - 0.0227501).abs() < 1e-6);
    }

    #[test]
    fn test_normal_pdf() {
        // Test standard normal PDF values
        assert!((normal_pdf(0.0) - 0.3989423).abs() < 1e-6);
        assert!((normal_pdf(1.0) - 0.2419707).abs() < 1e-6);
        assert!((normal_pdf(-1.0) - 0.2419707).abs() < 1e-6);
    }

    #[test]
    fn test_black_scholes_call_atm() {
        // At-the-money call option
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let price = black_scholes_price(&params).unwrap();

        // Expected price approximately 10.45 (from standard BS tables)
        assert!((price - 10.45).abs() < 0.1);
    }

    #[test]
    fn test_black_scholes_put_atm() {
        // At-the-money put option
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let price = black_scholes_price(&params).unwrap();

        // Expected price approximately 5.57 (from standard BS tables)
        assert!((price - 5.57).abs() < 0.1);
    }

    #[test]
    fn test_black_scholes_call_itm() {
        // In-the-money call option
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let price = black_scholes_price(&params).unwrap();

        // Price should be greater than intrinsic value
        assert!(price > 10.0);
        assert!(price < 20.0);
    }

    #[test]
    fn test_black_scholes_put_otm() {
        // Out-of-the-money put option
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let price = black_scholes_price(&params).unwrap();

        // Price should be small but positive
        assert!(price > 0.0);
        assert!(price < 5.0);
    }

    #[test]
    fn test_black_scholes_with_dividend() {
        // Call option with dividend yield
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        let price_with_div = black_scholes_price(&params).unwrap();

        let params_no_div = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let price_no_div = black_scholes_price(&params_no_div).unwrap();

        // Call price should be lower with dividend
        assert!(price_with_div < price_no_div);
    }

    #[test]
    fn test_black_scholes_zero_time() {
        // Option at expiry
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 0.0, OptionType::Call);
        let price = black_scholes_price(&params).unwrap();

        // Should equal intrinsic value
        assert!((price - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_black_scholes_zero_volatility() {
        // Option with zero volatility
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);
        let price = black_scholes_price(&params).unwrap();

        // Should be discounted forward intrinsic value
        let forward = 110.0 * (0.05_f64 * 1.0).exp();
        let intrinsic = (forward - 100.0).max(0.0);
        let expected = intrinsic * (-0.05_f64 * 1.0).exp();
        assert!((price - expected).abs() < 1e-10);
    }

    #[test]
    fn test_black_scholes_invalid_spot() {
        let params = OptionParams::new(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        assert!(black_scholes_price(&params).is_err());
    }

    #[test]
    fn test_black_scholes_invalid_strike() {
        let params = OptionParams::new(100.0, -100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        assert!(black_scholes_price(&params).is_err());
    }

    #[test]
    fn test_black_scholes_invalid_volatility() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, -0.2, 1.0, OptionType::Call);
        assert!(black_scholes_price(&params).is_err());
    }

    #[test]
    fn test_black_scholes_invalid_time() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, -1.0, OptionType::Call);
        assert!(black_scholes_price(&params).is_err());
    }

    #[test]
    fn test_greeks_call_atm() {
        // At-the-money call option
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let greeks = black_scholes_greeks(&params).unwrap();

        // Delta should be around 0.5-0.6 for ATM call
        assert!(greeks.delta > 0.5 && greeks.delta < 0.7);

        // Gamma should be positive
        assert!(greeks.gamma > 0.0);

        // Vega should be positive
        assert!(greeks.vega > 0.0);

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);

        // Rho should be positive for calls
        assert!(greeks.rho > 0.0);
    }

    #[test]
    fn test_greeks_put_atm() {
        // At-the-money put option
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let greeks = black_scholes_greeks(&params).unwrap();

        // Delta should be around -0.4 to -0.5 for ATM put
        assert!(greeks.delta < 0.0 && greeks.delta > -0.6);

        // Gamma should be positive (same as call)
        assert!(greeks.gamma > 0.0);

        // Vega should be positive (same as call)
        assert!(greeks.vega > 0.0);

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);

        // Rho should be negative for puts
        assert!(greeks.rho < 0.0);
    }

    #[test]
    fn test_greeks_call_itm() {
        // In-the-money call option
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let greeks = black_scholes_greeks(&params).unwrap();

        // Delta should be higher for ITM call
        assert!(greeks.delta > 0.7);

        // Gamma should be positive but smaller than ATM
        assert!(greeks.gamma > 0.0);
    }

    #[test]
    fn test_greeks_put_otm() {
        // Out-of-the-money put option
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let greeks = black_scholes_greeks(&params).unwrap();

        // Delta should be small negative for OTM put
        assert!(greeks.delta < 0.0 && greeks.delta > -0.3);

        // Gamma should be positive but small
        assert!(greeks.gamma > 0.0);
    }

    #[test]
    fn test_greeks_gamma_symmetry() {
        // Gamma should be the same for call and put with same parameters
        let call_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let put_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let call_greeks = black_scholes_greeks(&call_params).unwrap();
        let put_greeks = black_scholes_greeks(&put_params).unwrap();

        assert!((call_greeks.gamma - put_greeks.gamma).abs() < 1e-10);
    }

    #[test]
    fn test_greeks_vega_symmetry() {
        // Vega should be the same for call and put with same parameters
        let call_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let put_params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let call_greeks = black_scholes_greeks(&call_params).unwrap();
        let put_greeks = black_scholes_greeks(&put_params).unwrap();

        assert!((call_greeks.vega - put_greeks.vega).abs() < 1e-10);
    }

    #[test]
    fn test_greeks_delta_put_call_parity() {
        // Delta relationship: Delta_call - Delta_put = exp(-q*t)
        let call_params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        let put_params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Put);

        let call_greeks = black_scholes_greeks(&call_params).unwrap();
        let put_greeks = black_scholes_greeks(&put_params).unwrap();

        let expected_diff = (-0.02 * 1.0_f64).exp();
        let actual_diff = call_greeks.delta - put_greeks.delta;

        assert!((actual_diff - expected_diff).abs() < 1e-10);
    }

    #[test]
    fn test_greeks_zero_time() {
        // Greeks should be zero at expiry
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 0.0, OptionType::Call);
        let greeks = black_scholes_greeks(&params).unwrap();

        assert_eq!(greeks.delta, 0.0);
        assert_eq!(greeks.gamma, 0.0);
        assert_eq!(greeks.vega, 0.0);
        assert_eq!(greeks.theta, 0.0);
        assert_eq!(greeks.rho, 0.0);
    }

    #[test]
    fn test_greeks_zero_volatility() {
        // Greeks should be zero with zero volatility
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);
        let greeks = black_scholes_greeks(&params).unwrap();

        assert_eq!(greeks.delta, 0.0);
        assert_eq!(greeks.gamma, 0.0);
        assert_eq!(greeks.vega, 0.0);
        assert_eq!(greeks.theta, 0.0);
        assert_eq!(greeks.rho, 0.0);
    }
}
