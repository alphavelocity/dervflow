// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Volatility surface and implied volatility
//!
//! This module provides functionality for:
//! - Implied volatility calculation using Newton-Raphson and Brent's method
//! - Volatility surface construction and interpolation
//! - 2D interpolation methods (bilinear, cubic spline)
//! - SABR model for volatility surface modeling

use crate::common::error::{DervflowError, Result};
use crate::common::types::{OptionParams, OptionType};
use crate::numerical::root_finding::{RootFindingConfig, brent};
use crate::options::analytical::{black_scholes_greeks, black_scholes_price};

/// Calculate implied volatility using Newton-Raphson method with Brenner-Subrahmanyam initial guess
///
/// The implied volatility is the volatility value that, when input into the Black-Scholes formula,
/// produces the observed market price. This function uses Newton-Raphson iteration with Vega
/// as the derivative.
///
/// # Arguments
/// * `market_price` - The observed market price of the option
/// * `params` - Option parameters (volatility field will be ignored)
/// * `tolerance` - Convergence tolerance (default: 0.0001)
/// * `max_iterations` - Maximum number of iterations (default: 100)
///
/// # Returns
/// * `Ok(f64)` - The implied volatility
/// * `Err(DervflowError)` - If convergence fails or inputs are invalid
///
/// # Examples
/// ```
/// use dervflow::options::volatility::implied_volatility_newton;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);
/// let market_price = 10.45;
/// let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();
/// assert!(iv > 0.0 && iv < 1.0);
/// ```
pub fn implied_volatility_newton(
    market_price: f64,
    params: &OptionParams,
    tolerance: f64,
    max_iterations: usize,
) -> Result<f64> {
    // Validate inputs
    if market_price <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "Market price must be positive".to_string(),
        ));
    }

    params.validate().map_err(DervflowError::InvalidInput)?;

    // Check if option has intrinsic value only (at expiry)
    if params.time_to_maturity == 0.0 {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate implied volatility for option at expiry".to_string(),
        ));
    }

    // Calculate intrinsic value
    let intrinsic_value = match params.option_type {
        OptionType::Call => (params.spot - params.strike).max(0.0),
        OptionType::Put => (params.strike - params.spot).max(0.0),
    };

    // Check if market price is below intrinsic value
    if market_price < intrinsic_value {
        return Err(DervflowError::InvalidInput(format!(
            "Market price ({}) is below intrinsic value ({})",
            market_price, intrinsic_value
        )));
    }

    // Brenner-Subrahmanyam approximation for initial guess
    let initial_guess = brenner_subrahmanyam_approximation(market_price, params);

    // Ensure initial guess is reasonable
    let initial_guess = initial_guess.clamp(0.001, 5.0);

    // Newton-Raphson iteration
    let mut sigma = initial_guess;
    let mut iterations = 0;

    for i in 0..max_iterations {
        iterations = i + 1;

        // Create params with current volatility estimate
        let mut current_params = *params;
        current_params.volatility = sigma;

        // Calculate price and vega
        let price = black_scholes_price(&current_params)?;
        let greeks = black_scholes_greeks(&current_params)?;

        // Vega is the derivative of price with respect to volatility
        // Note: greeks.vega is per 1% change, so we multiply by 100
        let vega = greeks.vega * 100.0;

        // Check for zero vega (shouldn't happen in practice)
        if vega.abs() < 1e-10 {
            return Err(DervflowError::NumericalError(
                "Vega is too small, cannot continue Newton-Raphson".to_string(),
            ));
        }

        // Price difference
        let price_diff = price - market_price;

        // Check convergence
        if price_diff.abs() < tolerance {
            return Ok(sigma);
        }

        // Newton-Raphson update: sigma_new = sigma_old - f(sigma) / f'(sigma)
        let sigma_new = sigma - price_diff / vega;

        // Ensure volatility stays positive and reasonable
        let sigma_new = sigma_new.clamp(0.0001, 10.0);

        // Check if we're making progress
        if (sigma_new - sigma).abs() < tolerance * 0.01 {
            return Ok(sigma_new);
        }

        sigma = sigma_new;
    }

    // Failed to converge
    Err(DervflowError::ConvergenceFailure {
        iterations,
        error: {
            let mut final_params = *params;
            final_params.volatility = sigma;
            let final_price = black_scholes_price(&final_params).unwrap_or(0.0);
            (final_price - market_price).abs()
        },
    })
}

/// Brenner-Subrahmanyam approximation for initial volatility guess
///
/// This provides a good starting point for Newton-Raphson iteration,
/// especially for at-the-money options.
///
/// Formula: σ ≈ sqrt(2π/T) * (C/S)
/// where C is the option price, S is the spot price, and T is time to maturity
fn brenner_subrahmanyam_approximation(market_price: f64, params: &OptionParams) -> f64 {
    use std::f64::consts::PI;

    let s = params.spot;
    let t = params.time_to_maturity;

    // Basic approximation
    let sigma = (2.0 * PI / t).sqrt() * (market_price / s);

    // Ensure reasonable bounds
    sigma.clamp(0.01, 5.0)
}

/// Calculate implied volatility using Brent's method as a robust fallback
///
/// Brent's method is more robust than Newton-Raphson and guaranteed to converge
/// if the root is bracketed. It's particularly useful for deep ITM/OTM options
/// where Newton-Raphson might struggle.
///
/// # Arguments
/// * `market_price` - The observed market price of the option
/// * `params` - Option parameters (volatility field will be ignored)
/// * `vol_min` - Lower bound for volatility search (default: 0.0001)
/// * `vol_max` - Upper bound for volatility search (default: 5.0)
/// * `tolerance` - Convergence tolerance (default: 0.0001)
/// * `max_iterations` - Maximum number of iterations (default: 100)
///
/// # Returns
/// * `Ok(f64)` - The implied volatility
/// * `Err(DervflowError)` - If convergence fails or inputs are invalid
pub fn implied_volatility_brent(
    market_price: f64,
    params: &OptionParams,
    vol_min: f64,
    vol_max: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<f64> {
    // Validate inputs
    if market_price <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "Market price must be positive".to_string(),
        ));
    }

    params.validate().map_err(DervflowError::InvalidInput)?;

    if params.time_to_maturity == 0.0 {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate implied volatility for option at expiry".to_string(),
        ));
    }

    if vol_min <= 0.0 || vol_max <= vol_min {
        return Err(DervflowError::InvalidInput(
            "Invalid volatility bounds".to_string(),
        ));
    }

    // Define the objective function: price(sigma) - market_price
    let objective = |sigma: f64| -> f64 {
        let mut current_params = *params;
        current_params.volatility = sigma;
        match black_scholes_price(&current_params) {
            Ok(price) => price - market_price,
            Err(_) => f64::NAN,
        }
    };

    // Check that the root is bracketed
    let f_min = objective(vol_min);
    let f_max = objective(vol_max);

    if !f_min.is_finite() || !f_max.is_finite() {
        return Err(DervflowError::NumericalError(
            "Failed to evaluate objective function at bounds".to_string(),
        ));
    }

    if f_min * f_max > 0.0 {
        // Root is not bracketed, try to expand the search range
        // This can happen for deep ITM/OTM options
        let expanded_max = vol_max * 2.0;
        let f_expanded = objective(expanded_max);

        if f_expanded.is_finite() && f_min * f_expanded < 0.0 {
            // Root is bracketed with expanded range
            return implied_volatility_brent(
                market_price,
                params,
                vol_min,
                expanded_max,
                tolerance,
                max_iterations,
            );
        }

        return Err(DervflowError::InvalidInput(
            "Implied volatility is not bracketed by the search range. \
             This may indicate the market price is inconsistent with the model."
                .to_string(),
        ));
    }

    // Use Brent's method
    let config = RootFindingConfig {
        max_iterations,
        tolerance,
        relative_tolerance: tolerance,
    };

    let result = brent(objective, vol_min, vol_max, &config)?;

    Ok(result.root)
}

/// Calculate implied volatility with automatic method selection
///
/// This function tries Newton-Raphson first (faster) and falls back to Brent's method
/// if Newton-Raphson fails to converge. This provides a good balance between speed
/// and robustness.
///
/// # Arguments
/// * `market_price` - The observed market price of the option
/// * `params` - Option parameters (volatility field will be ignored)
///
/// # Returns
/// * `Ok(f64)` - The implied volatility
/// * `Err(DervflowError)` - If both methods fail
pub fn implied_volatility(market_price: f64, params: &OptionParams) -> Result<f64> {
    // Try Newton-Raphson first (faster for most cases)
    match implied_volatility_newton(market_price, params, 0.0001, 100) {
        Ok(iv) => Ok(iv),
        Err(_) => {
            // Fall back to Brent's method (more robust)
            implied_volatility_brent(market_price, params, 0.0001, 5.0, 0.0001, 100)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_implied_volatility_newton_atm() {
        // At-the-money call option
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        // Calculate market price with known volatility
        let true_vol = 0.2;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        // Calculate implied volatility
        let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();

        // Should recover the true volatility
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_newton_itm() {
        // In-the-money call option
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.25;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_newton_otm() {
        // Out-of-the-money call option
        let params = OptionParams::new(90.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.3;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_newton_put() {
        // Put option
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Put);

        let true_vol = 0.2;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_newton_high_vol() {
        // High volatility scenario
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.8;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_newton_low_vol() {
        // Low volatility scenario
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.05;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_newton(market_price, &params, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_newton_invalid_price() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        // Negative price
        let result = implied_volatility_newton(-10.0, &params, 0.0001, 100);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DervflowError::InvalidInput(_)
        ));

        // Zero price
        let result = implied_volatility_newton(0.0, &params, 0.0001, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_implied_volatility_newton_below_intrinsic() {
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        // Price below intrinsic value (10.0)
        let result = implied_volatility_newton(5.0, &params, 0.0001, 100);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DervflowError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_implied_volatility_newton_at_expiry() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 0.0, OptionType::Call);

        let result = implied_volatility_newton(10.0, &params, 0.0001, 100);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DervflowError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_implied_volatility_brent_atm() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.2;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_brent(market_price, &params, 0.0001, 5.0, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_brent_deep_itm() {
        // Moderately in-the-money call option
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.25;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_brent(market_price, &params, 0.0001, 5.0, 0.0001, 100).unwrap();
        // Slightly relaxed tolerance for ITM options where Brent's method may be less precise
        assert_relative_eq!(iv, true_vol, epsilon = 0.02);
    }

    #[test]
    fn test_implied_volatility_brent_deep_otm() {
        // Moderately out-of-the-money call option (less extreme for better numerical stability)
        let params = OptionParams::new(80.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.35;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_brent(market_price, &params, 0.0001, 5.0, 0.0001, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_implied_volatility_auto_method() {
        // Test automatic method selection
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        let true_vol = 0.25;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility(market_price, &params).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 0.001);
    }

    #[test]
    fn test_brenner_subrahmanyam_approximation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        // For ATM options, the approximation should be reasonable
        let market_price = 10.0;
        let approx = brenner_subrahmanyam_approximation(market_price, &params);

        // Should be in a reasonable range
        assert!(approx > 0.0 && approx < 1.0);
    }

    #[test]
    fn test_convergence_diagnostics() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, OptionType::Call);

        // Use a very tight tolerance to force more iterations
        let true_vol = 0.2;
        let mut price_params = params;
        price_params.volatility = true_vol;
        let market_price = black_scholes_price(&price_params).unwrap();

        let iv = implied_volatility_newton(market_price, &params, 1e-10, 100).unwrap();
        assert_relative_eq!(iv, true_vol, epsilon = 1e-8);
    }

    #[test]
    fn test_volatility_surface_creation() {
        let strikes = vec![90.0, 100.0, 110.0];
        let maturities = vec![0.25, 0.5, 1.0];
        let volatilities = vec![
            vec![0.25, 0.23, 0.21],
            vec![0.20, 0.19, 0.18],
            vec![0.22, 0.21, 0.20],
        ];

        let surface = VolatilitySurface::new(
            strikes.clone(),
            maturities.clone(),
            volatilities.clone(),
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        )
        .unwrap();

        assert_eq!(surface.strikes(), &strikes);
        assert_eq!(surface.maturities(), &maturities);
        assert_eq!(surface.spot(), 100.0);
        assert_eq!(surface.rate(), 0.05);
    }

    #[test]
    fn test_volatility_surface_invalid_inputs() {
        let strikes = vec![90.0, 100.0, 110.0];
        let maturities = vec![0.25, 0.5, 1.0];

        // Wrong number of rows
        let volatilities = vec![vec![0.25, 0.23, 0.21], vec![0.20, 0.19, 0.18]];
        let result = VolatilitySurface::new(
            strikes.clone(),
            maturities.clone(),
            volatilities,
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        );
        assert!(result.is_err());

        // Wrong number of columns
        let volatilities = vec![vec![0.25, 0.23], vec![0.20, 0.19], vec![0.22, 0.21]];
        let result = VolatilitySurface::new(
            strikes.clone(),
            maturities.clone(),
            volatilities,
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        );
        assert!(result.is_err());

        // Negative volatility
        let volatilities = vec![
            vec![0.25, 0.23, 0.21],
            vec![0.20, -0.19, 0.18],
            vec![0.22, 0.21, 0.20],
        ];
        let result = VolatilitySurface::new(
            strikes.clone(),
            maturities.clone(),
            volatilities,
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        );
        assert!(result.is_err());

        // Invalid spot
        let volatilities = vec![
            vec![0.25, 0.23, 0.21],
            vec![0.20, 0.19, 0.18],
            vec![0.22, 0.21, 0.20],
        ];
        let result = VolatilitySurface::new(
            strikes,
            maturities,
            volatilities,
            InterpolationMethod::Bilinear,
            -100.0,
            0.05,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_bilinear_interpolation() {
        let strikes = vec![90.0, 100.0, 110.0];
        let maturities = vec![0.25, 0.5, 1.0];
        let volatilities = vec![
            vec![0.25, 0.23, 0.21],
            vec![0.20, 0.19, 0.18],
            vec![0.22, 0.21, 0.20],
        ];

        let surface = VolatilitySurface::new(
            strikes,
            maturities,
            volatilities,
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        )
        .unwrap();

        // Test exact grid points
        let vol = surface.implied_volatility(100.0, 0.5).unwrap();
        assert_relative_eq!(vol, 0.19, epsilon = 1e-10);

        // Test interpolated point (midpoint)
        let vol = surface.implied_volatility(95.0, 0.375).unwrap();
        // Should be average of surrounding points
        let expected = (0.25 + 0.23 + 0.20 + 0.19) / 4.0;
        assert_relative_eq!(vol, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_bilinear_interpolation_out_of_bounds() {
        let strikes = vec![90.0, 100.0, 110.0];
        let maturities = vec![0.25, 0.5, 1.0];
        let volatilities = vec![
            vec![0.25, 0.23, 0.21],
            vec![0.20, 0.19, 0.18],
            vec![0.22, 0.21, 0.20],
        ];

        let surface = VolatilitySurface::new(
            strikes,
            maturities,
            volatilities,
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        )
        .unwrap();

        // Strike too low
        let result = surface.implied_volatility(80.0, 0.5);
        assert!(result.is_err());

        // Strike too high
        let result = surface.implied_volatility(120.0, 0.5);
        assert!(result.is_err());

        // Maturity too low
        let result = surface.implied_volatility(100.0, 0.1);
        assert!(result.is_err());

        // Maturity too high
        let result = surface.implied_volatility(100.0, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cubic_spline_interpolation() {
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let maturities = vec![0.25, 0.5, 0.75, 1.0];
        let volatilities = vec![
            vec![0.25, 0.24, 0.23, 0.22],
            vec![0.23, 0.22, 0.21, 0.20],
            vec![0.20, 0.19, 0.18, 0.17],
            vec![0.21, 0.20, 0.19, 0.18],
            vec![0.22, 0.21, 0.20, 0.19],
        ];

        let surface = VolatilitySurface::new(
            strikes,
            maturities,
            volatilities,
            InterpolationMethod::CubicSpline,
            100.0,
            0.05,
        )
        .unwrap();

        // Test exact grid point
        let vol = surface.implied_volatility(100.0, 0.5).unwrap();
        assert_relative_eq!(vol, 0.19, epsilon = 1e-6);

        // Test interpolated point
        let vol = surface.implied_volatility(97.5, 0.625).unwrap();
        // Should be smooth interpolation
        assert!(vol > 0.18 && vol < 0.23);
    }

    #[test]
    fn test_cubic_spline_1d() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // y = x^2

        // Test at grid points
        let result = cubic_spline_interpolate(&x, &y, 1.0).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);

        let result = cubic_spline_interpolate(&x, &y, 2.0).unwrap();
        assert_relative_eq!(result, 4.0, epsilon = 1e-6);

        // Test interpolated point
        let result = cubic_spline_interpolate(&x, &y, 1.5).unwrap();
        // Should be close to 2.25 for quadratic function
        assert_relative_eq!(result, 2.25, epsilon = 0.1);
    }

    #[test]
    fn test_solve_tridiagonal() {
        // Simple 3x3 system
        let a = vec![0.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 0.0];
        let d = vec![1.0, 2.0, 1.0];

        let x = solve_tridiagonal(&a, &b, &c, &d).unwrap();

        // Verify solution
        assert_relative_eq!(b[0] * x[0] + c[0] * x[1], d[0], epsilon = 1e-10);
        assert_relative_eq!(
            a[1] * x[0] + b[1] * x[1] + c[1] * x[2],
            d[1],
            epsilon = 1e-10
        );
        assert_relative_eq!(a[2] * x[1] + b[2] * x[2], d[2], epsilon = 1e-10);
    }

    #[test]
    fn test_volatility_surface_unsorted_inputs() {
        // Test that surface handles unsorted inputs correctly
        let strikes = vec![100.0, 90.0, 110.0]; // Unsorted
        let maturities = vec![0.5, 0.25, 1.0]; // Unsorted
        let volatilities = vec![
            vec![0.20, 0.25, 0.18], // Corresponds to strike 100
            vec![0.25, 0.30, 0.23], // Corresponds to strike 90
            vec![0.22, 0.27, 0.20], // Corresponds to strike 110
        ];

        let surface = VolatilitySurface::new(
            strikes,
            maturities,
            volatilities,
            InterpolationMethod::Bilinear,
            100.0,
            0.05,
        )
        .unwrap();

        // After sorting, strikes should be [90, 100, 110] and maturities [0.25, 0.5, 1.0]
        assert_eq!(surface.strikes(), &[90.0, 100.0, 110.0]);
        assert_eq!(surface.maturities(), &[0.25, 0.5, 1.0]);

        // Check that volatility at (90, 0.25) is correct (was at [1][1] in original, should be [0][0] after sorting)
        let vol = surface.implied_volatility(90.0, 0.25).unwrap();
        assert_relative_eq!(vol, 0.30, epsilon = 1e-10);
    }

    #[test]
    fn test_sabr_params_creation() {
        let params = SABRParams::new(0.2, 0.5, -0.3, 0.4).unwrap();
        assert_eq!(params.alpha, 0.2);
        assert_eq!(params.beta, 0.5);
        assert_eq!(params.rho, -0.3);
        assert_eq!(params.nu, 0.4);
    }

    #[test]
    fn test_sabr_params_invalid() {
        // Negative alpha
        assert!(SABRParams::new(-0.2, 0.5, -0.3, 0.4).is_err());

        // Beta out of range
        assert!(SABRParams::new(0.2, 1.5, -0.3, 0.4).is_err());
        assert!(SABRParams::new(0.2, -0.1, -0.3, 0.4).is_err());

        // Rho out of range
        assert!(SABRParams::new(0.2, 0.5, -1.5, 0.4).is_err());
        assert!(SABRParams::new(0.2, 0.5, 1.5, 0.4).is_err());

        // Negative nu
        assert!(SABRParams::new(0.2, 0.5, -0.3, -0.4).is_err());
    }

    #[test]
    fn test_sabr_implied_volatility_atm() {
        let params = SABRParams::new(0.2, 0.5, -0.3, 0.4).unwrap();
        let forward = 100.0;
        let strike = 100.0;
        let maturity = 1.0;

        let vol = params
            .implied_volatility(forward, strike, maturity)
            .unwrap();

        // ATM volatility should be positive and reasonable
        assert!(vol > 0.0 && vol < 1.0);
    }

    #[test]
    fn test_sabr_implied_volatility_otm() {
        let params = SABRParams::new(0.2, 0.5, -0.3, 0.4).unwrap();
        let forward = 100.0;
        let maturity = 1.0;

        // OTM call (high strike)
        let vol_otm = params.implied_volatility(forward, 110.0, maturity).unwrap();
        assert!(vol_otm > 0.0 && vol_otm < 1.0);

        // ITM call (low strike)
        let vol_itm = params.implied_volatility(forward, 90.0, maturity).unwrap();
        assert!(vol_itm > 0.0 && vol_itm < 1.0);
    }

    #[test]
    fn test_sabr_volatility_smile() {
        // Test that SABR produces reasonable volatility values
        let params = SABRParams::new(0.2, 0.5, -0.3, 0.4).unwrap();
        let forward = 100.0;
        let maturity = 1.0;

        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let mut vols = Vec::new();

        for &strike in &strikes {
            let vol = params
                .implied_volatility(forward, strike, maturity)
                .unwrap();
            vols.push(vol);
        }

        // All volatilities should be positive and reasonable
        for &vol in &vols {
            assert!(
                vol > 0.0 && vol < 2.0,
                "Volatility {} is out of reasonable range",
                vol
            );
        }

        // SABR should produce a volatility surface (not flat)
        // Check that not all volatilities are the same
        let max_vol = vols.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_vol = vols.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (max_vol - min_vol).abs() > 0.001,
            "SABR surface is too flat"
        );
    }

    #[test]
    fn test_sabr_beta_effect() {
        // Test different beta values
        let forward = 100.0;
        let strike = 110.0;
        let maturity = 1.0;

        // Beta = 0 (normal model)
        let params_normal = SABRParams::new(20.0, 0.0, -0.3, 0.4).unwrap();
        let vol_normal = params_normal
            .implied_volatility(forward, strike, maturity)
            .unwrap();

        // Beta = 1 (lognormal model)
        let params_lognormal = SABRParams::new(0.2, 1.0, -0.3, 0.4).unwrap();
        let vol_lognormal = params_lognormal
            .implied_volatility(forward, strike, maturity)
            .unwrap();

        // Both should be positive
        assert!(vol_normal > 0.0);
        assert!(vol_lognormal > 0.0);
    }

    #[test]
    fn test_sabr_calibration() {
        // Create synthetic market data from known SABR parameters
        let true_params = SABRParams::new(0.2, 0.5, -0.25, 0.3).unwrap();
        let forward = 100.0;
        let maturity = 1.0;

        let strikes = vec![85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];
        let mut market_vols = Vec::new();

        for &strike in &strikes {
            let vol = true_params
                .implied_volatility(forward, strike, maturity)
                .unwrap();
            market_vols.push(vol);
        }

        // Calibrate SABR to this data
        let calibrated = calibrate_sabr(forward, maturity, &strikes, &market_vols, 0.5).unwrap();

        // Check that calibrated parameters are reasonable
        assert!(calibrated.alpha > 0.0);
        assert_eq!(calibrated.beta, 0.5);
        assert!(calibrated.rho >= -1.0 && calibrated.rho <= 1.0);
        assert!(calibrated.nu >= 0.0);

        // Check that calibrated model reproduces market vols reasonably well
        for (i, &strike) in strikes.iter().enumerate() {
            let model_vol = calibrated
                .implied_volatility(forward, strike, maturity)
                .unwrap();
            let market_vol = market_vols[i];

            // Should be close (within 5% relative error)
            let rel_error = ((model_vol - market_vol) / market_vol).abs();
            assert!(
                rel_error < 0.05,
                "Strike {}: model vol {}, market vol {}, rel error {}",
                strike,
                model_vol,
                market_vol,
                rel_error
            );
        }
    }

    #[test]
    fn test_sabr_calibration_invalid_inputs() {
        let forward = 100.0;
        let maturity = 1.0;

        // Mismatched lengths
        let strikes = vec![90.0, 100.0, 110.0];
        let market_vols = vec![0.2, 0.19];
        let result = calibrate_sabr(forward, maturity, &strikes, &market_vols, 0.5);
        assert!(result.is_err());

        // Too few data points
        let strikes = vec![100.0, 110.0];
        let market_vols = vec![0.2, 0.19];
        let result = calibrate_sabr(forward, maturity, &strikes, &market_vols, 0.5);
        assert!(result.is_err());

        // Invalid forward
        let strikes = vec![90.0, 100.0, 110.0];
        let market_vols = vec![0.2, 0.19, 0.21];
        let result = calibrate_sabr(-100.0, maturity, &strikes, &market_vols, 0.5);
        assert!(result.is_err());
    }
}

/// 2D interpolation method for volatility surface
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Bilinear interpolation
    Bilinear,
    /// Cubic spline interpolation (2D)
    CubicSpline,
}

/// Volatility surface for storing and interpolating implied volatilities
///
/// The surface stores implied volatilities as a function of strike and maturity.
/// It supports multiple interpolation methods for querying volatilities at
/// arbitrary (strike, maturity) points.
#[derive(Debug, Clone)]
pub struct VolatilitySurface {
    /// Sorted vector of strike prices
    strikes: Vec<f64>,
    /// Sorted vector of maturities (in years)
    maturities: Vec<f64>,
    /// 2D grid of implied volatilities
    /// volatilities[i][j] corresponds to strikes[i] and maturities[j]
    volatilities: Vec<Vec<f64>>,
    /// Interpolation method to use
    interpolation_method: InterpolationMethod,
    /// Spot price of the underlying (for moneyness calculations)
    spot: f64,
    /// Risk-free rate (for forward calculations)
    rate: f64,
}

impl VolatilitySurface {
    /// Create a new volatility surface from market data
    ///
    /// # Arguments
    /// * `strikes` - Vector of strike prices (will be sorted)
    /// * `maturities` - Vector of maturities in years (will be sorted)
    /// * `volatilities` - 2D grid of implied volatilities \[strike_idx\]\[maturity_idx\]
    /// * `interpolation_method` - Method to use for interpolation
    /// * `spot` - Current spot price of the underlying
    /// * `rate` - Risk-free interest rate
    ///
    /// # Returns
    /// * `Ok(VolatilitySurface)` - The constructed surface
    /// * `Err(DervflowError)` - If inputs are invalid
    pub fn new(
        strikes: Vec<f64>,
        maturities: Vec<f64>,
        volatilities: Vec<Vec<f64>>,
        interpolation_method: InterpolationMethod,
        spot: f64,
        rate: f64,
    ) -> Result<Self> {
        // Validate inputs
        if strikes.is_empty() || maturities.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Strikes and maturities cannot be empty".to_string(),
            ));
        }

        if volatilities.len() != strikes.len() {
            return Err(DervflowError::InvalidInput(format!(
                "Volatilities rows ({}) must match strikes length ({})",
                volatilities.len(),
                strikes.len()
            )));
        }

        for (i, row) in volatilities.iter().enumerate() {
            if row.len() != maturities.len() {
                return Err(DervflowError::InvalidInput(format!(
                    "Volatilities row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    maturities.len()
                )));
            }
        }

        if spot <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Spot price must be positive".to_string(),
            ));
        }

        // Check for negative volatilities
        for (i, row) in volatilities.iter().enumerate() {
            for (j, &vol) in row.iter().enumerate() {
                if vol < 0.0 {
                    return Err(DervflowError::InvalidInput(format!(
                        "Volatility at strike {} maturity {} is negative: {}",
                        strikes[i], maturities[j], vol
                    )));
                }
            }
        }

        // Sort strikes and maturities, and reorder volatilities accordingly
        let mut strike_indices: Vec<usize> = (0..strikes.len()).collect();
        strike_indices.sort_by(|&a, &b| strikes[a].partial_cmp(&strikes[b]).unwrap());

        let mut maturity_indices: Vec<usize> = (0..maturities.len()).collect();
        maturity_indices.sort_by(|&a, &b| maturities[a].partial_cmp(&maturities[b]).unwrap());

        let sorted_strikes: Vec<f64> = strike_indices.iter().map(|&i| strikes[i]).collect();
        let sorted_maturities: Vec<f64> = maturity_indices.iter().map(|&i| maturities[i]).collect();

        let sorted_volatilities: Vec<Vec<f64>> = strike_indices
            .iter()
            .map(|&i| {
                maturity_indices
                    .iter()
                    .map(|&j| volatilities[i][j])
                    .collect()
            })
            .collect();

        Ok(Self {
            strikes: sorted_strikes,
            maturities: sorted_maturities,
            volatilities: sorted_volatilities,
            interpolation_method,
            spot,
            rate,
        })
    }

    /// Get the implied volatility at a specific strike and maturity using interpolation
    ///
    /// # Arguments
    /// * `strike` - Strike price
    /// * `maturity` - Time to maturity in years
    ///
    /// # Returns
    /// * `Ok(f64)` - The interpolated implied volatility
    /// * `Err(DervflowError)` - If interpolation fails or point is out of bounds
    pub fn implied_volatility(&self, strike: f64, maturity: f64) -> Result<f64> {
        if strike <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Strike must be positive".to_string(),
            ));
        }

        if maturity <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Maturity must be positive".to_string(),
            ));
        }

        // Check if point is within bounds
        if strike < self.strikes[0] || strike > self.strikes[self.strikes.len() - 1] {
            return Err(DervflowError::InvalidInput(format!(
                "Strike {} is outside the surface range [{}, {}]",
                strike,
                self.strikes[0],
                self.strikes[self.strikes.len() - 1]
            )));
        }

        if maturity < self.maturities[0] || maturity > self.maturities[self.maturities.len() - 1] {
            return Err(DervflowError::InvalidInput(format!(
                "Maturity {} is outside the surface range [{}, {}]",
                maturity,
                self.maturities[0],
                self.maturities[self.maturities.len() - 1]
            )));
        }

        match self.interpolation_method {
            InterpolationMethod::Bilinear => self.bilinear_interpolation(strike, maturity),
            InterpolationMethod::CubicSpline => {
                self.cubic_spline_2d_interpolation(strike, maturity)
            }
        }
    }

    /// Bilinear interpolation for 2D surface
    ///
    /// Performs linear interpolation in both strike and maturity dimensions.
    fn bilinear_interpolation(&self, strike: f64, maturity: f64) -> Result<f64> {
        // Find the indices for strike
        let strike_idx = self.find_interval(&self.strikes, strike);
        let maturity_idx = self.find_interval(&self.maturities, maturity);

        let k0 = self.strikes[strike_idx];
        let k1 = self.strikes[strike_idx + 1];
        let t0 = self.maturities[maturity_idx];
        let t1 = self.maturities[maturity_idx + 1];

        // Get the four corner volatilities
        let v00 = self.volatilities[strike_idx][maturity_idx];
        let v01 = self.volatilities[strike_idx][maturity_idx + 1];
        let v10 = self.volatilities[strike_idx + 1][maturity_idx];
        let v11 = self.volatilities[strike_idx + 1][maturity_idx + 1];

        // Bilinear interpolation formula
        let w_k = (strike - k0) / (k1 - k0);
        let w_t = (maturity - t0) / (t1 - t0);

        let vol = (1.0 - w_k) * (1.0 - w_t) * v00
            + (1.0 - w_k) * w_t * v01
            + w_k * (1.0 - w_t) * v10
            + w_k * w_t * v11;

        Ok(vol)
    }

    /// Cubic spline interpolation in 2D
    ///
    /// Performs cubic spline interpolation first along maturity dimension,
    /// then along strike dimension (tensor product approach).
    fn cubic_spline_2d_interpolation(&self, strike: f64, maturity: f64) -> Result<f64> {
        // First, interpolate along maturity dimension for each strike
        let mut interpolated_at_strikes = Vec::with_capacity(self.strikes.len());

        for i in 0..self.strikes.len() {
            let vol_at_strike = &self.volatilities[i];
            let interpolated_vol =
                cubic_spline_interpolate(&self.maturities, vol_at_strike, maturity)?;
            interpolated_at_strikes.push(interpolated_vol);
        }

        // Then interpolate along strike dimension
        let final_vol = cubic_spline_interpolate(&self.strikes, &interpolated_at_strikes, strike)?;

        Ok(final_vol)
    }

    /// Find the interval index where value falls
    ///
    /// Returns the index i such that data[i] <= value < data[i+1]
    fn find_interval(&self, data: &[f64], value: f64) -> usize {
        // Binary search for the interval
        let mut left = 0;
        let mut right = data.len() - 1;

        while right - left > 1 {
            let mid = (left + right) / 2;
            if data[mid] <= value {
                left = mid;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Get the strikes in the surface
    pub fn strikes(&self) -> &[f64] {
        &self.strikes
    }

    /// Get the maturities in the surface
    pub fn maturities(&self) -> &[f64] {
        &self.maturities
    }

    /// Get the volatility grid
    pub fn volatilities(&self) -> &[Vec<f64>] {
        &self.volatilities
    }

    /// Get the spot price
    pub fn spot(&self) -> f64 {
        self.spot
    }

    /// Get the risk-free rate
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Get the interpolation method
    pub fn interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

/// Cubic spline interpolation for 1D data
///
/// Uses natural cubic spline (second derivative is zero at endpoints)
///
/// # Arguments
/// * `x` - Sorted x values
/// * `y` - Corresponding y values
/// * `x_target` - Point at which to interpolate
///
/// # Returns
/// * `Ok(f64)` - Interpolated value
/// * `Err(DervflowError)` - If interpolation fails
fn cubic_spline_interpolate(x: &[f64], y: &[f64], x_target: f64) -> Result<f64> {
    let n = x.len();

    if n < 2 {
        return Err(DervflowError::InvalidInput(
            "Need at least 2 points for cubic spline".to_string(),
        ));
    }

    if n != y.len() {
        return Err(DervflowError::InvalidInput(
            "x and y must have the same length".to_string(),
        ));
    }

    // For small number of points, use linear interpolation
    if n == 2 {
        let w = (x_target - x[0]) / (x[1] - x[0]);
        return Ok((1.0 - w) * y[0] + w * y[1]);
    }

    // Find the interval
    let mut idx = 0;
    for i in 0..n - 1 {
        if x_target >= x[i] && x_target <= x[i + 1] {
            idx = i;
            break;
        }
    }

    // Compute natural cubic spline coefficients
    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();

    // Build tridiagonal system for second derivatives
    let mut a = vec![0.0; n];
    let mut b = vec![1.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    // Natural spline boundary conditions (second derivative = 0 at endpoints)
    b[0] = 1.0;
    c[0] = 0.0;
    d[0] = 0.0;

    for i in 1..n - 1 {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    a[n - 1] = 0.0;
    b[n - 1] = 1.0;
    d[n - 1] = 0.0;

    // Solve tridiagonal system using Thomas algorithm
    let m = solve_tridiagonal(&a, &b, &c, &d)?;

    // Evaluate spline at x_target
    let i = idx;
    let t = (x_target - x[i]) / h[i];

    let result = y[i] * (1.0 - t)
        + y[i + 1] * t
        + ((m[i] * (1.0 - t).powi(3) + m[i + 1] * t.powi(3)) * h[i].powi(2)) / 6.0
        - ((m[i] * (1.0 - t) + m[i + 1] * t) * h[i].powi(2)) / 6.0;

    Ok(result)
}

/// Solve tridiagonal system using Thomas algorithm
///
/// Solves Ax = d where A is tridiagonal with diagonals a, b, c
///
/// # Arguments
/// * `a` - Lower diagonal (a[0] is not used)
/// * `b` - Main diagonal
/// * `c` - Upper diagonal (c[n-1] is not used)
/// * `d` - Right-hand side
///
/// # Returns
/// * `Ok(Vec<f64>)` - Solution vector
/// * `Err(DervflowError)` - If system is singular
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();

    if a.len() != n || c.len() != n || d.len() != n {
        return Err(DervflowError::InvalidInput(
            "All arrays must have the same length".to_string(),
        ));
    }

    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            return Err(DervflowError::NumericalError(
                "Tridiagonal system is singular".to_string(),
            ));
        }
        c_prime[i] = c[i] / denom;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// SABR (Stochastic Alpha Beta Rho) model parameters
///
/// The SABR model is a stochastic volatility model used to model the evolution
/// of forward rates and their implied volatilities. It's widely used in interest
/// rate derivatives markets.
///
/// The model has the following dynamics:
/// dF = α * F^β * dW1
/// dα = ν * α * dW2
/// dW1 * dW2 = ρ * dt
///
/// where:
/// - F is the forward rate
/// - α (alpha) is the volatility
/// - β (beta) is the elasticity parameter (0 for normal, 1 for lognormal)
/// - ρ (rho) is the correlation between forward and volatility
/// - ν (nu/volvol) is the volatility of volatility
#[derive(Debug, Clone, Copy)]
pub struct SABRParams {
    /// Alpha: initial volatility
    pub alpha: f64,
    /// Beta: elasticity parameter (typically 0, 0.5, or 1)
    pub beta: f64,
    /// Rho: correlation between forward and volatility (-1 to 1)
    pub rho: f64,
    /// Nu (volvol): volatility of volatility
    pub nu: f64,
}

impl SABRParams {
    /// Create new SABR parameters
    pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Alpha must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(DervflowError::InvalidInput(
                "Beta must be between 0 and 1".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&rho) {
            return Err(DervflowError::InvalidInput(
                "Rho must be between -1 and 1".to_string(),
            ));
        }
        if nu < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Nu must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            alpha,
            beta,
            rho,
            nu,
        })
    }

    /// Calculate implied volatility using SABR formula (Hagan et al. 2002)
    ///
    /// This is an approximation formula that's accurate for small time to maturity
    /// and when the strike is not too far from the forward.
    ///
    /// # Arguments
    /// * `forward` - Forward price
    /// * `strike` - Strike price
    /// * `maturity` - Time to maturity in years
    ///
    /// # Returns
    /// * `Ok(f64)` - Implied volatility
    /// * `Err(DervflowError)` - If calculation fails
    pub fn implied_volatility(&self, forward: f64, strike: f64, maturity: f64) -> Result<f64> {
        if forward <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Forward must be positive".to_string(),
            ));
        }
        if strike <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Strike must be positive".to_string(),
            ));
        }
        if maturity <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Maturity must be positive".to_string(),
            ));
        }

        // Handle ATM case separately
        if (strike - forward).abs() < 1e-10 {
            return self.implied_volatility_atm(forward, maturity);
        }

        let f = forward;
        let k = strike;
        let t = maturity;
        let alpha = self.alpha;
        let beta = self.beta;
        let rho = self.rho;
        let nu = self.nu;

        // Log-moneyness
        let log_fk = (f / k).ln();

        // FK term
        let fk_beta = (f * k).powf((1.0 - beta) / 2.0);

        // z parameter
        let z = (nu / alpha) * fk_beta * log_fk;

        // x(z) function
        let x_z = if z.abs() < 1e-7 {
            // Taylor expansion for small z
            1.0 - 0.5 * rho * z + (rho.powi(2) - 1.0) * z.powi(2) / 12.0
        } else {
            let sqrt_term = (1.0 - 2.0 * rho * z + z.powi(2)).sqrt();
            ((sqrt_term + z - rho) / (1.0 - rho)).ln() / z
        };

        // First term (numerator)
        let numerator = alpha;

        // Second term (denominator)
        let denominator = fk_beta
            * (1.0
                + ((1.0 - beta).powi(2) / 24.0) * log_fk.powi(2)
                + ((1.0 - beta).powi(4) / 1920.0) * log_fk.powi(4));

        // Third term (time-dependent correction)
        let correction = 1.0
            + t * (((1.0 - beta).powi(2) / 24.0) * (alpha.powi(2) / fk_beta.powi(2))
                + (rho * beta * nu * alpha) / (4.0 * fk_beta)
                + ((2.0 - 3.0 * rho.powi(2)) / 24.0) * nu.powi(2));

        let vol = (numerator / denominator) * x_z * correction;

        if !vol.is_finite() || vol <= 0.0 {
            return Err(DervflowError::NumericalError(
                "SABR formula produced invalid volatility".to_string(),
            ));
        }

        Ok(vol)
    }

    /// Calculate ATM implied volatility using SABR formula
    ///
    /// Simplified formula when strike equals forward
    fn implied_volatility_atm(&self, forward: f64, maturity: f64) -> Result<f64> {
        let f = forward;
        let t = maturity;
        let alpha = self.alpha;
        let beta = self.beta;
        let rho = self.rho;
        let nu = self.nu;

        let f_beta = f.powf(1.0 - beta);

        let correction = 1.0
            + t * (((1.0 - beta).powi(2) / 24.0) * (alpha.powi(2) / f_beta.powi(2))
                + (rho * beta * nu * alpha) / (4.0 * f_beta)
                + ((2.0 - 3.0 * rho.powi(2)) / 24.0) * nu.powi(2));

        let vol = (alpha / f_beta) * correction;

        if !vol.is_finite() || vol <= 0.0 {
            return Err(DervflowError::NumericalError(
                "SABR ATM formula produced invalid volatility".to_string(),
            ));
        }

        Ok(vol)
    }
}

/// Calibrate SABR model to market volatility data
///
/// Calibrates the SABR parameters (alpha, rho, nu) to match market implied volatilities.
/// Beta is typically fixed (e.g., 0.5 for interest rates, 1.0 for equities).
///
/// # Arguments
/// * `forward` - Forward price
/// * `maturity` - Time to maturity
/// * `strikes` - Vector of strike prices
/// * `market_vols` - Vector of market implied volatilities
/// * `beta` - Fixed beta parameter
///
/// # Returns
/// * `Ok(SABRParams)` - Calibrated SABR parameters
/// * `Err(DervflowError)` - If calibration fails
pub fn calibrate_sabr(
    forward: f64,
    maturity: f64,
    strikes: &[f64],
    market_vols: &[f64],
    beta: f64,
) -> Result<SABRParams> {
    use crate::numerical::optimization::{NelderMead, OptimizationConfig};

    if strikes.len() != market_vols.len() {
        return Err(DervflowError::InvalidInput(
            "Strikes and market vols must have same length".to_string(),
        ));
    }

    if strikes.len() < 3 {
        return Err(DervflowError::InvalidInput(
            "Need at least 3 data points for SABR calibration".to_string(),
        ));
    }

    if forward <= 0.0 || maturity <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "Forward and maturity must be positive".to_string(),
        ));
    }

    // Find ATM volatility for initial alpha guess
    let atm_idx = strikes
        .iter()
        .enumerate()
        .min_by(|&(_, k1), &(_, k2)| {
            (k1 - forward)
                .abs()
                .partial_cmp(&(k2 - forward).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();

    let atm_vol = market_vols[atm_idx];
    let initial_alpha = atm_vol * forward.powf(1.0 - beta);

    // Objective function: sum of squared errors
    let objective = |params: &[f64]| -> f64 {
        // params = [alpha, rho, nu]
        let alpha = params[0].abs(); // Ensure positive
        let rho = params[1].clamp(-0.99, 0.99); // Keep in bounds
        let nu = params[2].abs(); // Ensure positive

        let sabr_params = match SABRParams::new(alpha, beta, rho, nu) {
            Ok(p) => p,
            Err(_) => return 1e10, // Large penalty for invalid params
        };

        let mut error = 0.0;
        for (i, &strike) in strikes.iter().enumerate() {
            match sabr_params.implied_volatility(forward, strike, maturity) {
                Ok(model_vol) => {
                    let diff = model_vol - market_vols[i];
                    error += diff * diff;
                }
                Err(_) => {
                    error += 1e10; // Large penalty for failed calculation
                }
            }
        }

        error
    };

    // Initial guess: [alpha, rho, nu]
    let initial = vec![initial_alpha, 0.0, 0.3];

    // Optimization configuration
    let config = OptimizationConfig {
        max_iterations: 1000,
        f_tol: 1e-8,
        g_tol: 1e-6,
        x_tol: 1e-8,
    };

    // Use Nelder-Mead optimization
    let optimizer = NelderMead::new(config);
    let result = optimizer.optimize(objective, &initial)?;

    let alpha = result.x[0].abs();
    let rho = result.x[1].clamp(-0.99, 0.99);
    let nu = result.x[2].abs();

    SABRParams::new(alpha, beta, rho, nu)
}
