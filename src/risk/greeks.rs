// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Greeks calculation using finite differences
//!
//! This module provides numerical Greeks calculation using finite difference methods.
//! Greeks are sensitivities of option prices to various parameters.

use crate::common::error::{DervflowError, Result};
use crate::common::types::{Greeks, OptionParams};

/// Extended Greeks including second and third order sensitivities
#[derive(Debug, Clone, Copy)]
pub struct ExtendedGreeks {
    /// First-order Greeks
    pub greeks: Greeks,
    /// Vanna: ∂²V/∂S∂σ (sensitivity of delta to volatility)
    pub vanna: f64,
    /// Volga (Vomma): ∂²V/∂σ² (sensitivity of vega to volatility)
    pub volga: f64,
    /// Speed: ∂³V/∂S³ (rate of change of gamma)
    pub speed: f64,
    /// Zomma: ∂³V/∂S²∂σ (sensitivity of gamma to volatility)
    pub zomma: f64,
    /// Color: ∂³V/∂S²∂t (sensitivity of gamma to time)
    pub color: f64,
    /// Ultima: ∂³V/∂σ³ (sensitivity of volga to volatility)
    pub ultima: f64,
}

impl ExtendedGreeks {
    /// Create a new ExtendedGreeks instance
    pub fn new(
        greeks: Greeks,
        vanna: f64,
        volga: f64,
        speed: f64,
        zomma: f64,
        color: f64,
        ultima: f64,
    ) -> Self {
        Self {
            greeks,
            vanna,
            volga,
            speed,
            zomma,
            color,
            ultima,
        }
    }

    /// Create an ExtendedGreeks instance with all values set to zero
    pub fn zero() -> Self {
        Self {
            greeks: Greeks::zero(),
            vanna: 0.0,
            volga: 0.0,
            speed: 0.0,
            zomma: 0.0,
            color: 0.0,
            ultima: 0.0,
        }
    }
}

/// Configuration for finite difference calculations
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceConfig {
    /// Bump size for spot price (relative, e.g., 0.01 = 1%)
    pub spot_bump: f64,
    /// Bump size for volatility (absolute, e.g., 0.01 = 1% vol)
    pub vol_bump: f64,
    /// Bump size for time (in years, e.g., 1/365 = 1 day)
    pub time_bump: f64,
    /// Bump size for interest rate (absolute, e.g., 0.0001 = 1 bp)
    pub rate_bump: f64,
}

impl Default for FiniteDifferenceConfig {
    fn default() -> Self {
        Self {
            spot_bump: 0.01,        // 1% spot bump
            vol_bump: 0.01,         // 1% volatility bump
            time_bump: 1.0 / 365.0, // 1 day time bump
            rate_bump: 0.0001,      // 1 basis point rate bump
        }
    }
}

/// Calculate Greeks using finite differences
///
/// This function computes first-order Greeks using numerical differentiation:
/// - Delta: central difference for spot
/// - Gamma: second derivative using central difference
/// - Vega: central difference for volatility
/// - Theta: forward difference for time
/// - Rho: central difference for interest rate
///
/// # Arguments
/// * `pricing_fn` - Function that prices the option given parameters
/// * `params` - Base option parameters
/// * `config` - Configuration for bump sizes (optional, uses defaults if None)
///
/// # Returns
/// * `Ok(Greeks)` - Struct containing all Greek values
/// * `Err(DervflowError)` - If pricing function fails or numerical issues occur
///
/// # Examples
/// ```
/// use dervflow::risk::greeks::{calculate_numerical_greeks, FiniteDifferenceConfig};
/// use dervflow::options::analytical::black_scholes_price;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
/// let config = FiniteDifferenceConfig::default();
/// let greeks = calculate_numerical_greeks(&black_scholes_price, &params, Some(config)).unwrap();
/// ```
pub fn calculate_numerical_greeks<F>(
    pricing_fn: &F,
    params: &OptionParams,
    config: Option<FiniteDifferenceConfig>,
) -> Result<Greeks>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    // Validate input parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    let config = config.unwrap_or_default();
    let spot_bump = ensure_positive_bump("spot", config.spot_bump)?;
    let vol_bump = ensure_positive_bump("volatility", config.vol_bump)?;
    let time_bump = ensure_positive_bump("time", config.time_bump)?;
    let rate_bump = ensure_positive_bump("rate", config.rate_bump)?;

    // Get base price
    let base_price = pricing_fn(params)?;

    let h_spot = params.spot * spot_bump;
    if !h_spot.is_finite() || h_spot <= 0.0 {
        return Err(DervflowError::InvalidInput(format!(
            "spot bump results in invalid step size: {h_spot}"
        )));
    }
    let (price_up, price_down) = spot_bumped_prices(pricing_fn, params, h_spot)?;

    // Calculate Delta using central difference: (V(S+h) - V(S-h)) / (2h)
    let delta = (price_up - price_down) / (2.0 * h_spot);
    if !delta.is_finite() {
        return Err(DervflowError::NumericalError(
            "Delta calculation resulted in non-finite value".to_string(),
        ));
    }

    // Calculate Gamma using central difference: (V(S+h) - 2V(S) + V(S-h)) / h²
    let gamma = (price_up - 2.0 * base_price + price_down) / (h_spot * h_spot);
    if !gamma.is_finite() {
        return Err(DervflowError::NumericalError(
            "Gamma calculation resulted in non-finite value".to_string(),
        ));
    }

    // Calculate Vega using central difference: (V(σ+h) - V(σ-h)) / (2h)
    let vega = calculate_vega(pricing_fn, params, vol_bump)?;

    // Calculate Theta using forward difference: (V(t-h) - V(t)) / h
    // Note: We use forward difference because we can't go back in time
    let theta = calculate_theta(pricing_fn, params, base_price, time_bump)?;

    // Calculate Rho using central difference: (V(r+h) - V(r-h)) / (2h)
    let rho = calculate_rho(pricing_fn, params, rate_bump)?;

    Ok(Greeks::new(delta, gamma, vega, theta, rho))
}

fn ensure_positive_bump(name: &str, bump: f64) -> Result<f64> {
    if !bump.is_finite() {
        return Err(DervflowError::InvalidInput(format!(
            "{name} bump must be finite, got {bump}"
        )));
    }
    if bump <= 0.0 {
        return Err(DervflowError::InvalidInput(format!(
            "{name} bump must be positive, got {bump}"
        )));
    }
    Ok(bump)
}

fn spot_bumped_prices<F>(pricing_fn: &F, params: &OptionParams, h: f64) -> Result<(f64, f64)>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    // Price with spot bumped up
    let mut params_up = *params;
    params_up.spot = params.spot + h;
    let price_up = pricing_fn(&params_up)?;

    // Price with spot bumped down
    let mut params_down = *params;
    params_down.spot = params.spot - h;
    let price_down = pricing_fn(&params_down)?;

    Ok((price_up, price_down))
}

/// Calculate Vega using central difference method
fn calculate_vega<F>(pricing_fn: &F, params: &OptionParams, vol_bump: f64) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h = vol_bump;

    // Price with volatility bumped up
    let mut params_up = *params;
    params_up.volatility = params.volatility + h;
    let price_up = pricing_fn(&params_up)?;

    // Price with volatility bumped down
    let mut params_down = *params;
    params_down.volatility = (params.volatility - h).max(0.0);
    let price_down = pricing_fn(&params_down)?;

    // Central difference
    // Note: Vega is typically expressed per 1% change in volatility
    let vega = (price_up - price_down) / (2.0 * h) / 100.0;

    if !vega.is_finite() {
        return Err(DervflowError::NumericalError(
            "Vega calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(vega)
}

/// Calculate Theta using forward difference method
fn calculate_theta<F>(
    pricing_fn: &F,
    params: &OptionParams,
    base_price: f64,
    time_bump: f64,
) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    // Handle edge case: option at or near expiry
    if params.time_to_maturity <= time_bump {
        return Ok(0.0);
    }

    let h = time_bump;

    // Price with time decreased (moving forward in time)
    let mut params_forward = *params;
    params_forward.time_to_maturity = params.time_to_maturity - h;
    let price_forward = pricing_fn(&params_forward)?;

    // Forward difference: (V(t-h) - V(t)) / h
    // Note: Theta is typically expressed per day, so we divide by 365
    let theta = (price_forward - base_price) / h / 365.0;

    if !theta.is_finite() {
        return Err(DervflowError::NumericalError(
            "Theta calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(theta)
}

/// Calculate Rho using central difference method
fn calculate_rho<F>(pricing_fn: &F, params: &OptionParams, rate_bump: f64) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h = rate_bump;

    // Price with rate bumped up
    let mut params_up = *params;
    params_up.rate = params.rate + h;
    let price_up = pricing_fn(&params_up)?;

    // Price with rate bumped down
    let mut params_down = *params;
    params_down.rate = params.rate - h;
    let price_down = pricing_fn(&params_down)?;

    // Central difference
    // Note: Rho is typically expressed per 1% change in interest rate
    let rho = (price_up - price_down) / (2.0 * h) / 100.0;

    if !rho.is_finite() {
        return Err(DervflowError::NumericalError(
            "Rho calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(rho)
}

/// Calculate extended Greeks including second and third order sensitivities
///
/// This function computes higher-order Greeks using numerical differentiation:
/// - Vanna: ∂²V/∂S∂σ (cross-derivative of delta and vega)
/// - Volga: ∂²V/∂σ² (second derivative with respect to volatility)
/// - Speed: ∂³V/∂S³ (third derivative with respect to spot)
/// - Zomma: ∂³V/∂S²∂σ (sensitivity of gamma to volatility)
/// - Color: ∂³V/∂S²∂t (sensitivity of gamma to time)
/// - Ultima: ∂³V/∂σ³ (third derivative with respect to volatility)
///
/// # Arguments
/// * `pricing_fn` - Function that prices the option given parameters
/// * `params` - Base option parameters
/// * `config` - Configuration for bump sizes (optional, uses defaults if None)
///
/// # Returns
/// * `Ok(ExtendedGreeks)` - Struct containing all Greek values including higher orders
/// * `Err(DervflowError)` - If pricing function fails or numerical issues occur
pub fn calculate_extended_greeks<F>(
    pricing_fn: &F,
    params: &OptionParams,
    config: Option<FiniteDifferenceConfig>,
) -> Result<ExtendedGreeks>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    // Validate input parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    let config = config.unwrap_or_default();

    // Calculate first-order Greeks
    let greeks = calculate_numerical_greeks(pricing_fn, params, Some(config))?;

    // Calculate second-order Greeks
    let vanna = calculate_vanna(pricing_fn, params, config.spot_bump, config.vol_bump)?;
    let volga = calculate_volga(pricing_fn, params, config.vol_bump)?;

    // Calculate third-order Greeks
    let speed = calculate_speed(pricing_fn, params, config.spot_bump)?;
    let zomma = calculate_zomma(pricing_fn, params, config.spot_bump, config.vol_bump)?;
    let color = calculate_color(pricing_fn, params, config.spot_bump, config.time_bump)?;
    let ultima = calculate_ultima(pricing_fn, params, config.vol_bump)?;

    Ok(ExtendedGreeks::new(
        greeks, vanna, volga, speed, zomma, color, ultima,
    ))
}

/// Calculate Vanna: ∂²V/∂S∂σ (cross-derivative)
fn calculate_vanna<F>(
    pricing_fn: &F,
    params: &OptionParams,
    spot_bump: f64,
    vol_bump: f64,
) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h_s = params.spot * spot_bump;
    let h_v = vol_bump;

    // V(S+h, σ+h)
    let mut params_up_up = *params;
    params_up_up.spot = params.spot + h_s;
    params_up_up.volatility = params.volatility + h_v;
    let price_up_up = pricing_fn(&params_up_up)?;

    // V(S+h, σ-h)
    let mut params_up_down = *params;
    params_up_down.spot = params.spot + h_s;
    params_up_down.volatility = (params.volatility - h_v).max(0.0);
    let price_up_down = pricing_fn(&params_up_down)?;

    // V(S-h, σ+h)
    let mut params_down_up = *params;
    params_down_up.spot = params.spot - h_s;
    params_down_up.volatility = params.volatility + h_v;
    let price_down_up = pricing_fn(&params_down_up)?;

    // V(S-h, σ-h)
    let mut params_down_down = *params;
    params_down_down.spot = params.spot - h_s;
    params_down_down.volatility = (params.volatility - h_v).max(0.0);
    let price_down_down = pricing_fn(&params_down_down)?;

    // Cross-derivative: (V(S+h,σ+h) - V(S+h,σ-h) - V(S-h,σ+h) + V(S-h,σ-h)) / (4*h_s*h_v)
    let vanna = (price_up_up - price_up_down - price_down_up + price_down_down) / (4.0 * h_s * h_v);

    if !vanna.is_finite() {
        return Err(DervflowError::NumericalError(
            "Vanna calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(vanna)
}

/// Calculate Volga (Vomma): ∂²V/∂σ² (second derivative with respect to volatility)
fn calculate_volga<F>(pricing_fn: &F, params: &OptionParams, vol_bump: f64) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h = vol_bump;

    // Base price
    let base_price = pricing_fn(params)?;

    // V(σ+h)
    let mut params_up = *params;
    params_up.volatility = params.volatility + h;
    let price_up = pricing_fn(&params_up)?;

    // V(σ-h)
    let mut params_down = *params;
    params_down.volatility = (params.volatility - h).max(0.0);
    let price_down = pricing_fn(&params_down)?;

    // Second derivative: (V(σ+h) - 2V(σ) + V(σ-h)) / h²
    let volga = (price_up - 2.0 * base_price + price_down) / (h * h);

    if !volga.is_finite() {
        return Err(DervflowError::NumericalError(
            "Volga calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(volga)
}

/// Calculate Speed: ∂³V/∂S³ (third derivative with respect to spot)
fn calculate_speed<F>(pricing_fn: &F, params: &OptionParams, spot_bump: f64) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h = params.spot * spot_bump;

    // V(S+2h)
    let mut params_up2 = *params;
    params_up2.spot = params.spot + 2.0 * h;
    let price_up2 = pricing_fn(&params_up2)?;

    // V(S+h)
    let mut params_up = *params;
    params_up.spot = params.spot + h;
    let price_up = pricing_fn(&params_up)?;

    // V(S-h)
    let mut params_down = *params;
    params_down.spot = params.spot - h;
    let price_down = pricing_fn(&params_down)?;

    // V(S-2h)
    let mut params_down2 = *params;
    params_down2.spot = params.spot - 2.0 * h;
    let price_down2 = pricing_fn(&params_down2)?;

    // Third derivative: (V(S+2h) - 2V(S+h) + 2V(S-h) - V(S-2h)) / (2h³)
    let speed = (price_up2 - 2.0 * price_up + 2.0 * price_down - price_down2) / (2.0 * h * h * h);

    if !speed.is_finite() {
        return Err(DervflowError::NumericalError(
            "Speed calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(speed)
}

/// Calculate Zomma: ∂³V/∂S²∂σ (sensitivity of gamma to volatility)
fn calculate_zomma<F>(
    pricing_fn: &F,
    params: &OptionParams,
    spot_bump: f64,
    vol_bump: f64,
) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h_s = params.spot * spot_bump;
    let h_v = vol_bump;

    // Calculate gamma at σ+h
    let mut params_vol_up = *params;
    params_vol_up.volatility = params.volatility + h_v;
    let base_price_vol_up = pricing_fn(&params_vol_up)?;

    let mut params_spot_up_vol_up = *params;
    params_spot_up_vol_up.spot = params.spot + h_s;
    params_spot_up_vol_up.volatility = params.volatility + h_v;
    let price_up_vol_up = pricing_fn(&params_spot_up_vol_up)?;

    let mut params_spot_down_vol_up = *params;
    params_spot_down_vol_up.spot = params.spot - h_s;
    params_spot_down_vol_up.volatility = params.volatility + h_v;
    let price_down_vol_up = pricing_fn(&params_spot_down_vol_up)?;

    let gamma_vol_up =
        (price_up_vol_up - 2.0 * base_price_vol_up + price_down_vol_up) / (h_s * h_s);

    // Calculate gamma at σ-h
    let mut params_vol_down = *params;
    params_vol_down.volatility = (params.volatility - h_v).max(0.0);
    let base_price_vol_down = pricing_fn(&params_vol_down)?;

    let mut params_spot_up_vol_down = *params;
    params_spot_up_vol_down.spot = params.spot + h_s;
    params_spot_up_vol_down.volatility = (params.volatility - h_v).max(0.0);
    let price_up_vol_down = pricing_fn(&params_spot_up_vol_down)?;

    let mut params_spot_down_vol_down = *params;
    params_spot_down_vol_down.spot = params.spot - h_s;
    params_spot_down_vol_down.volatility = (params.volatility - h_v).max(0.0);
    let price_down_vol_down = pricing_fn(&params_spot_down_vol_down)?;

    let gamma_vol_down =
        (price_up_vol_down - 2.0 * base_price_vol_down + price_down_vol_down) / (h_s * h_s);

    // Zomma: ∂Gamma/∂σ
    let zomma = (gamma_vol_up - gamma_vol_down) / (2.0 * h_v);

    if !zomma.is_finite() {
        return Err(DervflowError::NumericalError(
            "Zomma calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(zomma)
}

/// Calculate Color: ∂³V/∂S²∂t (sensitivity of gamma to time)
fn calculate_color<F>(
    pricing_fn: &F,
    params: &OptionParams,
    spot_bump: f64,
    time_bump: f64,
) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    // Handle edge case: option at or near expiry
    if params.time_to_maturity <= time_bump {
        return Ok(0.0);
    }

    let h_s = params.spot * spot_bump;
    let h_t = time_bump;

    // Calculate gamma at current time
    let base_price = pricing_fn(params)?;

    let mut params_spot_up = *params;
    params_spot_up.spot = params.spot + h_s;
    let price_up = pricing_fn(&params_spot_up)?;

    let mut params_spot_down = *params;
    params_spot_down.spot = params.spot - h_s;
    let price_down = pricing_fn(&params_spot_down)?;

    let gamma_now = (price_up - 2.0 * base_price + price_down) / (h_s * h_s);

    // Calculate gamma at t-h (forward in time)
    let mut params_time_forward = *params;
    params_time_forward.time_to_maturity = params.time_to_maturity - h_t;
    let base_price_forward = pricing_fn(&params_time_forward)?;

    let mut params_spot_up_time_forward = *params;
    params_spot_up_time_forward.spot = params.spot + h_s;
    params_spot_up_time_forward.time_to_maturity = params.time_to_maturity - h_t;
    let price_up_forward = pricing_fn(&params_spot_up_time_forward)?;

    let mut params_spot_down_time_forward = *params;
    params_spot_down_time_forward.spot = params.spot - h_s;
    params_spot_down_time_forward.time_to_maturity = params.time_to_maturity - h_t;
    let price_down_forward = pricing_fn(&params_spot_down_time_forward)?;

    let gamma_forward =
        (price_up_forward - 2.0 * base_price_forward + price_down_forward) / (h_s * h_s);

    // Color: ∂Gamma/∂t (note: negative because time decreases)
    let color = (gamma_forward - gamma_now) / h_t;

    if !color.is_finite() {
        return Err(DervflowError::NumericalError(
            "Color calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(color)
}

/// Calculate Ultima: ∂³V/∂σ³ (third derivative with respect to volatility)
fn calculate_ultima<F>(pricing_fn: &F, params: &OptionParams, vol_bump: f64) -> Result<f64>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let h = vol_bump;

    // V(σ+2h)
    let mut params_up2 = *params;
    params_up2.volatility = params.volatility + 2.0 * h;
    let price_up2 = pricing_fn(&params_up2)?;

    // V(σ+h)
    let mut params_up = *params;
    params_up.volatility = params.volatility + h;
    let price_up = pricing_fn(&params_up)?;

    // V(σ-h)
    let mut params_down = *params;
    params_down.volatility = (params.volatility - h).max(0.0);
    let price_down = pricing_fn(&params_down)?;

    // V(σ-2h)
    let mut params_down2 = *params;
    params_down2.volatility = (params.volatility - 2.0 * h).max(0.0);
    let price_down2 = pricing_fn(&params_down2)?;

    // Third derivative: (V(σ+2h) - 2V(σ+h) + 2V(σ-h) - V(σ-2h)) / (2h³)
    let ultima = (price_up2 - 2.0 * price_up + 2.0 * price_down - price_down2) / (2.0 * h * h * h);

    if !ultima.is_finite() {
        return Err(DervflowError::NumericalError(
            "Ultima calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(ultima)
}

/// Calculate portfolio-level Greeks by aggregating individual position Greeks
///
/// This function computes the total Greeks for a portfolio by summing the Greeks
/// of each position weighted by quantity. This is useful for understanding the
/// overall risk profile of a portfolio.
///
/// # Arguments
/// * `position_greeks` - Vector of tuples containing (Greeks, quantity) for each position
///
/// # Returns
/// * `Greeks` - Aggregated portfolio Greeks
///
/// # Examples
/// ```
/// use dervflow::risk::greeks::aggregate_portfolio_greeks;
/// use dervflow::common::types::Greeks;
///
/// let position1 = (Greeks::new(0.5, 0.02, 0.1, -0.01, 0.05), 100.0);
/// let position2 = (Greeks::new(0.3, 0.01, 0.08, -0.008, 0.03), 50.0);
/// let portfolio_greeks = aggregate_portfolio_greeks(&vec![position1, position2]);
/// ```
pub fn aggregate_portfolio_greeks(position_greeks: &[(Greeks, f64)]) -> Greeks {
    let mut total_delta = 0.0;
    let mut total_gamma = 0.0;
    let mut total_vega = 0.0;
    let mut total_theta = 0.0;
    let mut total_rho = 0.0;

    for (greeks, quantity) in position_greeks {
        total_delta += greeks.delta * quantity;
        total_gamma += greeks.gamma * quantity;
        total_vega += greeks.vega * quantity;
        total_theta += greeks.theta * quantity;
        total_rho += greeks.rho * quantity;
    }

    Greeks::new(total_delta, total_gamma, total_vega, total_theta, total_rho)
}

/// Calculate portfolio-level extended Greeks by aggregating individual position Greeks
///
/// This function computes the total extended Greeks for a portfolio by summing the Greeks
/// of each position weighted by quantity.
///
/// # Arguments
/// * `position_greeks` - Vector of tuples containing (ExtendedGreeks, quantity) for each position
///
/// # Returns
/// * `ExtendedGreeks` - Aggregated portfolio extended Greeks
pub fn aggregate_portfolio_extended_greeks(
    position_greeks: &[(ExtendedGreeks, f64)],
) -> ExtendedGreeks {
    let mut total_greeks = Greeks::zero();
    let mut total_vanna = 0.0;
    let mut total_volga = 0.0;
    let mut total_speed = 0.0;
    let mut total_zomma = 0.0;
    let mut total_color = 0.0;
    let mut total_ultima = 0.0;

    for (extended, quantity) in position_greeks {
        total_greeks.delta += extended.greeks.delta * quantity;
        total_greeks.gamma += extended.greeks.gamma * quantity;
        total_greeks.vega += extended.greeks.vega * quantity;
        total_greeks.theta += extended.greeks.theta * quantity;
        total_greeks.rho += extended.greeks.rho * quantity;

        total_vanna += extended.vanna * quantity;
        total_volga += extended.volga * quantity;
        total_speed += extended.speed * quantity;
        total_zomma += extended.zomma * quantity;
        total_color += extended.color * quantity;
        total_ultima += extended.ultima * quantity;
    }

    ExtendedGreeks::new(
        total_greeks,
        total_vanna,
        total_volga,
        total_speed,
        total_zomma,
        total_color,
        total_ultima,
    )
}

/// Calculate Greeks for a portfolio of options
///
/// This is a convenience function that calculates Greeks for each position in a portfolio
/// and aggregates them. It takes a pricing function and a list of (OptionParams, quantity) pairs.
///
/// # Arguments
/// * `pricing_fn` - Function that prices the option given parameters
/// * `positions` - Vector of tuples containing (OptionParams, quantity) for each position
/// * `config` - Configuration for bump sizes (optional, uses defaults if None)
///
/// # Returns
/// * `Ok(Greeks)` - Aggregated portfolio Greeks
/// * `Err(DervflowError)` - If any position fails to calculate Greeks
pub fn calculate_portfolio_greeks<F>(
    pricing_fn: &F,
    positions: &[(OptionParams, f64)],
    config: Option<FiniteDifferenceConfig>,
) -> Result<Greeks>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let mut position_greeks = Vec::with_capacity(positions.len());

    for (params, quantity) in positions {
        let greeks = calculate_numerical_greeks(pricing_fn, params, config)?;
        position_greeks.push((greeks, *quantity));
    }

    Ok(aggregate_portfolio_greeks(&position_greeks))
}

/// Calculate extended Greeks for a portfolio of options
///
/// This is a convenience function that calculates extended Greeks for each position in a portfolio
/// and aggregates them.
///
/// # Arguments
/// * `pricing_fn` - Function that prices the option given parameters
/// * `positions` - Vector of tuples containing (OptionParams, quantity) for each position
/// * `config` - Configuration for bump sizes (optional, uses defaults if None)
///
/// # Returns
/// * `Ok(ExtendedGreeks)` - Aggregated portfolio extended Greeks
/// * `Err(DervflowError)` - If any position fails to calculate Greeks
pub fn calculate_portfolio_extended_greeks<F>(
    pricing_fn: &F,
    positions: &[(OptionParams, f64)],
    config: Option<FiniteDifferenceConfig>,
) -> Result<ExtendedGreeks>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    let mut position_greeks = Vec::with_capacity(positions.len());

    for (params, quantity) in positions {
        let extended = calculate_extended_greeks(pricing_fn, params, config)?;
        position_greeks.push((extended, *quantity));
    }

    Ok(aggregate_portfolio_extended_greeks(&position_greeks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::OptionType;
    use crate::options::analytical::{black_scholes_greeks, black_scholes_price};

    #[test]
    fn test_numerical_greeks_call_atm() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let numerical =
            calculate_numerical_greeks(&black_scholes_price, &params, Some(config)).unwrap();
        let analytical = black_scholes_greeks(&params).unwrap();

        // Delta should be close (within 1%)
        assert!((numerical.delta - analytical.delta).abs() / analytical.delta < 0.01);

        // Gamma should be close (within 5% due to second derivative)
        assert!((numerical.gamma - analytical.gamma).abs() / analytical.gamma < 0.05);

        // Vega should be close (within 1%)
        assert!((numerical.vega - analytical.vega).abs() / analytical.vega < 0.01);

        // Theta is sensitive to time discretization - just check it's negative and reasonable
        assert!(numerical.theta < 0.0);
        assert!(numerical.theta > -1.0); // Should not be too large

        // Rho should be close (within 1%)
        assert!((numerical.rho - analytical.rho).abs() / analytical.rho < 0.01);
    }

    #[test]
    fn test_numerical_greeks_put_atm() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let config = FiniteDifferenceConfig::default();

        let numerical =
            calculate_numerical_greeks(&black_scholes_price, &params, Some(config)).unwrap();
        let analytical = black_scholes_greeks(&params).unwrap();

        // Delta should be close (within 1%)
        assert!((numerical.delta - analytical.delta).abs() / analytical.delta.abs() < 0.01);

        // Gamma should be close (within 5%)
        assert!((numerical.gamma - analytical.gamma).abs() / analytical.gamma < 0.05);

        // Vega should be close (within 1%)
        assert!((numerical.vega - analytical.vega).abs() / analytical.vega < 0.01);
    }

    #[test]
    fn test_numerical_greeks_call_itm() {
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let numerical =
            calculate_numerical_greeks(&black_scholes_price, &params, Some(config)).unwrap();

        // Delta should be high for ITM call
        assert!(numerical.delta > 0.7);

        // Gamma should be positive
        assert!(numerical.gamma > 0.0);

        // Vega should be positive
        assert!(numerical.vega > 0.0);
    }

    #[test]
    fn test_numerical_greeks_put_otm() {
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let config = FiniteDifferenceConfig::default();

        let numerical =
            calculate_numerical_greeks(&black_scholes_price, &params, Some(config)).unwrap();

        // Delta should be small negative for OTM put
        assert!(numerical.delta < 0.0 && numerical.delta > -0.3);

        // Gamma should be positive
        assert!(numerical.gamma > 0.0);
    }

    #[test]
    fn test_numerical_greeks_custom_bumps() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        // Use smaller bumps for more accuracy
        let config = FiniteDifferenceConfig {
            spot_bump: 0.001,
            vol_bump: 0.001,
            time_bump: 1.0 / 365.0,
            rate_bump: 0.00001,
        };

        let numerical =
            calculate_numerical_greeks(&black_scholes_price, &params, Some(config)).unwrap();
        let analytical = black_scholes_greeks(&params).unwrap();

        // With smaller bumps, accuracy should improve
        assert!((numerical.delta - analytical.delta).abs() / analytical.delta < 0.001);
    }

    #[test]
    fn test_numerical_greeks_near_expiry() {
        // Use time less than time_bump (1/365) to trigger zero theta
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 0.001, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let result = calculate_numerical_greeks(&black_scholes_price, &params, Some(config));
        assert!(result.is_ok());

        let greeks = result.unwrap();
        // When time to maturity < time bump, theta should be zero
        assert_eq!(greeks.theta, 0.0);
    }

    #[test]
    fn test_numerical_greeks_invalid_params() {
        let params = OptionParams::new(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let result = calculate_numerical_greeks(&black_scholes_price, &params, Some(config));
        assert!(result.is_err());
    }

    #[test]
    fn test_extended_greeks_call_atm() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let extended =
            calculate_extended_greeks(&black_scholes_price, &params, Some(config)).unwrap();

        // First-order Greeks should be reasonable
        assert!(extended.greeks.delta > 0.5 && extended.greeks.delta < 0.7);
        assert!(extended.greeks.gamma > 0.0);
        assert!(extended.greeks.vega > 0.0);

        // Vanna should be non-zero
        assert!(extended.vanna.abs() > 0.0);

        // Volga should be positive for ATM options
        assert!(extended.volga > 0.0);

        // Speed should be non-zero
        assert!(extended.speed.abs() > 0.0);

        // Zomma should be non-zero
        assert!(extended.zomma.abs() > 0.0);

        // Color should be non-zero
        assert!(extended.color.abs() > 0.0);

        // Ultima should be non-zero
        assert!(extended.ultima.abs() > 0.0);
    }

    #[test]
    fn test_extended_greeks_put_atm() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let config = FiniteDifferenceConfig::default();

        let extended =
            calculate_extended_greeks(&black_scholes_price, &params, Some(config)).unwrap();

        // First-order Greeks should be reasonable
        assert!(extended.greeks.delta < 0.0);
        assert!(extended.greeks.gamma > 0.0);
        assert!(extended.greeks.vega > 0.0);

        // Second and third order Greeks should be non-zero
        assert!(extended.vanna.abs() > 0.0);
        assert!(extended.volga > 0.0);
        assert!(extended.speed.abs() > 0.0);
    }

    #[test]
    fn test_extended_greeks_call_itm() {
        let params = OptionParams::new(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let extended =
            calculate_extended_greeks(&black_scholes_price, &params, Some(config)).unwrap();

        // Delta should be high for ITM call
        assert!(extended.greeks.delta > 0.7);

        // All higher-order Greeks should be calculable
        assert!(extended.vanna.is_finite());
        assert!(extended.volga.is_finite());
        assert!(extended.speed.is_finite());
        assert!(extended.zomma.is_finite());
        assert!(extended.color.is_finite());
        assert!(extended.ultima.is_finite());
    }

    #[test]
    fn test_extended_greeks_near_expiry() {
        // Use time less than time_bump (1/365) to trigger zero color
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 0.001, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let result = calculate_extended_greeks(&black_scholes_price, &params, Some(config));
        assert!(result.is_ok());

        let extended = result.unwrap();
        // When time to maturity < time bump, color should be zero
        assert_eq!(extended.color, 0.0);
    }

    #[test]
    fn test_vanna_calculation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let vanna = calculate_vanna(
            &black_scholes_price,
            &params,
            config.spot_bump,
            config.vol_bump,
        )
        .unwrap();

        // Vanna should be finite and non-zero for ATM option
        assert!(vanna.is_finite());
        assert!(vanna.abs() > 0.0);
    }

    #[test]
    fn test_volga_calculation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let volga = calculate_volga(&black_scholes_price, &params, config.vol_bump).unwrap();

        // Volga should be positive for ATM options
        assert!(volga > 0.0);
    }

    #[test]
    fn test_speed_calculation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let speed = calculate_speed(&black_scholes_price, &params, config.spot_bump).unwrap();

        // Speed should be finite
        assert!(speed.is_finite());
    }

    #[test]
    fn test_zomma_calculation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let zomma = calculate_zomma(
            &black_scholes_price,
            &params,
            config.spot_bump,
            config.vol_bump,
        )
        .unwrap();

        // Zomma should be finite
        assert!(zomma.is_finite());
    }

    #[test]
    fn test_color_calculation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let color = calculate_color(
            &black_scholes_price,
            &params,
            config.spot_bump,
            config.time_bump,
        )
        .unwrap();

        // Color should be finite
        assert!(color.is_finite());
    }

    #[test]
    fn test_ultima_calculation() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let config = FiniteDifferenceConfig::default();

        let ultima = calculate_ultima(&black_scholes_price, &params, config.vol_bump).unwrap();

        // Ultima should be finite
        assert!(ultima.is_finite());
    }

    #[test]
    fn test_aggregate_portfolio_greeks() {
        let greeks1 = Greeks::new(0.5, 0.02, 0.1, -0.01, 0.05);
        let greeks2 = Greeks::new(0.3, 0.01, 0.08, -0.008, 0.03);

        let position_greeks = vec![(greeks1, 100.0), (greeks2, 50.0)];

        let portfolio = aggregate_portfolio_greeks(&position_greeks);

        // Check aggregation
        assert!((portfolio.delta - (0.5 * 100.0 + 0.3 * 50.0)).abs() < 1e-10);
        assert!((portfolio.gamma - (0.02 * 100.0 + 0.01 * 50.0)).abs() < 1e-10);
        assert!((portfolio.vega - (0.1 * 100.0 + 0.08 * 50.0)).abs() < 1e-10);
        assert!((portfolio.theta - (-0.01 * 100.0 + -0.008 * 50.0)).abs() < 1e-10);
        assert!((portfolio.rho - (0.05 * 100.0 + 0.03 * 50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_portfolio_greeks_with_short_positions() {
        let greeks1 = Greeks::new(0.5, 0.02, 0.1, -0.01, 0.05);
        let greeks2 = Greeks::new(0.3, 0.01, 0.08, -0.008, 0.03);

        let position_greeks = vec![
            (greeks1, 100.0), // Long position
            (greeks2, -50.0), // Short position
        ];

        let portfolio = aggregate_portfolio_greeks(&position_greeks);

        // Check aggregation with short position
        assert!((portfolio.delta - (0.5 * 100.0 + 0.3 * -50.0)).abs() < 1e-10);
        assert!((portfolio.gamma - (0.02 * 100.0 + 0.01 * -50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_portfolio_extended_greeks() {
        let greeks1 = Greeks::new(0.5, 0.02, 0.1, -0.01, 0.05);
        let extended1 = ExtendedGreeks::new(greeks1, 0.001, 0.002, 0.0001, 0.0002, 0.0003, 0.0004);

        let greeks2 = Greeks::new(0.3, 0.01, 0.08, -0.008, 0.03);
        let extended2 =
            ExtendedGreeks::new(greeks2, 0.0008, 0.0015, 0.00008, 0.00015, 0.00025, 0.00035);

        let position_greeks = vec![(extended1, 100.0), (extended2, 50.0)];

        let portfolio = aggregate_portfolio_extended_greeks(&position_greeks);

        // Check first-order Greeks aggregation
        assert!((portfolio.greeks.delta - (0.5 * 100.0 + 0.3 * 50.0)).abs() < 1e-10);

        // Check second-order Greeks aggregation
        assert!((portfolio.vanna - (0.001 * 100.0 + 0.0008 * 50.0)).abs() < 1e-10);
        assert!((portfolio.volga - (0.002 * 100.0 + 0.0015 * 50.0)).abs() < 1e-10);

        // Check third-order Greeks aggregation
        assert!((portfolio.speed - (0.0001 * 100.0 + 0.00008 * 50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_portfolio_greeks() {
        let params1 = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params2 = OptionParams::new(100.0, 105.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let positions = vec![(params1, 100.0), (params2, 50.0)];

        let config = FiniteDifferenceConfig::default();
        let portfolio =
            calculate_portfolio_greeks(&black_scholes_price, &positions, Some(config)).unwrap();

        // Portfolio Greeks should be reasonable
        assert!(portfolio.delta.is_finite());
        assert!(portfolio.gamma.is_finite());
        assert!(portfolio.vega.is_finite());
        assert!(portfolio.theta.is_finite());
        assert!(portfolio.rho.is_finite());
    }

    #[test]
    fn test_calculate_portfolio_extended_greeks() {
        let params1 = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params2 = OptionParams::new(100.0, 105.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let positions = vec![(params1, 100.0), (params2, 50.0)];

        let config = FiniteDifferenceConfig::default();
        let portfolio =
            calculate_portfolio_extended_greeks(&black_scholes_price, &positions, Some(config))
                .unwrap();

        // Portfolio Greeks should be reasonable
        assert!(portfolio.greeks.delta.is_finite());
        assert!(portfolio.vanna.is_finite());
        assert!(portfolio.volga.is_finite());
        assert!(portfolio.speed.is_finite());
        assert!(portfolio.zomma.is_finite());
        assert!(portfolio.color.is_finite());
        assert!(portfolio.ultima.is_finite());
    }

    #[test]
    fn test_portfolio_greeks_hedged_position() {
        // Create a hedged position: long call + short call at different strikes
        let params_long = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params_short = OptionParams::new(100.0, 105.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        let positions = vec![
            (params_long, 100.0),   // Long 100 ATM calls
            (params_short, -100.0), // Short 100 OTM calls
        ];

        let config = FiniteDifferenceConfig::default();
        let portfolio =
            calculate_portfolio_greeks(&black_scholes_price, &positions, Some(config)).unwrap();

        // Delta should be reduced due to hedging
        let long_greeks =
            calculate_numerical_greeks(&black_scholes_price, &params_long, Some(config)).unwrap();
        assert!(portfolio.delta.abs() < long_greeks.delta.abs() * 100.0);
    }

    #[test]
    fn test_empty_portfolio() {
        let positions: Vec<(OptionParams, f64)> = vec![];
        let config = FiniteDifferenceConfig::default();

        let portfolio =
            calculate_portfolio_greeks(&black_scholes_price, &positions, Some(config)).unwrap();

        // Empty portfolio should have zero Greeks
        assert_eq!(portfolio.delta, 0.0);
        assert_eq!(portfolio.gamma, 0.0);
        assert_eq!(portfolio.vega, 0.0);
        assert_eq!(portfolio.theta, 0.0);
        assert_eq!(portfolio.rho, 0.0);
    }
}
