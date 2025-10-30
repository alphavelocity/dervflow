// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Bond analytics (duration, convexity, YTM)
//!
//! Provides bond analytics calculations:
//! - Yield to maturity (YTM)
//! - Macaulay duration
//! - Modified duration
//! - Convexity
//! - DV01 (dollar value of 01 basis point)

use crate::common::error::{DervflowError, Result};
use crate::common::types::{Rate, Time};
use crate::numerical::root_finding::{RootFindingConfig, brent};

/// Bond cashflow structure
#[derive(Debug, Clone, Copy)]
pub struct Cashflow {
    /// Time when cashflow occurs (in years)
    pub time: Time,
    /// Amount of cashflow
    pub amount: f64,
}

impl Cashflow {
    /// Create a new cashflow
    pub fn new(time: Time, amount: f64) -> Self {
        Self { time, amount }
    }
}

/// Calculate yield to maturity (YTM) for a bond
///
/// YTM is the internal rate of return of the bond's cashflows
///
/// # Arguments
/// * `price` - Current market price of the bond
/// * `cashflows` - Vector of bond cashflows
/// * `initial_guess` - Initial guess for YTM (optional, defaults to 0.05)
///
/// # Returns
/// * `Result<Rate>` - Yield to maturity
///
/// # Example
/// ```
/// use dervflow::yield_curve::bond_analytics::{yield_to_maturity, Cashflow};
///
/// let cashflows = vec![
///     Cashflow::new(1.0, 5.0),
///     Cashflow::new(2.0, 5.0),
///     Cashflow::new(2.0, 100.0),
/// ];
/// let ytm = yield_to_maturity(98.0, &cashflows, None).unwrap();
/// ```
pub fn yield_to_maturity(
    price: f64,
    cashflows: &[Cashflow],
    initial_guess: Option<Rate>,
) -> Result<Rate> {
    if cashflows.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cashflows cannot be empty".to_string(),
        ));
    }

    if price <= 0.0 {
        return Err(DervflowError::InvalidInput(format!(
            "Price must be positive, got {}",
            price
        )));
    }

    // Validate cashflows
    for cf in cashflows {
        if cf.time < 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Cashflow time must be non-negative, got {}",
                cf.time
            )));
        }
    }

    let _guess = initial_guess.unwrap_or(0.05);

    // Define the price function as a function of yield
    let price_fn = |y: f64| -> f64 {
        let mut pv = 0.0;
        for cf in cashflows {
            if cf.time > 0.0 {
                pv += cf.amount * (-y * cf.time).exp();
            } else {
                pv += cf.amount;
            }
        }
        pv - price
    };

    // Use Brent's method to find the root
    // Search in range [0.0, 0.5] which covers most realistic yields
    // Negative yields are rare, and yields above 50% are unrealistic
    let config = RootFindingConfig {
        max_iterations: 100,
        tolerance: 1e-8,
        relative_tolerance: 1e-8,
    };

    let result = brent(price_fn, 0.0, 0.5, &config)?;

    Ok(result.root)
}

/// Calculate Macaulay duration
///
/// Macaulay duration is the weighted average time to receive cashflows
///
/// # Arguments
/// * `yield_rate` - Yield rate (typically YTM)
/// * `cashflows` - Vector of bond cashflows
///
/// # Returns
/// * `Result<f64>` - Macaulay duration in years
pub fn macaulay_duration(yield_rate: Rate, cashflows: &[Cashflow]) -> Result<f64> {
    if cashflows.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cashflows cannot be empty".to_string(),
        ));
    }

    let mut weighted_time = 0.0;
    let mut total_pv = 0.0;

    for cf in cashflows {
        if cf.time < 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Cashflow time must be non-negative, got {}",
                cf.time
            )));
        }

        if cf.time > 0.0 {
            let pv = cf.amount * (-yield_rate * cf.time).exp();
            weighted_time += cf.time * pv;
            total_pv += pv;
        } else {
            total_pv += cf.amount;
        }
    }

    if total_pv <= 0.0 {
        return Err(DervflowError::NumericalError(
            "Total present value must be positive".to_string(),
        ));
    }

    Ok(weighted_time / total_pv)
}

/// Calculate Modified duration
///
/// Modified duration measures the price sensitivity to yield changes
/// Modified Duration = Macaulay Duration / (1 + y/m)
/// For continuous compounding: Modified Duration = Macaulay Duration
///
/// # Arguments
/// * `yield_rate` - Yield rate (typically YTM)
/// * `cashflows` - Vector of bond cashflows
/// * `compounding_frequency` - Compounding frequency per year (0 for continuous)
///
/// # Returns
/// * `Result<f64>` - Modified duration
pub fn modified_duration(
    yield_rate: Rate,
    cashflows: &[Cashflow],
    compounding_frequency: u32,
) -> Result<f64> {
    let mac_duration = macaulay_duration(yield_rate, cashflows)?;

    if compounding_frequency == 0 {
        // Continuous compounding
        Ok(mac_duration)
    } else {
        // Discrete compounding
        let m = compounding_frequency as f64;
        Ok(mac_duration / (1.0 + yield_rate / m))
    }
}

/// Calculate convexity
///
/// Convexity measures the curvature of the price-yield relationship
///
/// # Arguments
/// * `yield_rate` - Yield rate (typically YTM)
/// * `cashflows` - Vector of bond cashflows
///
/// # Returns
/// * `Result<f64>` - Convexity
pub fn convexity(yield_rate: Rate, cashflows: &[Cashflow]) -> Result<f64> {
    if cashflows.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cashflows cannot be empty".to_string(),
        ));
    }

    let mut weighted_time_squared = 0.0;
    let mut total_pv = 0.0;

    for cf in cashflows {
        if cf.time < 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Cashflow time must be non-negative, got {}",
                cf.time
            )));
        }

        if cf.time > 0.0 {
            let pv = cf.amount * (-yield_rate * cf.time).exp();
            weighted_time_squared += cf.time * cf.time * pv;
            total_pv += pv;
        } else {
            total_pv += cf.amount;
        }
    }

    if total_pv <= 0.0 {
        return Err(DervflowError::NumericalError(
            "Total present value must be positive".to_string(),
        ));
    }

    Ok(weighted_time_squared / total_pv)
}

/// Calculate DV01 (Dollar Value of 01 basis point)
///
/// DV01 measures the change in bond price for a 1 basis point change in yield
///
/// # Arguments
/// * `yield_rate` - Yield rate (typically YTM)
/// * `cashflows` - Vector of bond cashflows
///
/// # Returns
/// * `Result<f64>` - DV01
pub fn dv01(yield_rate: Rate, cashflows: &[Cashflow]) -> Result<f64> {
    if cashflows.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cashflows cannot be empty".to_string(),
        ));
    }

    // Calculate price at current yield
    let mut price = 0.0;
    for cf in cashflows {
        if cf.time > 0.0 {
            price += cf.amount * (-yield_rate * cf.time).exp();
        } else {
            price += cf.amount;
        }
    }

    // Calculate modified duration (continuous compounding)
    let mod_dur = modified_duration(yield_rate, cashflows, 0)?;

    // DV01 = Modified Duration * Price * 0.0001 (1 basis point = 0.01%)
    Ok(mod_dur * price * 0.0001)
}

/// Calculate bond price from yield
///
/// # Arguments
/// * `yield_rate` - Yield rate
/// * `cashflows` - Vector of bond cashflows
///
/// # Returns
/// * `Result<f64>` - Bond price
pub fn bond_price(yield_rate: Rate, cashflows: &[Cashflow]) -> Result<f64> {
    if cashflows.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cashflows cannot be empty".to_string(),
        ));
    }

    let mut price = 0.0;
    for cf in cashflows {
        if cf.time < 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Cashflow time must be non-negative, got {}",
                cf.time
            )));
        }

        if cf.time > 0.0 {
            price += cf.amount * (-yield_rate * cf.time).exp();
        } else {
            price += cf.amount;
        }
    }

    Ok(price)
}

/// Generate cashflows for a standard coupon bond
///
/// # Arguments
/// * `maturity` - Time to maturity in years
/// * `coupon_rate` - Annual coupon rate
/// * `face_value` - Face value of the bond
/// * `frequency` - Payment frequency per year
///
/// # Returns
/// * `Vec<Cashflow>` - Vector of cashflows
pub fn generate_bond_cashflows(
    maturity: Time,
    coupon_rate: Rate,
    face_value: f64,
    frequency: u32,
) -> Vec<Cashflow> {
    let mut cashflows = Vec::new();
    let coupon_payment = (coupon_rate * face_value) / frequency as f64;
    let dt = 1.0 / frequency as f64;
    let num_payments = (maturity * frequency as f64).round() as usize;

    for i in 1..=num_payments {
        let time = i as f64 * dt;
        if i == num_payments {
            // Last payment includes principal
            cashflows.push(Cashflow::new(time, coupon_payment + face_value));
        } else {
            cashflows.push(Cashflow::new(time, coupon_payment));
        }
    }

    cashflows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_bond_cashflows() {
        let cashflows = generate_bond_cashflows(2.0, 0.05, 100.0, 2);
        assert_eq!(cashflows.len(), 4);

        // Check coupon payments
        assert!((cashflows[0].amount - 2.5).abs() < 1e-10);
        assert!((cashflows[1].amount - 2.5).abs() < 1e-10);
        assert!((cashflows[2].amount - 2.5).abs() < 1e-10);

        // Check final payment (coupon + principal)
        assert!((cashflows[3].amount - 102.5).abs() < 1e-10);
    }

    #[test]
    fn test_bond_price() {
        let cashflows = vec![Cashflow::new(1.0, 5.0), Cashflow::new(2.0, 105.0)];

        let price = bond_price(0.05, &cashflows).unwrap();
        let expected = 5.0 * (-0.05_f64).exp() + 105.0 * (-0.10_f64).exp();
        assert!((price - expected).abs() < 1e-6);
    }

    #[test]
    fn test_yield_to_maturity() {
        let cashflows = generate_bond_cashflows(2.0, 0.05, 100.0, 2);
        let price = 98.0;

        let ytm = yield_to_maturity(price, &cashflows, None).unwrap();

        // YTM should be positive and reasonable
        assert!(ytm > 0.0);
        assert!(ytm < 0.2);

        // Verify: price calculated with YTM should match input price
        let calculated_price = bond_price(ytm, &cashflows).unwrap();
        assert!((calculated_price - price).abs() < 0.01);
    }

    #[test]
    fn test_macaulay_duration() {
        let cashflows = vec![Cashflow::new(1.0, 5.0), Cashflow::new(2.0, 105.0)];

        let duration = macaulay_duration(0.05, &cashflows).unwrap();

        // Duration should be between 1 and 2 years
        assert!(duration > 1.0);
        assert!(duration < 2.0);
    }

    #[test]
    fn test_modified_duration() {
        let cashflows = vec![Cashflow::new(1.0, 5.0), Cashflow::new(2.0, 105.0)];

        // Continuous compounding
        let mod_dur_cont = modified_duration(0.05, &cashflows, 0).unwrap();
        let mac_dur = macaulay_duration(0.05, &cashflows).unwrap();
        assert!((mod_dur_cont - mac_dur).abs() < 1e-10);

        // Semi-annual compounding
        let mod_dur_semi = modified_duration(0.05, &cashflows, 2).unwrap();
        assert!(mod_dur_semi < mac_dur);
    }

    #[test]
    fn test_convexity() {
        let cashflows = vec![Cashflow::new(1.0, 5.0), Cashflow::new(2.0, 105.0)];

        let conv = convexity(0.05, &cashflows).unwrap();

        // Convexity should be positive
        assert!(conv > 0.0);
    }

    #[test]
    fn test_dv01() {
        let cashflows = generate_bond_cashflows(5.0, 0.05, 100.0, 2);

        let dv01_value = dv01(0.05, &cashflows).unwrap();

        // DV01 should be positive
        assert!(dv01_value > 0.0);

        // For a 5-year bond, DV01 should be reasonable (typically 0.04-0.05 per 100 face value)
        assert!(dv01_value < 1.0);
    }

    #[test]
    fn test_zero_coupon_bond() {
        // Zero coupon bond: single payment at maturity
        let cashflows = vec![Cashflow::new(5.0, 100.0)];

        let ytm = yield_to_maturity(78.35, &cashflows, None).unwrap();

        // Verify the calculated YTM gives the correct price
        let verify_pv = 100.0 * (-ytm * 5.0).exp();

        // The YTM should give us back the original price within tolerance
        assert!(
            (verify_pv - 78.35).abs() < 0.01,
            "YTM should reproduce the original price. Got PV={}, Expected=78.35, YTM={}",
            verify_pv,
            ytm
        );

        let duration = macaulay_duration(0.05, &cashflows).unwrap();
        assert!((duration - 5.0).abs() < 1e-6); // Duration equals maturity for zero coupon
    }

    #[test]
    fn test_invalid_inputs() {
        let cashflows = vec![Cashflow::new(1.0, 100.0)];

        // Invalid price
        let result = yield_to_maturity(-10.0, &cashflows, None);
        assert!(result.is_err());

        // Empty cashflows
        let empty: Vec<Cashflow> = vec![];
        let result = macaulay_duration(0.05, &empty);
        assert!(result.is_err());

        // Negative time
        let invalid_cf = vec![Cashflow::new(-1.0, 100.0)];
        let result = bond_price(0.05, &invalid_cf);
        assert!(result.is_err());
    }
}
