// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Yield curve bootstrapping
//!
//! Bootstrap zero curves from bond prices and swap rates

use crate::common::error::{DervflowError, Result};
use crate::common::types::{BondQuote, Rate, Time};

/// Represents a swap quote for bootstrapping
#[derive(Debug, Clone, Copy)]
pub struct SwapQuote {
    /// Time to maturity (in years)
    pub maturity: Time,
    /// Fixed swap rate (annualized)
    pub rate: Rate,
    /// Payment frequency per year (e.g., 2 for semi-annual)
    pub frequency: u32,
}

impl SwapQuote {
    /// Create a new SwapQuote instance
    pub fn new(maturity: Time, rate: Rate, frequency: u32) -> Self {
        Self {
            maturity,
            rate,
            frequency,
        }
    }
}

/// Bootstrap zero rates from bond prices
///
/// Uses the bootstrapping algorithm to extract zero rates from a set of bond prices.
/// Bonds should be sorted by maturity in ascending order.
///
/// # Arguments
/// * `bonds` - Slice of bond quotes with prices
///
/// # Returns
/// * `Result<(Vec<Time>, Vec<Rate>)>` - Tuple of (maturities, zero_rates)
///
/// # Example
/// ```
/// use dervflow::yield_curve::bootstrap::bootstrap_from_bonds;
/// use dervflow::common::types::BondQuote;
///
/// let bonds = vec![
///     BondQuote::new(0.5, 0.03, 99.5, 2),
///     BondQuote::new(1.0, 0.04, 99.0, 2),
///     BondQuote::new(2.0, 0.05, 98.0, 2),
/// ];
///
/// let (maturities, rates) = bootstrap_from_bonds(&bonds).unwrap();
/// ```
pub fn bootstrap_from_bonds(bonds: &[BondQuote]) -> Result<(Vec<Time>, Vec<Rate>)> {
    if bonds.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Bond list cannot be empty".to_string(),
        ));
    }

    // Validate bonds
    for bond in bonds {
        if bond.maturity <= 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Bond maturity must be positive, got {}",
                bond.maturity
            )));
        }
        if bond.frequency == 0 {
            return Err(DervflowError::InvalidInput(
                "Bond frequency must be positive".to_string(),
            ));
        }
        if bond.price <= 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Bond price must be positive, got {}",
                bond.price
            )));
        }
    }

    let mut maturities = Vec::new();
    let mut zero_rates = Vec::new();

    // Bootstrap each bond sequentially
    for bond in bonds {
        // Assume face value = 100 (standard convention)
        let face_value = 100.0;
        let coupon_payment = (bond.coupon * face_value) / bond.frequency as f64;
        let dt = 1.0 / bond.frequency as f64;
        let num_payments = (bond.maturity * bond.frequency as f64).round() as usize;

        // Calculate present value of known cashflows using already bootstrapped rates
        let mut pv_known = 0.0;
        for i in 1..num_payments {
            let t = i as f64 * dt;
            let discount_factor = calculate_discount_factor(t, &maturities, &zero_rates);
            pv_known += coupon_payment * discount_factor;
        }

        // Final payment includes coupon and principal
        let final_payment = coupon_payment + face_value;
        let final_time = bond.maturity;

        // Solve for the zero rate at final maturity
        // bond.price = pv_known + final_payment * exp(-r * T)
        // exp(-r * T) = (bond.price - pv_known) / final_payment
        let discount_factor = (bond.price - pv_known) / final_payment;

        if discount_factor <= 0.0 {
            return Err(DervflowError::NumericalError(format!(
                "Invalid discount factor {} at maturity {}. Check bond prices.",
                discount_factor, final_time
            )));
        }

        let zero_rate = -discount_factor.ln() / final_time;

        maturities.push(final_time);
        zero_rates.push(zero_rate);
    }

    Ok((maturities, zero_rates))
}

/// Bootstrap zero rates from swap rates
///
/// Uses swap rates to construct a zero curve. Swaps should be sorted by maturity.
///
/// # Arguments
/// * `swaps` - Slice of swap quotes
///
/// # Returns
/// * `Result<(Vec<Time>, Vec<Rate>)>` - Tuple of (maturities, zero_rates)
///
/// # Example
/// ```
/// use dervflow::yield_curve::bootstrap::{bootstrap_from_swaps, SwapQuote};
///
/// let swaps = vec![
///     SwapQuote::new(1.0, 0.03, 2),
///     SwapQuote::new(2.0, 0.035, 2),
///     SwapQuote::new(5.0, 0.04, 2),
/// ];
///
/// let (maturities, rates) = bootstrap_from_swaps(&swaps).unwrap();
/// ```
pub fn bootstrap_from_swaps(swaps: &[SwapQuote]) -> Result<(Vec<Time>, Vec<Rate>)> {
    if swaps.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Swap list cannot be empty".to_string(),
        ));
    }

    // Validate swaps
    for swap in swaps {
        if swap.maturity <= 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Swap maturity must be positive, got {}",
                swap.maturity
            )));
        }
        if swap.frequency == 0 {
            return Err(DervflowError::InvalidInput(
                "Swap frequency must be positive".to_string(),
            ));
        }
    }

    let mut maturities = Vec::new();
    let mut zero_rates = Vec::new();

    // Bootstrap each swap sequentially
    for swap in swaps {
        let fixed_payment = swap.rate / swap.frequency as f64;
        let dt = 1.0 / swap.frequency as f64;
        let num_payments = (swap.maturity * swap.frequency as f64).round() as usize;

        // For a par swap, the present value of fixed leg equals the present value of floating leg
        // PV(fixed) = sum of (fixed_payment * DF(t_i)) = 1.0 (assuming notional = 1)
        // PV(floating) = 1.0 - DF(T) = 1.0 (at inception)
        // Therefore: sum of (fixed_payment * DF(t_i)) = 1.0 - DF(T)

        // Calculate sum of discount factors for known maturities
        let mut sum_df_known = 0.0;
        for i in 1..num_payments {
            let t = i as f64 * dt;
            let df = calculate_discount_factor(t, &maturities, &zero_rates);
            sum_df_known += df;
        }

        let final_time = swap.maturity;

        // Solve for final discount factor
        // fixed_payment * (sum_df_known + DF(T)) = 1.0 - DF(T)
        // fixed_payment * sum_df_known + fixed_payment * DF(T) = 1.0 - DF(T)
        // DF(T) * (fixed_payment + 1.0) = 1.0 - fixed_payment * sum_df_known
        let final_df = (1.0 - fixed_payment * sum_df_known) / (1.0 + fixed_payment);

        if final_df <= 0.0 {
            return Err(DervflowError::NumericalError(format!(
                "Invalid discount factor {} at maturity {}. Check swap rates.",
                final_df, final_time
            )));
        }

        let zero_rate = -final_df.ln() / final_time;

        maturities.push(final_time);
        zero_rates.push(zero_rate);
    }

    Ok((maturities, zero_rates))
}

/// Calculate discount factor at time t using linear interpolation of zero rates
fn calculate_discount_factor(t: Time, maturities: &[Time], zero_rates: &[Rate]) -> f64 {
    if maturities.is_empty() {
        return 1.0;
    }

    // Find the interpolation points
    let mut i = 0;
    while i < maturities.len() && maturities[i] < t {
        i += 1;
    }

    let rate = if i == 0 {
        // Extrapolate using first rate
        zero_rates[0]
    } else if i >= maturities.len() {
        // Extrapolate using last rate
        zero_rates[zero_rates.len() - 1]
    } else {
        // Linear interpolation
        let t0 = maturities[i - 1];
        let t1 = maturities[i];
        let r0 = zero_rates[i - 1];
        let r1 = zero_rates[i];
        r0 + (r1 - r0) * (t - t0) / (t1 - t0)
    };

    (-rate * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_from_bonds_single() {
        // Single zero-coupon bond
        // Price = 95 (95% of face value 100)
        // For zero coupon: Price = FV * e^(-r*T)
        // 95 = 100 * e^(-r*1)
        // e^(-r) = 0.95
        // r = -ln(0.95) ≈ 0.05129
        let bonds = vec![BondQuote::new(1.0, 0.0, 95.0, 1)];
        let (maturities, rates) = bootstrap_from_bonds(&bonds).unwrap();

        assert_eq!(maturities.len(), 1);
        assert_eq!(maturities[0], 1.0);
        // Expected rate: -ln(95/100) / 1.0 ≈ 0.05129
        assert!(
            (rates[0] - 0.05129).abs() < 0.001,
            "Expected rate ~0.05129, got {}",
            rates[0]
        );
    }

    #[test]
    fn test_bootstrap_from_bonds_multiple() {
        // Multiple bonds with coupons
        let bonds = vec![
            BondQuote::new(0.5, 0.03, 99.5, 2),
            BondQuote::new(1.0, 0.04, 99.0, 2),
        ];
        let (maturities, rates) = bootstrap_from_bonds(&bonds).unwrap();

        assert_eq!(maturities.len(), 2);
        assert_eq!(maturities[0], 0.5);
        assert_eq!(maturities[1], 1.0);
        // Rates should be positive
        assert!(rates[0] > 0.0);
        assert!(rates[1] > 0.0);
    }

    #[test]
    fn test_bootstrap_from_bonds_empty() {
        let bonds: Vec<BondQuote> = vec![];
        let result = bootstrap_from_bonds(&bonds);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_from_bonds_invalid_maturity() {
        let bonds = vec![BondQuote::new(-1.0, 0.03, 99.5, 2)];
        let result = bootstrap_from_bonds(&bonds);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_from_swaps_single() {
        let swaps = vec![SwapQuote::new(1.0, 0.05, 2)];
        let (maturities, rates) = bootstrap_from_swaps(&swaps).unwrap();

        assert_eq!(maturities.len(), 1);
        assert_eq!(maturities[0], 1.0);
        assert!(rates[0] > 0.0);
    }

    #[test]
    fn test_bootstrap_from_swaps_multiple() {
        let swaps = vec![
            SwapQuote::new(1.0, 0.03, 2),
            SwapQuote::new(2.0, 0.035, 2),
            SwapQuote::new(5.0, 0.04, 2),
        ];
        let (maturities, rates) = bootstrap_from_swaps(&swaps).unwrap();

        assert_eq!(maturities.len(), 3);
        assert_eq!(maturities[0], 1.0);
        assert_eq!(maturities[1], 2.0);
        assert_eq!(maturities[2], 5.0);
        // Rates should be positive and generally increasing
        assert!(rates[0] > 0.0);
        assert!(rates[1] > 0.0);
        assert!(rates[2] > 0.0);
    }

    #[test]
    fn test_bootstrap_from_swaps_empty() {
        let swaps: Vec<SwapQuote> = vec![];
        let result = bootstrap_from_swaps(&swaps);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_discount_factor() {
        let maturities = vec![1.0, 2.0, 3.0];
        let zero_rates = vec![0.03, 0.035, 0.04];

        // Test exact match
        let df1 = calculate_discount_factor(1.0, &maturities, &zero_rates);
        assert!((df1 - (-0.03_f64 * 1.0).exp()).abs() < 1e-10);

        // Test interpolation
        let df1_5 = calculate_discount_factor(1.5, &maturities, &zero_rates);
        let expected_rate: f64 = 0.03 + (0.035 - 0.03) * 0.5;
        assert!((df1_5 - (-expected_rate * 1.5).exp()).abs() < 1e-10);

        // Test extrapolation before first point
        let df0_5 = calculate_discount_factor(0.5, &maturities, &zero_rates);
        assert!((df0_5 - (-0.03_f64 * 0.5).exp()).abs() < 1e-10);

        // Test extrapolation after last point
        let df4 = calculate_discount_factor(4.0, &maturities, &zero_rates);
        assert!((df4 - (-0.04_f64 * 4.0).exp()).abs() < 1e-10);
    }
}
