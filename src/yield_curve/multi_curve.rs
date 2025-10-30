// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Multi-curve framework for interest rate modelling
//!
//! Supports separate discounting and forwarding curves which is essential in
//! modern interest-rate modelling (e.g. OIS discounting with LIBOR forward
//! curves). The module provides utilities to register multiple forward curves
//! and compute forward or swap rates using the appropriate curve combinations.

use super::YieldCurve;
use crate::common::error::{DervflowError, Result};
use crate::common::types::Time;
use std::collections::HashMap;

#[cfg(test)]
use super::InterpolationMethod;

/// Represents a swap period with start/end times and day-count factor
#[derive(Debug, Clone, Copy)]
pub struct SwapPeriod {
    /// Period start (in years)
    pub start: Time,
    /// Period end (in years)
    pub end: Time,
    /// Year fraction for the period
    pub year_fraction: f64,
}

impl SwapPeriod {
    /// Create a validated swap period instance
    pub fn new(start: Time, end: Time, year_fraction: f64) -> Result<Self> {
        let period = Self {
            start,
            end,
            year_fraction,
        };
        period.validate()?;
        Ok(period)
    }

    /// Validate the swap period parameters
    fn validate(&self) -> Result<()> {
        if self.start < 0.0 {
            return Err(DervflowError::InvalidInput(
                "Swap period start must be non-negative".to_string(),
            ));
        }
        if self.end <= self.start {
            return Err(DervflowError::InvalidInput(
                "Swap period end must be greater than start".to_string(),
            ));
        }
        if self.year_fraction <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Swap period year fraction must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Multi-curve container comprising a discount curve and named forward curves
pub struct MultiCurve {
    discount_curve: YieldCurve,
    forward_curves: HashMap<String, YieldCurve>,
}

impl MultiCurve {
    /// Create a new multi-curve with the specified discount curve
    pub fn new(discount_curve: YieldCurve) -> Self {
        Self {
            discount_curve,
            forward_curves: HashMap::new(),
        }
    }

    /// Register or replace a forward curve with the given name
    pub fn set_forward_curve<S: Into<String>>(&mut self, name: S, curve: YieldCurve) {
        self.forward_curves.insert(name.into(), curve);
    }

    /// Add a forward curve ensuring that the name does not already exist
    pub fn add_forward_curve<S: Into<String>>(&mut self, name: S, curve: YieldCurve) -> Result<()> {
        let key = name.into();
        if self.forward_curves.contains_key(&key) {
            return Err(DervflowError::InvalidInput(format!(
                "Forward curve '{}' already exists",
                key
            )));
        }
        self.forward_curves.insert(key, curve);
        Ok(())
    }

    /// Retrieve a forward curve by name
    pub fn forward_curve(&self, name: &str) -> Result<&YieldCurve> {
        self.forward_curves
            .get(name)
            .ok_or_else(|| DervflowError::DataError(format!("Forward curve '{}' not found", name)))
    }

    /// Return a reference to the discount curve
    pub fn discount_curve(&self) -> &YieldCurve {
        &self.discount_curve
    }

    /// Discount factor from the discount curve
    pub fn discount_factor(&self, t: Time) -> Result<f64> {
        self.discount_curve.discount_factor(t)
    }

    /// Forward rate using a specified forward curve between `start` and `end`
    pub fn forward_rate(&self, curve: &str, start: Time, end: Time) -> Result<f64> {
        let fwd_curve = self.forward_curve(curve)?;
        fwd_curve.forward_rate(start, end)
    }

    /// Present value of cashflows discounted using the discount curve
    pub fn present_value(&self, cashflows: &[(Time, f64)]) -> Result<f64> {
        self.discount_curve.price_bond(cashflows)
    }

    /// Compute the par swap rate for a given forward curve and schedule
    pub fn par_swap_rate(&self, curve: &str, periods: &[SwapPeriod]) -> Result<f64> {
        if periods.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Swap schedule cannot be empty".to_string(),
            ));
        }

        for period in periods {
            period.validate()?;
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for period in periods {
            let forward = self.forward_rate(curve, period.start, period.end)?;
            let df = self.discount_factor(period.end)?;
            numerator += forward * period.year_fraction * df;
            denominator += period.year_fraction * df;
        }

        if denominator.abs() < 1e-12 {
            return Err(DervflowError::NumericalError(
                "Swap denominator is numerically zero".to_string(),
            ));
        }

        Ok(numerator / denominator)
    }

    /// Price a fixed-for-floating interest rate swap
    ///
    /// Returns the present value of a payer swap (pay fixed, receive floating).
    pub fn price_payer_swap(
        &self,
        curve: &str,
        periods: &[SwapPeriod],
        fixed_rate: f64,
        notional: f64,
    ) -> Result<f64> {
        if periods.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Swap schedule cannot be empty".to_string(),
            ));
        }
        if notional <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Notional must be positive".to_string(),
            ));
        }

        let mut pv_floating = 0.0;
        let mut pv_fixed = 0.0;

        for period in periods {
            period.validate()?;
            let forward = self.forward_rate(curve, period.start, period.end)?;
            let df = self.discount_factor(period.end)?;
            pv_floating += forward * period.year_fraction * df * notional;
            pv_fixed += fixed_rate * period.year_fraction * df * notional;
        }

        // Include notional exchange at maturity if modelling a standard swap
        let final_df = self.discount_factor(periods.last().unwrap().end)?;
        pv_floating += final_df * notional;
        pv_fixed += final_df * notional;

        Ok(pv_floating - pv_fixed)
    }

    /// List all registered forward curve names
    pub fn forward_curve_names(&self) -> Vec<String> {
        self.forward_curves.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_curve(rate: f64) -> YieldCurve {
        let times = vec![0.5, 1.0, 5.0];
        let rates = vec![rate; times.len()];
        YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap()
    }

    fn build_multicurve() -> MultiCurve {
        let discount = flat_curve(0.02);
        let mut mc = MultiCurve::new(discount);
        mc.add_forward_curve("LIBOR3M", flat_curve(0.03)).unwrap();
        mc
    }

    #[test]
    fn test_forward_curve_registration() {
        let mut mc = build_multicurve();
        assert!(mc.add_forward_curve("LIBOR3M", flat_curve(0.025)).is_err());
        mc.set_forward_curve("LIBOR6M", flat_curve(0.035));
        assert!(mc.forward_curve("LIBOR6M").is_ok());
    }

    #[test]
    fn test_discount_factor_delegation() {
        let mc = build_multicurve();
        let df = mc.discount_factor(1.0).unwrap();
        assert!((df - (-0.02_f64 * 1.0).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_forward_rate() {
        let mc = build_multicurve();
        let rate = mc.forward_rate("LIBOR3M", 0.0, 0.5).unwrap();
        assert!((rate - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_par_swap_rate() {
        let mc = build_multicurve();
        let periods = vec![
            SwapPeriod {
                start: 0.0,
                end: 0.5,
                year_fraction: 0.5,
            },
            SwapPeriod {
                start: 0.5,
                end: 1.0,
                year_fraction: 0.5,
            },
        ];

        let par_rate = mc.par_swap_rate("LIBOR3M", &periods).unwrap();
        assert!((par_rate - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_price_payer_swap_at_par() {
        let mc = build_multicurve();
        let periods = vec![
            SwapPeriod {
                start: 0.0,
                end: 0.5,
                year_fraction: 0.5,
            },
            SwapPeriod {
                start: 0.5,
                end: 1.0,
                year_fraction: 0.5,
            },
        ];

        let par_rate = mc.par_swap_rate("LIBOR3M", &periods).unwrap();
        let pv = mc
            .price_payer_swap("LIBOR3M", &periods, par_rate, 1_000_000.0)
            .unwrap();
        assert!(pv.abs() < 1e-2);
    }
}
