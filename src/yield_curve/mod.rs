// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Yield curve construction and analysis
//!
//! Provides tools for fixed income analysis:
//! - Yield curve bootstrapping
//! - Interpolation methods (linear, cubic spline, Nelson-Siegel)
//! - Multi-curve framework
//! - Bond analytics (duration, convexity, YTM)

pub mod bond_analytics;
pub mod bootstrap;
pub mod interpolation;
pub mod multi_curve;
use crate::common::error::{DervflowError, Result};
use crate::common::types::{Rate, Time};
use interpolation::{
    CubicSpline, InterpolationMethod, NelsonSiegelParams, NelsonSiegelSvenssonParams,
};

/// Yield curve representation with interpolation
#[derive(Clone)]
pub struct YieldCurve {
    times: Vec<Time>,
    rates: Vec<Rate>,
    interpolation_method: InterpolationMethod,
    spline: Option<CubicSpline>,
    nelson_siegel: Option<NelsonSiegelParams>,
    nelson_siegel_svensson: Option<NelsonSiegelSvenssonParams>,
}

impl YieldCurve {
    /// Create a new yield curve with specified interpolation method
    ///
    /// # Arguments
    /// * `times` - Time points (maturities in years)
    /// * `rates` - Zero rates at each time point
    /// * `method` - Interpolation method to use
    ///
    /// # Example
    /// ```
    /// use dervflow::yield_curve::{YieldCurve, interpolation::InterpolationMethod};
    ///
    /// let times = vec![1.0, 2.0, 5.0, 10.0];
    /// let rates = vec![0.03, 0.035, 0.04, 0.045];
    /// let curve = YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap();
    /// ```
    pub fn new(times: Vec<Time>, rates: Vec<Rate>, method: InterpolationMethod) -> Result<Self> {
        if times.is_empty() || rates.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Times and rates cannot be empty".to_string(),
            ));
        }

        if times.len() != rates.len() {
            return Err(DervflowError::InvalidInput(
                "Times and rates must have the same length".to_string(),
            ));
        }

        // Validate times are positive and sorted
        for i in 0..times.len() {
            if times[i] <= 0.0 {
                return Err(DervflowError::InvalidInput(format!(
                    "All times must be positive, got {} at index {}",
                    times[i], i
                )));
            }
            if i > 0 && times[i] <= times[i - 1] {
                return Err(DervflowError::InvalidInput(
                    "Times must be strictly increasing".to_string(),
                ));
            }
        }

        let mut curve = Self {
            times: times.clone(),
            rates: rates.clone(),
            interpolation_method: method,
            spline: None,
            nelson_siegel: None,
            nelson_siegel_svensson: None,
        };

        // Pre-compute interpolation structures
        match method {
            InterpolationMethod::CubicSplineNatural | InterpolationMethod::CubicSplineClamped => {
                curve.spline = Some(CubicSpline::new(times, rates, method)?);
            }
            _ => {}
        }

        Ok(curve)
    }

    /// Create a yield curve from Nelson-Siegel parameters
    pub fn from_nelson_siegel(params: NelsonSiegelParams, times: Vec<Time>) -> Result<Self> {
        if times.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Times cannot be empty".to_string(),
            ));
        }

        let rates: Vec<Rate> = times.iter().map(|&t| params.rate(t)).collect();

        Ok(Self {
            times,
            rates,
            interpolation_method: InterpolationMethod::NelsonSiegel,
            spline: None,
            nelson_siegel: Some(params),
            nelson_siegel_svensson: None,
        })
    }

    /// Create a yield curve from Nelson-Siegel-Svensson parameters
    pub fn from_nelson_siegel_svensson(
        params: NelsonSiegelSvenssonParams,
        times: Vec<Time>,
    ) -> Result<Self> {
        if times.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Times cannot be empty".to_string(),
            ));
        }

        let rates: Vec<Rate> = times.iter().map(|&t| params.rate(t)).collect();

        Ok(Self {
            times,
            rates,
            interpolation_method: InterpolationMethod::NelsonSiegelSvensson,
            spline: None,
            nelson_siegel: None,
            nelson_siegel_svensson: Some(params),
        })
    }

    /// Get zero rate at time t
    ///
    /// # Arguments
    /// * `t` - Time (maturity in years)
    ///
    /// # Returns
    /// * `Result<Rate>` - Zero rate at time t
    pub fn zero_rate(&self, t: Time) -> Result<Rate> {
        if t <= 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Time must be positive, got {}",
                t
            )));
        }

        match self.interpolation_method {
            InterpolationMethod::Linear => {
                interpolation::linear_interpolate(t, &self.times, &self.rates)
            }
            InterpolationMethod::CubicSplineNatural | InterpolationMethod::CubicSplineClamped => {
                self.spline
                    .as_ref()
                    .ok_or_else(|| {
                        DervflowError::NumericalError("Spline not initialized".to_string())
                    })?
                    .interpolate(t)
            }
            InterpolationMethod::NelsonSiegel => {
                if let Some(params) = &self.nelson_siegel {
                    Ok(params.rate(t))
                } else {
                    interpolation::linear_interpolate(t, &self.times, &self.rates)
                }
            }
            InterpolationMethod::NelsonSiegelSvensson => {
                if let Some(params) = &self.nelson_siegel_svensson {
                    Ok(params.rate(t))
                } else {
                    interpolation::linear_interpolate(t, &self.times, &self.rates)
                }
            }
        }
    }

    /// Calculate forward rate between two times
    ///
    /// Forward rate f(t1, t2) is the rate for borrowing from t1 to t2
    ///
    /// # Arguments
    /// * `t1` - Start time
    /// * `t2` - End time
    ///
    /// # Returns
    /// * `Result<Rate>` - Forward rate
    pub fn forward_rate(&self, t1: Time, t2: Time) -> Result<Rate> {
        if t1 < 0.0 || t2 <= t1 {
            return Err(DervflowError::InvalidInput(format!(
                "Invalid time range: t1={}, t2={} (need 0 <= t1 < t2)",
                t1, t2
            )));
        }

        if t1 == 0.0 {
            // Forward rate from now to t2 is just the zero rate at t2
            return self.zero_rate(t2);
        }

        // Forward rate formula: f(t1,t2) = (r2*t2 - r1*t1) / (t2 - t1)
        let r1 = self.zero_rate(t1)?;
        let r2 = self.zero_rate(t2)?;

        let forward = (r2 * t2 - r1 * t1) / (t2 - t1);
        Ok(forward)
    }

    /// Calculate discount factor at time t
    ///
    /// Discount factor DF(t) = exp(-r(t) * t)
    ///
    /// # Arguments
    /// * `t` - Time (maturity in years)
    ///
    /// # Returns
    /// * `Result<f64>` - Discount factor
    pub fn discount_factor(&self, t: Time) -> Result<f64> {
        if t < 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Time must be non-negative, got {}",
                t
            )));
        }

        if t == 0.0 {
            return Ok(1.0);
        }

        let rate = self.zero_rate(t)?;
        Ok((-rate * t).exp())
    }

    /// Price a bond using the yield curve
    ///
    /// # Arguments
    /// * `cashflows` - Vector of (time, amount) tuples representing bond cashflows
    ///
    /// # Returns
    /// * `Result<f64>` - Present value of the bond
    pub fn price_bond(&self, cashflows: &[(Time, f64)]) -> Result<f64> {
        if cashflows.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Cashflows cannot be empty".to_string(),
            ));
        }

        let mut pv = 0.0;
        for &(time, amount) in cashflows {
            if time < 0.0 {
                return Err(DervflowError::InvalidInput(format!(
                    "Cashflow time must be non-negative, got {}",
                    time
                )));
            }
            if time > 0.0 {
                let df = self.discount_factor(time)?;
                pv += amount * df;
            } else {
                // Immediate cashflow
                pv += amount;
            }
        }

        Ok(pv)
    }

    /// Get the underlying time points
    pub fn times(&self) -> &[Time] {
        &self.times
    }

    /// Get the underlying rates
    pub fn rates(&self) -> &[Rate] {
        &self.rates
    }

    /// Get the interpolation method
    pub fn interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yield_curve_creation() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap();

        assert_eq!(curve.times().len(), 4);
        assert_eq!(curve.rates().len(), 4);
    }

    #[test]
    fn test_yield_curve_zero_rate() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap();

        // Exact point
        let r = curve.zero_rate(2.0).unwrap();
        assert!((r - 0.035).abs() < 1e-10);

        // Interpolated point
        let r = curve.zero_rate(3.0).unwrap();
        assert!(r > 0.035 && r < 0.04);
    }

    #[test]
    fn test_yield_curve_forward_rate() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap();

        // Forward rate from year 1 to year 2
        let f = curve.forward_rate(1.0, 2.0).unwrap();
        // f(1,2) = (r2*2 - r1*1) / (2-1) = (0.035*2 - 0.03*1) / 1 = 0.04
        assert!((f - 0.04).abs() < 1e-10);

        // Forward rate from now to year 2
        let f = curve.forward_rate(0.0, 2.0).unwrap();
        assert!((f - 0.035).abs() < 1e-10);
    }

    #[test]
    fn test_yield_curve_discount_factor() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap();

        // DF at t=0 should be 1
        let df = curve.discount_factor(0.0).unwrap();
        assert!((df - 1.0).abs() < 1e-10);

        // DF at t=1 with r=0.03
        let df = curve.discount_factor(1.0).unwrap();
        let expected = (-0.03_f64 * 1.0).exp();
        assert!((df - expected).abs() < 1e-10);
    }

    #[test]
    fn test_yield_curve_price_bond() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = YieldCurve::new(times, rates, InterpolationMethod::Linear).unwrap();

        // Simple bond with annual coupons
        let cashflows = vec![
            (1.0, 5.0),   // Coupon at year 1
            (2.0, 5.0),   // Coupon at year 2
            (2.0, 100.0), // Principal at year 2
        ];

        let price = curve.price_bond(&cashflows).unwrap();
        assert!(price > 0.0);
        assert!(price < 110.0); // Should be less than sum of cashflows
    }

    #[test]
    fn test_yield_curve_cubic_spline() {
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let rates = vec![0.03, 0.035, 0.04, 0.045];
        let curve = YieldCurve::new(times, rates, InterpolationMethod::CubicSplineNatural).unwrap();

        // Test interpolation
        let r = curve.zero_rate(3.0).unwrap();
        assert!(r > 0.035 && r < 0.045);
    }

    #[test]
    fn test_yield_curve_nelson_siegel() {
        let params = NelsonSiegelParams::new(0.05, -0.02, 0.01, 1.0);
        let times = vec![1.0, 2.0, 5.0, 10.0];
        let curve = YieldCurve::from_nelson_siegel(params, times).unwrap();

        let r = curve.zero_rate(3.0).unwrap();
        assert!(r > 0.0);
    }

    #[test]
    fn test_yield_curve_invalid_times() {
        let times = vec![1.0, 0.5, 2.0]; // Not sorted
        let rates = vec![0.03, 0.035, 0.04];
        let result = YieldCurve::new(times, rates, InterpolationMethod::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_yield_curve_empty() {
        let times: Vec<Time> = vec![];
        let rates: Vec<Rate> = vec![];
        let result = YieldCurve::new(times, rates, InterpolationMethod::Linear);
        assert!(result.is_err());
    }
}
