// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Yield curve interpolation methods
//!
//! Provides various interpolation methods for yield curves:
//! - Linear interpolation
//! - Cubic spline (natural, clamped)
//! - Nelson-Siegel model
//! - Nelson-Siegel-Svensson extension

use crate::common::error::{DervflowError, Result};
use crate::common::types::{Rate, Time};

/// Interpolation method for yield curves
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Natural cubic spline (second derivative = 0 at endpoints)
    CubicSplineNatural,
    /// Clamped cubic spline (first derivative specified at endpoints)
    CubicSplineClamped,
    /// Nelson-Siegel parametric model
    NelsonSiegel,
    /// Nelson-Siegel-Svensson extension
    NelsonSiegelSvensson,
}

/// Linear interpolation of rates
///
/// # Arguments
/// * `t` - Time at which to interpolate
/// * `times` - Known time points (must be sorted)
/// * `rates` - Known rates at time points
///
/// # Returns
/// * `Result<Rate>` - Interpolated rate
pub fn linear_interpolate(t: Time, times: &[Time], rates: &[Rate]) -> Result<Rate> {
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

    // Find the interpolation interval
    if t <= times[0] {
        // Flat extrapolation before first point
        return Ok(rates[0]);
    }

    if t >= times[times.len() - 1] {
        // Flat extrapolation after last point
        return Ok(rates[rates.len() - 1]);
    }

    // Binary search for the interval
    let mut left = 0;
    let mut right = times.len() - 1;

    while right - left > 1 {
        let mid = (left + right) / 2;
        if times[mid] <= t {
            left = mid;
        } else {
            right = mid;
        }
    }

    // Linear interpolation
    let t0 = times[left];
    let t1 = times[right];
    let r0 = rates[left];
    let r1 = rates[right];

    let rate = r0 + (r1 - r0) * (t - t0) / (t1 - t0);
    Ok(rate)
}

/// Cubic spline interpolation
#[derive(Clone)]
pub struct CubicSpline {
    times: Vec<Time>,
    rates: Vec<Rate>,
    coefficients: Vec<SplineCoefficients>,
    #[allow(dead_code)]
    method: InterpolationMethod,
}

#[derive(Debug, Clone, Copy)]
struct SplineCoefficients {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl CubicSpline {
    /// Create a new cubic spline interpolator
    ///
    /// # Arguments
    /// * `times` - Known time points (must be sorted)
    /// * `rates` - Known rates at time points
    /// * `method` - Spline type (Natural or Clamped)
    pub fn new(times: Vec<Time>, rates: Vec<Rate>, method: InterpolationMethod) -> Result<Self> {
        if times.len() != rates.len() {
            return Err(DervflowError::InvalidInput(
                "Times and rates must have the same length".to_string(),
            ));
        }

        if times.len() < 2 {
            return Err(DervflowError::InvalidInput(
                "Need at least 2 points for cubic spline".to_string(),
            ));
        }

        // Verify times are sorted
        for i in 1..times.len() {
            if times[i] <= times[i - 1] {
                return Err(DervflowError::InvalidInput(
                    "Times must be strictly increasing".to_string(),
                ));
            }
        }

        let n = times.len();
        let mut h = vec![0.0; n - 1];
        let mut alpha = vec![0.0; n - 1];

        // Calculate h and alpha
        for i in 0..n - 1 {
            h[i] = times[i + 1] - times[i];
        }

        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (rates[i + 1] - rates[i])
                - (3.0 / h[i - 1]) * (rates[i] - rates[i - 1]);
        }

        // Solve tridiagonal system for second derivatives
        let mut l = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut c = vec![0.0; n];

        match method {
            InterpolationMethod::CubicSplineNatural => {
                // Natural spline: second derivative = 0 at endpoints
                l[0] = 1.0;
                mu[0] = 0.0;
                z[0] = 0.0;
            }
            InterpolationMethod::CubicSplineClamped => {
                // Clamped spline: use finite difference for first derivative at endpoints
                l[0] = 2.0 * h[0];
                mu[0] = 0.5;
                let deriv_0 = (rates[1] - rates[0]) / h[0];
                z[0] = (3.0 / h[0]) * (rates[1] - rates[0]) - 3.0 * deriv_0;
            }
            _ => {
                return Err(DervflowError::InvalidInput(
                    "Invalid spline method".to_string(),
                ));
            }
        }

        // Forward elimination
        for i in 1..n - 1 {
            l[i] = 2.0 * (times[i + 1] - times[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        match method {
            InterpolationMethod::CubicSplineNatural => {
                l[n - 1] = 1.0;
                z[n - 1] = 0.0;
                c[n - 1] = 0.0;
            }
            InterpolationMethod::CubicSplineClamped => {
                let deriv_n = (rates[n - 1] - rates[n - 2]) / h[n - 2];
                l[n - 1] = h[n - 2] * (2.0 - mu[n - 2]);
                z[n - 1] = (3.0 * deriv_n
                    - 3.0 * (rates[n - 1] - rates[n - 2]) / h[n - 2]
                    - h[n - 2] * z[n - 2])
                    / l[n - 1];
                c[n - 1] = z[n - 1];
            }
            _ => {}
        }

        // Back substitution
        for i in (0..n - 1).rev() {
            c[i] = z[i] - mu[i] * c[i + 1];
        }

        // Calculate spline coefficients
        let mut coefficients = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let b = (rates[i + 1] - rates[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            let d = (c[i + 1] - c[i]) / (3.0 * h[i]);

            coefficients.push(SplineCoefficients {
                a: rates[i],
                b,
                c: c[i],
                d,
            });
        }

        Ok(Self {
            times,
            rates,
            coefficients,
            method,
        })
    }

    /// Interpolate rate at time t
    pub fn interpolate(&self, t: Time) -> Result<Rate> {
        if t <= self.times[0] {
            return Ok(self.rates[0]);
        }

        if t >= self.times[self.times.len() - 1] {
            return Ok(self.rates[self.rates.len() - 1]);
        }

        // Find the interval
        let mut i = 0;
        while i < self.times.len() - 1 && self.times[i + 1] < t {
            i += 1;
        }

        // Evaluate cubic polynomial
        let dt = t - self.times[i];
        let coef = &self.coefficients[i];
        let rate = coef.a + coef.b * dt + coef.c * dt * dt + coef.d * dt * dt * dt;

        Ok(rate)
    }
}

/// Nelson-Siegel model parameters
#[derive(Debug, Clone, Copy)]
pub struct NelsonSiegelParams {
    /// Level parameter (long-term rate)
    pub beta0: f64,
    /// Slope parameter
    pub beta1: f64,
    /// Curvature parameter
    pub beta2: f64,
    /// Decay parameter (controls where curvature peaks)
    pub lambda: f64,
}

impl NelsonSiegelParams {
    /// Create new Nelson-Siegel parameters
    pub fn new(beta0: f64, beta1: f64, beta2: f64, lambda: f64) -> Self {
        Self {
            beta0,
            beta1,
            beta2,
            lambda,
        }
    }

    /// Calculate rate at time t using Nelson-Siegel formula
    pub fn rate(&self, t: Time) -> Rate {
        if t <= 0.0 {
            return self.beta0 + self.beta1;
        }

        let lambda_t = self.lambda * t;
        let exp_term = (-lambda_t).exp();
        let factor1 = (1.0 - exp_term) / lambda_t;
        let factor2 = factor1 - exp_term;

        self.beta0 + self.beta1 * factor1 + self.beta2 * factor2
    }
}

/// Nelson-Siegel-Svensson model parameters
#[derive(Debug, Clone, Copy)]
pub struct NelsonSiegelSvenssonParams {
    /// Level parameter
    pub beta0: f64,
    /// Slope parameter
    pub beta1: f64,
    /// Curvature parameter 1
    pub beta2: f64,
    /// Curvature parameter 2
    pub beta3: f64,
    /// Decay parameter 1
    pub lambda1: f64,
    /// Decay parameter 2
    pub lambda2: f64,
}

impl NelsonSiegelSvenssonParams {
    /// Create new Nelson-Siegel-Svensson parameters
    pub fn new(beta0: f64, beta1: f64, beta2: f64, beta3: f64, lambda1: f64, lambda2: f64) -> Self {
        Self {
            beta0,
            beta1,
            beta2,
            beta3,
            lambda1,
            lambda2,
        }
    }

    /// Calculate rate at time t using Nelson-Siegel-Svensson formula
    pub fn rate(&self, t: Time) -> Rate {
        if t <= 0.0 {
            return self.beta0 + self.beta1;
        }

        let lambda1_t = self.lambda1 * t;
        let lambda2_t = self.lambda2 * t;
        let exp1 = (-lambda1_t).exp();
        let exp2 = (-lambda2_t).exp();

        let factor1 = (1.0 - exp1) / lambda1_t;
        let factor2 = factor1 - exp1;
        let factor3 = (1.0 - exp2) / lambda2_t - exp2;

        self.beta0 + self.beta1 * factor1 + self.beta2 * factor2 + self.beta3 * factor3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolate() {
        let times = vec![1.0, 2.0, 3.0];
        let rates = vec![0.03, 0.04, 0.05];

        // Exact point
        let r = linear_interpolate(2.0, &times, &rates).unwrap();
        assert!((r - 0.04).abs() < 1e-10);

        // Interpolation
        let r = linear_interpolate(1.5, &times, &rates).unwrap();
        assert!((r - 0.035).abs() < 1e-10);

        // Extrapolation before
        let r = linear_interpolate(0.5, &times, &rates).unwrap();
        assert!((r - 0.03).abs() < 1e-10);

        // Extrapolation after
        let r = linear_interpolate(4.0, &times, &rates).unwrap();
        assert!((r - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolate_empty() {
        let times: Vec<Time> = vec![];
        let rates: Vec<Rate> = vec![];
        let result = linear_interpolate(1.0, &times, &rates);
        assert!(result.is_err());
    }

    #[test]
    fn test_cubic_spline_natural() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let rates = vec![0.03, 0.04, 0.045, 0.05];

        let spline = CubicSpline::new(
            times.clone(),
            rates.clone(),
            InterpolationMethod::CubicSplineNatural,
        )
        .unwrap();

        // Test exact points
        for i in 0..times.len() {
            let r = spline.interpolate(times[i]).unwrap();
            assert!((r - rates[i]).abs() < 1e-6);
        }

        // Test interpolation
        let r = spline.interpolate(2.5).unwrap();
        assert!(r > 0.04 && r < 0.045);
    }

    #[test]
    fn test_cubic_spline_clamped() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let rates = vec![0.03, 0.04, 0.045, 0.05];

        let spline = CubicSpline::new(
            times.clone(),
            rates.clone(),
            InterpolationMethod::CubicSplineClamped,
        )
        .unwrap();

        // Test exact points
        for i in 0..times.len() {
            let r = spline.interpolate(times[i]).unwrap();
            assert!((r - rates[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cubic_spline_insufficient_points() {
        let times = vec![1.0];
        let rates = vec![0.03];
        let result = CubicSpline::new(times, rates, InterpolationMethod::CubicSplineNatural);
        assert!(result.is_err());
    }

    #[test]
    fn test_cubic_spline_unsorted() {
        let times = vec![1.0, 3.0, 2.0];
        let rates = vec![0.03, 0.05, 0.04];
        let result = CubicSpline::new(times, rates, InterpolationMethod::CubicSplineNatural);
        assert!(result.is_err());
    }

    #[test]
    fn test_nelson_siegel() {
        let params = NelsonSiegelParams::new(0.05, -0.02, 0.01, 1.0);

        // Test at various maturities
        let r1 = params.rate(1.0);
        let r5 = params.rate(5.0);
        let r10 = params.rate(10.0);

        // Rates should be positive
        assert!(r1 > 0.0);
        assert!(r5 > 0.0);
        assert!(r10 > 0.0);

        // Long-term rate should approach beta0
        let r_long = params.rate(100.0);
        assert!((r_long - params.beta0).abs() < 0.01);
    }

    #[test]
    fn test_nelson_siegel_zero_time() {
        let params = NelsonSiegelParams::new(0.05, -0.02, 0.01, 1.0);
        let r = params.rate(0.0);
        assert!((r - (params.beta0 + params.beta1)).abs() < 1e-10);
    }

    #[test]
    fn test_nelson_siegel_svensson() {
        let params = NelsonSiegelSvenssonParams::new(0.05, -0.02, 0.01, 0.005, 1.0, 3.0);

        // Test at various maturities
        let r1 = params.rate(1.0);
        let r5 = params.rate(5.0);
        let r10 = params.rate(10.0);

        // Rates should be positive
        assert!(r1 > 0.0);
        assert!(r5 > 0.0);
        assert!(r10 > 0.0);

        // Long-term rate should approach beta0
        let r_long = params.rate(100.0);
        assert!((r_long - params.beta0).abs() < 0.01);
    }

    #[test]
    fn test_nelson_siegel_svensson_zero_time() {
        let params = NelsonSiegelSvenssonParams::new(0.05, -0.02, 0.01, 0.005, 1.0, 3.0);
        let r = params.rate(0.0);
        assert!((r - (params.beta0 + params.beta1)).abs() < 1e-10);
    }
}
