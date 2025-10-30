// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Root finding algorithms
//!
//! Provides robust root finding methods including Newton-Raphson, Brent's method,
//! and bisection. All methods include convergence diagnostics.

use crate::common::error::{DervflowError, Result};

/// Configuration for root finding algorithms
#[derive(Debug, Clone)]
pub struct RootFindingConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Relative tolerance (for Brent's method)
    pub relative_tolerance: f64,
}

impl Default for RootFindingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            relative_tolerance: 1e-8,
        }
    }
}

/// Result of a root finding operation with diagnostics
#[derive(Debug, Clone)]
pub struct RootFindingResult {
    /// The root found
    pub root: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final error estimate
    pub error: f64,
    /// Whether the method converged
    pub converged: bool,
}

/// Newton-Raphson method for finding roots
///
/// Requires both the function and its derivative.
/// Converges quadratically when close to the root.
///
/// # Arguments
/// * `f` - The function to find the root of
/// * `df` - The derivative of the function
/// * `initial_guess` - Starting point for the iteration
/// * `config` - Configuration parameters
///
/// # Returns
/// Result containing the root and convergence diagnostics
pub fn newton_raphson<F, DF>(
    f: F,
    df: DF,
    initial_guess: f64,
    config: &RootFindingConfig,
) -> Result<RootFindingResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = initial_guess;
    let mut iterations = 0;
    let mut error = f64::INFINITY;

    for i in 0..config.max_iterations {
        iterations = i + 1;

        let fx = f(x);
        let dfx = df(x);

        // Check for zero derivative
        if dfx.abs() < 1e-15 {
            return Err(DervflowError::NumericalError(
                "Derivative is zero, cannot continue Newton-Raphson".to_string(),
            ));
        }

        // Newton-Raphson update
        let x_new = x - fx / dfx;

        // Check for NaN or infinity
        if !x_new.is_finite() {
            return Err(DervflowError::NumericalError(
                "Newton-Raphson produced non-finite value".to_string(),
            ));
        }

        error = (x_new - x).abs();
        x = x_new;

        // Check convergence
        if error < config.tolerance {
            return Ok(RootFindingResult {
                root: x,
                iterations,
                error,
                converged: true,
            });
        }
    }

    Err(DervflowError::ConvergenceFailure { iterations, error })
}

/// Brent's method for finding roots
///
/// Combines bisection, secant, and inverse quadratic interpolation.
/// Very robust and guaranteed to converge if the root is bracketed.
///
/// # Arguments
/// * `f` - The function to find the root of
/// * `a` - Lower bound of the bracket
/// * `b` - Upper bound of the bracket
/// * `config` - Configuration parameters
///
/// # Returns
/// Result containing the root and convergence diagnostics
pub fn brent<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    config: &RootFindingConfig,
) -> Result<RootFindingResult>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    // Check that root is bracketed
    if fa * fb > 0.0 {
        return Err(DervflowError::InvalidInput(
            "Root is not bracketed: f(a) and f(b) must have opposite signs".to_string(),
        ));
    }

    // Ensure |f(a)| >= |f(b)| (swap if needed so b is the better approximation)
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0;
    let mut iterations = 0;

    for i in 0..config.max_iterations {
        iterations = i + 1;

        // Check convergence on function value
        if fb.abs() < config.tolerance {
            return Ok(RootFindingResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        // Check convergence on interval size
        if (b - a).abs() < config.tolerance {
            return Ok(RootFindingResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        let mut s;

        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Check if we should use bisection instead
        let use_bisection =
            // s is not between (3a+b)/4 and b
            !(s > (3.0 * a + b) / 4.0 && s < b || s < (3.0 * a + b) / 4.0 && s > b) ||
            // mflag is set and |s-b| >= |b-c|/2
            (mflag && (s - b).abs() >= (b - c).abs() / 2.0) ||
            // mflag is clear and |s-b| >= |c-d|/2
            (!mflag && (s - b).abs() >= (c - d).abs() / 2.0) ||
            // mflag is set and |b-c| < tolerance
            (mflag && (b - c).abs() < config.tolerance) ||
            // mflag is clear and |c-d| < tolerance
            (!mflag && (c - d).abs() < config.tolerance);

        if use_bisection {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);

        // Update d before c is updated
        d = c;
        c = b;
        fc = fb;

        // Update the bracket
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    Err(DervflowError::ConvergenceFailure {
        iterations,
        error: fb.abs(),
    })
}

/// Bisection method for finding roots
///
/// Simple and robust method that always converges if the root is bracketed.
/// Converges linearly (slower than Newton-Raphson or Brent).
///
/// # Arguments
/// * `f` - The function to find the root of
/// * `a` - Lower bound of the bracket
/// * `b` - Upper bound of the bracket
/// * `config` - Configuration parameters
///
/// # Returns
/// Result containing the root and convergence diagnostics
pub fn bisection<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    config: &RootFindingConfig,
) -> Result<RootFindingResult>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    // Check that root is bracketed
    if fa * fb > 0.0 {
        return Err(DervflowError::InvalidInput(
            "Root is not bracketed: f(a) and f(b) must have opposite signs".to_string(),
        ));
    }

    let mut iterations = 0;
    let mut error = (b - a).abs();

    for i in 0..config.max_iterations {
        iterations = i + 1;

        // Compute midpoint
        let c = 0.5 * (a + b);
        let fc = f(c);

        error = 0.5 * (b - a).abs();

        // Check convergence
        if error < config.tolerance || fc.abs() < config.tolerance {
            return Ok(RootFindingResult {
                root: c,
                iterations,
                error,
                converged: true,
            });
        }

        // Update bracket
        if fa * fc < 0.0 {
            b = c;
            #[allow(unused_assignments)]
            {
                fb = fc;
            }
        } else {
            a = c;
            fa = fc;
        }
    }

    Err(DervflowError::ConvergenceFailure { iterations, error })
}

/// Secant method for finding roots
///
/// Similar to Newton-Raphson but uses finite differences to approximate the derivative.
/// Doesn't require explicit derivative but converges slightly slower.
///
/// # Arguments
/// * `f` - The function to find the root of
/// * `x0` - First initial guess
/// * `x1` - Second initial guess
/// * `config` - Configuration parameters
///
/// # Returns
/// Result containing the root and convergence diagnostics
pub fn secant<F>(
    f: F,
    mut x0: f64,
    mut x1: f64,
    config: &RootFindingConfig,
) -> Result<RootFindingResult>
where
    F: Fn(f64) -> f64,
{
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    let mut iterations = 0;
    let mut error = f64::INFINITY;

    for i in 0..config.max_iterations {
        iterations = i + 1;

        // Check for zero denominator
        if (f1 - f0).abs() < 1e-15 {
            return Err(DervflowError::NumericalError(
                "Secant method: function values too close".to_string(),
            ));
        }

        // Secant update
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);

        // Check for NaN or infinity
        if !x2.is_finite() {
            return Err(DervflowError::NumericalError(
                "Secant method produced non-finite value".to_string(),
            ));
        }

        error = (x2 - x1).abs();

        // Check convergence
        if error < config.tolerance {
            return Ok(RootFindingResult {
                root: x2,
                iterations,
                error,
                converged: true,
            });
        }

        // Update for next iteration
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x2);
    }

    Err(DervflowError::ConvergenceFailure { iterations, error })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_newton_raphson_simple() {
        // Find root of f(x) = x^2 - 4 (root at x = 2)
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;

        let config = RootFindingConfig::default();
        let result = newton_raphson(f, df, 1.0, &config).unwrap();

        assert_relative_eq!(result.root, 2.0, epsilon = 1e-6);
        assert!(result.converged);
        assert!(result.iterations < 10);
    }

    #[test]
    fn test_newton_raphson_cubic() {
        // Find root of f(x) = x^3 - x - 2 (root at x ≈ 1.5214)
        let f = |x: f64| x.powi(3) - x - 2.0;
        let df = |x: f64| 3.0 * x.powi(2) - 1.0;

        let config = RootFindingConfig::default();
        let result = newton_raphson(f, df, 2.0, &config).unwrap();

        assert_relative_eq!(f(result.root), 0.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_brent_simple() {
        // Find root of f(x) = x^2 - 4 (root at x = 2)
        let f = |x: f64| x * x - 4.0;

        let config = RootFindingConfig::default();
        let result = brent(f, 0.0, 3.0, &config).unwrap();

        assert_relative_eq!(result.root, 2.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_brent_transcendental() {
        // Find root of f(x) = cos(x) - x (root at x ≈ 0.7391)
        let f = |x: f64| x.cos() - x;

        let config = RootFindingConfig::default();
        let result = brent(f, 0.0, 1.0, &config).unwrap();

        assert_relative_eq!(f(result.root), 0.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_brent_not_bracketed() {
        let f = |x: f64| x * x - 4.0;
        let config = RootFindingConfig::default();

        let result = brent(f, 3.0, 5.0, &config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DervflowError::InvalidInput(_)));
    }

    #[test]
    fn test_bisection_simple() {
        // Find root of f(x) = x^2 - 4 (root at x = 2)
        let f = |x: f64| x * x - 4.0;

        let config = RootFindingConfig::default();
        let result = bisection(f, 0.0, 3.0, &config).unwrap();

        assert_relative_eq!(result.root, 2.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_bisection_not_bracketed() {
        let f = |x: f64| x * x - 4.0;
        let config = RootFindingConfig::default();

        let result = bisection(f, 3.0, 5.0, &config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DervflowError::InvalidInput(_)));
    }

    #[test]
    fn test_secant_simple() {
        // Find root of f(x) = x^2 - 4 (root at x = 2)
        let f = |x: f64| x * x - 4.0;

        let config = RootFindingConfig::default();
        let result = secant(f, 1.0, 3.0, &config).unwrap();

        assert_relative_eq!(result.root, 2.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_convergence_diagnostics() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;

        let config = RootFindingConfig::default();
        let result = newton_raphson(f, df, 1.0, &config).unwrap();

        assert!(result.iterations > 0);
        assert!(result.error < config.tolerance);
        assert!(result.converged);
    }

    #[test]
    fn test_custom_tolerance() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;

        let config = RootFindingConfig {
            max_iterations: 100,
            tolerance: 1e-12,
            relative_tolerance: 1e-12,
        };

        let result = newton_raphson(f, df, 1.0, &config).unwrap();
        assert!(result.error < 1e-12);
    }
}
