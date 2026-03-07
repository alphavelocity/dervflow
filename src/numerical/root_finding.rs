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

#[inline]
fn scaled_tolerance(estimate: f64, config: &RootFindingConfig) -> f64 {
    config
        .tolerance
        .max(config.relative_tolerance * estimate.abs().max(1.0))
}

#[inline]
fn has_converged(step_error: f64, estimate: f64, config: &RootFindingConfig) -> bool {
    step_error <= scaled_tolerance(estimate, config)
}

#[inline]
fn residual_converged(residual: f64, estimate: f64, config: &RootFindingConfig) -> bool {
    residual.abs() <= scaled_tolerance(estimate, config)
}

#[inline]
fn derivative_too_small(derivative: f64, estimate: f64) -> bool {
    let scale = estimate.abs().max(1.0);
    derivative.abs() <= f64::EPSILON * scale
}

#[inline]
fn secant_denominator_too_small(f0: f64, f1: f64) -> bool {
    let scale = f0.abs().max(f1.abs()).max(1.0);
    (f1 - f0).abs() <= f64::EPSILON * scale
}

#[inline]
fn values_distinct(a: f64, b: f64, c: f64) -> bool {
    !secant_denominator_too_small(a, b)
        && !secant_denominator_too_small(a, c)
        && !secant_denominator_too_small(b, c)
}

#[inline]
fn opposite_signs(a: f64, b: f64) -> bool {
    (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0)
}

#[inline]
fn stable_midpoint(a: f64, b: f64) -> f64 {
    // For opposite-sign inputs, (a + b) is safe from overflow and usually the most stable choice.
    // For same-sign inputs, use a + (b - a)/2 to avoid overflow in a + b.
    if a.is_sign_positive() != b.is_sign_positive() {
        0.5 * (a + b)
    } else {
        a + 0.5 * (b - a)
    }
}

#[inline]
fn three_quarters_toward_b(a: f64, b: f64) -> f64 {
    // (3a + b)/4 computed via nested stable midpoints for better overflow behavior.
    stable_midpoint(a, stable_midpoint(a, b))
}

#[inline]
fn finite_difference_derivative<F>(f: &F, x: f64) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let h = f64::EPSILON.sqrt() * x.abs().max(1.0);
    let x_plus = ensure_finite(x + h, "Finite-difference stencil point")?;
    let x_minus = ensure_finite(x - h, "Finite-difference stencil point")?;

    let f_plus = ensure_finite(f(x_plus), "Finite-difference upper function evaluation")?;
    let f_minus = ensure_finite(f(x_minus), "Finite-difference lower function evaluation")?;

    Ok((f_plus - f_minus) / (2.0 * h))
}

#[inline]
fn validate_config(config: &RootFindingConfig) -> Result<()> {
    if config.max_iterations == 0 {
        return Err(DervflowError::InvalidInput(
            "max_iterations must be greater than 0".to_string(),
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "tolerance must be positive and finite".to_string(),
        ));
    }
    if !config.relative_tolerance.is_finite() || config.relative_tolerance < 0.0 {
        return Err(DervflowError::InvalidInput(
            "relative_tolerance must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn ensure_finite(value: f64, context: &str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DervflowError::NumericalError(format!(
            "{context} produced non-finite value"
        )))
    }
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
    validate_config(config)?;
    if !initial_guess.is_finite() {
        return Err(DervflowError::InvalidInput(
            "initial_guess must be finite".to_string(),
        ));
    }

    let mut x = initial_guess;
    let mut fx = ensure_finite(f(x), "Function evaluation")?;
    if residual_converged(fx, x, config) {
        return Ok(RootFindingResult {
            root: x,
            iterations: 0,
            error: fx.abs(),
            converged: true,
        });
    }

    let mut iterations = 0;
    let mut error = f64::INFINITY;

    for i in 0..config.max_iterations {
        iterations = i + 1;

        let mut dfx = ensure_finite(df(x), "Derivative evaluation")?;

        // Safeguard against tiny analytical derivatives by using finite differences.
        if derivative_too_small(dfx, x) {
            dfx = finite_difference_derivative(&f, x)?;
            if derivative_too_small(dfx, x) {
                return Err(DervflowError::NumericalError(
                    "Derivative is too small, cannot continue Newton-Raphson".to_string(),
                ));
            }
        }

        // Newton-Raphson update
        let x_new = x - fx / dfx;

        // Check for NaN or infinity
        if !x_new.is_finite() {
            return Err(DervflowError::NumericalError(
                "Newton-Raphson produced non-finite value".to_string(),
            ));
        }

        let fx_new = ensure_finite(f(x_new), "Function evaluation after Newton update")?;
        x = x_new;
        fx = fx_new;

        error = fx.abs();
        if residual_converged(fx, x, config) {
            return Ok(RootFindingResult {
                root: x,
                iterations,
                error: fx.abs(),
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
    validate_config(config)?;
    if !a.is_finite() || !b.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Bracket endpoints must be finite".to_string(),
        ));
    }

    let mut fa = ensure_finite(f(a), "Function evaluation at lower bracket")?;
    let mut fb = ensure_finite(f(b), "Function evaluation at upper bracket")?;

    if residual_converged(fa, a, config) {
        return Ok(RootFindingResult {
            root: a,
            iterations: 0,
            error: fa.abs(),
            converged: true,
        });
    }
    if residual_converged(fb, b, config) {
        return Ok(RootFindingResult {
            root: b,
            iterations: 0,
            error: fb.abs(),
            converged: true,
        });
    }

    // Check that root is bracketed
    if !opposite_signs(fa, fb) {
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
        if residual_converged(fb, b, config) {
            return Ok(RootFindingResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        // Check convergence on interval size
        if has_converged((b - a).abs(), b, config) {
            return Ok(RootFindingResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        let mut s;

        if values_distinct(fa, fb, fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else if !secant_denominator_too_small(fa, fb) {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        } else {
            // Degenerate secant denominator, fall back to midpoint
            s = stable_midpoint(a, b);
        }

        // Check if we should use bisection instead
        let threshold = three_quarters_toward_b(a, b);
        let use_bisection =
            // s is not between (3a+b)/4 and b
            !(s > threshold.min(b) && s < threshold.max(b)) ||
            // mflag is set and |s-b| >= |b-c|/2
            (mflag && (s - b).abs() >= (b - c).abs() / 2.0) ||
            // mflag is clear and |s-b| >= |c-d|/2
            (!mflag && (s - b).abs() >= (c - d).abs() / 2.0) ||
            // mflag is set and |b-c| < tolerance
            (mflag && (b - c).abs() < scaled_tolerance(b, config)) ||
            // mflag is clear and |c-d| < tolerance
            (!mflag && (c - d).abs() < scaled_tolerance(c, config));

        if use_bisection {
            s = stable_midpoint(a, b);
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = ensure_finite(f(s), "Function evaluation during Brent iteration")?;
        if residual_converged(fs, s, config) {
            return Ok(RootFindingResult {
                root: s,
                iterations,
                error: fs.abs(),
                converged: true,
            });
        }

        // Update d before c is updated
        d = c;
        c = b;
        fc = fb;

        // Update the bracket
        if opposite_signs(fa, fs) {
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
    validate_config(config)?;
    if !a.is_finite() || !b.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Bracket endpoints must be finite".to_string(),
        ));
    }

    let mut fa = ensure_finite(f(a), "Function evaluation at lower bracket")?;
    let fb = ensure_finite(f(b), "Function evaluation at upper bracket")?;

    if residual_converged(fa, a, config) {
        return Ok(RootFindingResult {
            root: a,
            iterations: 0,
            error: fa.abs(),
            converged: true,
        });
    }
    if residual_converged(fb, b, config) {
        return Ok(RootFindingResult {
            root: b,
            iterations: 0,
            error: fb.abs(),
            converged: true,
        });
    }

    // Check that root is bracketed
    if !opposite_signs(fa, fb) {
        return Err(DervflowError::InvalidInput(
            "Root is not bracketed: f(a) and f(b) must have opposite signs".to_string(),
        ));
    }

    let mut iterations = 0;
    let mut error = (b - a).abs();

    for i in 0..config.max_iterations {
        iterations = i + 1;

        // Compute midpoint
        let c = stable_midpoint(a, b);
        let fc = ensure_finite(f(c), "Function evaluation at bisection midpoint")?;

        error = 0.5 * (b - a).abs();

        // Check convergence
        if has_converged(error, c, config) || residual_converged(fc, c, config) {
            return Ok(RootFindingResult {
                root: c,
                iterations,
                error,
                converged: true,
            });
        }

        // Update bracket
        if opposite_signs(fa, fc) {
            b = c;
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
    validate_config(config)?;
    if !x0.is_finite() || !x1.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Initial guesses must be finite".to_string(),
        ));
    }
    if (x1 - x0).abs() <= f64::EPSILON * x0.abs().max(x1.abs()).max(1.0) {
        return Err(DervflowError::InvalidInput(
            "Initial guesses must be distinct for secant method".to_string(),
        ));
    }

    let mut f0 = ensure_finite(f(x0), "Function evaluation at first secant guess")?;
    if residual_converged(f0, x0, config) {
        return Ok(RootFindingResult {
            root: x0,
            iterations: 0,
            error: f0.abs(),
            converged: true,
        });
    }

    let mut f1 = ensure_finite(f(x1), "Function evaluation at second secant guess")?;
    if residual_converged(f1, x1, config) {
        return Ok(RootFindingResult {
            root: x1,
            iterations: 0,
            error: f1.abs(),
            converged: true,
        });
    }

    let mut iterations = 0;
    let mut error = f64::INFINITY;

    for i in 0..config.max_iterations {
        iterations = i + 1;

        // Check for zero denominator
        if secant_denominator_too_small(f0, f1) {
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

        let f2 = ensure_finite(f(x2), "Function evaluation during secant iteration")?;
        error = f2.abs();

        // Check convergence (residual-based to prevent tiny-step false positives)
        if residual_converged(f2, x2, config) {
            return Ok(RootFindingResult {
                root: x2,
                iterations,
                error: f2.abs(),
                converged: true,
            });
        }

        // Update for next iteration
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
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
        assert!(matches!(
            result.unwrap_err(),
            DervflowError::InvalidInput(_)
        ));
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
        assert!(matches!(
            result.unwrap_err(),
            DervflowError::InvalidInput(_)
        ));
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
        assert!(result.error < config.tolerance);
    }

    #[test]
    fn test_zero_max_iterations_rejected() {
        let config = RootFindingConfig {
            max_iterations: 0,
            ..RootFindingConfig::default()
        };
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;

        assert!(matches!(
            newton_raphson(f, df, 1.0, &config),
            Err(DervflowError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_relative_tolerance_triggers_convergence() {
        let f = |x: f64| x - 1_000_000.0;
        let df = |_x: f64| 1.0;
        let config = RootFindingConfig {
            max_iterations: 4,
            tolerance: 1e-14,
            relative_tolerance: 1e-8,
        };

        let result = newton_raphson(f, df, 999_999.99, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.root, 1_000_000.0, epsilon = 1e-8);
    }

    #[test]
    fn test_bracket_endpoint_root_returns_immediately() {
        let f = |x: f64| x - 2.0;
        let config = RootFindingConfig::default();

        let brent_result = brent(f, 2.0, 5.0, &config).unwrap();
        assert_eq!(brent_result.iterations, 0);
        assert_eq!(brent_result.root, 2.0);

        let bisection_result = bisection(f, 2.0, 5.0, &config).unwrap();
        assert_eq!(bisection_result.iterations, 0);
        assert_eq!(bisection_result.root, 2.0);
    }

    #[test]
    fn test_newton_rejects_non_finite_function_evaluation() {
        let f = |_x: f64| f64::NAN;
        let df = |_x: f64| 1.0;
        let config = RootFindingConfig::default();

        assert!(matches!(
            newton_raphson(f, df, 0.0, &config),
            Err(DervflowError::NumericalError(_))
        ));
    }

    #[test]
    fn test_brent_rejects_non_finite_midpoint_evaluation() {
        let f = |x: f64| if x > 1.0 { f64::NAN } else { x - 0.5 };
        let config = RootFindingConfig::default();

        assert!(matches!(
            brent(f, 0.0, 2.0, &config),
            Err(DervflowError::NumericalError(_))
        ));
    }

    #[test]
    fn test_secant_rejects_non_finite_function_evaluation() {
        let f = |x: f64| if x > 1.0 { f64::INFINITY } else { x - 0.5 };
        let config = RootFindingConfig::default();

        assert!(matches!(
            secant(f, 0.0, 2.0, &config),
            Err(DervflowError::NumericalError(_))
        ));
    }

    #[test]
    fn test_newton_converges_on_residual_when_step_is_stagnant() {
        let f = |x: f64| (x - 1.0).powi(3);
        let df = |x: f64| 3.0 * (x - 1.0).powi(2);
        let config = RootFindingConfig::default();

        let result = newton_raphson(f, df, 1.0 + 1e-14, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.root, 1.0 + 1e-14, epsilon = 1e-12);
    }

    #[test]
    fn test_brent_accepts_residual_convergence_with_strict_absolute_tol() {
        let f = |x: f64| x - 1_000_000.0;
        let config = RootFindingConfig {
            max_iterations: 100,
            tolerance: 1e-14,
            relative_tolerance: 1e-8,
        };

        let result = brent(f, 999_999.0, 1_000_001.0, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.root, 1_000_000.0, epsilon = 1e-8);
    }

    #[test]
    fn test_secant_denominator_scale_aware_detection() {
        assert!(secant_denominator_too_small(1.0, 1.0 + f64::EPSILON));
        assert!(!secant_denominator_too_small(1.0, 1.0 + 1e-8));
    }

    #[test]
    fn test_derivative_small_threshold_scales_with_estimate() {
        assert!(derivative_too_small(f64::EPSILON, 1.0));
        assert!(!derivative_too_small(1e-8, 1.0));
    }

    #[test]
    fn test_values_distinct_rejects_near_equal_values() {
        assert!(!values_distinct(1.0, 1.0 + f64::EPSILON, 2.0));
        assert!(values_distinct(1.0, 1.1, 2.0));
    }

    #[test]
    fn test_opposite_signs_without_multiplication_overflow() {
        let a = 1.0e308;
        let b = -1.0e308;
        assert!(opposite_signs(a, b));
        assert!(!opposite_signs(a, a));
    }

    #[test]
    fn test_opposite_signs_handles_signed_zero_consistently() {
        assert!(!opposite_signs(0.0, -1.0));
        assert!(!opposite_signs(-0.0, 1.0));
        assert!(opposite_signs(1.0, -1.0));
    }

    #[test]
    fn test_secant_initial_guess_root_returns_immediately() {
        let f = |x: f64| x - 2.0;
        let config = RootFindingConfig::default();

        let at_x0 = secant(f, 2.0, 3.0, &config).unwrap();
        assert_eq!(at_x0.iterations, 0);
        assert_eq!(at_x0.root, 2.0);

        let at_x1 = secant(f, 1.0, 2.0, &config).unwrap();
        assert_eq!(at_x1.iterations, 0);
        assert_eq!(at_x1.root, 2.0);
    }

    #[test]
    fn test_secant_rejects_nearly_identical_initial_guesses() {
        let f = |x: f64| x * x - 2.0;
        let config = RootFindingConfig::default();

        let result = secant(f, 1.0, 1.0 + f64::EPSILON, &config);
        assert!(matches!(result, Err(DervflowError::InvalidInput(_))));
    }

    #[test]
    fn test_newton_does_not_false_converge_on_tiny_step_non_root() {
        let f = |_x: f64| 1.0;
        let df = |_x: f64| 1e30;
        let config = RootFindingConfig {
            max_iterations: 5,
            tolerance: 1e-12,
            relative_tolerance: 0.0,
        };

        let result = newton_raphson(f, df, 0.0, &config);
        assert!(matches!(
            result,
            Err(DervflowError::ConvergenceFailure { .. }) | Err(DervflowError::NumericalError(_))
        ));
    }

    #[test]
    fn test_secant_does_not_false_converge_on_tiny_step_non_root() {
        let f = |_x: f64| 1.0;
        let config = RootFindingConfig {
            max_iterations: 5,
            tolerance: 1e-12,
            relative_tolerance: 0.0,
        };

        let result = secant(f, 0.0, 1e-8, &config);
        assert!(matches!(result, Err(DervflowError::NumericalError(_))));
    }

    #[test]
    fn test_stable_midpoint_handles_large_opposite_sign_inputs() {
        let m = stable_midpoint(-1.0e308, 1.0e308);
        assert!(m.is_finite());
        assert_eq!(m, 0.0);
    }

    #[test]
    fn test_three_quarters_toward_b_stays_finite_for_large_opposite_sign_inputs() {
        let t = three_quarters_toward_b(-1.0e308, 1.0e308);
        assert!(t.is_finite());
        assert_eq!(t, -5.0e307);
    }

    #[test]
    fn test_newton_fallback_to_finite_difference_derivative() {
        let f = |x: f64| x * x - 2.0;
        let df = |_x: f64| 0.0;
        let config = RootFindingConfig::default();

        let result = newton_raphson(f, df, 1.5, &config).unwrap();
        assert!(result.converged);
        assert_relative_eq!(result.root, 2.0_f64.sqrt(), epsilon = 1e-6);
    }
}
