// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Numerical integration methods
//!
//! Provides robust numerical integration algorithms including adaptive Simpson's rule
//! and Gauss-Legendre quadrature. All methods include configurable tolerance and
//! maximum iteration limits.

use crate::common::error::{DervflowError, Result};

/// Configuration for numerical integration algorithms
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Maximum number of iterations for adaptive methods
    pub max_iterations: usize,
    /// Absolute tolerance for convergence
    pub tolerance: f64,
    /// Relative tolerance for convergence
    pub relative_tolerance: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            relative_tolerance: 1e-8,
        }
    }
}

/// Result of a numerical integration operation with diagnostics
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// The computed integral value
    pub value: f64,
    /// Estimated error
    pub error_estimate: f64,
    /// Number of function evaluations used
    pub function_evaluations: usize,
    /// Whether the method converged within tolerance
    pub converged: bool,
}

#[derive(Debug, Clone)]
struct SimpsonResult {
    value: f64,
    error_estimate: f64,
}

#[derive(Debug, Clone)]
struct AdaptiveGaussLegendreResult {
    value: f64,
    error_estimate: f64,
    function_evaluations: usize,
}

/// Adaptive Simpson's rule for numerical integration
///
/// Uses recursive subdivision to achieve desired accuracy.
/// Efficient for smooth functions with automatic error estimation.
///
/// # Arguments
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `config` - Configuration parameters
///
/// # Returns
/// Result containing the integral value and convergence diagnostics
pub fn adaptive_simpsons<F>(
    f: F,
    a: f64,
    b: f64,
    config: &IntegrationConfig,
) -> Result<IntegrationResult>
where
    F: Fn(f64) -> f64,
{
    // Validate bounds
    if !a.is_finite() || !b.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Integration bounds must be finite".to_string(),
        ));
    }

    if a >= b {
        return Err(DervflowError::InvalidInput(format!(
            "Lower bound ({}) must be less than upper bound ({})",
            a, b
        )));
    }

    let mut function_evaluations = 0;

    // Helper function for Simpson's rule on an interval
    let simpsons_rule = |x0: f64, x2: f64, f0: f64, f1: f64, f2: f64| -> f64 {
        let h = (x2 - x0) / 2.0;
        h / 3.0 * (f0 + 4.0 * f1 + f2)
    };

    // Evaluate function at initial points
    let mid = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fc = f(mid);
    function_evaluations += 3;

    // Check for non-finite values
    if !fa.is_finite() || !fb.is_finite() || !fc.is_finite() {
        return Err(DervflowError::NumericalError(
            "Function returned non-finite value during integration".to_string(),
        ));
    }

    let whole = simpsons_rule(a, b, fa, fc, fb);

    // Recursive adaptive integration
    let result = adaptive_simpsons_recursive(
        &f,
        a,
        b,
        config.tolerance,
        config.relative_tolerance,
        whole,
        fa,
        fc,
        fb,
        config.max_iterations,
        &mut function_evaluations,
    )?;

    Ok(IntegrationResult {
        value: result.value,
        error_estimate: result.error_estimate,
        function_evaluations,
        converged: true,
    })
}

/// Recursive helper for adaptive Simpson's rule
#[allow(clippy::too_many_arguments)]
fn adaptive_simpsons_recursive<F>(
    f: &F,
    a: f64,
    b: f64,
    tolerance: f64,
    relative_tolerance: f64,
    whole: f64,
    fa: f64,
    fc: f64,
    fb: f64,
    max_depth: usize,
    function_evaluations: &mut usize,
) -> Result<SimpsonResult>
where
    F: Fn(f64) -> f64,
{
    if max_depth == 0 {
        return Err(DervflowError::ConvergenceFailure {
            iterations: *function_evaluations,
            error: tolerance,
        });
    }

    let c = (a + b) / 2.0;
    let d = (a + c) / 2.0;
    let e = (c + b) / 2.0;

    let fd = f(d);
    let fe = f(e);
    *function_evaluations += 2;

    // Check for non-finite values
    if !fd.is_finite() || !fe.is_finite() {
        return Err(DervflowError::NumericalError(
            "Function returned non-finite value during integration".to_string(),
        ));
    }

    let h = (b - a) / 2.0;
    let left = h / 6.0 * (fa + 4.0 * fd + fc);
    let right = h / 6.0 * (fc + 4.0 * fe + fb);
    let sum = left + right;

    // Error estimate using Richardson extrapolation
    let error = (sum - whole).abs() / 15.0;
    let tol = tolerance + relative_tolerance * sum.abs();

    if error <= tol {
        Ok(SimpsonResult {
            value: sum + (sum - whole) / 15.0,
            error_estimate: error,
        })
    } else {
        let left_result = adaptive_simpsons_recursive(
            f,
            a,
            c,
            tolerance / 2.0,
            relative_tolerance,
            left,
            fa,
            fd,
            fc,
            max_depth - 1,
            function_evaluations,
        )?;

        let right_result = adaptive_simpsons_recursive(
            f,
            c,
            b,
            tolerance / 2.0,
            relative_tolerance,
            right,
            fc,
            fe,
            fb,
            max_depth - 1,
            function_evaluations,
        )?;

        Ok(SimpsonResult {
            value: left_result.value + right_result.value,
            error_estimate: left_result.error_estimate + right_result.error_estimate,
        })
    }
}

/// Gauss-Legendre quadrature nodes and weights
///
/// Pre-computed nodes and weights for various orders of Gauss-Legendre quadrature
struct GaussLegendreRule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendreRule {
    /// Get Gauss-Legendre rule for specified number of points
    fn get(n: usize) -> Result<Self> {
        match n {
            2 => Ok(Self {
                nodes: vec![-0.5773502691896257, 0.5773502691896257],
                weights: vec![1.0, 1.0],
            }),
            3 => Ok(Self {
                nodes: vec![-0.7745966692414834, 0.0, 0.7745966692414834],
                weights: vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
            }),
            4 => Ok(Self {
                nodes: vec![
                    -0.8611363115940526,
                    -0.3399810435848563,
                    0.3399810435848563,
                    0.8611363115940526,
                ],
                weights: vec![
                    0.3478548451374538,
                    0.6521451548625461,
                    0.6521451548625461,
                    0.3478548451374538,
                ],
            }),
            5 => Ok(Self {
                nodes: vec![
                    -0.906_179_845_938_664,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.906_179_845_938_664,
                ],
                weights: vec![
                    0.2369268850561891,
                    0.4786286704993665,
                    0.5688888888888889,
                    0.4786286704993665,
                    0.2369268850561891,
                ],
            }),
            10 => Ok(Self {
                nodes: vec![
                    -0.9739065285171717,
                    -0.8650633666889845,
                    -0.6794095682990244,
                    -0.4333953941292472,
                    -0.1488743389816312,
                    0.1488743389816312,
                    0.4333953941292472,
                    0.6794095682990244,
                    0.8650633666889845,
                    0.9739065285171717,
                ],
                weights: vec![
                    0.0666713443086881,
                    0.1494513491505806,
                    0.219_086_362_515_982,
                    0.2692667193099963,
                    0.2955242247147529,
                    0.2955242247147529,
                    0.2692667193099963,
                    0.219_086_362_515_982,
                    0.1494513491505806,
                    0.0666713443086881,
                ],
            }),
            20 => Ok(Self {
                nodes: vec![
                    -0.9931285991850949,
                    -0.9639719272779138,
                    -0.912_234_428_251_326,
                    -0.8391169718222188,
                    -0.7463319064601508,
                    -0.636_053_680_726_515,
                    -0.5108670019508271,
                    -0.3737060887154195,
                    -0.2277858511416451,
                    -0.0765265211334973,
                    0.0765265211334973,
                    0.2277858511416451,
                    0.3737060887154195,
                    0.5108670019508271,
                    0.636_053_680_726_515,
                    0.7463319064601508,
                    0.8391169718222188,
                    0.912_234_428_251_326,
                    0.9639719272779138,
                    0.9931285991850949,
                ],
                weights: vec![
                    0.0176140071391521,
                    0.0406014298003869,
                    0.0626720483341091,
                    0.0832767415767048,
                    0.1019301198172404,
                    0.1181945319615184,
                    0.1316886384491766,
                    0.142_096_109_318_382,
                    0.1491729864726037,
                    0.1527533871307258,
                    0.1527533871307258,
                    0.1491729864726037,
                    0.142_096_109_318_382,
                    0.1316886384491766,
                    0.1181945319615184,
                    0.1019301198172404,
                    0.0832767415767048,
                    0.0626720483341091,
                    0.0406014298003869,
                    0.0176140071391521,
                ],
            }),
            _ => Err(DervflowError::InvalidInput(format!(
                "Gauss-Legendre quadrature not available for {} points. Supported: 2, 3, 4, 5, 10, 20",
                n
            ))),
        }
    }
}

/// Gauss-Legendre quadrature for numerical integration
///
/// Uses pre-computed nodes and weights for high-accuracy integration.
/// Very efficient for smooth functions, especially polynomials.
///
/// # Arguments
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n_points` - Number of quadrature points (2, 3, 4, 5, 10, or 20)
///
/// # Returns
/// Result containing the integral value
pub fn gauss_legendre<F>(f: F, a: f64, b: f64, n_points: usize) -> Result<IntegrationResult>
where
    F: Fn(f64) -> f64,
{
    // Validate bounds
    if !a.is_finite() || !b.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Integration bounds must be finite".to_string(),
        ));
    }

    if a >= b {
        return Err(DervflowError::InvalidInput(format!(
            "Lower bound ({}) must be less than upper bound ({})",
            a, b
        )));
    }

    let rule = GaussLegendreRule::get(n_points)?;

    // Transform from [-1, 1] to [a, b]
    let mid = (b + a) / 2.0;
    let half_length = (b - a) / 2.0;

    let mut sum = 0.0;
    let mut function_evaluations = 0;

    for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
        let x = mid + half_length * node;
        let fx = f(x);
        function_evaluations += 1;

        // Check for non-finite values
        if !fx.is_finite() {
            return Err(DervflowError::NumericalError(
                "Function returned non-finite value during integration".to_string(),
            ));
        }

        sum += weight * fx;
    }

    let value = half_length * sum;

    Ok(IntegrationResult {
        value,
        error_estimate: 0.0, // Gauss-Legendre doesn't provide error estimate
        function_evaluations,
        converged: true,
    })
}

/// Adaptive Gauss-Legendre quadrature with automatic subdivision
///
/// Combines Gauss-Legendre quadrature with adaptive subdivision for
/// improved accuracy on non-smooth functions.
///
/// # Arguments
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `config` - Configuration parameters
///
/// # Returns
/// Result containing the integral value and convergence diagnostics
pub fn adaptive_gauss_legendre<F>(
    f: F,
    a: f64,
    b: f64,
    config: &IntegrationConfig,
) -> Result<IntegrationResult>
where
    F: Fn(f64) -> f64,
{
    // Validate bounds
    if !a.is_finite() || !b.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Integration bounds must be finite".to_string(),
        ));
    }

    if a >= b {
        return Err(DervflowError::InvalidInput(format!(
            "Lower bound ({}) must be less than upper bound ({})",
            a, b
        )));
    }

    let result = adaptive_gauss_legendre_recursive(
        &f,
        a,
        b,
        config.tolerance,
        config.relative_tolerance,
        config.max_iterations,
    )?;

    Ok(IntegrationResult {
        value: result.value,
        error_estimate: result.error_estimate,
        function_evaluations: result.function_evaluations,
        converged: true,
    })
}

/// Recursive helper for adaptive Gauss-Legendre quadrature
fn adaptive_gauss_legendre_recursive<F>(
    f: &F,
    a: f64,
    b: f64,
    tolerance: f64,
    relative_tolerance: f64,
    max_depth: usize,
) -> Result<AdaptiveGaussLegendreResult>
where
    F: Fn(f64) -> f64,
{
    if max_depth == 0 {
        return Err(DervflowError::ConvergenceFailure {
            iterations: max_depth,
            error: tolerance,
        });
    }

    let result_5 = gauss_legendre(f, a, b, 5)?;
    let result_10 = gauss_legendre(f, a, b, 10)?;

    let error = (result_10.value - result_5.value).abs();
    let tol = tolerance + relative_tolerance * result_10.value.abs();
    let function_evaluations = result_5.function_evaluations + result_10.function_evaluations;

    if error <= tol {
        Ok(AdaptiveGaussLegendreResult {
            value: result_10.value,
            error_estimate: error,
            function_evaluations,
        })
    } else {
        let mid = (a + b) / 2.0;
        let left = adaptive_gauss_legendre_recursive(
            f,
            a,
            mid,
            tolerance / 2.0,
            relative_tolerance,
            max_depth - 1,
        )?;

        let right = adaptive_gauss_legendre_recursive(
            f,
            mid,
            b,
            tolerance / 2.0,
            relative_tolerance,
            max_depth - 1,
        )?;

        Ok(AdaptiveGaussLegendreResult {
            value: left.value + right.value,
            error_estimate: left.error_estimate + right.error_estimate,
            function_evaluations: function_evaluations
                + left.function_evaluations
                + right.function_evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_adaptive_simpsons_polynomial() {
        // Integrate x^2 from 0 to 1, exact answer = 1/3
        let f = |x: f64| x * x;
        let config = IntegrationConfig::default();

        let result = adaptive_simpsons(f, 0.0, 1.0, &config).unwrap();

        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_adaptive_simpsons_sine() {
        // Integrate sin(x) from 0 to π, exact answer = 2
        let f = |x: f64| x.sin();
        let config = IntegrationConfig::default();

        let result = adaptive_simpsons(f, 0.0, std::f64::consts::PI, &config).unwrap();

        assert_relative_eq!(result.value, 2.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_adaptive_simpsons_exponential() {
        // Integrate e^x from 0 to 1, exact answer = e - 1
        let f = |x: f64| x.exp();
        let config = IntegrationConfig::default();

        let result = adaptive_simpsons(f, 0.0, 1.0, &config).unwrap();
        let expected = 1.0_f64.exp() - 1.0;

        assert_relative_eq!(result.value, expected, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_gauss_legendre_polynomial() {
        // Integrate x^2 from 0 to 1, exact answer = 1/3
        let f = |x: f64| x * x;

        let result = gauss_legendre(f, 0.0, 1.0, 5).unwrap();

        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-10);
        assert_eq!(result.function_evaluations, 5);
    }

    #[test]
    fn test_gauss_legendre_sine() {
        // Integrate sin(x) from 0 to π, exact answer = 2
        let f = |x: f64| x.sin();

        let result = gauss_legendre(f, 0.0, std::f64::consts::PI, 10).unwrap();

        assert_relative_eq!(result.value, 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_gauss_legendre_high_order() {
        // Integrate x^4 from -1 to 1, exact answer = 2/5
        let f = |x: f64| x.powi(4);

        let result = gauss_legendre(f, -1.0, 1.0, 5).unwrap();

        assert_relative_eq!(result.value, 0.4, epsilon = 1e-12);
    }

    #[test]
    fn test_gauss_legendre_different_orders() {
        let f = |x: f64| x * x;

        for n in &[2, 3, 4, 5, 10, 20] {
            let result = gauss_legendre(f, 0.0, 1.0, *n).unwrap();
            assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gauss_legendre_invalid_order() {
        let f = |x: f64| x * x;
        let result = gauss_legendre(f, 0.0, 1.0, 7);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DervflowError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_adaptive_gauss_legendre() {
        // Integrate a function with varying smoothness
        let f = |x: f64| (10.0 * x).sin() / (1.0 + x * x);
        let config = IntegrationConfig::default();

        let result = adaptive_gauss_legendre(f, 0.0, 5.0, &config).unwrap();

        assert!(result.converged);
        assert!(result.value.is_finite());
    }

    #[test]
    fn test_invalid_bounds() {
        let f = |x: f64| x * x;
        let config = IntegrationConfig::default();

        // a >= b
        let result = adaptive_simpsons(f, 1.0, 0.0, &config);
        assert!(result.is_err());

        // Non-finite bounds
        let result = adaptive_simpsons(f, f64::INFINITY, 1.0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_tolerance() {
        let f = |x: f64| x * x;
        let config = IntegrationConfig {
            max_iterations: 1000,
            tolerance: 1e-12,
            relative_tolerance: 1e-12,
        };

        let result = adaptive_simpsons(f, 0.0, 1.0, &config).unwrap();
        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_function_evaluations_counted() {
        let f = |x: f64| x * x;
        let config = IntegrationConfig::default();

        let result = adaptive_simpsons(f, 0.0, 1.0, &config).unwrap();
        assert!(result.function_evaluations > 0);

        let result = gauss_legendre(f, 0.0, 1.0, 10).unwrap();
        assert_eq!(result.function_evaluations, 10);
    }

    #[test]
    fn test_convergence_diagnostics() {
        let f = |x: f64| x.exp();
        let config = IntegrationConfig::default();

        let result = adaptive_simpsons(f, 0.0, 1.0, &config).unwrap();
        let tolerance = config.tolerance + config.relative_tolerance * result.value.abs();

        assert!(result.converged);
        assert!(result.function_evaluations > 0);
        assert!(result.error_estimate <= tolerance);
    }

    #[test]
    fn test_comparison_methods() {
        // Compare adaptive Simpson's and Gauss-Legendre on same integral
        let f = |x: f64| (x * x + 1.0).sqrt();
        let config = IntegrationConfig::default();

        let simpson_result = adaptive_simpsons(f, 0.0, 2.0, &config).unwrap();
        let gauss_result = adaptive_gauss_legendre(f, 0.0, 2.0, &config).unwrap();

        // Both methods should give similar results
        assert_relative_eq!(simpson_result.value, gauss_result.value, epsilon = 1e-5);
    }
}
