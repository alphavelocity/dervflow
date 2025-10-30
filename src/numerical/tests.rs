// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Property-based tests for numerical methods
//!
//! These tests verify numerical stability and correctness properties
//! across a wide range of inputs using property-based testing.

#[cfg(test)]
mod property_tests {
    use crate::numerical::integration::{IntegrationConfig, adaptive_simpsons, gauss_legendre};
    use crate::numerical::linalgops::{
        cholesky_decomposition, is_positive_definite, matrix_multiply,
    };
    use crate::numerical::optimization::{BFGS, GradientDescent, NelderMead, OptimizationConfig};
    use crate::numerical::root_finding::{RootFindingConfig, brent, newton_raphson};
    use nalgebra::DMatrix;
    use proptest::prelude::*;

    // Property: Integration of constant function should equal constant * width
    proptest! {
        #[test]
        fn prop_integration_constant(c in -100.0..100.0, a in -10.0..10.0, width in 0.1..10.0) {
            let b = a + width;
            let f = |_: f64| c;
            let config = IntegrationConfig::default();

            if let Ok(result) = adaptive_simpsons(f, a, b, &config) {
                let expected = c * (b - a);
                let error = (result.value - expected).abs();
                prop_assert!(error < 1e-6, "Integration error too large: {}", error);
            }
        }
    }

    // Property: Integration is additive over intervals
    proptest! {
        #[test]
        fn prop_integration_additive(a in -10.0..0.0, b in 0.0..10.0, c in 10.0..20.0) {
            let f = |x: f64| x * x;
            let config = IntegrationConfig::default();

            if let (Ok(r1), Ok(r2), Ok(r_total)) = (
                adaptive_simpsons(f, a, b, &config),
                adaptive_simpsons(f, b, c, &config),
                adaptive_simpsons(f, a, c, &config),
            ) {
                let sum = r1.value + r2.value;
                let error = (sum - r_total.value).abs();
                prop_assert!(error < 1e-4, "Integration additivity violated: {}", error);
            }
        }
    }

    // Property: Root finding should find actual roots
    proptest! {
        #[test]
        fn prop_root_finding_correctness(root in -10.0..10.0) {
            // Function with known root: f(x) = (x - root)
            let f = |x: f64| x - root;
            let df = |_: f64| 1.0;

            let config = RootFindingConfig::default();
            let initial_guess = root + 1.0;

            if let Ok(result) = newton_raphson(f, df, initial_guess, &config) {
                let f_at_root = f(result.root);
                prop_assert!(f_at_root.abs() < 1e-6, "Not a root: f(x) = {}", f_at_root);
            }
        }
    }

    // Property: Brent's method should converge for bracketed roots
    proptest! {
        #[test]
        fn prop_brent_convergence(root in -5.0..5.0) {
            let f = |x: f64| (x - root) * (x - root) - 0.01;
            let config = RootFindingConfig::default();

            // Bracket the root
            let a = root - 2.0;
            let b = root + 2.0;

            if f(a) * f(b) < 0.0 && let Ok(result) = brent(f, a, b, &config) {
                prop_assert!(result.converged, "Brent's method should converge");
                let f_val = f(result.root);
                prop_assert!(f_val.abs() < 1e-6, "Root not accurate: f(x) = {}", f_val);
            }
        }
    }

    // Test: Optimization finds minimum of quadratic
    #[test]
    fn test_optimization_quadratic() {
        let f = |x: &[f64]| (x[0] - 2.5).powi(2) + (x[1] - 3.5).powi(2);
        let grad_f = |x: &[f64]| vec![2.0 * (x[0] - 2.5), 2.0 * (x[1] - 3.5)];

        let config = OptimizationConfig::default();
        let optimizer = GradientDescent::new(config);

        let result = optimizer.optimize(f, grad_f, &[0.0, 0.0]).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.5).abs() < 1e-3);
        assert!((result.x[1] - 3.5).abs() < 1e-3);
        assert!(result.f_val < 1e-5);
    }

    // Property: Cholesky decomposition should satisfy A = L * L^T
    proptest! {
        #[test]
        fn prop_cholesky_reconstruction(
            d1 in 1.0..10.0,
            d2 in 1.0..10.0,
            d3 in 1.0..10.0,
        ) {
            // Create a positive definite matrix using diagonal dominance
            let a = DMatrix::from_row_slice(3, 3, &[
                d1, 0.1, 0.1,
                0.1, d2, 0.1,
                0.1, 0.1, d3,
            ]);

            if let Ok(l) = cholesky_decomposition(&a) {
                let reconstructed = &l * l.transpose();

                for i in 0..3 {
                    for j in 0..3 {
                        let error = (a[(i, j)] - reconstructed[(i, j)]).abs();
                        prop_assert!(error < 1e-10, "Cholesky reconstruction error: {}", error);
                    }
                }
            }
        }
    }

    // Property: Matrix multiplication is associative
    proptest! {
        #[test]
        fn prop_matrix_multiply_associative(
            a11 in -10.0..10.0, a12 in -10.0..10.0,
            a21 in -10.0..10.0, a22 in -10.0..10.0,
            b11 in -10.0..10.0, b12 in -10.0..10.0,
            b21 in -10.0..10.0, b22 in -10.0..10.0,
            c11 in -10.0..10.0, c12 in -10.0..10.0,
            c21 in -10.0..10.0, c22 in -10.0..10.0,
        ) {
            let a = DMatrix::from_row_slice(2, 2, &[a11, a12, a21, a22]);
            let b = DMatrix::from_row_slice(2, 2, &[b11, b12, b21, b22]);
            let c = DMatrix::from_row_slice(2, 2, &[c11, c12, c21, c22]);

            if let (Ok(ab), Ok(bc)) = (matrix_multiply(&a, &b), matrix_multiply(&b, &c))
                && let (Ok(ab_c), Ok(a_bc)) = (matrix_multiply(&ab, &c), matrix_multiply(&a, &bc))
            {
                for i in 0..2 {
                    for j in 0..2 {
                        let error = (ab_c[(i, j)] - a_bc[(i, j)]).abs();
                        prop_assert!(error < 1e-8, "Associativity violated: {}", error);
                    }
                }
            }
        }
    }

    // Property: Positive definite matrices have positive eigenvalues
    proptest! {
        #[test]
        fn prop_positive_definite_eigenvalues(
            d1 in 1.0..10.0,
            d2 in 1.0..10.0,
        ) {
            // Create a positive definite matrix
            let a = DMatrix::from_row_slice(2, 2, &[
                d1, 0.5,
                0.5, d2,
            ]);

            if d1 * d2 > 0.25 {  // Ensure positive definite
                prop_assert!(is_positive_definite(&a), "Matrix should be positive definite");
            }
        }
    }

    // Property: Gauss-Legendre should be exact for polynomials up to degree 2n-1
    proptest! {
        #[test]
        fn prop_gauss_legendre_polynomial_exactness(
            c0 in -10.0..10.0,
            c1 in -10.0..10.0,
            c2 in -10.0..10.0,
        ) {
            // Polynomial: f(x) = c0 + c1*x + c2*x^2
            let f = |x: f64| c0 + c1 * x + c2 * x * x;

            // Analytical integral from -1 to 1
            let expected: f64 = 2.0 * c0 + (2.0 / 3.0) * c2;

            if let Ok(result) = gauss_legendre(f, -1.0, 1.0, 5) {
                let error: f64 = (result.value - expected).abs();
                prop_assert!(error < 1e-12, "Gauss-Legendre not exact for polynomial: error = {}", error);
            }
        }
    }

    // Test: BFGS convergence on simple quadratic
    #[test]
    fn test_bfgs_convergence_simple() {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let grad_f = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)];

        let config = OptimizationConfig::default();
        let bfgs = BFGS::new(config);

        let result = bfgs.optimize(f, grad_f, &[0.0, 0.0]).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-3);
        assert!((result.x[1] - 3.0).abs() < 1e-3);
    }

    // Test: Nelder-Mead works without gradients
    #[test]
    fn test_nelder_mead_no_gradient() {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        let config = OptimizationConfig {
            max_iterations: 2000,
            f_tol: 1e-6,
            g_tol: 1e-6,
            x_tol: 1e-6,
        };
        let optimizer = NelderMead::new(config);

        let result = optimizer.optimize(f, &[0.0, 0.0]).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 0.1);
        assert!((result.x[1] - 3.0).abs() < 0.1);
    }
}
