// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Optimization algorithms
//!
//! Provides unconstrained optimization methods:
//! - Gradient descent with line search
//! - BFGS quasi-Newton method
//! - Nelder-Mead simplex method (derivative-free)

use crate::common::error::{DervflowError, Result};
use nalgebra::{DMatrix, DVector};

/// Configuration for optimization algorithms
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for function value
    pub f_tol: f64,
    /// Convergence tolerance for gradient norm
    pub g_tol: f64,
    /// Convergence tolerance for step size
    pub x_tol: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            f_tol: 1e-8,
            g_tol: 1e-6,
            x_tol: 1e-8,
        }
    }
}

/// Result of an optimization run
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal point found
    pub x: Vec<f64>,
    /// Function value at optimal point
    pub f_val: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the optimization converged
    pub converged: bool,
    /// Final gradient norm (if applicable)
    pub gradient_norm: Option<f64>,
}

/// Gradient descent optimizer with backtracking line search
pub struct GradientDescent {
    config: OptimizationConfig,
    /// Initial step size for line search
    alpha_init: f64,
    /// Line search backtracking factor (0 < beta < 1)
    beta: f64,
    /// Armijo condition parameter (0 < c < 1)
    c: f64,
}

impl GradientDescent {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            alpha_init: 1.0,
            beta: 0.5,
            c: 1e-4,
        }
    }

    /// Optimize a function using gradient descent
    ///
    /// # Arguments
    /// * `f` - Objective function
    /// * `grad_f` - Gradient of objective function
    /// * `x0` - Initial point
    pub fn optimize<F, G>(&self, f: F, grad_f: G, x0: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut x = x0.to_vec();
        let mut f_val = f(&x);

        for iter in 0..self.config.max_iterations {
            let grad = grad_f(&x);
            let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Check gradient convergence
            if grad_norm < self.config.g_tol {
                return Ok(OptimizationResult {
                    x,
                    f_val,
                    iterations: iter,
                    converged: true,
                    gradient_norm: Some(grad_norm),
                });
            }

            // Backtracking line search
            let mut alpha = self.alpha_init;
            let descent_dir: Vec<f64> = grad.iter().map(|g| -g).collect();
            let grad_dot_dir: f64 = grad.iter().zip(&descent_dir).map(|(g, d)| g * d).sum();

            loop {
                let x_new: Vec<f64> = x
                    .iter()
                    .zip(&descent_dir)
                    .map(|(xi, di)| xi + alpha * di)
                    .collect();

                let f_new = f(&x_new);

                // Armijo condition
                if f_new <= f_val + self.c * alpha * grad_dot_dir {
                    let f_diff = (f_val - f_new).abs();
                    let x_diff = x
                        .iter()
                        .zip(&x_new)
                        .map(|(xi, xn)| (xi - xn).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    x = x_new;
                    f_val = f_new;

                    // Check function and step convergence
                    if f_diff < self.config.f_tol && x_diff < self.config.x_tol {
                        return Ok(OptimizationResult {
                            x,
                            f_val,
                            iterations: iter + 1,
                            converged: true,
                            gradient_norm: Some(grad_norm),
                        });
                    }

                    break;
                }

                alpha *= self.beta;

                if alpha < 1e-16 {
                    return Err(DervflowError::ConvergenceFailure {
                        iterations: iter,
                        error: grad_norm,
                    });
                }
            }
        }

        let grad = grad_f(&x);
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

        Err(DervflowError::ConvergenceFailure {
            iterations: self.config.max_iterations,
            error: grad_norm,
        })
    }
}

/// BFGS quasi-Newton optimizer
pub struct BFGS {
    config: OptimizationConfig,
}

impl BFGS {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Optimize a function using BFGS
    ///
    /// # Arguments
    /// * `f` - Objective function
    /// * `grad_f` - Gradient of objective function
    /// * `x0` - Initial point
    pub fn optimize<F, G>(&self, f: F, grad_f: G, x0: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x0.len();
        let mut x = DVector::from_vec(x0.to_vec());
        let mut f_val = f(x.as_slice());

        // Initialize inverse Hessian approximation as identity
        let mut h_inv = DMatrix::identity(n, n);

        let mut grad = DVector::from_vec(grad_f(x.as_slice()));

        for iter in 0..self.config.max_iterations {
            let grad_norm = grad.norm();

            // Check gradient convergence
            if grad_norm < self.config.g_tol {
                return Ok(OptimizationResult {
                    x: x.as_slice().to_vec(),
                    f_val,
                    iterations: iter,
                    converged: true,
                    gradient_norm: Some(grad_norm),
                });
            }

            // Compute search direction: p = -H * grad
            let p = -(&h_inv * &grad);

            // Line search with backtracking
            let mut alpha = 1.0;
            let grad_dot_p = grad.dot(&p);

            let x_new = loop {
                let x_trial = &x + alpha * &p;
                let f_trial = f(x_trial.as_slice());

                // Armijo condition
                if f_trial <= f_val + 1e-4 * alpha * grad_dot_p {
                    break x_trial;
                }

                alpha *= 0.5;

                if alpha < 1e-16 {
                    return Err(DervflowError::ConvergenceFailure {
                        iterations: iter,
                        error: grad_norm,
                    });
                }
            };

            let f_new = f(x_new.as_slice());
            let grad_new = DVector::from_vec(grad_f(x_new.as_slice()));

            // BFGS update
            let s = &x_new - &x;
            let y = &grad_new - &grad;
            let rho = 1.0 / y.dot(&s);

            if rho.is_finite() && rho > 0.0 {
                let i = DMatrix::identity(n, n);
                let rho_sy = rho * &s * y.transpose();
                let term1 = &i - &rho_sy;
                let term2 = &i - rho * &y * s.transpose();
                h_inv = &term1 * &h_inv * &term2 + rho * &s * s.transpose();
            }

            // Check convergence
            let f_diff = (f_val - f_new).abs();
            let x_diff = (&x_new - &x).norm();

            x = x_new;
            f_val = f_new;
            grad = grad_new;

            if f_diff < self.config.f_tol && x_diff < self.config.x_tol {
                return Ok(OptimizationResult {
                    x: x.as_slice().to_vec(),
                    f_val,
                    iterations: iter + 1,
                    converged: true,
                    gradient_norm: Some(grad.norm()),
                });
            }
        }

        Err(DervflowError::ConvergenceFailure {
            iterations: self.config.max_iterations,
            error: grad.norm(),
        })
    }
}

/// Nelder-Mead simplex optimizer (derivative-free)
pub struct NelderMead {
    config: OptimizationConfig,
    /// Reflection coefficient
    alpha: f64,
    /// Expansion coefficient
    gamma: f64,
    /// Contraction coefficient
    rho: f64,
    /// Shrink coefficient
    sigma: f64,
}

impl NelderMead {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }

    /// Optimize a function using Nelder-Mead simplex method
    ///
    /// # Arguments
    /// * `f` - Objective function
    /// * `x0` - Initial point
    pub fn optimize<F>(&self, f: F, x0: &[f64]) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();

        // Initialize simplex
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(x0.to_vec());

        // Create initial simplex using coordinate directions
        for i in 0..n {
            let mut vertex = x0.to_vec();
            vertex[i] += if x0[i].abs() > 1e-8 {
                0.05 * x0[i]
            } else {
                0.00025
            };
            simplex.push(vertex);
        }

        // Evaluate function at simplex vertices
        let mut f_vals: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

        for iter in 0..self.config.max_iterations {
            // Sort simplex by function values
            let mut indices: Vec<usize> = (0..=n).collect();
            indices.sort_by(|&i, &j| f_vals[i].partial_cmp(&f_vals[j]).unwrap());

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            // Check convergence
            let f_range = f_vals[worst_idx] - f_vals[best_idx];
            if f_range < self.config.f_tol {
                return Ok(OptimizationResult {
                    x: simplex[best_idx].clone(),
                    f_val: f_vals[best_idx],
                    iterations: iter,
                    converged: true,
                    gradient_norm: None,
                });
            }

            // Compute centroid (excluding worst point)
            let centroid: Vec<f64> = (0..n)
                .map(|j| indices[..n].iter().map(|&i| simplex[i][j]).sum::<f64>() / n as f64)
                .collect();

            // Reflection
            let x_r: Vec<f64> = centroid
                .iter()
                .zip(&simplex[worst_idx])
                .map(|(c, w)| c + self.alpha * (c - w))
                .collect();
            let f_r = f(&x_r);

            if f_vals[best_idx] <= f_r && f_r < f_vals[second_worst_idx] {
                // Accept reflection
                simplex[worst_idx] = x_r;
                f_vals[worst_idx] = f_r;
                continue;
            }

            // Expansion
            if f_r < f_vals[best_idx] {
                let x_e: Vec<f64> = centroid
                    .iter()
                    .zip(&x_r)
                    .map(|(c, r)| c + self.gamma * (r - c))
                    .collect();
                let f_e = f(&x_e);

                if f_e < f_r {
                    simplex[worst_idx] = x_e;
                    f_vals[worst_idx] = f_e;
                } else {
                    simplex[worst_idx] = x_r;
                    f_vals[worst_idx] = f_r;
                }
                continue;
            }

            // Contraction
            if f_r < f_vals[worst_idx] {
                // Outside contraction
                let x_c: Vec<f64> = centroid
                    .iter()
                    .zip(&x_r)
                    .map(|(c, r)| c + self.rho * (r - c))
                    .collect();
                let f_c = f(&x_c);

                if f_c < f_r {
                    simplex[worst_idx] = x_c;
                    f_vals[worst_idx] = f_c;
                    continue;
                }
            } else {
                // Inside contraction
                let x_c: Vec<f64> = centroid
                    .iter()
                    .zip(&simplex[worst_idx])
                    .map(|(c, w)| c + self.rho * (w - c))
                    .collect();
                let f_c = f(&x_c);

                if f_c < f_vals[worst_idx] {
                    simplex[worst_idx] = x_c;
                    f_vals[worst_idx] = f_c;
                    continue;
                }
            }

            // Shrink
            for i in 1..=n {
                simplex[i] = simplex[best_idx]
                    .iter()
                    .zip(&simplex[i])
                    .map(|(b, s)| b + self.sigma * (s - b))
                    .collect();
                f_vals[i] = f(&simplex[i]);
            }
        }

        // Find best point
        let best_idx = f_vals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        Err(DervflowError::ConvergenceFailure {
            iterations: self.config.max_iterations,
            error: f_vals[best_idx],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gradient_descent_quadratic() {
        // Minimize f(x) = (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let grad_f = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)];

        let config = OptimizationConfig::default();
        let optimizer = GradientDescent::new(config);

        let result = optimizer.optimize(f, grad_f, &[0.0, 0.0]).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-4);
        assert_relative_eq!(result.f_val, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let grad_f = |x: &[f64]| {
            vec![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
                200.0 * (x[1] - x[0].powi(2)),
            ]
        };

        let config = OptimizationConfig {
            max_iterations: 2000,
            f_tol: 1e-8,
            g_tol: 1e-6,
            x_tol: 1e-8,
        };
        let optimizer = BFGS::new(config);

        let result = optimizer.optimize(f, grad_f, &[0.0, 0.0]).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.f_val, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nelder_mead_quadratic() {
        // Minimize f(x) = (x-2)^2 + (y-3)^2
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        let config = OptimizationConfig::default();
        let optimizer = NelderMead::new(config);

        let result = optimizer.optimize(f, &[0.0, 0.0]).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-3);
        assert_relative_eq!(result.f_val, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Rosenbrock function
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);

        let config = OptimizationConfig {
            max_iterations: 5000,
            f_tol: 1e-6,
            g_tol: 1e-6,
            x_tol: 1e-6,
        };
        let optimizer = NelderMead::new(config);

        let result = optimizer.optimize(f, &[-1.0, 1.0]).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-2);
    }
}
