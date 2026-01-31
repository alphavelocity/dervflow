// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Linear algebra operations
//!
//! Provides essential linear algebra operations with numerical stability:
//! - Cholesky decomposition for positive definite matrices
//! - Matrix operations (multiplication, inversion)
//! - Numerical stability checks and guards

use crate::common::error::{DervflowError, Result};
use nalgebra::{DMatrix, DVector};
use std::str::FromStr;

/// Supported matrix norms
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MatrixNorm {
    /// Maximum absolute column sum
    One,
    /// Maximum absolute row sum
    Infinity,
    /// Frobenius norm
    #[default]
    Frobenius,
    /// Spectral (2) norm
    Spectral,
}

impl FromStr for MatrixNorm {
    type Err = DervflowError;

    fn from_str(value: &str) -> Result<Self> {
        let normalized = value.trim().to_lowercase();
        match normalized.as_str() {
            "1" | "one" | "l1" => Ok(MatrixNorm::One),
            "inf" | "infinity" | "linf" | "l_inf" => Ok(MatrixNorm::Infinity),
            "fro" | "frobenius" | "frob" => Ok(MatrixNorm::Frobenius),
            "2" | "two" | "l2" | "spectral" => Ok(MatrixNorm::Spectral),
            _ => Err(DervflowError::InvalidInput(format!(
                "Unsupported matrix norm '{}'. Expected one of '1', 'inf', 'fro', or '2/spectral'",
                value
            ))),
        }
    }
}

/// Perform Cholesky decomposition of a positive definite matrix
///
/// Decomposes A = L * L^T where L is lower triangular
///
/// # Arguments
/// * `matrix` - Symmetric positive definite matrix (n x n)
///
/// # Returns
/// Lower triangular matrix L such that A = L * L^T
///
/// # Errors
/// Returns error if matrix is not positive definite
pub fn cholesky_decomposition(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let n = matrix.nrows();

    if n != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square for Cholesky decomposition".to_string(),
        ));
    }

    // Check symmetry
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-10 {
                return Err(DervflowError::InvalidInput(
                    "Matrix must be symmetric for Cholesky decomposition".to_string(),
                ));
            }
        }
    }

    let mut l = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum: f64 = 0.0;

            if i == j {
                // Diagonal element
                for k in 0..j {
                    sum += l[(j, k)].powi(2);
                }

                let diag_val = matrix[(j, j)] - sum;

                if diag_val <= 0.0 {
                    return Err(DervflowError::NumericalError(format!(
                        "Matrix is not positive definite: diagonal element {} is {}",
                        j, diag_val
                    )));
                }

                l[(j, j)] = diag_val.sqrt();
            } else {
                // Off-diagonal element
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }

                if l[(j, j)].abs() < 1e-15 {
                    return Err(DervflowError::NumericalError(
                        "Near-zero diagonal element in Cholesky decomposition".to_string(),
                    ));
                }

                l[(i, j)] = (matrix[(i, j)] - sum) / l[(j, j)];
            }
        }
    }

    Ok(l)
}

/// Generate correlated random samples using Cholesky decomposition
///
/// # Arguments
/// * `correlation` - Correlation matrix (n x n)
/// * `samples` - Independent standard normal samples (n x m)
///
/// # Returns
/// Correlated samples (n x m)
pub fn correlate_samples(
    correlation: &DMatrix<f64>,
    samples: &DMatrix<f64>,
) -> Result<DMatrix<f64>> {
    if correlation.nrows() != correlation.ncols() {
        return Err(DervflowError::InvalidInput(
            "Correlation matrix must be square".to_string(),
        ));
    }

    if correlation.nrows() != samples.nrows() {
        return Err(DervflowError::InvalidInput(format!(
            "Sample matrix row count ({}) must match correlation dimension ({})",
            samples.nrows(),
            correlation.nrows()
        )));
    }

    let l = cholesky_decomposition(correlation)?;
    Ok(&l * samples)
}

/// Multiply two matrices with dimension checking
///
/// # Arguments
/// * `a` - First matrix (m x n)
/// * `b` - Second matrix (n x p)
///
/// # Returns
/// Product matrix (m x p)
pub fn matrix_multiply(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if a.ncols() != b.nrows() {
        return Err(DervflowError::InvalidInput(format!(
            "Matrix dimensions incompatible for multiplication: ({}, {}) x ({}, {})",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }

    Ok(a * b)
}

/// Invert a matrix with condition number checking
///
/// # Arguments
/// * `matrix` - Square matrix to invert
/// * `check_condition` - Whether to check condition number
///
/// # Returns
/// Inverse matrix
///
/// # Errors
/// Returns error if matrix is singular or ill-conditioned
pub fn matrix_inverse(matrix: &DMatrix<f64>, check_condition: bool) -> Result<DMatrix<f64>> {
    let n = matrix.nrows();

    if n != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    // Use LU decomposition for inversion
    let lu = matrix.clone().lu();

    if !lu.is_invertible() {
        return Err(DervflowError::NumericalError(
            "Matrix is singular and cannot be inverted".to_string(),
        ));
    }

    let inverse = lu.try_inverse().ok_or_else(|| {
        DervflowError::NumericalError("Failed to compute matrix inverse".to_string())
    })?;

    // Check condition number if requested
    if check_condition {
        let cond = estimate_condition_number(matrix);

        if cond > 1e12 {
            return Err(DervflowError::NumericalError(format!(
                "Matrix is ill-conditioned (condition number: {:.2e}). Inversion may be numerically unstable",
                cond
            )));
        }
    }

    Ok(inverse)
}

/// Compute the matrix exponential using scaling and squaring via nalgebra
pub fn matrix_exponential(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square to compute the exponential".to_string(),
        ));
    }

    Ok(matrix.clone().exp())
}

/// Compute the determinant of a square matrix using LU decomposition
pub fn matrix_determinant(matrix: &DMatrix<f64>) -> Result<f64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square to compute determinant".to_string(),
        ));
    }

    Ok(matrix.clone().lu().determinant())
}

/// Compute the trace (sum of diagonal elements) of a square matrix
pub fn matrix_trace(matrix: &DMatrix<f64>) -> Result<f64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square to compute trace".to_string(),
        ));
    }

    Ok((0..matrix.nrows()).map(|i| matrix[(i, i)]).sum())
}

/// Raise a square matrix to a non-negative integer power using exponentiation by squaring
pub fn matrix_power(matrix: &DMatrix<f64>, power: u32) -> Result<DMatrix<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square to compute matrix powers".to_string(),
        ));
    }

    let n = matrix.nrows();
    if n == 0 {
        return Err(DervflowError::InvalidInput(
            "Matrix must have positive dimensions".to_string(),
        ));
    }

    if power == 0 {
        return Ok(DMatrix::<f64>::identity(n, n));
    }

    let mut result = DMatrix::<f64>::identity(n, n);
    let mut base = matrix.clone();
    let mut exp = power;

    while exp > 0 {
        if exp & 1 == 1 {
            result = &result * &base;
        }
        exp >>= 1;
        if exp > 0 {
            base = &base * &base;
        }
    }

    Ok(result)
}

/// Estimate the condition number of a matrix (1-norm based)
///
/// Condition number = ||A|| * ||A^-1||
/// A high condition number indicates the matrix is ill-conditioned
fn estimate_condition_number(matrix: &DMatrix<f64>) -> f64 {
    // Compute 1-norm of matrix
    let norm_a = matrix_1_norm(matrix);

    // Try to compute inverse
    if let Some(inv) = matrix.clone().try_inverse() {
        let norm_inv = matrix_1_norm(&inv);
        norm_a * norm_inv
    } else {
        f64::INFINITY
    }
}

/// Compute the 1-norm of a matrix (maximum absolute column sum)
fn matrix_1_norm(matrix: &DMatrix<f64>) -> f64 {
    let mut max_sum: f64 = 0.0;

    for j in 0..matrix.ncols() {
        let col_sum: f64 = (0..matrix.nrows()).map(|i| matrix[(i, j)].abs()).sum();
        max_sum = max_sum.max(col_sum);
    }

    max_sum
}

fn matrix_inf_norm(matrix: &DMatrix<f64>) -> f64 {
    let mut max_sum: f64 = 0.0;

    for i in 0..matrix.nrows() {
        let row_sum: f64 = (0..matrix.ncols()).map(|j| matrix[(i, j)].abs()).sum();
        max_sum = max_sum.max(row_sum);
    }

    max_sum
}

/// Compute a requested matrix norm
pub fn matrix_norm(matrix: &DMatrix<f64>, norm: MatrixNorm) -> Result<f64> {
    if matrix.is_empty() {
        return Ok(0.0);
    }

    match norm {
        MatrixNorm::One => Ok(matrix_1_norm(matrix)),
        MatrixNorm::Infinity => Ok(matrix_inf_norm(matrix)),
        MatrixNorm::Frobenius => Ok(matrix.norm()),
        MatrixNorm::Spectral => {
            let svd = matrix.clone().svd(false, false);
            Ok(svd.singular_values.iter().copied().fold(0.0_f64, f64::max))
        }
    }
}

/// Compute the condition number of a matrix for a requested norm
pub fn matrix_condition_number(matrix: &DMatrix<f64>, norm: MatrixNorm) -> Result<f64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square to compute the condition number".to_string(),
        ));
    }

    if matrix.is_empty() {
        return Ok(0.0);
    }

    match norm {
        MatrixNorm::Spectral => {
            let svd = matrix.clone().svd(false, false);
            let mut max_sv = 0.0_f64;

            for &sv in svd.singular_values.iter() {
                max_sv = max_sv.max(sv);
            }

            if !max_sv.is_finite() {
                return Err(DervflowError::NumericalError(
                    "Matrix has non-finite singular values; condition number is undefined"
                        .to_string(),
                ));
            }

            let zero_threshold = f64::EPSILON * max_sv.max(1.0);
            let mut min_sv = f64::INFINITY;

            for &sv in svd.singular_values.iter() {
                if sv <= zero_threshold {
                    return Ok(f64::INFINITY);
                }
                min_sv = min_sv.min(sv);
            }

            Ok(max_sv / min_sv)
        }
        other => {
            let inv = matrix_inverse(matrix, false)?;
            let norm_a = matrix_norm(matrix, other)?;
            let norm_inv = matrix_norm(&inv, other)?;
            if norm_inv == 0.0 {
                return Err(DervflowError::NumericalError(
                    "Matrix inverse has zero norm; condition number is undefined".to_string(),
                ));
            }
            Ok(norm_a * norm_inv)
        }
    }
}

/// Solve a linear system Ax = b using LU decomposition
///
/// # Arguments
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
///
/// # Returns
/// Solution vector x
pub fn solve_linear_system(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>> {
    if a.nrows() != a.ncols() {
        return Err(DervflowError::InvalidInput(
            "Coefficient matrix must be square".to_string(),
        ));
    }

    if a.nrows() != b.len() {
        return Err(DervflowError::InvalidInput(
            "Dimensions of matrix and vector do not match".to_string(),
        ));
    }

    let lu = a.clone().lu();

    if !lu.is_invertible() {
        return Err(DervflowError::NumericalError(
            "Coefficient matrix is singular".to_string(),
        ));
    }

    lu.solve(b)
        .ok_or_else(|| DervflowError::NumericalError("Failed to solve linear system".to_string()))
}

/// Solve an over- or under-determined linear system in the least squares sense
pub fn solve_least_squares(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>> {
    if a.nrows() == 0 || a.ncols() == 0 {
        return Err(DervflowError::InvalidInput(
            "Coefficient matrix must have positive dimensions".to_string(),
        ));
    }

    if a.nrows() != b.len() {
        return Err(DervflowError::InvalidInput(
            "Dimensions of matrix and vector do not match".to_string(),
        ));
    }

    let svd = a.clone().svd(true, true);
    let eps = f64::EPSILON
        * (a.nrows().max(a.ncols()) as f64)
        * svd.singular_values.iter().cloned().fold(0.0, f64::max);

    let rhs = DMatrix::from_column_slice(b.len(), 1, b.as_slice());
    let solution = svd
        .solve(&rhs, eps)
        .map_err(|msg| DervflowError::NumericalError(msg.to_string()))?
        .column(0)
        .into();

    Ok(solution)
}

/// Compute the LU decomposition (PA = LU) of a square matrix
pub fn lu_decomposition(
    matrix: &DMatrix<f64>,
) -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square for LU decomposition".to_string(),
        ));
    }

    if n == 0 {
        return Err(DervflowError::InvalidInput(
            "Matrix must have positive dimensions".to_string(),
        ));
    }

    let lu = matrix.clone().lu();

    if !lu.is_invertible() {
        return Err(DervflowError::NumericalError(
            "Matrix is singular and LU decomposition is not unique".to_string(),
        ));
    }

    let l = lu.l();
    let u = lu.u();
    let mut p = DMatrix::<f64>::identity(n, n);
    lu.p().permute_rows(&mut p);

    Ok((l, u, p))
}

/// Compute eigenvalues and eigenvectors of a symmetric matrix
///
/// # Arguments
/// * `matrix` - Symmetric matrix (n x n)
///
/// # Returns
/// Tuple of (eigenvalues, eigenvectors) where eigenvectors are column vectors
pub fn eigen_decomposition(matrix: &DMatrix<f64>) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let n = matrix.nrows();

    if n != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square for eigendecomposition".to_string(),
        ));
    }

    // Check symmetry
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-10 {
                return Err(DervflowError::InvalidInput(
                    "Matrix must be symmetric for eigendecomposition".to_string(),
                ));
            }
        }
    }

    let eigen = matrix.clone().symmetric_eigen();

    Ok((eigen.eigenvalues, eigen.eigenvectors))
}

/// Compute the QR decomposition of a matrix, returning orthogonal Q and upper triangular R
pub fn qr_decomposition(matrix: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>)> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(DervflowError::InvalidInput(
            "Matrix must have positive dimensions".to_string(),
        ));
    }

    let qr = matrix.clone().qr();
    let (q, r) = qr.unpack();
    Ok((q, r))
}

/// Compute the singular value decomposition of a matrix
pub fn svd_decomposition(
    matrix: &DMatrix<f64>,
) -> Result<(DMatrix<f64>, DVector<f64>, DMatrix<f64>)> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(DervflowError::InvalidInput(
            "Matrix must have positive dimensions".to_string(),
        ));
    }

    let svd = matrix.clone().svd(true, true);
    let u = svd.u.ok_or_else(|| {
        DervflowError::NumericalError("Failed to compute left singular vectors".to_string())
    })?;
    let v_t = svd.v_t.ok_or_else(|| {
        DervflowError::NumericalError("Failed to compute right singular vectors".to_string())
    })?;

    Ok((u, svd.singular_values, v_t))
}

/// Compute the Moore-Penrose pseudo-inverse of a matrix using SVD
pub fn pseudo_inverse(matrix: &DMatrix<f64>, tolerance: Option<f64>) -> Result<DMatrix<f64>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(DervflowError::InvalidInput(
            "Matrix must have positive dimensions".to_string(),
        ));
    }

    let (u, singular_values, v_t) = svd_decomposition(matrix)?;

    let max_sv = singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let eps = f64::EPSILON * (matrix.nrows().max(matrix.ncols()) as f64) * max_sv;
    let tol = tolerance.unwrap_or(eps);

    let mut sigma_inv = DMatrix::<f64>::zeros(singular_values.len(), singular_values.len());
    for (i, &sigma) in singular_values.iter().enumerate() {
        if sigma > tol {
            sigma_inv[(i, i)] = 1.0 / sigma;
        }
    }

    let v = v_t.transpose();
    Ok(v * sigma_inv * u.transpose())
}

/// Estimate the numerical rank of a matrix using its singular values
pub fn matrix_rank(matrix: &DMatrix<f64>, tolerance: Option<f64>) -> Result<usize> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Ok(0);
    }

    let svd = matrix.clone().svd(false, false);
    let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let eps = f64::EPSILON * (matrix.nrows().max(matrix.ncols()) as f64) * max_sv;
    let tol = tolerance.unwrap_or(eps);

    Ok(svd
        .singular_values
        .iter()
        .filter(|&&sigma| sigma > tol)
        .count())
}

/// Check if a matrix is positive definite
///
/// A matrix is positive definite if all eigenvalues are positive
pub fn is_positive_definite(matrix: &DMatrix<f64>) -> bool {
    match eigen_decomposition(matrix) {
        Ok((eigenvalues, _)) => eigenvalues.iter().all(|&e| e > 0.0),
        Err(_) => false,
    }
}

/// Make a correlation matrix positive definite by adjusting eigenvalues
///
/// This is useful when a correlation matrix becomes non-positive definite
/// due to numerical errors or missing data
pub fn nearest_positive_definite(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DervflowError::InvalidInput(
            "Matrix must be square to compute the nearest positive definite approximation"
                .to_string(),
        ));
    }

    if matrix.is_empty() {
        return Ok(matrix.clone());
    }

    // Symmetrize the input to guard against minor asymmetries.
    let symmetrized = 0.5 * (matrix + matrix.transpose());

    let (mut eigenvalues, eigenvectors) = eigen_decomposition(&symmetrized)?;

    // Set negative eigenvalues to small positive value
    let min_eigenvalue = 1e-8;
    for val in eigenvalues.iter_mut() {
        if *val < min_eigenvalue {
            *val = min_eigenvalue;
        }
    }

    // Reconstruct matrix: A = V * D * V^T
    let d = DMatrix::from_diagonal(&eigenvalues);
    let result = &eigenvectors * &d * eigenvectors.transpose();

    // Rescale to ensure diagonal is 1 (for correlation matrices)
    let n = result.nrows();
    let mut scaled = result.clone();

    for i in 0..n {
        for j in 0..n {
            let scale = (result[(i, i)] * result[(j, j)]).sqrt();
            if scale > 1e-15 {
                scaled[(i, j)] = result[(i, j)] / scale;
            }
        }
    }

    // Enforce symmetry in case of accumulated rounding error.
    Ok(0.5 * (&scaled + scaled.transpose()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cholesky_decomposition() {
        // Create a positive definite matrix
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);

        let l = cholesky_decomposition(&a).unwrap();

        // Verify A = L * L^T
        let reconstructed = &l * l.transpose();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[(i, j)], reconstructed[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Create a non-positive definite matrix
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);

        let result = cholesky_decomposition(&a);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_multiply() {
        let a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DMatrix::from_row_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let c = matrix_multiply(&a, &b).unwrap();

        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert_relative_eq!(c[(0, 0)], 58.0, epsilon = 1e-10);
        assert_relative_eq!(c[(0, 1)], 64.0, epsilon = 1e-10);
        assert_relative_eq!(c[(1, 0)], 139.0, epsilon = 1e-10);
        assert_relative_eq!(c[(1, 1)], 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0]);

        let inv = matrix_inverse(&a, true).unwrap();

        // Verify A * A^-1 = I
        let identity = &a * &inv;

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(identity[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_determinant_and_trace() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let det = matrix_determinant(&a).unwrap();
        let trace = matrix_trace(&a).unwrap();

        assert_relative_eq!(det, 11.0, epsilon = 1e-12);
        assert_relative_eq!(trace, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_matrix_power() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let a3 = matrix_power(&a, 3).unwrap();
        let expected = &a * &a * &a;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(a3[(i, j)], expected[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve: 2x + y = 5, x + 3y = 7
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let b = DVector::from_vec(vec![5.0, 7.0]);

        let x = solve_linear_system(&a, &b).unwrap();

        assert_relative_eq!(x[0], 1.6, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.8, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_least_squares() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0, 2.0]);

        let x = solve_least_squares(&a, &b).unwrap();

        let normal = a.transpose() * &a;
        let rhs = a.transpose() * b;
        let expected = normal.lu().solve(&rhs).unwrap();

        for i in 0..x.len() {
            assert_relative_eq!(x[i], expected[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_eigen_decomposition() {
        // Symmetric matrix
        let a = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0]);

        let (eigenvalues, eigenvectors) = eigen_decomposition(&a).unwrap();

        // Verify A * v = λ * v for each eigenpair
        for i in 0..3 {
            let v = eigenvectors.column(i);
            let av = &a * v;
            let lambda_v = eigenvalues[i] * v;

            for j in 0..3 {
                assert_relative_eq!(av[j], lambda_v[j], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_decomposition() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let (q, r) = qr_decomposition(&a).unwrap();

        let reconstructed = &q * &r;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }

        let qtq = q.transpose() * &q;
        let identity = DMatrix::<f64>::identity(qtq.nrows(), qtq.ncols());
        for i in 0..qtq.nrows() {
            for j in 0..qtq.ncols() {
                assert_relative_eq!(qtq[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_svd_decomposition() {
        let a = DMatrix::from_row_slice(2, 3, &[3.0, 1.0, 1.0, -1.0, 3.0, 1.0]);
        let (u, singular, v_t) = svd_decomposition(&a).unwrap();

        let sigma = DMatrix::from_diagonal(&singular);
        let reconstructed = &u * &sigma * &v_t;

        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_pseudo_inverse() {
        let a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 4.0]);
        let pinv = pseudo_inverse(&a, None).unwrap();

        let reconstructed = &a * &pinv * &a;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_rank() {
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0]);
        let rank = matrix_rank(&a, None).unwrap();
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_matrix_norms() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, -2.0, 3.0, 4.0]);

        let one_norm = matrix_norm(&a, MatrixNorm::One).unwrap();
        assert_relative_eq!(one_norm, 6.0, epsilon = 1e-12);

        let inf_norm = matrix_norm(&a, MatrixNorm::Infinity).unwrap();
        assert_relative_eq!(inf_norm, 7.0, epsilon = 1e-12);

        let fro_norm = matrix_norm(&a, MatrixNorm::Frobenius).unwrap();
        assert_relative_eq!(fro_norm, f64::sqrt(1.0 + 4.0 + 9.0 + 16.0), epsilon = 1e-12);

        let spectral = matrix_norm(&a, MatrixNorm::Spectral).unwrap();
        assert!(spectral >= 0.0);
    }

    #[test]
    fn test_matrix_condition_number() {
        let a = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0]));

        let cond_spectral = matrix_condition_number(&a, MatrixNorm::Spectral).unwrap();
        assert_relative_eq!(cond_spectral, 2.0, epsilon = 1e-12);

        let cond_one = matrix_condition_number(&a, MatrixNorm::One).unwrap();
        assert_relative_eq!(cond_one, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_matrix_exponential() {
        let a = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0]));
        let exp_a = matrix_exponential(&a).unwrap();

        assert_relative_eq!(exp_a[(0, 0)], (1.0f64).exp(), epsilon = 1e-12);
        assert_relative_eq!(exp_a[(1, 1)], (2.0f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_lu_decomposition() {
        let a = DMatrix::from_row_slice(3, 3, &[2.0, 1.0, 1.0, 4.0, -6.0, 0.0, -2.0, 7.0, 2.0]);
        let (l, u, p) = lu_decomposition(&a).unwrap();

        let reconstructed = l * u;
        let permuted = p * a;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[(i, j)], permuted[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_is_positive_definite() {
        // Positive definite matrix
        let pd = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        assert!(is_positive_definite(&pd));

        // Non-positive definite matrix
        let npd = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(!is_positive_definite(&npd));
    }

    #[test]
    fn test_nearest_positive_definite() {
        // Create a correlation matrix that's not quite positive definite
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 0.9, 1.0]);

        let pd = nearest_positive_definite(&a).unwrap();

        // Check that result is positive definite
        assert!(is_positive_definite(&pd));

        // Check that diagonal is 1 (correlation matrix property)
        for i in 0..3 {
            assert_relative_eq!(pd[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_correlate_samples() {
        // Create a simple correlation matrix
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);

        // Create independent samples
        let samples = DMatrix::from_row_slice(2, 100, &vec![0.0; 200]);

        let correlated = correlate_samples(&corr, &samples).unwrap();

        assert_eq!(correlated.nrows(), 2);
        assert_eq!(correlated.ncols(), 100);
    }
}
