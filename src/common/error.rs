// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Error types and handling for the dervflow library
//!
//! This module defines the error types used throughout dervflow and provides
//! conversions to Python exceptions for the PyO3 bindings.
//!
//! # Error Types
//!
//! The main error type is [`DervflowError`], which covers:
//! - Invalid input parameters
//! - Convergence failures in numerical methods
//! - Numerical computation errors
//! - Optimization infeasibility
//! - Data-related errors
//!
//! # Examples
//!
//! ```rust
//! use dervflow::common::error::{DervflowError, Result};
//!
//! fn validate_price(price: f64) -> Result<f64> {
//!     if price <= 0.0 {
//!         Err(DervflowError::InvalidInput(
//!             format!("Price must be positive, got {}", price)
//!         ))
//!     } else {
//!         Ok(price)
//!     }
//! }
//!
//! assert!(validate_price(100.0).is_ok());
//! assert!(validate_price(-10.0).is_err());
//! ```

#[cfg(feature = "python")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::error::Error as StdError;
use std::fmt;

/// Result type alias for dervflow operations
///
/// This is a convenience type alias that uses [`DervflowError`] as the error type.
///
/// # Examples
///
/// ```rust
/// use dervflow::common::error::{Result, DervflowError};
///
/// fn compute_something(x: f64) -> Result<f64> {
///     if x < 0.0 {
///         Err(DervflowError::InvalidInput("x must be non-negative".to_string()))
///     } else {
///         Ok(x.sqrt())
///     }
/// }
/// ```
pub type Result<T> = std::result::Result<T, DervflowError>;

/// Main error type for the dervflow library
///
/// This enum represents all possible errors that can occur in dervflow operations.
/// Each variant provides context-specific information about the error.
///
/// # Variants
///
/// * `InvalidInput` - Input parameters are invalid or out of range
/// * `ConvergenceFailure` - Numerical method failed to converge within limits
/// * `NumericalError` - Numerical computation error (overflow, NaN, etc.)
/// * `OptimizationInfeasible` - Optimization problem has no solution
/// * `DataError` - Data-related error (missing, malformed, etc.)
///
/// # Examples
///
/// ```rust
/// use dervflow::common::error::DervflowError;
///
/// // Invalid input error
/// let err = DervflowError::InvalidInput("spot price must be positive".to_string());
/// println!("{}", err);
///
/// // Convergence failure with diagnostics
/// let err = DervflowError::ConvergenceFailure {
///     iterations: 100,
///     error: 0.001,
/// };
/// println!("{}", err);
/// ```
#[derive(Debug, Clone)]
pub enum DervflowError {
    /// Invalid input parameters provided
    ///
    /// This error occurs when input parameters fail validation checks,
    /// such as negative prices, invalid dates, or out-of-range values.
    InvalidInput(String),

    /// Numerical method failed to converge
    ///
    /// This error occurs when iterative numerical methods (root finding,
    /// optimization, etc.) fail to converge within the specified tolerance
    /// and maximum iterations.
    ///
    /// # Fields
    ///
    /// * `iterations` - Number of iterations attempted before failure
    /// * `error` - Final error value at termination
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
        /// Final error value
        error: f64,
    },

    /// Numerical computation error (overflow, underflow, NaN, etc.)
    ///
    /// This error occurs when numerical computations produce invalid results
    /// such as NaN, infinity, or encounter numerical instability.
    NumericalError(String),

    /// Optimization problem is infeasible or unbounded
    ///
    /// This error occurs when an optimization problem has no feasible solution
    /// (constraints are contradictory) or is unbounded (objective can be
    /// improved indefinitely).
    OptimizationInfeasible(String),

    /// Data-related error (missing data, invalid format, etc.)
    ///
    /// This error occurs when working with data that is missing, malformed,
    /// or doesn't meet expected format requirements.
    DataError(String),
}

impl fmt::Display for DervflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DervflowError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
            DervflowError::ConvergenceFailure { iterations, error } => {
                write!(
                    f,
                    "Convergence failure: failed to converge after {} iterations (final error: {:.6e})",
                    iterations, error
                )
            }
            DervflowError::NumericalError(msg) => {
                write!(f, "Numerical error: {}", msg)
            }
            DervflowError::OptimizationInfeasible(msg) => {
                write!(f, "Optimization infeasible: {}", msg)
            }
            DervflowError::DataError(msg) => {
                write!(f, "Data error: {}", msg)
            }
        }
    }
}

impl StdError for DervflowError {}

// Convert DervflowError to PyErr for Python exception handling
#[cfg(feature = "python")]
impl From<DervflowError> for PyErr {
    fn from(err: DervflowError) -> PyErr {
        match err {
            DervflowError::InvalidInput(msg) => {
                PyValueError::new_err(format!("Invalid input: {}", msg))
            }
            DervflowError::ConvergenceFailure { iterations, error } => {
                PyRuntimeError::new_err(format!(
                    "Convergence failure: failed to converge after {} iterations (final error: {:.6e})",
                    iterations, error
                ))
            }
            DervflowError::NumericalError(msg) => {
                PyRuntimeError::new_err(format!("Numerical error: {}", msg))
            }
            DervflowError::OptimizationInfeasible(msg) => {
                PyRuntimeError::new_err(format!("Optimization infeasible: {}", msg))
            }
            DervflowError::DataError(msg) => PyValueError::new_err(format!("Data error: {}", msg)),
        }
    }
}

// Python exception creation helpers
// Note: PyO3 with abi3 doesn't support custom exception subclassing,
// so we use standard Python exceptions with descriptive messages

/// Create a Python exception for invalid input
#[cfg(feature = "python")]
pub fn create_invalid_input_error(msg: &str) -> PyErr {
    PyValueError::new_err(format!("InvalidInputError: {}", msg))
}

/// Create a Python exception for convergence failure
#[cfg(feature = "python")]
pub fn create_convergence_error(iterations: usize, error: f64) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ConvergenceError: failed to converge after {} iterations (final error: {:.6e})",
        iterations, error
    ))
}

/// Create a Python exception for optimization infeasibility
#[cfg(feature = "python")]
pub fn create_optimization_error(msg: &str) -> PyErr {
    PyRuntimeError::new_err(format!("OptimizationError: {}", msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_input_error_display() {
        let err = DervflowError::InvalidInput("spot price must be positive".to_string());
        assert_eq!(
            format!("{}", err),
            "Invalid input: spot price must be positive"
        );
    }

    #[test]
    fn test_convergence_failure_display() {
        let err = DervflowError::ConvergenceFailure {
            iterations: 100,
            error: 0.001,
        };
        let display = format!("{}", err);
        assert!(display.contains("Convergence failure"));
        assert!(display.contains("100 iterations"));
        assert!(display.contains("1.000000e-3"));
    }

    #[test]
    fn test_numerical_error_display() {
        let err = DervflowError::NumericalError("division by zero".to_string());
        assert_eq!(format!("{}", err), "Numerical error: division by zero");
    }

    #[test]
    fn test_optimization_infeasible_display() {
        let err = DervflowError::OptimizationInfeasible("constraints are contradictory".to_string());
        assert_eq!(
            format!("{}", err),
            "Optimization infeasible: constraints are contradictory"
        );
    }

    #[test]
    fn test_data_error_display() {
        let err = DervflowError::DataError("missing required field".to_string());
        assert_eq!(format!("{}", err), "Data error: missing required field");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DervflowError>();
    }
}
