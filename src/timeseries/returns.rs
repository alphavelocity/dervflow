// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Return calculations
//!
//! This module provides functions for calculating various types of returns
//! from price data, including simple returns, log returns, and rolling returns.

use crate::common::error::{DervflowError, Result};

/// Type of return calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnType {
    /// Simple returns: (P_t - P_{t-1}) / P_{t-1}
    Simple,
    /// Log returns: ln(P_t / P_{t-1})
    Log,
    /// Continuously compounded returns (same as log returns)
    Continuous,
}

/// Calculate returns from a price series
///
/// # Arguments
///
/// * `prices` - Slice of price data (must have at least 2 elements)
/// * `return_type` - Type of return to calculate
///
/// # Returns
///
/// Vector of returns (length = prices.len() - 1)
///
/// # Errors
///
/// Returns error if:
/// - Price series has fewer than 2 elements
/// - Any price is non-positive
/// - Calculation results in NaN or infinity
pub fn calculate_returns(prices: &[f64], return_type: ReturnType) -> Result<Vec<f64>> {
    if prices.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "Price series must have at least 2 elements".to_string(),
        ));
    }

    // Validate all prices are positive
    for (i, &price) in prices.iter().enumerate() {
        if price <= 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Price at index {} must be positive, got {}",
                i, price
            )));
        }
    }

    let mut returns = Vec::with_capacity(prices.len() - 1);

    for i in 1..prices.len() {
        let ret = match return_type {
            ReturnType::Simple => (prices[i] - prices[i - 1]) / prices[i - 1],
            ReturnType::Log | ReturnType::Continuous => (prices[i] / prices[i - 1]).ln(),
        };

        if !ret.is_finite() {
            return Err(DervflowError::NumericalError(format!(
                "Return calculation resulted in non-finite value at index {}",
                i
            )));
        }

        returns.push(ret);
    }

    Ok(returns)
}

/// Calculate rolling returns over a specified window
///
/// # Arguments
///
/// * `prices` - Slice of price data
/// * `window` - Window size for rolling calculation (must be >= 2)
/// * `return_type` - Type of return to calculate
///
/// # Returns
///
/// Vector of rolling returns (length = prices.len() - window)
///
/// # Errors
///
/// Returns error if:
/// - Window size is less than 2
/// - Price series has fewer elements than window size
/// - Any price is non-positive
pub fn calculate_rolling_returns(
    prices: &[f64],
    window: usize,
    return_type: ReturnType,
) -> Result<Vec<f64>> {
    if window < 2 {
        return Err(DervflowError::InvalidInput(
            "Window size must be at least 2".to_string(),
        ));
    }

    if prices.len() < window {
        return Err(DervflowError::InvalidInput(format!(
            "Price series length ({}) must be at least window size ({})",
            prices.len(),
            window
        )));
    }

    // Validate all prices are positive
    for (i, &price) in prices.iter().enumerate() {
        if price <= 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Price at index {} must be positive, got {}",
                i, price
            )));
        }
    }

    let mut rolling_returns = Vec::with_capacity(prices.len() - window + 1);

    for i in window..=prices.len() {
        let start_price = prices[i - window];
        let end_price = prices[i - 1];

        let ret = match return_type {
            ReturnType::Simple => (end_price - start_price) / start_price,
            ReturnType::Log | ReturnType::Continuous => (end_price / start_price).ln(),
        };

        if !ret.is_finite() {
            return Err(DervflowError::NumericalError(format!(
                "Rolling return calculation resulted in non-finite value at index {}",
                i
            )));
        }

        rolling_returns.push(ret);
    }

    Ok(rolling_returns)
}

/// Calculate cumulative returns from a return series
///
/// # Arguments
///
/// * `returns` - Slice of return data
/// * `return_type` - Type of returns in the input
///
/// # Returns
///
/// Vector of cumulative returns (same length as input)
pub fn calculate_cumulative_returns(returns: &[f64], return_type: ReturnType) -> Result<Vec<f64>> {
    if returns.is_empty() {
        return Ok(Vec::new());
    }

    let mut cumulative = Vec::with_capacity(returns.len());

    match return_type {
        ReturnType::Simple => {
            let mut cum_ret = 1.0;
            for &ret in returns {
                cum_ret *= 1.0 + ret;
                cumulative.push(cum_ret - 1.0);
            }
        }
        ReturnType::Log | ReturnType::Continuous => {
            let mut cum_ret = 0.0;
            for &ret in returns {
                cum_ret += ret;
                cumulative.push(cum_ret);
            }
        }
    }

    Ok(cumulative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = calculate_returns(&prices, ReturnType::Simple).unwrap();

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.05).abs() < 1e-10); // (105-100)/100 = 0.05
        assert!((returns[1] - (-0.019047619)).abs() < 1e-6); // (103-105)/105 ≈ -0.019
        assert!((returns[2] - 0.048543689).abs() < 1e-6); // (108-103)/103 ≈ 0.0485
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = calculate_returns(&prices, ReturnType::Log).unwrap();

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (103.0_f64 / 105.0).ln()).abs() < 1e-10);
        assert!((returns[2] - (108.0_f64 / 103.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_returns_same_as_log() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let log_returns = calculate_returns(&prices, ReturnType::Log).unwrap();
        let cont_returns = calculate_returns(&prices, ReturnType::Continuous).unwrap();

        assert_eq!(log_returns.len(), cont_returns.len());
        for (log_ret, cont_ret) in log_returns.iter().zip(cont_returns.iter()) {
            assert!((log_ret - cont_ret).abs() < 1e-15);
        }
    }

    #[test]
    fn test_returns_insufficient_data() {
        let prices = vec![100.0];
        let result = calculate_returns(&prices, ReturnType::Simple);
        assert!(result.is_err());
    }

    #[test]
    fn test_returns_negative_price() {
        let prices = vec![100.0, -105.0, 103.0];
        let result = calculate_returns(&prices, ReturnType::Simple);
        assert!(result.is_err());
    }

    #[test]
    fn test_returns_zero_price() {
        let prices = vec![100.0, 0.0, 103.0];
        let result = calculate_returns(&prices, ReturnType::Simple);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0, 110.0];
        let rolling = calculate_rolling_returns(&prices, 3, ReturnType::Simple).unwrap();

        assert_eq!(rolling.len(), 3); // 5 - 3 + 1 = 3
        assert!((rolling[0] - 0.03).abs() < 1e-10); // (103-100)/100 = 0.03
        assert!((rolling[1] - (3.0 / 105.0)).abs() < 1e-10); // (108-105)/105 = 0.02857...
        assert!((rolling[2] - (7.0 / 103.0)).abs() < 1e-10); // (110-103)/103 = 0.06796...
    }

    #[test]
    fn test_rolling_returns_correct_calculation() {
        let prices = vec![100.0, 110.0, 120.0, 130.0];
        let rolling = calculate_rolling_returns(&prices, 2, ReturnType::Simple).unwrap();

        assert_eq!(rolling.len(), 3);
        assert!((rolling[0] - 0.10).abs() < 1e-10); // (110-100)/100 = 0.10
        assert!((rolling[1] - (10.0 / 110.0)).abs() < 1e-10); // (120-110)/110 = 0.0909...
        assert!((rolling[2] - (10.0 / 120.0)).abs() < 1e-10); // (130-120)/120 = 0.0833...
    }

    #[test]
    fn test_rolling_returns_window_2() {
        let prices = vec![100.0, 110.0, 121.0];
        let rolling = calculate_rolling_returns(&prices, 2, ReturnType::Simple).unwrap();

        assert_eq!(rolling.len(), 2);
        assert!((rolling[0] - 0.10).abs() < 1e-10); // (110-100)/100
        assert!((rolling[1] - 0.10).abs() < 1e-10); // (121-110)/110
    }

    #[test]
    fn test_rolling_returns_invalid_window() {
        let prices = vec![100.0, 105.0, 103.0];
        let result = calculate_rolling_returns(&prices, 1, ReturnType::Simple);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_returns_insufficient_data() {
        let prices = vec![100.0, 105.0];
        let result = calculate_rolling_returns(&prices, 3, ReturnType::Simple);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumulative_simple_returns() {
        let returns = vec![0.1, -0.05, 0.08];
        let cumulative = calculate_cumulative_returns(&returns, ReturnType::Simple).unwrap();

        assert_eq!(cumulative.len(), 3);
        assert!((cumulative[0] - 0.1).abs() < 1e-10); // 1.1 - 1 = 0.1
        assert!((cumulative[1] - 0.045).abs() < 1e-10); // 1.1 * 0.95 - 1 = 0.045
        assert!((cumulative[2] - 0.1286).abs() < 1e-4); // 1.045 * 1.08 - 1 ≈ 0.1286
    }

    #[test]
    fn test_cumulative_log_returns() {
        let returns = vec![0.1, -0.05, 0.08];
        let cumulative = calculate_cumulative_returns(&returns, ReturnType::Log).unwrap();

        assert_eq!(cumulative.len(), 3);
        assert!((cumulative[0] - 0.1).abs() < 1e-10);
        assert!((cumulative[1] - 0.05).abs() < 1e-10); // 0.1 - 0.05
        assert!((cumulative[2] - 0.13).abs() < 1e-10); // 0.1 - 0.05 + 0.08
    }

    #[test]
    fn test_cumulative_empty_returns() {
        let returns: Vec<f64> = vec![];
        let cumulative = calculate_cumulative_returns(&returns, ReturnType::Simple).unwrap();
        assert!(cumulative.is_empty());
    }
}
