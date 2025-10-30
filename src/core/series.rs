// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

use super::validation::{validate_finite, validate_min_length, validate_non_empty};

pub fn cumulative_sum(data: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_sum")?;
    validate_finite(data, "cumulative_sum")?;

    let mut result = Vec::with_capacity(data.len());
    let mut running = 0.0;
    for &value in data {
        running += value;
        if !running.is_finite() {
            return Err(DervflowError::NumericalError(
                "Cumulative sum produced non-finite value".to_string(),
            ));
        }
        result.push(running);
    }
    Ok(result)
}

pub fn cumulative_product(data: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_product")?;
    validate_finite(data, "cumulative_product")?;

    let mut result = Vec::with_capacity(data.len());
    let mut running = 1.0;
    for &value in data {
        running *= value;
        if !running.is_finite() {
            return Err(DervflowError::NumericalError(
                "Cumulative product produced non-finite value".to_string(),
            ));
        }
        result.push(running);
    }
    Ok(result)
}

pub fn cumulative_max(data: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_max")?;
    validate_finite(data, "cumulative_max")?;

    let mut result = Vec::with_capacity(data.len());
    let mut current_max = f64::NEG_INFINITY;
    for &value in data {
        current_max = current_max.max(value);
        result.push(current_max);
    }
    Ok(result)
}

pub fn cumulative_min(data: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_min")?;
    validate_finite(data, "cumulative_min")?;

    let mut result = Vec::with_capacity(data.len());
    let mut current_min = f64::INFINITY;
    for &value in data {
        current_min = current_min.min(value);
        result.push(current_min);
    }
    Ok(result)
}

pub fn first_difference(data: &[f64]) -> Result<Vec<f64>> {
    validate_min_length(data, 2, "first_difference")?;
    validate_finite(data, "first_difference")?;

    Ok(data
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cumulative_operations() {
        let data = [1.0, 2.0, 3.0];
        assert_eq!(cumulative_sum(&data).unwrap(), vec![1.0, 3.0, 6.0]);
        assert_eq!(cumulative_product(&data).unwrap(), vec![1.0, 2.0, 6.0]);
        assert_eq!(cumulative_max(&data).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cumulative_min(&data).unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_first_difference() {
        let data = [1.0, 4.0, 9.0, 16.0];
        let diff = first_difference(&data).unwrap();
        assert_abs_diff_eq!(diff[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diff[1], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diff[2], 7.0, epsilon = 1e-12);
    }
}
