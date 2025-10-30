// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

pub(crate) fn validate_non_empty(data: &[f64], context: &str) -> Result<()> {
    if data.is_empty() {
        Err(DervflowError::InvalidInput(format!(
            "Input slice for {} must not be empty",
            context
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_min_length(data: &[f64], min_len: usize, context: &str) -> Result<()> {
    if data.len() < min_len {
        Err(DervflowError::InvalidInput(format!(
            "Input slice for {} must contain at least {} elements",
            context, min_len
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_finite(data: &[f64], context: &str) -> Result<()> {
    if let Some(value) = data.iter().find(|x| !x.is_finite()) {
        Err(DervflowError::InvalidInput(format!(
            "Input slice for {} contains non-finite value {}",
            context, value
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_positive(data: &[f64], context: &str) -> Result<()> {
    if let Some(value) = data.iter().find(|x| **x <= 0.0) {
        Err(DervflowError::InvalidInput(format!(
            "Input slice for {} must contain strictly positive values, found {}",
            context, value
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_same_length(a: &[f64], b: &[f64], context: &str) -> Result<()> {
    if a.len() != b.len() {
        Err(DervflowError::InvalidInput(format!(
            "Input slices for {} must have the same length",
            context
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_window_size(len: usize, window_size: usize, context: &str) -> Result<()> {
    if window_size == 0 || window_size > len {
        Err(DervflowError::InvalidInput(format!(
            "Window size for {} must be in the range [1, {}]",
            context, len
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_dimension(data: &[f64], dimension: usize, context: &str) -> Result<()> {
    if data.len() != dimension {
        Err(DervflowError::InvalidInput(format!(
            "Input slice for {} must have length {}",
            context, dimension
        )))
    } else {
        Ok(())
    }
}
