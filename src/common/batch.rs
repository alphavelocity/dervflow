// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Batch processing utilities for performance optimization
//!
//! This module provides batch processing capabilities for common operations
//! to improve performance through parallelization and reduced overhead.

use crate::common::error::{DervflowError, Result};
use crate::common::types::OptionParams;
use rayon::prelude::*;

/// Batch process a function over multiple option parameters in parallel
///
/// This function applies a pricing or calculation function to multiple option
/// parameters in parallel using Rayon, providing significant performance benefits
/// for large batches.
///
/// # Arguments
/// * `params_batch` - Slice of option parameters to process
/// * `func` - Function to apply to each parameter set
///
/// # Returns
/// * `Ok(Vec<T>)` - Vector of results in the same order as input
/// * `Err(DervflowError)` - If any calculation fails
///
/// # Examples
/// ```
/// use dervflow::common::batch::batch_process;
/// use dervflow::options::analytical::black_scholes_price;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = vec![
///     OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call),
///     OptionParams::new(100.0, 105.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call),
/// ];
///
/// let prices = batch_process(&params, &black_scholes_price).unwrap();
/// assert_eq!(prices.len(), 2);
/// ```
pub fn batch_process<F, T>(params_batch: &[OptionParams], func: &F) -> Result<Vec<T>>
where
    F: Fn(&OptionParams) -> Result<T> + Sync,
    T: Send,
{
    if params_batch.is_empty() {
        return Ok(Vec::new());
    }

    if params_batch.len() == 1 {
        return std::iter::once(&params_batch[0]).map(func).collect();
    }

    // Process in parallel
    let results: Result<Vec<T>> = params_batch.par_iter().map(func).collect();

    results
}

/// Batch process with error handling that collects all results
///
/// Unlike `batch_process`, this function continues processing even if some
/// calculations fail, returning a vector of Results.
///
/// # Arguments
/// * `params_batch` - Slice of option parameters to process
/// * `func` - Function to apply to each parameter set
///
/// # Returns
/// * `Vec<Result<T>>` - Vector of results, one for each input parameter
///
/// # Examples
/// ```
/// use dervflow::common::batch::batch_process_with_errors;
/// use dervflow::options::analytical::black_scholes_price;
/// use dervflow::common::types::{OptionParams, OptionType};
///
/// let params = vec![
///     OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call),
///     OptionParams::new(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call), // Invalid
/// ];
///
/// let results = batch_process_with_errors(&params, &black_scholes_price);
/// assert_eq!(results.len(), 2);
/// assert!(results[0].is_ok());
/// assert!(results[1].is_err());
/// ```
pub fn batch_process_with_errors<F, T>(params_batch: &[OptionParams], func: &F) -> Vec<Result<T>>
where
    F: Fn(&OptionParams) -> Result<T> + Sync,
    T: Send,
{
    if params_batch.is_empty() {
        return Vec::new();
    }

    if params_batch.len() == 1 {
        return std::iter::once(&params_batch[0]).map(func).collect();
    }

    // Process in parallel, collecting all results (including errors)
    params_batch.par_iter().map(func).collect()
}

/// Batch process with a threshold for parallel execution
///
/// For small batches, parallel overhead may exceed benefits. This function
/// uses sequential processing for small batches and parallel for large ones.
///
/// # Arguments
/// * `params_batch` - Slice of option parameters to process
/// * `func` - Function to apply to each parameter set
/// * `parallel_threshold` - Minimum batch size for parallel processing (default: 10)
///
/// # Returns
/// * `Ok(Vec<T>)` - Vector of results in the same order as input
/// * `Err(DervflowError)` - If any calculation fails
pub fn batch_process_adaptive<F, T>(
    params_batch: &[OptionParams],
    func: &F,
    parallel_threshold: Option<usize>,
) -> Result<Vec<T>>
where
    F: Fn(&OptionParams) -> Result<T> + Sync,
    T: Send,
{
    let threshold = parallel_threshold.unwrap_or(10);

    if params_batch.len() < threshold {
        // Sequential processing for small batches
        params_batch.iter().map(func).collect()
    } else {
        // Parallel processing for large batches
        batch_process(params_batch, func)
    }
}

/// Chunk-based batch processing for very large datasets
///
/// Processes data in chunks to balance parallelism with memory usage.
/// Useful for processing millions of options without excessive memory allocation.
///
/// # Arguments
/// * `params_batch` - Slice of option parameters to process
/// * `func` - Function to apply to each parameter set
/// * `chunk_size` - Size of each processing chunk (default: 1000)
///
/// # Returns
/// * `Ok(Vec<T>)` - Vector of results in the same order as input
/// * `Err(DervflowError)` - If any calculation fails
pub fn batch_process_chunked<F, T>(
    params_batch: &[OptionParams],
    func: &F,
    chunk_size: Option<usize>,
) -> Result<Vec<T>>
where
    F: Fn(&OptionParams) -> Result<T> + Sync,
    T: Send,
{
    if params_batch.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = chunk_size.unwrap_or(1000);

    if chunk_size == 0 {
        return Err(DervflowError::InvalidInput(
            "chunk_size must be greater than 0".to_string(),
        ));
    }

    if chunk_size >= params_batch.len() {
        return params_batch.iter().map(func).collect();
    }

    // Process in chunks
    let results: Result<Vec<Vec<T>>> = params_batch
        .par_chunks(chunk_size)
        .map(|chunk| chunk.iter().map(func).collect())
        .collect();

    // Flatten results while preserving order and minimizing reallocations.
    // Total output length equals the input length when all chunk computations succeed.
    results.map(|chunks| {
        let mut flattened = Vec::with_capacity(params_batch.len());
        for mut chunk in chunks {
            flattened.append(&mut chunk);
        }
        flattened
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::OptionType;
    use crate::options::analytical::black_scholes_price;

    fn create_test_params(count: usize) -> Vec<OptionParams> {
        (0..count)
            .map(|i| {
                let strike = 100.0 + i as f64;
                OptionParams::new(100.0, strike, 0.05, 0.0, 0.2, 1.0, OptionType::Call)
            })
            .collect()
    }

    #[test]
    fn test_batch_process_empty() {
        let params: Vec<OptionParams> = vec![];
        let results = batch_process(&params, &black_scholes_price).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_batch_process_single() {
        let params = create_test_params(1);
        let results = batch_process(&params, &black_scholes_price).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0] > 0.0);
    }

    #[test]
    fn test_batch_process_with_errors_single() {
        let params = create_test_params(1);
        let results = batch_process_with_errors(&params, &black_scholes_price);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_batch_process_multiple() {
        let params = create_test_params(10);
        let results = batch_process(&params, &black_scholes_price).unwrap();
        assert_eq!(results.len(), 10);

        // All prices should be positive
        for price in &results {
            assert!(*price > 0.0);
        }

        // Prices should generally decrease as strike increases (for calls)
        for i in 0..results.len() - 1 {
            assert!(results[i] >= results[i + 1]);
        }
    }

    #[test]
    fn test_batch_process_with_errors_all_valid() {
        let params = create_test_params(5);
        let results = batch_process_with_errors(&params, &black_scholes_price);
        assert_eq!(results.len(), 5);

        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_batch_process_with_errors_some_invalid() {
        let mut params = create_test_params(3);
        // Add an invalid parameter (negative spot)
        params.push(OptionParams::new(
            -100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
        ));

        let results = batch_process_with_errors(&params, &black_scholes_price);
        assert_eq!(results.len(), 4);

        // First 3 should be ok
        for result in results.iter().take(3) {
            assert!(result.is_ok());
        }

        // Last one should be error
        assert!(results[3].is_err());
    }

    #[test]
    fn test_batch_process_adaptive_small() {
        let params = create_test_params(5);
        let results = batch_process_adaptive(&params, &black_scholes_price, Some(10)).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_batch_process_adaptive_large() {
        let params = create_test_params(20);
        let results = batch_process_adaptive(&params, &black_scholes_price, Some(10)).unwrap();
        assert_eq!(results.len(), 20);
    }

    #[test]
    fn test_batch_process_chunked() {
        let params = create_test_params(100);
        let results = batch_process_chunked(&params, &black_scholes_price, Some(25)).unwrap();
        assert_eq!(results.len(), 100);

        // All prices should be positive
        for price in &results {
            assert!(*price > 0.0);
        }
    }

    #[test]
    fn test_batch_process_chunked_large() {
        let params = create_test_params(5000);
        let results = batch_process_chunked(&params, &black_scholes_price, Some(1000)).unwrap();
        assert_eq!(results.len(), 5000);
    }

    #[test]
    fn test_batch_process_chunked_zero_chunk_size() {
        let params = create_test_params(3);
        let result = batch_process_chunked(&params, &black_scholes_price, Some(0));

        assert!(matches!(
            result,
            Err(DervflowError::InvalidInput(message)) if message == "chunk_size must be greater than 0"
        ));
    }

    #[test]
    fn test_batch_process_chunked_empty_input_short_circuits() {
        let params: Vec<OptionParams> = Vec::new();
        let result = batch_process_chunked(&params, &black_scholes_price, Some(0)).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_process_chunked_single_chunk_matches_direct_processing() {
        let params = create_test_params(8);
        let chunked =
            batch_process_chunked(&params, &black_scholes_price, Some(params.len() * 2)).unwrap();
        let direct: Vec<f64> = params
            .iter()
            .map(|p| black_scholes_price(p).unwrap())
            .collect();

        assert_eq!(chunked.len(), direct.len());
        for i in 0..direct.len() {
            assert!((chunked[i] - direct[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_process_chunked_exact_chunk_size_matches_direct_processing() {
        let params = create_test_params(8);
        let chunked =
            batch_process_chunked(&params, &black_scholes_price, Some(params.len())).unwrap();
        let direct: Vec<f64> = params
            .iter()
            .map(|p| black_scholes_price(p).unwrap())
            .collect();

        assert_eq!(chunked.len(), direct.len());
        for i in 0..direct.len() {
            assert!((chunked[i] - direct[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_consistency() {
        // Verify that batch processing gives same results as sequential
        let params = create_test_params(10);

        let sequential: Vec<f64> = params
            .iter()
            .map(|p| black_scholes_price(p).unwrap())
            .collect();

        let parallel = batch_process(&params, &black_scholes_price).unwrap();

        assert_eq!(sequential.len(), parallel.len());
        for i in 0..sequential.len() {
            assert!((sequential[i] - parallel[i]).abs() < 1e-10);
        }
    }
}
