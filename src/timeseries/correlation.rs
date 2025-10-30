// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Correlation analysis
//!
//! This module provides functions for calculating various correlation measures
//! including autocorrelation, partial autocorrelation, and cross-correlation.

use crate::common::error::{DervflowError, Result};
use crate::timeseries::stat::{mean, std_dev};

/// Calculate Pearson correlation coefficient between two series
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Series must have same length: x={}, y={}",
            x.len(),
            y.len()
        )));
    }

    if x.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "Need at least 2 data points for correlation".to_string(),
        ));
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let std_x = std_dev(x, 1)?;
    let std_y = std_dev(y, 1)?;

    if std_x == 0.0 || std_y == 0.0 {
        return Ok(0.0);
    }

    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }

    let corr = sum / ((x.len() - 1) as f64 * std_x * std_y);
    Ok(corr)
}

/// Calculate Spearman rank correlation coefficient
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Series must have same length: x={}, y={}",
            x.len(),
            y.len()
        )));
    }

    if x.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "Need at least 2 data points for correlation".to_string(),
        ));
    }

    // Convert to ranks
    let rank_x = rank(x);
    let rank_y = rank(y);

    // Calculate Pearson correlation on ranks
    pearson_correlation(&rank_x, &rank_y)
}

/// Calculate Kendall's tau correlation coefficient
pub fn kendall_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Series must have same length: x={}, y={}",
            x.len(),
            y.len()
        )));
    }

    if x.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "Need at least 2 data points for correlation".to_string(),
        ));
    }

    let n = x.len();
    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let sign_x = (x[j] - x[i]).signum();
            let sign_y = (y[j] - y[i]).signum();
            let product = sign_x * sign_y;

            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
        }
    }

    let tau = (concordant - discordant) as f64 / (n * (n - 1) / 2) as f64;
    Ok(tau)
}

/// Calculate autocorrelation function (ACF) up to max_lag
///
/// # Arguments
///
/// * `data` - Time series data
/// * `max_lag` - Maximum lag to calculate (must be less than data length)
///
/// # Returns
///
/// Vector of autocorrelation values from lag 0 to max_lag
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if data.len() <= max_lag {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be greater than max_lag ({})",
            data.len(),
            max_lag
        )));
    }

    let m = mean(data)?;
    let n = data.len();

    // Calculate variance (lag 0 autocovariance)
    let var: f64 = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n as f64;

    if var == 0.0 {
        return Ok(vec![1.0; max_lag + 1]);
    }

    let mut acf = Vec::with_capacity(max_lag + 1);

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += (data[i] - m) * (data[i + lag] - m);
        }
        let autocov = sum / n as f64;
        acf.push(autocov / var);
    }

    Ok(acf)
}

/// Calculate partial autocorrelation function (PACF) up to max_lag
///
/// Uses the Durbin-Levinson algorithm
pub fn partial_autocorrelation(data: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if data.len() <= max_lag {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be greater than max_lag ({})",
            data.len(),
            max_lag
        )));
    }

    let acf = autocorrelation(data, max_lag)?;
    let mut pacf = Vec::with_capacity(max_lag + 1);

    // PACF at lag 0 is always 1
    pacf.push(1.0);

    if max_lag == 0 {
        return Ok(pacf);
    }

    // PACF at lag 1 equals ACF at lag 1
    pacf.push(acf[1]);

    // Durbin-Levinson recursion for higher lags
    let mut phi = vec![vec![0.0; max_lag]; max_lag];
    phi[0][0] = acf[1];

    for k in 1..max_lag {
        // Calculate numerator and denominator
        let mut num = acf[k + 1];
        let mut den = 1.0;

        for j in 0..k {
            num -= phi[k - 1][j] * acf[k - j];
            den -= phi[k - 1][j] * acf[j + 1];
        }

        phi[k][k] = num / den;
        pacf.push(phi[k][k]);

        // Update phi values
        for j in 0..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - 1 - j];
        }
    }

    Ok(pacf)
}

/// Calculate cross-correlation between two series up to max_lag
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `max_lag` - Maximum lag to calculate
///
/// # Returns
///
/// Vector of cross-correlation values from -max_lag to +max_lag
pub fn cross_correlation(x: &[f64], y: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Series must have same length: x={}, y={}",
            x.len(),
            y.len()
        )));
    }

    if x.len() <= max_lag {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be greater than max_lag ({})",
            x.len(),
            max_lag
        )));
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let std_x = std_dev(x, 1)?;
    let std_y = std_dev(y, 1)?;
    let n = x.len();

    if std_x == 0.0 || std_y == 0.0 {
        return Ok(vec![0.0; 2 * max_lag + 1]);
    }

    let mut ccf = Vec::with_capacity(2 * max_lag + 1);

    // Negative lags (y leads x)
    for lag in (1..=max_lag).rev() {
        let mut sum = 0.0;
        for i in lag..n {
            sum += (x[i] - mean_x) * (y[i - lag] - mean_y);
        }
        let cross_cov = sum / (n as f64 * std_x * std_y);
        ccf.push(cross_cov);
    }

    // Zero lag
    let corr_0 = pearson_correlation(x, y)?;
    ccf.push(corr_0);

    // Positive lags (x leads y)
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        for i in lag..n {
            sum += (x[i - lag] - mean_x) * (y[i] - mean_y);
        }
        let cross_cov = sum / (n as f64 * std_x * std_y);
        ccf.push(cross_cov);
    }

    Ok(ccf)
}

/// Calculate rolling correlation between two series
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `window` - Window size for rolling calculation
pub fn rolling_correlation(x: &[f64], y: &[f64], window: usize) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return Err(DervflowError::InvalidInput(format!(
            "Series must have same length: x={}, y={}",
            x.len(),
            y.len()
        )));
    }

    if window < 2 {
        return Err(DervflowError::InvalidInput(
            "Window size must be at least 2".to_string(),
        ));
    }

    if x.len() < window {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be at least window size ({})",
            x.len(),
            window
        )));
    }

    let mut result = Vec::with_capacity(x.len() - window + 1);

    for i in 0..=(x.len() - window) {
        let window_x = &x[i..i + window];
        let window_y = &y[i..i + window];
        let corr = pearson_correlation(window_x, window_y)?;
        result.push(corr);
    }

    Ok(result)
}

/// Convert data to ranks (for Spearman correlation)
fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Handle ties by averaging ranks
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }
        let avg_rank = ((i + j - 1) as f64) / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = pearson_correlation(&x, &y).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let corr = pearson_correlation(&x, &y).unwrap();
        assert!((corr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = spearman_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = kendall_correlation(&x, &y).unwrap();
        assert!((tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tau = kendall_correlation(&x, &y).unwrap();
        assert!((tau - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let acf = autocorrelation(&data, 3).unwrap();

        assert_eq!(acf.len(), 4);
        assert!((acf[0] - 1.0).abs() < 1e-10); // ACF at lag 0 is always 1
        assert!(acf[1] < 1.0); // ACF decreases with lag
    }

    #[test]
    fn test_partial_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let pacf = partial_autocorrelation(&data, 3).unwrap();

        assert_eq!(pacf.len(), 4);
        assert!((pacf[0] - 1.0).abs() < 1e-10); // PACF at lag 0 is always 1
    }

    #[test]
    fn test_cross_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ccf = cross_correlation(&x, &y, 2).unwrap();

        assert_eq!(ccf.len(), 5); // 2*max_lag + 1
        // Middle value (zero lag) should be close to 1
        assert!((ccf[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = rolling_correlation(&x, &y, 3).unwrap();

        assert_eq!(rolling.len(), 3);
        // Perfect correlation in all windows
        for &corr in &rolling {
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_correlation_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(pearson_correlation(&x, &y).is_err());
    }

    #[test]
    fn test_autocorrelation_insufficient_data() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(autocorrelation(&data, 5).is_err());
    }

    #[test]
    fn test_rank() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = rank(&data);

        // Expected ranks: [3, 1.5, 4, 1.5, 5] (ties get average rank)
        assert!((ranks[0] - 3.0).abs() < 1e-10);
        assert!((ranks[1] - 1.5).abs() < 1e-10);
        assert!((ranks[2] - 4.0).abs() < 1e-10);
        assert!((ranks[3] - 1.5).abs() < 1e-10);
        assert!((ranks[4] - 5.0).abs() < 1e-10);
    }
}
