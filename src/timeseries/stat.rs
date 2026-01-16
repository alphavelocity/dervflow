// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Statistical measures
//!
//! This module provides functions for calculating statistical measures
//! of financial time series, including moments, quantiles, and rolling stat metrics.

use crate::common::error::{DervflowError, Result};
use crate::core::stat::{
    mean_absolute_deviation as core_mean_absolute_deviation,
    median_absolute_deviation as core_median_absolute_deviation,
    root_mean_square as core_root_mean_square,
};

fn ensure_finite(name: &str, data: &[f64]) -> Result<()> {
    if let Some(value) = data.iter().find(|value| !value.is_finite()) {
        return Err(DervflowError::InvalidInput(format!(
            "{name} must contain only finite values, found {value}"
        )));
    }
    Ok(())
}

fn mean_unchecked(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

#[derive(Debug, Clone, Copy)]
struct Moments {
    count: usize,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl Moments {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    fn update(&mut self, value: f64) {
        let count_prev = self.count as f64;
        self.count += 1;
        let count = self.count as f64;

        let delta = value - self.mean;
        let delta_n = delta / count;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * count_prev;

        self.m4 += term1 * delta_n2 * (count * count - 3.0 * count + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (count - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.mean += delta_n;
    }

    fn variance(&self, ddof: usize) -> f64 {
        let variance = self.m2 / (self.count - ddof) as f64;
        if variance < 0.0 { 0.0 } else { variance }
    }

    fn sample_std(&self) -> f64 {
        self.variance(1).sqrt()
    }

    fn sample_skewness(&self) -> f64 {
        let n = self.count as f64;
        let std = self.sample_std();

        if std == 0.0 {
            return 0.0;
        }

        let sum_cubed = self.m3 / std.powi(3);
        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
    }

    fn sample_kurtosis(&self) -> f64 {
        let n = self.count as f64;
        let std = self.sample_std();

        if std == 0.0 {
            return 0.0;
        }

        let sum_fourth = self.m4 / std.powi(4);
        ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth
            - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0))
    }
}

fn moments_unchecked(data: &[f64]) -> Moments {
    let mut moments = Moments::new();
    for &value in data {
        moments.update(value);
    }
    moments
}

fn variance_unchecked(data: &[f64], ddof: usize) -> f64 {
    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (i, &value) in data.iter().enumerate() {
        let delta = value - mean;
        mean += delta / (i as f64 + 1.0);
        m2 += delta * (value - mean);
    }

    let variance = m2 / (data.len() - ddof) as f64;
    if variance < 0.0 { 0.0 } else { variance }
}

fn skewness_unchecked(data: &[f64]) -> f64 {
    moments_unchecked(data).sample_skewness()
}

fn kurtosis_unchecked(data: &[f64]) -> f64 {
    moments_unchecked(data).sample_kurtosis()
}

fn quantiles_unchecked(data: &[f64], qs: &[f64]) -> Result<Vec<f64>> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut results = Vec::with_capacity(qs.len());
    let n = sorted.len();

    for &q in qs {
        if !(0.0..=1.0).contains(&q) {
            return Err(DervflowError::InvalidInput(format!(
                "Quantile must be between 0 and 1, got {}",
                q
            )));
        }

        let index = q * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let fraction = index - lower as f64;

        let value = if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
        };

        results.push(value);
    }

    Ok(results)
}

fn ewma_unchecked(data: &[f64], alpha: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);

    for i in 1..data.len() {
        let ema = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }

    result
}

/// Statistical moments and measures for a time series
#[derive(Debug, Clone, Copy)]
pub struct TimeSeriesStats {
    /// Number of observations
    pub count: usize,
    /// Sum of all observations
    pub sum: f64,
    /// Mean (first moment)
    pub mean: f64,
    /// Variance (second central moment, sample estimator)
    pub variance: f64,
    /// Standard deviation (square root of variance)
    pub std_dev: f64,
    /// Standard error of the mean
    pub std_error: f64,
    /// Skewness (third standardized moment)
    pub skewness: f64,
    /// Kurtosis (fourth standardized moment)
    pub kurtosis: f64,
    /// Minimum observation
    pub min: f64,
    /// Maximum observation
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
    /// Median value (50th percentile)
    pub median: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Interquartile range (q3 - q1)
    pub iqr: f64,
    /// Mean absolute deviation around the mean
    pub mean_abs_dev: f64,
    /// Median absolute deviation (unscaled)
    pub median_abs_dev: f64,
    /// Root mean square value
    pub root_mean_square: f64,
}

/// Calculate mean of a data series
pub fn mean(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate mean of empty data".to_string(),
        ));
    }
    ensure_finite("mean", data)?;

    Ok(mean_unchecked(data))
}

/// Calculate variance of a data series
///
/// # Arguments
///
/// * `data` - Data series
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
pub fn variance(data: &[f64], ddof: usize) -> Result<f64> {
    if data.len() <= ddof {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be greater than ddof ({})",
            data.len(),
            ddof
        )));
    }
    ensure_finite("variance", data)?;

    Ok(variance_unchecked(data, ddof))
}

/// Calculate standard deviation of a data series
pub fn std_dev(data: &[f64], ddof: usize) -> Result<f64> {
    Ok(variance(data, ddof)?.sqrt())
}

/// Calculate skewness of a data series
///
/// Skewness measures the asymmetry of the distribution.
/// Uses the sample skewness formula with bias correction.
pub fn skewness(data: &[f64]) -> Result<f64> {
    if data.len() < 3 {
        return Err(DervflowError::InvalidInput(
            "Need at least 3 data points to calculate skewness".to_string(),
        ));
    }
    ensure_finite("skewness", data)?;

    Ok(skewness_unchecked(data))
}

/// Calculate kurtosis of a data series
///
/// Kurtosis measures the "tailedness" of the distribution.
/// Returns excess kurtosis (kurtosis - 3), where normal distribution has excess kurtosis of 0.
pub fn kurtosis(data: &[f64]) -> Result<f64> {
    if data.len() < 4 {
        return Err(DervflowError::InvalidInput(
            "Need at least 4 data points to calculate kurtosis".to_string(),
        ));
    }
    ensure_finite("kurtosis", data)?;

    Ok(kurtosis_unchecked(data))
}

/// Calculate all statistical moments at once
pub fn calculate_stat(data: &[f64]) -> Result<TimeSeriesStats> {
    if data.len() < 4 {
        return Err(DervflowError::InvalidInput(
            "Need at least 4 data points to calculate all stat metrics".to_string(),
        ));
    }
    ensure_finite("calculate_stat", data)?;

    let count = data.len();
    let mut sum = 0.0;
    let mut moments = Moments::new();

    // Compute moments, extrema, and sum in a single pass.
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for &value in data {
        sum += value;
        moments.update(value);

        if value < min_value {
            min_value = value;
        }
        if value > max_value {
            max_value = value;
        }
    }

    let mean_value = moments.mean;
    let variance_value = moments.variance(1);
    let std_dev = variance_value.sqrt();
    let skew = moments.sample_skewness();
    let kurt = moments.sample_kurtosis();

    let range = max_value - min_value;

    // Quantile-based metrics reuse a single sort pass.
    let quartiles = quantiles_unchecked(data, &[0.25, 0.5, 0.75])?;
    let q1 = quartiles[0];
    let median = quartiles[1];
    let q3 = quartiles[2];
    let iqr = q3 - q1;

    // Dispersion metrics derived from absolute deviations.
    let mean_abs_dev = core_mean_absolute_deviation(data)?;
    let median_abs_dev = core_median_absolute_deviation(data, Some(1.0))?;
    let root_mean_square = core_root_mean_square(data)?;

    let std_error = std_dev / (count as f64).sqrt();

    Ok(TimeSeriesStats {
        count,
        sum,
        mean: mean_value,
        variance: variance_value,
        std_dev,
        std_error,
        skewness: skew,
        kurtosis: kurt,
        min: min_value,
        max: max_value,
        range,
        median,
        q1,
        q3,
        iqr,
        mean_abs_dev,
        median_abs_dev,
        root_mean_square,
    })
}

/// Calculate quantile of a data series
///
/// # Arguments
///
/// * `data` - Data series (will be sorted internally)
/// * `q` - Quantile to calculate (between 0 and 1)
///
/// # Returns
///
/// The value at the specified quantile
pub fn quantile(data: &[f64], q: f64) -> Result<f64> {
    if data.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate quantile of empty data".to_string(),
        ));
    }
    ensure_finite("quantile", data)?;

    if !(0.0..=1.0).contains(&q) {
        return Err(DervflowError::InvalidInput(format!(
            "Quantile must be between 0 and 1, got {}",
            q
        )));
    }

    Ok(quantiles_unchecked(data, &[q])?
        .into_iter()
        .next()
        .expect("quantiles_unchecked should return one value"))
}

/// Calculate multiple quantiles at once
pub fn quantiles(data: &[f64], qs: &[f64]) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate quantiles of empty data".to_string(),
        ));
    }
    ensure_finite("quantiles", data)?;

    quantiles_unchecked(data, qs)
}

/// Calculate rolling mean over a window
pub fn rolling_mean(data: &[f64], window: usize) -> Result<Vec<f64>> {
    if window == 0 {
        return Err(DervflowError::InvalidInput(
            "Window size must be positive".to_string(),
        ));
    }

    if data.len() < window {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be at least window size ({})",
            data.len(),
            window
        )));
    }
    ensure_finite("rolling_mean", data)?;

    let mut result = Vec::with_capacity(data.len() - window + 1);

    // Calculate first window
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);

    // Rolling calculation
    for i in window..data.len() {
        sum = sum - data[i - window] + data[i];
        result.push(sum / window as f64);
    }

    Ok(result)
}

/// Calculate rolling standard deviation over a window
pub fn rolling_std(data: &[f64], window: usize) -> Result<Vec<f64>> {
    if window < 2 {
        return Err(DervflowError::InvalidInput(
            "Window size must be at least 2 for standard deviation".to_string(),
        ));
    }

    if data.len() < window {
        return Err(DervflowError::InvalidInput(format!(
            "Data length ({}) must be at least window size ({})",
            data.len(),
            window
        )));
    }
    ensure_finite("rolling_std", data)?;

    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let window_f = window as f64;
    let denom = (window - 1) as f64;

    for &value in &data[..window] {
        sum += value;
        sum_sq += value * value;
    }

    let mut variance = (sum_sq - (sum * sum) / window_f) / denom;
    if variance < 0.0 {
        variance = 0.0;
    }
    result.push(variance.sqrt());

    for i in window..data.len() {
        let outgoing = data[i - window];
        let incoming = data[i];
        sum += incoming - outgoing;
        sum_sq += incoming * incoming - outgoing * outgoing;
        let mut variance = (sum_sq - (sum * sum) / window_f) / denom;
        if variance < 0.0 {
            variance = 0.0;
        }
        result.push(variance.sqrt());
    }

    Ok(result)
}

/// Calculate exponentially weighted moving average (EWMA)
///
/// # Arguments
///
/// * `data` - Data series
/// * `alpha` - Smoothing factor (0 < alpha <= 1), higher values give more weight to recent observations
pub fn ewma(data: &[f64], alpha: f64) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate EWMA of empty data".to_string(),
        ));
    }
    ensure_finite("ewma", data)?;

    if !(0.0 < alpha && alpha <= 1.0) {
        return Err(DervflowError::InvalidInput(format!(
            "Alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    Ok(ewma_unchecked(data, alpha))
}

/// Calculate exponentially weighted moving standard deviation
///
/// # Arguments
///
/// * `data` - Data series
/// * `alpha` - Smoothing factor (0 < alpha <= 1)
pub fn ewm_std(data: &[f64], alpha: f64) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Cannot calculate EWM std of empty data".to_string(),
        ));
    }
    ensure_finite("ewm_std", data)?;

    if !(0.0 < alpha && alpha <= 1.0) {
        return Err(DervflowError::InvalidInput(format!(
            "Alpha must be between 0 and 1, got {}",
            alpha
        )));
    }

    let ema = ewma_unchecked(data, alpha);
    let mut result = Vec::with_capacity(data.len());

    // First value: use zero variance
    result.push(0.0);

    let mut ewm_var = 0.0;
    for i in 1..data.len() {
        let diff = data[i] - ema[i - 1];
        ewm_var = alpha * diff * diff + (1.0 - alpha) * ewm_var;
        result.push(ewm_var.sqrt());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&data).unwrap();
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_empty() {
        let data: Vec<f64> = vec![];
        assert!(mean(&data).is_err());
    }

    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = variance(&data, 1).unwrap(); // Sample variance
        assert!((var - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = std_dev(&data, 1).unwrap();
        assert!((std - 2.5_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_skewness() {
        // Symmetric data should have skewness near 0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let skew = skewness(&data).unwrap();
        assert!(skew.abs() < 0.1);
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Right-skewed data
        let data = vec![1.0, 1.0, 1.0, 2.0, 10.0];
        let skew = skewness(&data).unwrap();
        assert!(skew > 0.0); // Positive skewness
    }

    #[test]
    fn test_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kurt = kurtosis(&data).unwrap();
        // Uniform-ish distribution should have negative excess kurtosis
        assert!(kurt < 0.0);
    }

    #[test]
    fn test_calculate_stat() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_stat(&data).unwrap();

        assert_eq!(stats.count, 5);
        assert!((stats.sum - 15.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.variance - 2.5).abs() < 1e-10);
        assert!((stats.std_dev - 2.5_f64.sqrt()).abs() < 1e-10);
        assert!((stats.std_error - (2.5_f64.sqrt() / 5.0_f64.sqrt())).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.range - 4.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert!((stats.q1 - 2.0).abs() < 1e-10);
        assert!((stats.q3 - 4.0).abs() < 1e-10);
        assert!((stats.iqr - 2.0).abs() < 1e-10);
        assert!((stats.mean_abs_dev - 1.2).abs() < 1e-10);
        assert!((stats.median_abs_dev - 1.0).abs() < 1e-10);
        assert!((stats.root_mean_square - 11.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let q0 = quantile(&data, 0.0).unwrap();
        assert!((q0 - 1.0).abs() < 1e-10);

        let q50 = quantile(&data, 0.5).unwrap();
        assert!((q50 - 3.0).abs() < 1e-10);

        let q100 = quantile(&data, 1.0).unwrap();
        assert!((q100 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qs = vec![0.25, 0.5, 0.75];
        let results = quantiles(&data, &qs).unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0] - 2.0).abs() < 1e-10);
        assert!((results[1] - 3.0).abs() < 1e-10);
        assert!((results[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = rolling_mean(&data, 3).unwrap();

        assert_eq!(rolling.len(), 3);
        assert!((rolling[0] - 2.0).abs() < 1e-10); // (1+2+3)/3
        assert!((rolling[1] - 3.0).abs() < 1e-10); // (2+3+4)/3
        assert!((rolling[2] - 4.0).abs() < 1e-10); // (3+4+5)/3
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = rolling_std(&data, 3).unwrap();

        assert_eq!(rolling.len(), 3);
        // Each window should have std dev of 1.0
        for &std in &rolling {
            assert!((std - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ewma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = ewma(&data, 0.5).unwrap();

        assert_eq!(ema.len(), 5);
        assert!((ema[0] - 1.0).abs() < 1e-10);
        // Each subsequent value should be weighted average
        assert!(ema[1] > 1.0 && ema[1] < 2.0);
    }

    #[test]
    fn test_ewm_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = ewm_std(&data, 0.5).unwrap();

        assert_eq!(std.len(), 5);
        assert!((std[0] - 0.0).abs() < 1e-10); // First value is 0
        assert!(std[1] > 0.0); // Subsequent values should be positive
    }

    #[test]
    fn test_quantile_invalid() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(quantile(&data, -0.1).is_err());
        assert!(quantile(&data, 1.1).is_err());
    }

    #[test]
    fn test_ewma_invalid_alpha() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(ewma(&data, 0.0).is_err());
        assert!(ewma(&data, 1.5).is_err());
    }
}
