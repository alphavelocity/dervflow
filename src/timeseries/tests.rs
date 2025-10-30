// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Statistical tests (stationarity, normality)
//!
//! Implements various statistical tests for time series analysis:
//! - Augmented Dickey-Fuller (ADF) test for stationarity
//! - KPSS test for stationarity
//! - Ljung-Box test for autocorrelation
//! - Jarque-Bera test for normality

use crate::common::error::{DervflowError, Result};

/// Result of a statistical test
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical values at different significance levels
    pub critical_values: Vec<(f64, f64)>, // (significance_level, critical_value)
    /// Whether to reject the null hypothesis at 5% significance
    pub reject_null: bool,
}

/// Augmented Dickey-Fuller test for stationarity
///
/// Null hypothesis: The series has a unit root (non-stationary)
/// Alternative: The series is stationary
///
/// # Arguments
/// * `data` - Time series data
/// * `max_lag` - Maximum number of lags to include (if None, uses 12*(n/100)^0.25)
/// * `regression` - Type of regression: "c" (constant), "ct" (constant+trend), "nc" (no constant)
///
/// # Returns
/// Test result with statistic, p-value, and critical values
pub fn adf_test(data: &[f64], max_lag: Option<usize>, regression: &str) -> Result<TestResult> {
    if data.len() < 10 {
        return Err(DervflowError::InvalidInput(
            "ADF test requires at least 10 observations".to_string(),
        ));
    }

    let n = data.len();
    let lag = max_lag.unwrap_or_else(|| {
        let n_f64 = n as f64;
        (12.0 * (n_f64 / 100.0).powf(0.25)).floor() as usize
    });

    // Compute first differences
    let mut diff = vec![0.0; n - 1];
    for i in 0..n - 1 {
        diff[i] = data[i + 1] - data[i];
    }

    // Build regression matrices
    let n_obs = n - lag - 1;
    let mut y = vec![0.0; n_obs];
    let mut x = Vec::new();

    for i in 0..n_obs {
        y[i] = diff[lag + i];

        let mut row = Vec::new();

        // Add constant if needed
        if regression == "c" || regression == "ct" {
            row.push(1.0);
        }

        // Add trend if needed
        if regression == "ct" {
            row.push((lag + i + 1) as f64);
        }

        // Add lagged level
        row.push(data[lag + i]);

        // Add lagged differences
        for j in 0..lag {
            row.push(diff[lag + i - j - 1]);
        }

        x.push(row);
    }

    // Perform OLS regression
    let (coefficients, std_errors) = ols_regression(&y, &x)?;

    // The test statistic is the t-statistic for the lagged level coefficient
    let level_coef_idx = if regression == "c" {
        1
    } else if regression == "ct" {
        2
    } else {
        0
    };

    let test_stat = coefficients[level_coef_idx] / std_errors[level_coef_idx];

    // Get critical values (MacKinnon approximation)
    let critical_values = adf_critical_values(n, regression);

    // Approximate p-value using critical values
    let p_value = approximate_adf_pvalue(test_stat, n, regression);

    let reject_null = p_value < 0.05;

    Ok(TestResult {
        statistic: test_stat,
        p_value,
        critical_values,
        reject_null,
    })
}

/// KPSS test for stationarity
///
/// Null hypothesis: The series is stationary
/// Alternative: The series has a unit root (non-stationary)
///
/// # Arguments
/// * `data` - Time series data
/// * `regression` - Type of regression: "c" (level stationary) or "ct" (trend stationary)
/// * `lags` - Number of lags for Newey-West correction (if None, uses 4*(n/100)^0.25)
///
/// # Returns
/// Test result with statistic, p-value, and critical values
pub fn kpss_test(data: &[f64], regression: &str, lags: Option<usize>) -> Result<TestResult> {
    if data.len() < 10 {
        return Err(DervflowError::InvalidInput(
            "KPSS test requires at least 10 observations".to_string(),
        ));
    }

    let n = data.len();
    let n_lags = lags.unwrap_or_else(|| {
        let n_f64 = n as f64;
        (4.0 * (n_f64 / 100.0).powf(0.25)).floor() as usize
    });

    // Detrend the data
    let residuals = if regression == "ct" {
        detrend_linear(data)
    } else {
        detrend_mean(data)
    };

    // Compute partial sums
    let mut partial_sums = vec![0.0; n];
    partial_sums[0] = residuals[0];
    for i in 1..n {
        partial_sums[i] = partial_sums[i - 1] + residuals[i];
    }

    // Compute sum of squared partial sums
    let s_squared: f64 = partial_sums.iter().map(|x| x * x).sum::<f64>() / (n as f64).powi(2);

    // Compute long-run variance using Newey-West
    let mut variance = residuals.iter().map(|x| x * x).sum::<f64>() / n as f64;

    for lag in 1..=n_lags {
        let weight = 1.0 - (lag as f64) / ((n_lags + 1) as f64);
        let mut autocovariance = 0.0;
        for i in lag..n {
            autocovariance += residuals[i] * residuals[i - lag];
        }
        autocovariance /= n as f64;
        variance += 2.0 * weight * autocovariance;
    }

    let test_stat = s_squared / variance;

    // Get critical values
    let critical_values = kpss_critical_values(regression);

    // Approximate p-value
    let p_value = approximate_kpss_pvalue(test_stat, regression);

    let reject_null = p_value < 0.05;

    Ok(TestResult {
        statistic: test_stat,
        p_value,
        critical_values,
        reject_null,
    })
}

/// Ljung-Box test for autocorrelation
///
/// Null hypothesis: No autocorrelation up to lag h
/// Alternative: At least one autocorrelation is non-zero
///
/// # Arguments
/// * `data` - Time series data
/// * `lags` - Number of lags to test
///
/// # Returns
/// Test result with statistic and p-value
pub fn ljung_box_test(data: &[f64], lags: usize) -> Result<TestResult> {
    if data.len() < lags + 2 {
        return Err(DervflowError::InvalidInput(
            "Insufficient data for Ljung-Box test".to_string(),
        ));
    }

    let n = data.len();

    // Compute mean
    let mean = data.iter().sum::<f64>() / n as f64;

    // Compute autocorrelations
    let mut autocorr = vec![0.0; lags];
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();

    for lag in 1..=lags {
        let mut sum = 0.0;
        for i in lag..n {
            sum += (data[i] - mean) * (data[i - lag] - mean);
        }
        autocorr[lag - 1] = sum / variance;
    }

    // Compute Ljung-Box statistic
    let mut q_stat = 0.0;
    for (k, &rho) in autocorr.iter().enumerate() {
        let lag = k + 1;
        q_stat += rho.powi(2) / (n - lag) as f64;
    }
    q_stat *= (n as f64) * (n as f64 + 2.0);

    // P-value from chi-squared distribution with 'lags' degrees of freedom
    let p_value = chi_squared_survival(q_stat, lags as f64);

    let reject_null = p_value < 0.05;

    Ok(TestResult {
        statistic: q_stat,
        p_value,
        critical_values: vec![], // Chi-squared critical values not typically reported
        reject_null,
    })
}

/// Jarque-Bera test for normality
///
/// Null hypothesis: Data is normally distributed
/// Alternative: Data is not normally distributed
///
/// # Arguments
/// * `data` - Time series data
///
/// # Returns
/// Test result with statistic and p-value
pub fn jarque_bera_test(data: &[f64]) -> Result<TestResult> {
    if data.len() < 4 {
        return Err(DervflowError::InvalidInput(
            "Jarque-Bera test requires at least 4 observations".to_string(),
        ));
    }

    let n = data.len() as f64;

    // Compute mean
    let mean = data.iter().sum::<f64>() / n;

    // Compute moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &x in data {
        let dev = x - mean;
        let dev2 = dev * dev;
        m2 += dev2;
        m3 += dev2 * dev;
        m4 += dev2 * dev2;
    }

    m2 /= n;
    m3 /= n;
    m4 /= n;

    // Compute skewness and kurtosis
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2);

    // Jarque-Bera statistic
    let jb_stat = (n / 6.0) * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);

    // P-value from chi-squared distribution with 2 degrees of freedom
    let p_value = chi_squared_survival(jb_stat, 2.0);

    let reject_null = p_value < 0.05;

    Ok(TestResult {
        statistic: jb_stat,
        p_value,
        critical_values: vec![(0.05, 5.99), (0.01, 9.21)], // Chi-squared(2) critical values
        reject_null,
    })
}

// Helper functions

/// Ordinary Least Squares regression
fn ols_regression(y: &[f64], x: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = x[0].len();

    if n < k {
        return Err(DervflowError::NumericalError(
            "Insufficient observations for regression".to_string(),
        ));
    }

    // Build X'X matrix
    let mut xtx = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            for row in x {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    // Build X'y vector
    let mut xty = vec![0.0; k];
    for i in 0..k {
        for (idx, row) in x.iter().enumerate() {
            xty[i] += row[i] * y[idx];
        }
    }

    // Solve (X'X)^-1 X'y using Gaussian elimination
    let coefficients = solve_linear_system(&xtx, &xty)?;

    // Compute residuals
    let mut residuals = vec![0.0; n];
    for (i, row) in x.iter().enumerate() {
        let mut fitted = 0.0;
        for (j, &coef) in coefficients.iter().enumerate() {
            fitted += coef * row[j];
        }
        residuals[i] = y[i] - fitted;
    }

    // Compute standard errors
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma_squared = rss / (n - k) as f64;

    // Invert X'X for variance-covariance matrix
    let xtx_inv = invert_matrix(&xtx)?;

    let mut std_errors = vec![0.0; k];
    for i in 0..k {
        std_errors[i] = (sigma_squared * xtx_inv[i][i]).sqrt();
    }

    Ok((coefficients, std_errors))
}

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-10 {
            return Err(DervflowError::NumericalError(
                "Singular matrix in linear system".to_string(),
            ));
        }

        // Eliminate column
        for k in i + 1..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in i + 1..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

/// Invert a matrix using Gaussian elimination
fn invert_matrix(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-10 {
            return Err(DervflowError::NumericalError(
                "Singular matrix cannot be inverted".to_string(),
            ));
        }

        // Scale row
        let pivot = aug[i][i];
        for j in 0..2 * n {
            aug[i][j] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Ok(inv)
}

/// Detrend data by removing mean
fn detrend_mean(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| x - mean).collect()
}

/// Detrend data by removing linear trend
fn detrend_linear(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;

    // Compute linear regression y = a + b*t
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_ty = 0.0;
    let mut sum_t2 = 0.0;

    for (i, &y) in data.iter().enumerate() {
        let t = i as f64;
        sum_t += t;
        sum_y += y;
        sum_ty += t * y;
        sum_t2 += t * t;
    }

    let b = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t * sum_t);
    let a = (sum_y - b * sum_t) / n;

    // Remove trend
    data.iter()
        .enumerate()
        .map(|(i, &y)| y - (a + b * (i as f64)))
        .collect()
}

/// ADF critical values (MacKinnon approximation)
fn adf_critical_values(_n: usize, regression: &str) -> Vec<(f64, f64)> {
    // Simplified critical values for common significance levels
    // These are approximate values for large samples
    match regression {
        "nc" => vec![(0.10, -1.62), (0.05, -1.95), (0.01, -2.58)],
        "c" => vec![(0.10, -2.57), (0.05, -2.86), (0.01, -3.43)],
        "ct" => vec![(0.10, -3.12), (0.05, -3.41), (0.01, -3.96)],
        _ => vec![(0.10, -2.57), (0.05, -2.86), (0.01, -3.43)],
    }
}

/// Approximate ADF p-value
fn approximate_adf_pvalue(stat: f64, n: usize, regression: &str) -> f64 {
    let critical = adf_critical_values(n, regression);

    // Simple linear interpolation between critical values
    if stat > critical[0].1 {
        0.10 + (stat - critical[0].1) * 0.10 / (0.0 - critical[0].1)
    } else if stat > critical[1].1 {
        0.05 + (stat - critical[1].1) * 0.05 / (critical[0].1 - critical[1].1)
    } else if stat > critical[2].1 {
        0.01 + (stat - critical[2].1) * 0.04 / (critical[1].1 - critical[2].1)
    } else {
        0.001
    }
}

/// KPSS critical values
fn kpss_critical_values(regression: &str) -> Vec<(f64, f64)> {
    match regression {
        "c" => vec![(0.10, 0.347), (0.05, 0.463), (0.01, 0.739)],
        "ct" => vec![(0.10, 0.119), (0.05, 0.146), (0.01, 0.216)],
        _ => vec![(0.10, 0.347), (0.05, 0.463), (0.01, 0.739)],
    }
}

/// Approximate KPSS p-value
fn approximate_kpss_pvalue(stat: f64, regression: &str) -> f64 {
    let critical = kpss_critical_values(regression);

    // Simple linear interpolation
    if stat < critical[0].1 {
        0.10 + (critical[0].1 - stat) * 0.10 / critical[0].1
    } else if stat < critical[1].1 {
        0.05 + (critical[1].1 - stat) * 0.05 / (critical[1].1 - critical[0].1)
    } else if stat < critical[2].1 {
        0.01 + (critical[2].1 - stat) * 0.04 / (critical[2].1 - critical[1].1)
    } else {
        0.001
    }
}

/// Chi-squared survival function (1 - CDF)
fn chi_squared_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    // Use incomplete gamma function approximation
    // P(X > x) = 1 - P(X <= x) = 1 - gamma_cdf(x/2, df/2)
    let k = df / 2.0;
    let x_half = x / 2.0;

    // Simple approximation using series expansion for small x or asymptotic for large x
    if x < df {
        // Series expansion
        let mut term = (-x_half).exp() * x_half.powf(k) / gamma_function(k + 1.0);
        let mut sum = term;

        for i in 1..100 {
            term *= x_half / (k + i as f64);
            sum += term;
            if term.abs() < 1e-10 * sum.abs() {
                break;
            }
        }

        1.0 - sum
    } else {
        // Asymptotic approximation
        let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
        standard_normal_survival(z)
    }
}

/// Gamma function approximation
fn gamma_function(x: f64) -> f64 {
    // Stirling's approximation for x > 1
    if x > 1.0 {
        (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    } else {
        // Use recursion: Gamma(x) = Gamma(x+1) / x
        gamma_function(x + 1.0) / x
    }
}

/// Standard normal survival function
fn standard_normal_survival(z: f64) -> f64 {
    0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod test_module {
    use super::*;

    #[test]
    fn test_adf_stationary() {
        // Stationary series with some variation
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos() * 0.5)
            .collect();
        let result = adf_test(&data, Some(3), "c");
        // Test should complete without error
        assert!(result.is_ok());
        if let Ok(res) = result {
            assert!(res.statistic.is_finite());
        }
    }

    #[test]
    fn test_kpss_stationary() {
        // Stationary series
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = kpss_test(&data, "c", None).unwrap();
        assert!(result.statistic < 0.5); // Should be small for stationary series
    }

    #[test]
    fn test_ljung_box() {
        // White noise should not reject null
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = ljung_box_test(&data, 10).unwrap();
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_jarque_bera_normal() {
        // Approximately normal data
        let data: Vec<f64> = vec![
            -1.0, -0.5, 0.0, 0.5, 1.0, -0.8, 0.3, -0.2, 0.7, -0.3, 0.1, -0.6, 0.4, -0.1, 0.2, -0.4,
            0.6, -0.7, 0.8, 0.0,
        ];
        let result = jarque_bera_test(&data).unwrap();
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
