// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

use super::validation::{
    validate_finite, validate_min_length, validate_non_empty, validate_positive,
    validate_same_length, validate_window_size,
};

const DEFAULT_MAD_SCALE: f64 = 1.482_602_218_505_602; // Consistent with normal distribution MAD.

#[inline]
fn stable_lerp(lower: f64, upper: f64, weight: f64) -> f64 {
    lower * (1.0 - weight) + upper * weight
}

#[derive(Clone, Copy, Debug)]
struct BivariateMoments {
    m2_x: f64,
    m2_y: f64,
    co_moment: f64,
}

#[inline]
fn accumulate_bivariate_moments(
    x: &[f64],
    y: &[f64],
    metric_name: &str,
) -> Result<BivariateMoments> {
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut m2_x = 0.0;
    let mut m2_y = 0.0;
    let mut co_moment = 0.0;

    for (index, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        let n = index as f64 + 1.0;
        let inv_n = 1.0 / n;
        let delta_x = xi - mean_x;
        let delta_y = yi - mean_y;

        mean_x += delta_x * inv_n;
        mean_y += delta_y * inv_n;

        let scale = 1.0 - inv_n;
        m2_x += scale * delta_x * delta_x;
        m2_y += scale * delta_y * delta_y;
        co_moment += scale * delta_x * delta_y;
    }

    if !m2_x.is_finite() || !m2_y.is_finite() || !co_moment.is_finite() {
        return Err(DervflowError::NumericalError(format!(
            "{metric_name} accumulation overflowed or became non-finite"
        )));
    }

    Ok(BivariateMoments {
        m2_x,
        m2_y,
        co_moment,
    })
}

pub fn mean(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "mean")?;
    validate_finite(data, "mean")?;

    Ok(data.iter().sum::<f64>() / data.len() as f64)
}

pub fn geometric_mean(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "geometric_mean")?;
    validate_finite(data, "geometric_mean")?;
    validate_positive(data, "geometric_mean")?;

    let log_sum: f64 = data.iter().map(|x| x.ln()).sum();
    Ok((log_sum / data.len() as f64).exp())
}

pub fn harmonic_mean(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "harmonic_mean")?;
    validate_finite(data, "harmonic_mean")?;
    validate_positive(data, "harmonic_mean")?;

    let reciprocal_sum: f64 = data.iter().map(|x| 1.0 / x).sum();
    Ok(data.len() as f64 / reciprocal_sum)
}

pub fn weighted_mean(data: &[f64], weights: &[f64]) -> Result<f64> {
    validate_same_length(data, weights, "weighted_mean")?;
    validate_non_empty(data, "weighted_mean")?;
    validate_finite(data, "weighted_mean")?;
    validate_finite(weights, "weighted_mean")?;

    let weight_sum: f64 = weights.iter().sum();
    if !weight_sum.is_finite() || weight_sum.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Sum of weights must be finite and non-zero".to_string(),
        ));
    }

    let numerator: f64 = data.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();

    Ok(numerator / weight_sum)
}

pub fn root_mean_square(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "root_mean_square")?;
    validate_finite(data, "root_mean_square")?;

    let mut scale = 0.0;
    let mut scaled_sum = 0.0;

    for &value in data {
        let abs_value = value.abs();
        if abs_value == 0.0 {
            continue;
        }

        if scale == 0.0 {
            scale = abs_value;
            scaled_sum = 1.0;
            continue;
        }

        if abs_value > scale {
            let ratio = scale / abs_value;
            scaled_sum = 1.0 + scaled_sum * ratio * ratio;
            scale = abs_value;
        } else {
            let ratio = abs_value / scale;
            scaled_sum += ratio * ratio;
        }
    }

    if scale == 0.0 {
        Ok(0.0)
    } else {
        Ok(scale * (scaled_sum / data.len() as f64).sqrt())
    }
}

pub fn mean_absolute_deviation(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "mean_absolute_deviation")?;
    validate_finite(data, "mean_absolute_deviation")?;

    let mean_value = mean(data)?;
    let total_dev: f64 = data.iter().map(|x| (x - mean_value).abs()).sum();
    Ok(total_dev / data.len() as f64)
}

pub fn median_absolute_deviation(data: &[f64], scale: Option<f64>) -> Result<f64> {
    validate_non_empty(data, "median_absolute_deviation")?;
    validate_finite(data, "median_absolute_deviation")?;

    let median_value = median(data)?;
    let deviations: Vec<f64> = data.iter().map(|x| (x - median_value).abs()).collect();

    // The median of absolute deviations exists because deviations mirrors input length.
    let raw_mad = median(&deviations)?;
    let scale = scale.unwrap_or(DEFAULT_MAD_SCALE);

    if !scale.is_finite() || scale <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "MAD scale factor must be a finite, positive number".to_string(),
        ));
    }

    Ok(raw_mad * scale)
}

pub fn coefficient_of_variation(data: &[f64], unbiased: bool) -> Result<f64> {
    validate_non_empty(data, "coefficient_of_variation")?;
    validate_finite(data, "coefficient_of_variation")?;

    let mean_value = mean(data)?;
    if mean_value.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Coefficient of variation is undefined when mean is zero".to_string(),
        ));
    }

    let std_dev = standard_deviation(data, unbiased)?;
    Ok(std_dev / mean_value.abs())
}

pub fn central_moment(data: &[f64], order: u32) -> Result<f64> {
    validate_non_empty(data, "central_moment")?;
    validate_finite(data, "central_moment")?;

    if order == 0 {
        return Ok(1.0);
    }

    if order == 1 {
        return Ok(0.0);
    }

    if order == 2 {
        return variance(data, false);
    }

    let mean_value = mean(data)?;
    let order_i32 = if order <= i32::MAX as u32 {
        Some(order as i32)
    } else {
        None
    };

    let moment_sum: f64 = if let Some(order_i32) = order_i32 {
        data.iter()
            .map(|&value| (value - mean_value).powi(order_i32))
            .sum()
    } else {
        let order_f64 = order as f64;
        data.iter()
            .map(|&value| (value - mean_value).powf(order_f64))
            .sum()
    };

    Ok(moment_sum / data.len() as f64)
}

pub fn variance(data: &[f64], unbiased: bool) -> Result<f64> {
    validate_non_empty(data, "variance")?;
    validate_finite(data, "variance")?;

    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (i, &value) in data.iter().enumerate() {
        let delta = value - mean;
        mean += delta / (i as f64 + 1.0);
        m2 += delta * (value - mean);
    }

    let n = data.len();
    if unbiased {
        if n < 2 {
            return Err(DervflowError::InvalidInput(
                "At least two observations are required for sample variance".to_string(),
            ));
        }
        Ok(m2 / (n as f64 - 1.0))
    } else {
        Ok(m2 / n as f64)
    }
}

pub fn standard_deviation(data: &[f64], unbiased: bool) -> Result<f64> {
    variance(data, unbiased).map(|v| v.sqrt())
}

pub fn median(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "median")?;
    validate_finite(data, "median")?;

    let mut values = data.to_vec();
    let upper_mid = values.len() / 2;

    if values.len() % 2 == 1 {
        let (_, median_value, _) = values.select_nth_unstable_by(upper_mid, f64::total_cmp);
        Ok(*median_value)
    } else {
        let (lower_partition, upper_value, _) =
            values.select_nth_unstable_by(upper_mid, f64::total_cmp);
        let (_, lower_value, _) =
            lower_partition.select_nth_unstable_by(upper_mid - 1, f64::total_cmp);
        Ok(stable_lerp(*lower_value, *upper_value, 0.5))
    }
}

pub fn percentile(data: &[f64], percentile: f64) -> Result<f64> {
    validate_non_empty(data, "percentile")?;
    validate_finite(data, "percentile")?;

    if !(0.0..=1.0).contains(&percentile) {
        return Err(DervflowError::InvalidInput(
            "Percentile must be in the range [0, 1]".to_string(),
        ));
    }

    let mut values = data.to_vec();
    let position = percentile * ((values.len() - 1) as f64);
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;

    if lower_index == upper_index {
        let (_, percentile_value, _) = values.select_nth_unstable_by(lower_index, f64::total_cmp);
        Ok(*percentile_value)
    } else {
        let (lower_partition, upper_value, _) =
            values.select_nth_unstable_by(upper_index, f64::total_cmp);
        let (_, lower_value, _) =
            lower_partition.select_nth_unstable_by(lower_index, f64::total_cmp);
        let weight = position - lower_index as f64;
        Ok(stable_lerp(*lower_value, *upper_value, weight))
    }
}

pub fn moving_average(data: &[f64], window_size: usize) -> Result<Vec<f64>> {
    validate_non_empty(data, "moving_average")?;
    validate_finite(data, "moving_average")?;
    validate_window_size(data.len(), window_size, "moving_average")?;

    let mut result = Vec::with_capacity(data.len() - window_size + 1);
    let mut window_sum: f64 = data.iter().take(window_size).sum();
    result.push(window_sum / window_size as f64);

    for i in window_size..data.len() {
        window_sum += data[i] - data[i - window_size];
        result.push(window_sum / window_size as f64);
    }

    Ok(result)
}

pub fn z_scores(data: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(data, "z_scores")?;
    validate_finite(data, "z_scores")?;

    let mean_value = mean(data)?;
    let variance_value = variance(data, false)?;

    if variance_value.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Cannot compute z-scores with zero variance".to_string(),
        ));
    }

    let std_dev = variance_value.sqrt();
    Ok(data.iter().map(|x| (x - mean_value) / std_dev).collect())
}

pub fn covariance(x: &[f64], y: &[f64], unbiased: bool) -> Result<f64> {
    validate_non_empty(x, "covariance")?;
    validate_same_length(x, y, "covariance")?;

    let n = x.len();
    if unbiased && n < 2 {
        return Err(DervflowError::InvalidInput(
            "At least two observations are required for unbiased covariance".to_string(),
        ));
    }

    validate_finite(x, "covariance")?;
    validate_finite(y, "covariance")?;

    // Online covariance accumulation (Welford-style update) to reduce extra
    // passes and improve numerical stability for large-magnitude inputs.
    let co_moment = accumulate_bivariate_moments(x, y, "Covariance")?.co_moment;

    let covariance = if unbiased {
        co_moment / (n as f64 - 1.0)
    } else {
        co_moment / n as f64
    };

    if !covariance.is_finite() {
        return Err(DervflowError::NumericalError(
            "Covariance computation overflowed or became non-finite".to_string(),
        ));
    }

    Ok(covariance)
}

pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    validate_non_empty(x, "correlation")?;
    validate_same_length(x, y, "correlation")?;

    if x.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "At least two observations are required for correlation".to_string(),
        ));
    }

    validate_finite(x, "correlation")?;
    validate_finite(y, "correlation")?;

    // Single-pass, numerically stable updates for means, variances, and
    // covariance avoid redundant traversals over the same slices.
    let moments = accumulate_bivariate_moments(x, y, "Correlation")?;
    let m2_x = moments.m2_x;
    let m2_y = moments.m2_y;
    let co_moment = moments.co_moment;

    if m2_x <= 0.0 || m2_y <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "Cannot compute correlation when variance is zero".to_string(),
        ));
    }

    // Divide in two stages to avoid unnecessary overflow in `sqrt(m2_x * m2_y)`.
    let std_x = m2_x.sqrt();
    let std_y = m2_y.sqrt();
    let corr = (co_moment / std_x) / std_y;

    if !corr.is_finite() {
        return Err(DervflowError::NumericalError(
            "Correlation computation overflowed or became non-finite".to_string(),
        ));
    }

    // Numerical round-off can move correlation slightly outside [-1, 1].
    const CORR_ROUNDING_TOLERANCE: f64 = 1e-12;
    let abs_corr = corr.abs();

    if abs_corr <= 1.0 + CORR_ROUNDING_TOLERANCE {
        Ok(corr.clamp(-1.0, 1.0))
    } else {
        Err(DervflowError::NumericalError(
            "Correlation result fell outside the valid range [-1, 1]".to_string(),
        ))
    }
}

pub fn skewness(data: &[f64]) -> Result<f64> {
    validate_min_length(data, 3, "skewness")?;
    validate_finite(data, "skewness")?;

    let mean_value = mean(data)?;
    let n = data.len() as f64;

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &value in data {
        let diff = value - mean_value;
        m2 += diff.powi(2);
        m3 += diff.powi(3);
    }

    if m2.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Skewness is undefined when variance is zero".to_string(),
        ));
    }

    let m2 = m2 / n;
    let m3 = m3 / n;

    Ok(m3 / m2.powf(1.5))
}

pub fn kurtosis(data: &[f64]) -> Result<f64> {
    validate_min_length(data, 4, "kurtosis")?;
    validate_finite(data, "kurtosis")?;

    let mean_value = mean(data)?;
    let n = data.len() as f64;

    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &value in data {
        let diff = value - mean_value;
        m2 += diff.powi(2);
        m4 += diff.powi(4);
    }

    if m2.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Kurtosis is undefined when variance is zero".to_string(),
        ));
    }

    let m2 = m2 / n;
    let m4 = m4 / n;

    Ok(m4 / m2.powi(2) - 3.0)
}

pub fn interquartile_range(data: &[f64]) -> Result<f64> {
    let q1 = percentile(data, 0.25)?;
    let q3 = percentile(data, 0.75)?;
    Ok(q3 - q1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    fn percentile_reference(data: &[f64], percentile: f64) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);

        let position = percentile * ((sorted.len() - 1) as f64);
        let lower_index = position.floor() as usize;
        let upper_index = position.ceil() as usize;

        if lower_index == upper_index {
            sorted[lower_index]
        } else {
            let weight = position - lower_index as f64;
            sorted[lower_index] * (1.0 - weight) + sorted[upper_index] * weight
        }
    }

    #[test]
    fn test_mean_variants() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(mean(&data).unwrap(), 2.5);

        let weights = [1.0, 2.0, 3.0, 4.0];
        let weighted = weighted_mean(&data, &weights).unwrap();
        assert_relative_eq!(weighted, 3.0);

        let positive = [1.0, 3.0, 9.0];
        assert_abs_diff_eq!(geometric_mean(&positive).unwrap(), 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            harmonic_mean(&positive).unwrap(),
            27.0 / 13.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_variance_and_standard_deviation() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var_population = variance(&data, false).unwrap();
        let var_sample = variance(&data, true).unwrap();

        assert_abs_diff_eq!(var_population, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(var_sample, 4.571_428_571_428_571, epsilon = 1e-12);

        let std_population = standard_deviation(&data, false).unwrap();
        assert_abs_diff_eq!(std_population, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_additional_dispersion_measures() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_abs_diff_eq!(
            root_mean_square(&data).unwrap(),
            7.5f64.sqrt(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            mean_absolute_deviation(&data).unwrap(),
            1.0,
            epsilon = 1e-12
        );

        let scaled_mad = median_absolute_deviation(&data, None).unwrap();
        assert_abs_diff_eq!(scaled_mad, DEFAULT_MAD_SCALE, epsilon = 1e-12);

        let unscaled_mad = median_absolute_deviation(&data, Some(1.0)).unwrap();
        assert_abs_diff_eq!(unscaled_mad, 1.0, epsilon = 1e-12);

        let cv = coefficient_of_variation(&data, true).unwrap();
        assert_abs_diff_eq!(cv, 0.516_397_779_494_322, epsilon = 1e-12);

        let huge = [1.0e200, -1.0e200];
        let rms = root_mean_square(&huge).unwrap();
        assert_relative_eq!(rms, 1.0e200, max_relative = 1e-12);
    }

    #[test]
    fn test_median_percentiles_and_iqr() {
        let odd = [7.0, 1.0, 3.0];
        assert_abs_diff_eq!(median(&odd).unwrap(), 3.0, epsilon = 1e-12);

        let even = [1.0, 3.0, 5.0, 7.0];
        assert_abs_diff_eq!(median(&even).unwrap(), 4.0, epsilon = 1e-12);

        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(percentile(&data, 0.0).unwrap(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&data, 0.5).unwrap(), 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&data, 1.0).unwrap(), 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(interquartile_range(&data).unwrap(), 2.0, epsilon = 1e-12);

        let signed_zero = [-0.0, 0.0, 1.0, -1.0];
        assert_abs_diff_eq!(median(&signed_zero).unwrap(), 0.0, epsilon = 1e-12);

        let with_duplicates = [5.0, 3.0, 3.0, 2.0, 9.0, 8.0, 8.0, 1.0];
        assert_abs_diff_eq!(median(&with_duplicates).unwrap(), 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            percentile(&with_duplicates, 0.25).unwrap(),
            2.75,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            percentile(&with_duplicates, 0.75).unwrap(),
            8.0,
            epsilon = 1e-12
        );

        let single = [42.0];
        assert_abs_diff_eq!(median(&single).unwrap(), 42.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&single, 0.0).unwrap(), 42.0, epsilon = 1e-12);
        assert_abs_diff_eq!(percentile(&single, 1.0).unwrap(), 42.0, epsilon = 1e-12);

        let extreme_pair = [-1.0e308, 1.0e308];
        assert_abs_diff_eq!(median(&extreme_pair).unwrap(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            percentile(&extreme_pair, 0.5).unwrap(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            percentile(&extreme_pair, 0.25).unwrap(),
            -5.0e307,
            epsilon = 1.0e292
        );
        assert_abs_diff_eq!(
            percentile(&extreme_pair, 0.75).unwrap(),
            5.0e307,
            epsilon = 1.0e292
        );
    }

    #[test]
    fn test_percentile_invariants_monotonic_and_bounded() {
        let datasets = [
            vec![-1.0e308, -3.0, -3.0, 0.0, 4.0, 9.0, 1.0e308],
            vec![42.0, 42.0, 42.0, 42.0],
            vec![-5.0, -1.0, 0.0, 2.0, 8.0],
        ];

        for data in datasets {
            let min = data.iter().copied().fold(f64::INFINITY, f64::min);
            let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let mut prev = percentile(&data, 0.0).unwrap();
            assert!(prev >= min - 1e-12 && prev <= max + 1e-12);

            for step in 1..=100 {
                let q = step as f64 / 100.0;
                let current = percentile(&data, q).unwrap();
                assert!(current >= min - 1e-10 && current <= max + 1e-10);
                assert!(current + 1e-10 >= prev);
                prev = current;
            }
        }
    }

    #[test]
    fn test_percentile_matches_sorted_reference_across_samples() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let quantiles = [0.0, 0.01, 0.1, 0.25, 0.5, 0.73, 0.9, 0.99, 1.0];

        for len in 2..=64 {
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let centered = ((state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5) * 1.0e6;
                data.push(centered);
            }

            for &q in &quantiles {
                let observed = percentile(&data, q).unwrap();
                let expected = percentile_reference(&data, q);
                assert_abs_diff_eq!(observed, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_moments() {
        let data = [2.0, 8.0, 0.0, 4.0, 1.0];
        let skew = skewness(&data).unwrap();
        let kurt = kurtosis(&data).unwrap();
        assert!(skew.is_finite());
        assert!(kurt.is_finite());
    }

    #[test]
    fn test_central_moments() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_abs_diff_eq!(central_moment(&data, 0).unwrap(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(central_moment(&data, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(central_moment(&data, 2).unwrap(), 1.25, epsilon = 1e-12);
        assert_abs_diff_eq!(central_moment(&data, 4).unwrap(), 2.5625, epsilon = 1e-12);
    }

    #[test]
    fn test_covariance_correlation_and_zscores() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        assert_abs_diff_eq!(covariance(&x, &y, false).unwrap(), 2.5, epsilon = 1e-12);
        assert_abs_diff_eq!(
            covariance(&x, &y, true).unwrap(),
            10.0 / 3.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(correlation(&x, &y).unwrap(), 1.0, epsilon = 1e-12);

        let z = z_scores(&x).unwrap();
        assert_abs_diff_eq!(z.iter().copied().sum::<f64>(), 0.0, epsilon = 1e-12);

        let constant = [5.0, 5.0, 5.0, 5.0];
        let err = correlation(&constant, &y).unwrap_err();
        assert!(
            err.to_string()
                .contains("Cannot compute correlation when variance is zero")
        );

        let single_x = [1.0];
        let single_y = [2.0];
        let err = correlation(&single_x, &single_y).unwrap_err();
        assert!(
            err.to_string()
                .contains("At least two observations are required for correlation")
        );
    }

    #[test]
    fn test_covariance_large_magnitude_stability() {
        let x = [1.0e12 + 1.0, 1.0e12 + 2.0, 1.0e12 + 3.0, 1.0e12 + 4.0];
        let y = [2.0e12 + 2.0, 2.0e12 + 4.0, 2.0e12 + 6.0, 2.0e12 + 8.0];

        // y = 2x + c => cov(x, y) = 2 * var(x), corr(x, y) = 1.
        assert_abs_diff_eq!(covariance(&x, &y, false).unwrap(), 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(
            covariance(&x, &y, true).unwrap(),
            10.0 / 3.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(correlation(&x, &y).unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_with_tiny_nonzero_variance_is_finite() {
        let x = [1.0, 1.0 + 1.0e-9, 1.0 + 2.0e-9, 1.0 + 3.0e-9];
        let y = [2.0, 2.0 + 1.0e-9, 2.0 + 2.0e-9, 2.0 + 3.0e-9];

        let corr = correlation(&x, &y).unwrap();
        assert!(corr.is_finite());
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_large_scale_avoids_denominator_overflow() {
        let x = [1.0e150, 2.0e150, 3.0e150, 4.0e150];
        let y = [2.0e150, 4.0e150, 6.0e150, 8.0e150];

        let corr = correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_reports_numerical_error_on_overflowing_moments() {
        let x = [1.0e308, -1.0e308, 1.0e308, -1.0e308];
        let y = x;

        let err = correlation(&x, &y).unwrap_err();
        assert!(matches!(err, DervflowError::NumericalError(_)));
    }

    #[test]
    fn test_correlation_bounds_and_symmetry() {
        let x = [3.0, -1.0, 4.0, -2.0, 5.0, -3.0];
        let y = [-7.0, 2.0, -6.0, 1.0, -5.0, 0.0];

        let corr_xy = correlation(&x, &y).unwrap();
        let corr_yx = correlation(&y, &x).unwrap();

        assert!((-1.0..=1.0).contains(&corr_xy));
        assert_abs_diff_eq!(corr_xy, corr_yx, epsilon = 1e-12);

        let anti = correlation(&x, &x.map(|v| -v)).unwrap();
        assert_abs_diff_eq!(anti, -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_covariance_unbiased_requires_two_observations() {
        let x = [42.0];
        let y = [84.0];

        let err = covariance(&x, &y, true).unwrap_err();
        assert!(
            err.to_string()
                .contains("At least two observations are required for unbiased covariance")
        );
    }

    #[test]
    fn test_covariance_reports_numerical_error_on_overflowing_moments() {
        let x = [1.0e308, -1.0e308, 1.0e308, -1.0e308];
        let y = x;

        let err = covariance(&x, &y, false).unwrap_err();
        assert!(matches!(err, DervflowError::NumericalError(_)));
    }
}
