// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

use super::validation::{validate_finite, validate_min_length, validate_non_empty};

#[derive(Default)]
struct RunningMomentState {
    count: usize,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl RunningMomentState {
    fn update(&mut self, value: f64) -> Result<(usize, f64, f64, f64, f64)> {
        self.count += 1;
        let n = self.count as f64;

        let delta = value - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        let new_m4 =
            self.m4 + term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
                - 4.0 * delta_n * self.m3;
        let new_m3 = self.m3 + term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        let new_m2 = self.m2 + term1;
        let new_mean = self.mean + delta_n;

        if !new_mean.is_finite()
            || !new_m2.is_finite()
            || !new_m3.is_finite()
            || !new_m4.is_finite()
        {
            return Err(DervflowError::NumericalError(
                "Running moment update produced a non-finite value".to_string(),
            ));
        }

        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;

        Ok((self.count, self.mean, self.m2, self.m3, self.m4))
    }
}

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

pub fn cumulative_mean(data: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_mean")?;
    validate_finite(data, "cumulative_mean")?;

    let mut state = RunningMomentState::default();
    let mut result = Vec::with_capacity(data.len());

    for &value in data {
        let (_, mean, _, _, _) = state.update(value)?;
        result.push(mean);
    }

    Ok(result)
}

fn cumulative_variance_internal(data: &[f64], unbiased: bool) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_variance")?;
    validate_finite(data, "cumulative_variance")?;

    let mut state = RunningMomentState::default();
    let mut result = Vec::with_capacity(data.len());

    for &value in data {
        let (count, _mean, m2, _m3, _m4) = state.update(value)?;

        if unbiased {
            if count < 2 {
                result.push(0.0);
            } else {
                result.push(m2 / (count as f64 - 1.0));
            }
        } else {
            result.push(m2 / count as f64);
        }
    }

    Ok(result)
}

pub fn cumulative_variance(data: &[f64], unbiased: bool) -> Result<Vec<f64>> {
    cumulative_variance_internal(data, unbiased)
}

pub fn cumulative_std(data: &[f64], unbiased: bool) -> Result<Vec<f64>> {
    let variances = cumulative_variance_internal(data, unbiased)?;
    Ok(variances
        .into_iter()
        .map(|var| var.max(0.0).sqrt())
        .collect())
}

pub fn cumulative_skewness(data: &[f64], unbiased: bool) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_skewness")?;
    validate_finite(data, "cumulative_skewness")?;

    let mut state = RunningMomentState::default();
    let mut result = Vec::with_capacity(data.len());

    for &value in data {
        let (count, _mean, m2, m3, _m4) = state.update(value)?;
        if count < 2 || m2.abs() <= f64::EPSILON {
            result.push(0.0);
            continue;
        }

        let n = count as f64;
        let skewness = if unbiased {
            if count < 3 {
                0.0
            } else {
                let denominator = m2.powf(1.5);
                if denominator.abs() <= f64::EPSILON {
                    0.0
                } else {
                    let correction = n * (n - 1.0).sqrt() / (n - 2.0);
                    correction * (m3 / denominator)
                }
            }
        } else {
            let denominator = m2.powf(1.5);
            if denominator.abs() <= f64::EPSILON {
                0.0
            } else {
                (n.sqrt() * m3) / denominator
            }
        };

        if !skewness.is_finite() {
            return Err(DervflowError::NumericalError(
                "Cumulative skewness produced a non-finite value".to_string(),
            ));
        }

        result.push(skewness);
    }

    Ok(result)
}

pub fn cumulative_kurtosis(data: &[f64], unbiased: bool) -> Result<Vec<f64>> {
    validate_non_empty(data, "cumulative_kurtosis")?;
    validate_finite(data, "cumulative_kurtosis")?;

    let mut state = RunningMomentState::default();
    let mut result = Vec::with_capacity(data.len());

    for &value in data {
        let (count, _mean, m2, _m3, m4) = state.update(value)?;
        if count < 2 || m2.abs() <= f64::EPSILON {
            result.push(0.0);
            continue;
        }

        let n = count as f64;
        let kurtosis = if unbiased {
            if count < 4 {
                0.0
            } else {
                let s2 = m2 / (n - 1.0);
                if s2.abs() <= f64::EPSILON {
                    0.0
                } else {
                    let numerator = n * (n + 1.0) * m4;
                    let denominator = (n - 1.0) * (n - 2.0) * (n - 3.0) * s2 * s2;
                    let correction = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
                    numerator / denominator - correction
                }
            }
        } else {
            let mu2 = m2 / n;
            let mu4 = m4 / n;
            if mu2.abs() <= f64::EPSILON {
                0.0
            } else {
                mu4 / (mu2 * mu2) - 3.0
            }
        };

        if !kurtosis.is_finite() {
            return Err(DervflowError::NumericalError(
                "Cumulative kurtosis produced a non-finite value".to_string(),
            ));
        }

        result.push(kurtosis);
    }

    Ok(result)
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

    #[test]
    fn test_cumulative_mean() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let means = cumulative_mean(&data).unwrap();
        assert_abs_diff_eq!(means[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(means[1], 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(means[2], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(means[3], 2.5, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_variance_unbiased() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let variance = cumulative_variance(&data, true).unwrap();
        assert_abs_diff_eq!(variance[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(variance[1], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(variance[2], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(variance[3], 5.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_variance_biased() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let variance = cumulative_variance(&data, false).unwrap();
        assert_abs_diff_eq!(variance[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(variance[1], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(variance[2], 2.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(variance[3], 5.0 / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_std_unbiased() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let std = cumulative_std(&data, true).unwrap();
        assert_abs_diff_eq!(std[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(std[1], (0.5f64).sqrt(), epsilon = 1e-12);
        assert_abs_diff_eq!(std[2], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(std[3], (5.0f64 / 3.0).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_std_biased() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let std = cumulative_std(&data, false).unwrap();
        assert_abs_diff_eq!(std[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            std[1],
            0.5f64.sqrt() / std::f64::consts::SQRT_2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(std[2], (2.0f64 / 3.0).sqrt(), epsilon = 1e-12);
        assert_abs_diff_eq!(std[3], (5.0f64 / 4.0).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_skewness_unbiased() {
        let data = [1.0, 2.0, 3.0, 6.0];
        let skewness = cumulative_skewness(&data, true).unwrap();
        assert_abs_diff_eq!(skewness[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(skewness[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(skewness[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            skewness[3],
            1.190_340_128_278_994_7,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_cumulative_skewness_biased() {
        let data = [1.0, 2.0, 3.0, 6.0];
        let skewness = cumulative_skewness(&data, false).unwrap();
        assert_abs_diff_eq!(skewness[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(skewness[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(skewness[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(skewness[3], 0.687_243_193_489_091_2, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_kurtosis_unbiased() {
        let data = [1.0, 2.0, 3.0, 6.0];
        let kurtosis = cumulative_kurtosis(&data, true).unwrap();
        assert_abs_diff_eq!(kurtosis[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(kurtosis[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(kurtosis[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(kurtosis[3], 1.5, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_kurtosis_biased() {
        let data = [1.0, 2.0, 3.0, 6.0];
        let kurtosis = cumulative_kurtosis(&data, false).unwrap();
        assert_abs_diff_eq!(kurtosis[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(kurtosis[1], -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(kurtosis[2], -1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(kurtosis[3], -1.0, epsilon = 1e-12);
    }
}
