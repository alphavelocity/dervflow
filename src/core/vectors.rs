// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

use super::validation::{
    validate_dimension, validate_finite, validate_non_empty, validate_same_length,
};

pub fn dot(a: &[f64], b: &[f64]) -> Result<f64> {
    validate_non_empty(a, "dot")?;
    validate_same_length(a, b, "dot")?;
    validate_finite(a, "dot")?;
    validate_finite(b, "dot")?;

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

pub fn hadamard_product(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(a, "hadamard_product")?;
    validate_same_length(a, b, "hadamard_product")?;
    validate_finite(a, "hadamard_product")?;
    validate_finite(b, "hadamard_product")?;

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

pub fn norm(data: &[f64]) -> Result<f64> {
    validate_non_empty(data, "norm")?;
    validate_finite(data, "norm")?;
    Ok(data.iter().map(|x| x * x).sum::<f64>().sqrt())
}

pub fn lp_norm(data: &[f64], p: f64) -> Result<f64> {
    validate_non_empty(data, "lp_norm")?;
    validate_finite(data, "lp_norm")?;

    if p.is_nan() {
        return Err(DervflowError::InvalidInput(
            "Norm order must be a finite number or positive infinity".to_string(),
        ));
    }

    if p.is_sign_negative() {
        return Err(DervflowError::InvalidInput(
            "Norm order must be greater than or equal to 1".to_string(),
        ));
    }

    if p.is_infinite() {
        return Ok(data
            .iter()
            .map(|x| x.abs())
            .fold(0.0, |acc, value| acc.max(value)));
    }

    if p < 1.0 {
        return Err(DervflowError::InvalidInput(
            "Norm order must be greater than or equal to 1".to_string(),
        ));
    }

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
            scaled_sum = 1.0 + scaled_sum * ratio.powf(p);
            scale = abs_value;
        } else {
            let ratio = abs_value / scale;
            scaled_sum += ratio.powf(p);
        }
    }

    if scale == 0.0 {
        Ok(0.0)
    } else {
        Ok(scale * scaled_sum.powf(1.0 / p))
    }
}

pub fn normalize(data: &[f64]) -> Result<Vec<f64>> {
    let norm_value = norm(data)?;
    if norm_value.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Cannot normalise a zero vector".to_string(),
        ));
    }
    Ok(data.iter().map(|x| x / norm_value).collect())
}

fn validate_distance_inputs<'a>(
    a: &'a [f64],
    b: &'a [f64],
    context: &str,
) -> Result<(&'a [f64], &'a [f64])> {
    validate_non_empty(a, context)?;
    validate_same_length(a, b, context)?;
    validate_finite(a, context)?;
    validate_finite(b, context)?;
    Ok((a, b))
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    let (a, b) = validate_distance_inputs(a, b, "euclidean_distance")?;
    let sum = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f64>();
    Ok(sum.sqrt())
}

pub fn manhattan_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    let (a, b) = validate_distance_inputs(a, b, "manhattan_distance")?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>())
}

pub fn chebyshev_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    let (a, b) = validate_distance_inputs(a, b, "chebyshev_distance")?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |acc, value| acc.max(value)))
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64> {
    validate_non_empty(a, "cosine_similarity")?;
    validate_same_length(a, b, "cosine_similarity")?;
    validate_finite(a, "cosine_similarity")?;
    validate_finite(b, "cosine_similarity")?;

    let dot = dot(a, b)?;
    let norm_a = norm(a)?;
    let norm_b = norm(b)?;

    if norm_a.abs() < f64::EPSILON || norm_b.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Cannot compute cosine similarity for zero vectors".to_string(),
        ));
    }

    Ok(dot / (norm_a * norm_b))
}

pub fn vector_add(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(a, "vector_add")?;
    validate_same_length(a, b, "vector_add")?;
    validate_finite(a, "vector_add")?;
    validate_finite(b, "vector_add")?;

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

pub fn vector_subtract(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(a, "vector_subtract")?;
    validate_same_length(a, b, "vector_subtract")?;
    validate_finite(a, "vector_subtract")?;
    validate_finite(b, "vector_subtract")?;

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
}

pub fn scalar_multiply(data: &[f64], scalar: f64) -> Result<Vec<f64>> {
    validate_non_empty(data, "scalar_multiply")?;
    validate_finite(data, "scalar_multiply")?;
    if !scalar.is_finite() {
        return Err(DervflowError::InvalidInput(
            "Scalar multiplier must be finite".to_string(),
        ));
    }

    Ok(data.iter().map(|x| x * scalar).collect())
}

pub fn cross_product(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    validate_dimension(a, 3, "cross_product")?;
    validate_dimension(b, 3, "cross_product")?;
    validate_finite(a, "cross_product")?;
    validate_finite(b, "cross_product")?;

    let result = vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    Ok(result)
}

pub fn projection(a: &[f64], onto: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(a, "projection")?;
    validate_same_length(a, onto, "projection")?;
    validate_finite(a, "projection")?;
    validate_finite(onto, "projection")?;

    let denom = dot(onto, onto)?;
    if denom.abs() < f64::EPSILON {
        return Err(DervflowError::InvalidInput(
            "Cannot project onto a zero vector".to_string(),
        ));
    }

    let scalar = dot(a, onto)? / denom;
    Ok(onto.iter().map(|v| v * scalar).collect())
}

pub fn angle_between(a: &[f64], b: &[f64]) -> Result<f64> {
    let cosine = cosine_similarity(a, b)?;
    if !cosine.is_finite() {
        return Err(DervflowError::NumericalError(
            "Cosine similarity produced a non-finite value".to_string(),
        ));
    }

    let clamped = cosine.clamp(-1.0, 1.0);

    Ok(clamped.acos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_vector_operations() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, -5.0, 6.0];
        assert_abs_diff_eq!(dot(&a, &b).unwrap(), 12.0, epsilon = 1e-12);

        let norm_a = norm(&a).unwrap();
        assert_abs_diff_eq!(norm_a, (14.0f64).sqrt(), epsilon = 1e-12);

        assert_abs_diff_eq!(
            lp_norm(&a, 3.0).unwrap(),
            36.0f64.powf(1.0 / 3.0),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(lp_norm(&a, f64::INFINITY).unwrap(), 3.0, epsilon = 1e-12);

        let normalised = normalize(&a).unwrap();
        assert_abs_diff_eq!(
            normalised.iter().map(|x| x * x).sum::<f64>(),
            1.0,
            epsilon = 1e-12
        );

        let hadamard = hadamard_product(&a, &b).unwrap();
        assert_eq!(hadamard, vec![4.0, -10.0, 18.0]);

        let added = vector_add(&a, &b).unwrap();
        assert_eq!(added, vec![5.0, -3.0, 9.0]);
    }

    #[test]
    fn test_advanced_vector_operations() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let cross = cross_product(&a, &b).unwrap();
        assert_abs_diff_eq!(cross[2], 1.0, epsilon = 1e-12);

        let proj = projection(&[2.0, 2.0], &[1.0, 0.0]).unwrap();
        assert_abs_diff_eq!(proj[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(proj[1], 0.0, epsilon = 1e-12);

        let angle = angle_between(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert_abs_diff_eq!(angle, std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn test_distances() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 9.0];

        assert_abs_diff_eq!(
            euclidean_distance(&a, &b).unwrap(),
            61.0f64.sqrt(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(manhattan_distance(&a, &b).unwrap(), 13.0, epsilon = 1e-12);
        assert_abs_diff_eq!(chebyshev_distance(&a, &b).unwrap(), 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lp_norm_large_magnitude() {
        let values = [1.0e200, -1.0e200];
        let norm_two = lp_norm(&values, 2.0).unwrap();
        assert_relative_eq!(norm_two, 2.0f64.sqrt() * 1.0e200, max_relative = 1e-12);

        let norm_three = lp_norm(&values, 3.0).unwrap();
        let expected = (2.0f64).powf(1.0 / 3.0) * 1.0e200;
        assert_relative_eq!(norm_three, expected, max_relative = 1e-12);
    }

    #[test]
    fn test_angle_between_clamping() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 1.0e-16, 0.0];
        let angle = angle_between(&a, &b).unwrap();
        assert!(angle.is_finite());
        assert!(angle >= 0.0);
    }
}
