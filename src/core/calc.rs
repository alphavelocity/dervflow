// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

use super::validation::{validate_finite, validate_min_length, validate_non_empty};

const MIN_POINTS_DERIVATIVE: usize = 2;
const MIN_POINTS_SECOND_DERIVATIVE: usize = 3;

fn validate_spacing(spacing: f64, context: &str) -> Result<()> {
    if !spacing.is_finite() || spacing <= 0.0 {
        Err(DervflowError::InvalidInput(format!(
            "Spacing for {} must be a finite, positive value",
            context
        )))
    } else {
        Ok(())
    }
}

fn validate_shape(shape: &[usize], context: &str) -> Result<()> {
    if shape.is_empty() {
        return Err(DervflowError::InvalidInput(format!(
            "Shape for {} must contain at least one dimension",
            context
        )));
    }

    if shape.iter().any(|&dim| dim == 0) {
        return Err(DervflowError::InvalidInput(format!(
            "Dimensions for {} must be non-zero",
            context
        )));
    }

    Ok(())
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len()).rev() {
        if i + 1 < shape.len() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

fn unravel_index(mut index: usize, shape: &[usize], coords: &mut [usize]) {
    for (coord, &dim) in coords.iter_mut().rev().zip(shape.iter().rev()) {
        *coord = index % dim;
        index /= dim;
    }
}

fn first_derivative_along_axis(
    values: &[f64],
    shape: &[usize],
    strides: &[usize],
    coords: &[usize],
    axis: usize,
    spacing: f64,
) -> f64 {
    let stride = strides[axis];
    let axis_len = shape[axis];
    let index = coords
        .iter()
        .rev()
        .zip(strides.iter().rev())
        .fold(0usize, |acc, (&coord, &stride)| acc + coord * stride);

    if axis_len == 1 {
        return 0.0;
    }

    let idx = index;
    let h = spacing;

    if coords[axis] == 0 {
        if axis_len > 2 {
            let forward = values[idx + stride];
            let forward_two = values[idx + 2 * stride];
            (-3.0 * values[idx] + 4.0 * forward - forward_two) / (2.0 * h)
        } else {
            (values[idx + stride] - values[idx]) / h
        }
    } else if coords[axis] == axis_len - 1 {
        if axis_len > 2 {
            let backward = values[idx - stride];
            let backward_two = values[idx - 2 * stride];
            (3.0 * values[idx] - 4.0 * backward + backward_two) / (2.0 * h)
        } else {
            (values[idx] - values[idx - stride]) / h
        }
    } else {
        (values[idx + stride] - values[idx - stride]) / (2.0 * h)
    }
}

fn second_derivative_along_axis(
    values: &[f64],
    shape: &[usize],
    strides: &[usize],
    coords: &[usize],
    axis: usize,
    spacing: f64,
) -> f64 {
    let stride = strides[axis];
    let axis_len = shape[axis];
    let index = coords
        .iter()
        .rev()
        .zip(strides.iter().rev())
        .fold(0usize, |acc, (&coord, &stride)| acc + coord * stride);

    if axis_len <= 1 {
        return 0.0;
    }

    let h2 = spacing * spacing;

    if axis_len == 2 {
        // With only two points along an axis we cannot form a second derivative stencil.
        // Treat the curvature as zero which corresponds to the quadratic passing through
        // both points with minimum curvature.
        return 0.0;
    }

    if axis_len == 3 {
        return match coords[axis] {
            0 => {
                let f0 = values[index];
                let f1 = values[index + stride];
                let f2 = values[index + 2 * stride];
                (f2 - 2.0 * f1 + f0) / h2
            }
            x if x == axis_len - 1 => {
                let f0 = values[index];
                let f1 = values[index - stride];
                let f2 = values[index - 2 * stride];
                (f0 - 2.0 * f1 + f2) / h2
            }
            _ => {
                let forward = values[index + stride];
                let backward = values[index - stride];
                (forward - 2.0 * values[index] + backward) / h2
            }
        };
    }

    if coords[axis] == 0 {
        let f0 = values[index];
        let f1 = values[index + stride];
        let f2 = values[index + 2 * stride];
        let f3 = values[index + 3 * stride];
        (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) / h2
    } else if coords[axis] == axis_len - 1 {
        let f0 = values[index];
        let f1 = values[index - stride];
        let f2 = values[index - 2 * stride];
        let f3 = values[index - 3 * stride];
        (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) / h2
    } else {
        let forward = values[index + stride];
        let backward = values[index - stride];
        (forward - 2.0 * values[index] + backward) / h2
    }
}

pub fn derivative(data: &[f64], spacing: f64) -> Result<Vec<f64>> {
    validate_min_length(data, MIN_POINTS_DERIVATIVE, "derivative")?;
    validate_finite(data, "derivative")?;
    validate_spacing(spacing, "derivative")?;

    let mut result = vec![0.0; data.len()];
    let n = data.len();

    if n == 2 {
        let slope = (data[1] - data[0]) / spacing;
        result[0] = slope;
        result[1] = slope;
        return Ok(result);
    }

    result[0] = (-3.0 * data[0] + 4.0 * data[1] - data[2]) / (2.0 * spacing);
    for i in 1..n - 1 {
        result[i] = (data[i + 1] - data[i - 1]) / (2.0 * spacing);
    }
    result[n - 1] = (3.0 * data[n - 1] - 4.0 * data[n - 2] + data[n - 3]) / (2.0 * spacing);

    Ok(result)
}

pub fn second_derivative(data: &[f64], spacing: f64) -> Result<Vec<f64>> {
    validate_min_length(data, MIN_POINTS_SECOND_DERIVATIVE, "second_derivative")?;
    validate_finite(data, "second_derivative")?;
    validate_spacing(spacing, "second_derivative")?;

    let mut result = vec![0.0; data.len()];
    let n = data.len();

    if n == 3 {
        let value = (data[2] - 2.0 * data[1] + data[0]) / (spacing * spacing);
        result.fill(value);
        return Ok(result);
    }

    result[0] = (2.0 * data[0] - 5.0 * data[1] + 4.0 * data[2] - data[3]) / (spacing * spacing);
    for i in 1..n - 1 {
        result[i] = (data[i + 1] - 2.0 * data[i] + data[i - 1]) / (spacing * spacing);
    }
    result[n - 1] = (2.0 * data[n - 1] - 5.0 * data[n - 2] + 4.0 * data[n - 3] - data[n - 4])
        / (spacing * spacing);

    Ok(result)
}

pub fn definite_integral(data: &[f64], spacing: f64) -> Result<f64> {
    validate_min_length(data, 2, "definite_integral")?;
    validate_finite(data, "definite_integral")?;
    validate_spacing(spacing, "definite_integral")?;

    let mut area = 0.0;
    for window in data.windows(2) {
        area += (window[0] + window[1]) * 0.5 * spacing;
    }
    Ok(area)
}

pub fn cumulative_integral(data: &[f64], spacing: f64) -> Result<Vec<f64>> {
    validate_min_length(data, 2, "cumulative_integral")?;
    validate_finite(data, "cumulative_integral")?;
    validate_spacing(spacing, "cumulative_integral")?;

    let mut result = vec![0.0; data.len()];
    for i in 1..data.len() {
        result[i] = result[i - 1] + (data[i - 1] + data[i]) * 0.5 * spacing;
    }
    Ok(result)
}

pub fn gradient(values: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(values, "gradient")?;
    validate_finite(values, "gradient")?;
    validate_shape(shape, "gradient")?;

    if shape.len() != spacings.len() {
        return Err(DervflowError::InvalidInput(
            "Spacings must have the same length as shape".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("gradient axis {}", i))?;
    }

    let total_size = product(shape);
    if total_size != values.len() {
        return Err(DervflowError::InvalidInput(
            "Values length must equal product of shape".to_string(),
        ));
    }

    let dims = shape.len();
    let strides = compute_strides(shape);
    let mut result = vec![0.0; values.len() * dims];
    let mut coords = vec![0usize; dims];

    for linear_idx in 0..values.len() {
        unravel_index(linear_idx, shape, &mut coords);
        let storage_index = coords
            .iter()
            .zip(strides.iter())
            .fold(0usize, |acc, (&coord, &stride)| acc + coord * stride);
        for axis in 0..dims {
            let spacing = spacings[axis];
            let deriv =
                first_derivative_along_axis(values, shape, &strides, &coords, axis, spacing);
            result[storage_index * dims + axis] = deriv;
        }
    }

    Ok(result)
}

pub fn divergence(field: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(field, "divergence")?;
    validate_finite(field, "divergence")?;
    validate_shape(shape, "divergence")?;

    if shape.len() != spacings.len() {
        return Err(DervflowError::InvalidInput(
            "Spacings must match the number of field dimensions".to_string(),
        ));
    }

    let dims = shape.len();
    if field.len() != product(shape) * dims {
        return Err(DervflowError::InvalidInput(
            "Vector field must have len == product(shape) * dimensions".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("divergence axis {}", i))?;
    }

    let strides = compute_strides(shape);
    let mut result = vec![0.0; product(shape)];
    let mut coords = vec![0usize; dims];

    let total_points = product(shape);

    for linear_idx in 0..total_points {
        unravel_index(linear_idx, shape, &mut coords);
        let storage_index = coords
            .iter()
            .zip(strides.iter())
            .fold(0usize, |acc, (&coord, &stride)| acc + coord * stride);
        let mut divergence_value = 0.0;
        for axis in 0..dims {
            let spacing = spacings[axis];
            let component_offset = axis;
            let component_slice =
                &field[component_offset * product(shape)..(component_offset + 1) * product(shape)];
            divergence_value += first_derivative_along_axis(
                component_slice,
                shape,
                &strides,
                &coords,
                axis,
                spacing,
            );
        }
        result[storage_index] = divergence_value;
    }

    Ok(result)
}

pub fn curl(field: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(field, "curl")?;
    validate_finite(field, "curl")?;

    if shape.len() != 3 {
        return Err(DervflowError::InvalidInput(
            "Curl is defined for three-dimensional vector fields".to_string(),
        ));
    }

    if spacings.len() != 3 {
        return Err(DervflowError::InvalidInput(
            "Curl spacings must have length three".to_string(),
        ));
    }

    let total = product(shape);
    if field.len() != 3 * total {
        return Err(DervflowError::InvalidInput(
            "Vector field must have three components of equal length".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("curl axis {}", i))?;
    }

    let strides = compute_strides(shape);
    let mut result = vec![0.0; 3 * total];
    let mut coords = vec![0usize; 3];

    let component =
        |component: usize| -> &[f64] { &field[component * total..(component + 1) * total] };

    for linear_idx in 0..total {
        unravel_index(linear_idx, shape, &mut coords);
        let storage_index = coords
            .iter()
            .zip(strides.iter())
            .fold(0usize, |acc, (&coord, &stride)| acc + coord * stride);

        let dvy_dz =
            first_derivative_along_axis(component(1), shape, &strides, &coords, 2, spacings[2]);
        let dvz_dy =
            first_derivative_along_axis(component(2), shape, &strides, &coords, 1, spacings[1]);
        let dvz_dx =
            first_derivative_along_axis(component(2), shape, &strides, &coords, 0, spacings[0]);
        let dvx_dz =
            first_derivative_along_axis(component(0), shape, &strides, &coords, 2, spacings[2]);
        let dvx_dy =
            first_derivative_along_axis(component(0), shape, &strides, &coords, 1, spacings[1]);
        let dvy_dx =
            first_derivative_along_axis(component(1), shape, &strides, &coords, 0, spacings[0]);

        result[storage_index * 3] = dvz_dy - dvy_dz;
        result[storage_index * 3 + 1] = dvx_dz - dvz_dx;
        result[storage_index * 3 + 2] = dvy_dx - dvx_dy;
    }

    Ok(result)
}

pub fn laplacian(values: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(values, "laplacian")?;
    validate_finite(values, "laplacian")?;
    validate_shape(shape, "laplacian")?;

    if shape.len() != spacings.len() {
        return Err(DervflowError::InvalidInput(
            "Spacings must match the number of dimensions".to_string(),
        ));
    }

    if shape.iter().any(|&dim| dim < 3) {
        return Err(DervflowError::InvalidInput(
            "Laplacian requires at least three points along every dimension".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("laplacian axis {}", i))?;
    }

    let total_points = product(shape);
    if total_points != values.len() {
        return Err(DervflowError::InvalidInput(
            "Values length must equal product of shape".to_string(),
        ));
    }

    let dims = shape.len();
    let strides = compute_strides(shape);
    let mut coords = vec![0usize; dims];
    let mut result = vec![0.0; total_points];

    for linear_idx in 0..total_points {
        unravel_index(linear_idx, shape, &mut coords);
        let mut value = 0.0;
        for axis in 0..dims {
            value += second_derivative_along_axis(
                values,
                shape,
                &strides,
                &coords,
                axis,
                spacings[axis],
            );
        }
        result[linear_idx] = value;
    }

    Ok(result)
}

pub fn gradient_magnitude(values: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    let gradient_values = gradient(values, shape, spacings)?;
    let dims = shape.len();
    let total_points = product(shape);

    let mut magnitudes = vec![0.0; total_points];
    for point in 0..total_points {
        let mut sum_sq = 0.0;
        for axis in 0..dims {
            let value = gradient_values[point * dims + axis];
            sum_sq += value * value;
        }
        magnitudes[point] = sum_sq.sqrt();
    }

    Ok(magnitudes)
}

pub fn normalized_gradient(values: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    let gradient_values = gradient(values, shape, spacings)?;
    let dims = shape.len();
    let total_points = product(shape);

    let mut normalized = vec![0.0; gradient_values.len()];
    for point in 0..total_points {
        let mut sum_sq = 0.0;
        for axis in 0..dims {
            let value = gradient_values[point * dims + axis];
            sum_sq += value * value;
        }

        if sum_sq > 0.0 {
            let inv_norm = sum_sq.sqrt().recip();
            for axis in 0..dims {
                let idx = point * dims + axis;
                normalized[idx] = gradient_values[idx] * inv_norm;
            }
        }
    }

    Ok(normalized)
}

pub fn directional_derivative(
    values: &[f64],
    shape: &[usize],
    spacings: &[f64],
    direction: &[f64],
) -> Result<Vec<f64>> {
    if direction.len() != shape.len() {
        return Err(DervflowError::InvalidInput(
            "Direction vector must match the number of dimensions".to_string(),
        ));
    }

    if direction.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Direction vector must contain at least one component".to_string(),
        ));
    }

    if direction.iter().any(|value| !value.is_finite()) {
        return Err(DervflowError::InvalidInput(
            "Direction vector must contain only finite values".to_string(),
        ));
    }

    let norm_sq: f64 = direction.iter().map(|value| value * value).sum();
    if norm_sq <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "Direction vector must have a non-zero magnitude".to_string(),
        ));
    }

    let norm = norm_sq.sqrt();
    let gradient_values = gradient(values, shape, spacings)?;
    let dims = shape.len();
    let total_points = product(shape);
    let mut result = vec![0.0; total_points];

    for point in 0..total_points {
        let mut projection = 0.0;
        for axis in 0..dims {
            let grad_component = gradient_values[point * dims + axis];
            projection += grad_component * direction[axis] / norm;
        }
        result[point] = projection;
    }

    Ok(result)
}

pub fn vector_laplacian(field: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(field, "vector_laplacian")?;
    validate_finite(field, "vector_laplacian")?;
    validate_shape(shape, "vector_laplacian")?;

    if shape.len() != spacings.len() {
        return Err(DervflowError::InvalidInput(
            "Spacings must match the number of dimensions".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("vector_laplacian axis {}", i))?;
    }

    let dims = shape.len();
    let total_points = product(shape);
    if field.len() != total_points * dims {
        return Err(DervflowError::InvalidInput(
            "Vector field must have len == product(shape) * dimensions".to_string(),
        ));
    }

    let mut result = vec![0.0; field.len()];
    for component in 0..dims {
        let start = component * total_points;
        let end = start + total_points;
        let component_result = laplacian(&field[start..end], shape, spacings)?;
        result[start..end].copy_from_slice(&component_result);
    }

    Ok(result)
}

pub fn jacobian(field: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(field, "jacobian")?;
    validate_finite(field, "jacobian")?;
    validate_shape(shape, "jacobian")?;

    if shape.len() != spacings.len() {
        return Err(DervflowError::InvalidInput(
            "Spacings must match the number of field dimensions".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("jacobian axis {}", i))?;
    }

    let dims = shape.len();
    let total_points = product(shape);
    if total_points == 0 {
        return Err(DervflowError::InvalidInput(
            "Shape must describe at least one point".to_string(),
        ));
    }

    if field.len() % total_points != 0 {
        return Err(DervflowError::InvalidInput(
            "Vector field length must be a multiple of the number of grid points".to_string(),
        ));
    }

    let components = field.len() / total_points;
    let strides = compute_strides(shape);
    let mut coords = vec![0usize; dims];
    let mut result = vec![0.0; total_points * components * dims];

    for linear_idx in 0..total_points {
        unravel_index(linear_idx, shape, &mut coords);
        for component in 0..components {
            let component_slice = &field[component * total_points..(component + 1) * total_points];
            for axis in 0..dims {
                let derivative = first_derivative_along_axis(
                    component_slice,
                    shape,
                    &strides,
                    &coords,
                    axis,
                    spacings[axis],
                );
                let offset = (linear_idx * components + component) * dims + axis;
                result[offset] = derivative;
            }
        }
    }

    Ok(result)
}

pub fn hessian(values: &[f64], shape: &[usize], spacings: &[f64]) -> Result<Vec<f64>> {
    validate_non_empty(values, "hessian")?;
    validate_finite(values, "hessian")?;
    validate_shape(shape, "hessian")?;

    if shape.len() != spacings.len() {
        return Err(DervflowError::InvalidInput(
            "Spacings must match the number of dimensions".to_string(),
        ));
    }

    for (i, &spacing) in spacings.iter().enumerate() {
        validate_spacing(spacing, &format!("hessian axis {}", i))?;
    }

    let dims = shape.len();
    let total_points = product(shape);
    if total_points != values.len() {
        return Err(DervflowError::InvalidInput(
            "Values length must equal product of shape".to_string(),
        ));
    }

    if dims == 0 {
        return Err(DervflowError::InvalidInput(
            "Shape must contain at least one dimension".to_string(),
        ));
    }

    let gradient_values = gradient(values, shape, spacings)?;
    let mut component_slices = vec![vec![0.0; total_points]; dims];
    for linear_idx in 0..total_points {
        for axis in 0..dims {
            component_slices[axis][linear_idx] = gradient_values[linear_idx * dims + axis];
        }
    }

    let strides = compute_strides(shape);
    let mut coords = vec![0usize; dims];
    let mut result = vec![0.0; total_points * dims * dims];

    for linear_idx in 0..total_points {
        unravel_index(linear_idx, shape, &mut coords);
        let base = linear_idx * dims * dims;

        for axis in 0..dims {
            let diag = second_derivative_along_axis(
                values,
                shape,
                &strides,
                &coords,
                axis,
                spacings[axis],
            );
            result[base + axis * dims + axis] = diag;
        }

        for axis in 0..dims {
            for other in axis + 1..dims {
                let partial_a = first_derivative_along_axis(
                    &component_slices[axis],
                    shape,
                    &strides,
                    &coords,
                    other,
                    spacings[other],
                );
                let partial_b = first_derivative_along_axis(
                    &component_slices[other],
                    shape,
                    &strides,
                    &coords,
                    axis,
                    spacings[axis],
                );
                let mixed = 0.5 * (partial_a + partial_b);
                result[base + axis * dims + other] = mixed;
                result[base + other * dims + axis] = mixed;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_linear() {
        let data = vec![1.0, 3.0, 5.0, 7.0];
        let result = derivative(&data, 1.0).unwrap();
        for value in result {
            assert!((value - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_second_derivative_quadratic() {
        let data = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let result = second_derivative(&data, 1.0).unwrap();
        for value in result.iter().skip(1).take(3) {
            assert!((value - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_definite_integral_trapezoid() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let integral = definite_integral(&data, 1.0).unwrap();
        assert!((integral - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_cumulative_integral_monotonic() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let cumulative = cumulative_integral(&data, 1.0).unwrap();
        assert_eq!(cumulative[0], 0.0);
        assert!(cumulative.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn test_gradient_sine_wave() {
        let data: Vec<f64> = (0..100).map(|i| ((i as f64) * 0.1).sin()).collect();
        let grad = gradient(&data, &[100], &[0.1]).unwrap();
        let max_error = grad
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let x = i as f64 * 0.1;
                (g - x.cos()).abs()
            })
            .fold(0.0, f64::max);
        assert!(max_error < 5e-2);
    }

    #[test]
    fn test_divergence_constant_field_zero() {
        let shape = [5, 5];
        let total = product(&shape);
        let field = vec![1.0; total * shape.len()];
        let div = divergence(&field, &shape, &[1.0, 1.0]).unwrap();
        for value in div {
            assert!(value.abs() < 1e-12);
        }
    }

    #[test]
    fn test_curl_linear_field_zero() {
        let shape = [3, 3, 3];
        let total = product(&shape);
        let mut field = vec![0.0; 3 * total];
        for i in 0..total {
            field[i] = 1.0;
            field[total + i] = -2.0;
            field[2 * total + i] = 0.5;
        }
        let curl_res = curl(&field, &shape, &[1.0, 1.0, 1.0]).unwrap();
        assert!(curl_res.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn test_laplacian_quadratic_surface() {
        let shape = [5, 5];
        let dx = 0.25;
        let dy = 0.25;
        let mut values = vec![0.0; product(&shape)];

        for (idx, value) in values.iter_mut().enumerate() {
            let x = (idx / shape[1]) as f64;
            let y = (idx % shape[1]) as f64;
            let x_coord = (x - 2.0) * dx;
            let y_coord = (y - 2.0) * dy;
            *value = x_coord * x_coord + y_coord * y_coord;
        }

        let laplacian = laplacian(&values, &shape, &[dx, dy]).unwrap();
        let mut max_error = 0.0f64;

        for i in 1..shape[0] - 1 {
            for j in 1..shape[1] - 1 {
                let idx = i * shape[1] + j;
                max_error = max_error.max((laplacian[idx] - 4.0).abs());
            }
        }

        assert!(max_error < 5e-2);
    }

    #[test]
    fn test_jacobian_linear_vector_field() {
        let shape = [5, 5];
        let dx = 0.25;
        let dy = 0.25;
        let total = product(&shape);
        let mut component_one = vec![0.0; total];
        let mut component_two = vec![0.0; total];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x_coord = (i as f64 - 2.0) * dx;
                let y_coord = (j as f64 - 2.0) * dy;
                component_one[idx] = 2.0 * x_coord + y_coord;
                component_two[idx] = x_coord - 3.0 * y_coord;
            }
        }

        let mut field = component_one;
        field.extend(component_two);

        let jacobian = jacobian(&field, &shape, &[dx, dy]).unwrap();
        let components = 2;
        let dims = 2;
        let centre_idx = (shape[0] / 2) * shape[1] + (shape[1] / 2);
        let base = centre_idx * components * dims;

        let j11 = jacobian[base];
        let j12 = jacobian[base + 1];
        let j21 = jacobian[base + dims];
        let j22 = jacobian[base + dims + 1];

        assert!((j11 - 2.0).abs() < 5e-2);
        assert!((j12 - 1.0).abs() < 5e-2);
        assert!((j21 - 1.0).abs() < 5e-2);
        assert!((j22 + 3.0).abs() < 5e-2);
    }

    #[test]
    fn test_hessian_quadratic_surface() {
        let shape = [7, 7];
        let dx = 0.2;
        let dy = 0.2;
        let total = product(&shape);
        let mut values = vec![0.0; total];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x = (i as f64 - (shape[0] as f64 - 1.0) / 2.0) * dx;
                let y = (j as f64 - (shape[1] as f64 - 1.0) / 2.0) * dy;
                values[idx] = 3.0 * x * x + x * y + 2.0 * y * y;
            }
        }

        let hessian = hessian(&values, &shape, &[dx, dy]).unwrap();
        let dims = shape.len();
        let centre_idx = (shape[0] / 2) * shape[1] + (shape[1] / 2);
        let base = centre_idx * dims * dims;

        let h11 = hessian[base];
        let h12 = hessian[base + 1];
        let h21 = hessian[base + dims];
        let h22 = hessian[base + dims + 1];

        assert!((h11 - 6.0).abs() < 5e-2);
        assert!((h12 - 1.0).abs() < 5e-2);
        assert!((h21 - 1.0).abs() < 5e-2);
        assert!((h22 - 4.0).abs() < 5e-2);
    }

    #[test]
    fn test_hessian_reduces_to_second_derivative_in_1d() {
        let n = 21;
        let dx = 0.1;
        let mut values = vec![0.0; n];
        for (i, value) in values.iter_mut().enumerate() {
            let x = (i as f64 - (n as f64 - 1.0) / 2.0) * dx;
            *value = 0.5 * x * x;
        }

        let hessian = hessian(&values, &[n], &[dx]).unwrap();
        let second = second_derivative(&values, dx).unwrap();

        for i in 0..n {
            let h_value = hessian[i];
            assert!((h_value - second[i]).abs() < 5e-2);
        }
    }

    #[test]
    fn test_gradient_magnitude_matches_analytical_norm() {
        let shape = [9, 9];
        let dx = 0.25;
        let dy = 0.25;
        let total = product(&shape);
        let mut values = vec![0.0; total];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x = (i as f64 - (shape[0] as f64 - 1.0) / 2.0) * dx;
                let y = (j as f64 - (shape[1] as f64 - 1.0) / 2.0) * dy;
                values[idx] = x * x + y * y;
            }
        }

        let magnitudes = gradient_magnitude(&values, &shape, &[dx, dy]).unwrap();

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x = (i as f64 - (shape[0] as f64 - 1.0) / 2.0) * dx;
                let y = (j as f64 - (shape[1] as f64 - 1.0) / 2.0) * dy;
                let expected = 2.0 * (x * x + y * y).sqrt();
                assert!((magnitudes[idx] - expected).abs() < 1e-1);
            }
        }
    }

    #[test]
    fn test_normalized_gradient_yields_unit_vectors() {
        let shape = [9, 7];
        let dx = 0.25;
        let dy = 0.2;
        let total = product(&shape);
        let mut values = vec![0.0; total];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x = (i as f64 - (shape[0] as f64 - 1.0) / 2.0) * dx;
                let y = (j as f64 - (shape[1] as f64 - 1.0) / 2.0) * dy;
                values[idx] = 1.5 * x + 0.75 * y;
            }
        }

        let unit_grad = normalized_gradient(&values, &shape, &[dx, dy]).unwrap();
        let dims = shape.len();
        let centre_idx = (shape[0] / 2) * shape[1] + (shape[1] / 2);
        let gx = unit_grad[centre_idx * dims];
        let gy = unit_grad[centre_idx * dims + 1];

        let expected_norm = (1.5_f64 * 1.5 + 0.75_f64 * 0.75).sqrt();
        assert!((gx - 1.5 / expected_norm).abs() < 5e-3);
        assert!((gy - 0.75 / expected_norm).abs() < 5e-3);

        for point in 0..total {
            let mut norm_sq = 0.0;
            for axis in 0..dims {
                let value = unit_grad[point * dims + axis];
                norm_sq += value * value;
            }
            if norm_sq > 0.0 {
                assert!((norm_sq.sqrt() - 1.0).abs() < 5e-3);
            }
        }

        let constant = vec![3.0; total];
        let zeros = normalized_gradient(&constant, &shape, &[dx, dy]).unwrap();
        assert!(zeros.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_vector_laplacian_operates_componentwise() {
        let shape = [9, 9];
        let dx = 0.2;
        let dy = 0.2;
        let total = product(&shape);
        let dims = shape.len();

        let mut component_one = vec![0.0; total];
        let mut component_two = vec![0.0; total];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x = (i as f64 - (shape[0] as f64 - 1.0) / 2.0) * dx;
                let y = (j as f64 - (shape[1] as f64 - 1.0) / 2.0) * dy;
                component_one[idx] = x * x + y * y;
                component_two[idx] = 2.0 * x - 3.0 * y;
            }
        }

        let mut field = component_one.clone();
        field.extend(component_two.clone());

        let vector_lap = vector_laplacian(&field, &shape, &[dx, dy]).unwrap();

        let centre_idx = (shape[0] / 2) * shape[1] + (shape[1] / 2);
        let lap_component_one = vector_lap[centre_idx];
        let lap_component_two = vector_lap[total + centre_idx];

        assert!((lap_component_one - 4.0).abs() < 1e-1);
        assert!(lap_component_two.abs() < 1e-2);

        let zero_field = vec![0.0; total * dims];
        let zero_result = vector_laplacian(&zero_field, &shape, &[dx, dy]).unwrap();
        assert!(zero_result.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn test_directional_derivative_consistent_with_gradient_projection() {
        let shape = [7, 5];
        let dx = 0.2;
        let dy = 0.3;
        let total = product(&shape);
        let mut values = vec![0.0; total];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let idx = i * shape[1] + j;
                let x = (i as f64 - (shape[0] as f64 - 1.0) / 2.0) * dx;
                let y = (j as f64 - (shape[1] as f64 - 1.0) / 2.0) * dy;
                values[idx] = 1.5 * x * x + 0.5 * x * y + 2.5 * y * y;
            }
        }

        let direction = [2.0_f64, -1.0_f64];
        let dir_derivative =
            directional_derivative(&values, &shape, &[dx, dy], &direction).unwrap();
        let grad = gradient(&values, &shape, &[dx, dy]).unwrap();

        let dims = shape.len();
        for point in 0..total {
            let mut dot = 0.0;
            let mut norm = 0.0;
            for axis in 0..dims {
                let grad_component = grad[point * dims + axis];
                dot += grad_component * direction[axis];
                norm += direction[axis] * direction[axis];
            }
            let expected = dot / norm.sqrt();
            assert!((dir_derivative[point] - expected).abs() < 1e-2);
        }
    }
}
