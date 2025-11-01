// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for the core mathematics module.

use crate::common::error::DervflowError;
use crate::core::calc::{
    cumulative_integral as cumulative_integral_fn, curl as curl_fn,
    definite_integral as definite_integral_fn, derivative as derivative_fn,
    directional_derivative as directional_derivative_fn, divergence as divergence_fn,
    gradient as gradient_fn, gradient_magnitude as gradient_magnitude_fn, hessian as hessian_fn,
    jacobian as jacobian_fn, laplacian as laplacian_fn,
    normalized_gradient as normalized_gradient_fn, second_derivative as second_derivative_fn,
    vector_laplacian as vector_laplacian_fn,
};
use crate::core::combinatorics::{
    bell_number as bell_number_fn, binomial_probability as binomial_probability_fn,
    catalan_number as catalan_number_fn, combination, factorial,
    falling_factorial as falling_factorial_fn, lah_number as lah_number_fn,
    multinomial as multinomial_fn, permutation, rising_factorial as rising_factorial_fn,
    stirling_number_first as stirling_number_first_fn,
    stirling_number_second as stirling_number_second_fn,
};
use crate::core::series::{
    cumulative_max as cumulative_max_fn, cumulative_min as cumulative_min_fn,
    cumulative_product as cumulative_product_fn, cumulative_sum as cumulative_sum_fn,
    first_difference as first_difference_fn,
};
use crate::core::stat::{
    central_moment as central_moment_fn, coefficient_of_variation as coefficient_of_variation_fn,
    correlation as correlation_fn, covariance as covariance_fn,
    geometric_mean as geometric_mean_fn, harmonic_mean as harmonic_mean_fn,
    interquartile_range as interquartile_range_fn, kurtosis as kurtosis_fn, mean as mean_fn,
    mean_absolute_deviation as mean_absolute_deviation_fn, median as median_fn,
    median_absolute_deviation as median_absolute_deviation_fn, moving_average as moving_average_fn,
    percentile as percentile_fn, root_mean_square as root_mean_square_fn, skewness as skewness_fn,
    standard_deviation as std_dev_fn, variance as variance_fn, weighted_mean as weighted_mean_fn,
    z_scores as z_scores_fn,
};
use crate::core::vectors::{
    angle_between as angle_between_fn, chebyshev_distance as chebyshev_distance_fn,
    cosine_similarity as cosine_similarity_fn, cross_product as cross_product_fn, dot as dot_fn,
    euclidean_distance as euclidean_distance_fn, hadamard_product as hadamard_product_fn,
    lp_norm as lp_norm_fn, manhattan_distance as manhattan_distance_fn, norm as norm_fn,
    normalize as normalize_fn, projection as projection_fn, scalar_multiply as scalar_multiply_fn,
    vector_add as vector_add_fn, vector_subtract as vector_subtract_fn,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py_err(err: DervflowError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(name = "Core")]
pub struct PyCore;

#[pymethods]
impl PyCore {
    #[new]
    fn new() -> Self {
        Self
    }

    fn mean(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        mean_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn geometric_mean(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        geometric_mean_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn harmonic_mean(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        harmonic_mean_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn weighted_mean(
        &self,
        data: PyReadonlyArray1<f64>,
        weights: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        weighted_mean_fn(data.as_slice()?, weights.as_slice()?).map_err(to_py_err)
    }

    fn root_mean_square(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        root_mean_square_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn mean_absolute_deviation(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        mean_absolute_deviation_fn(data.as_slice()?).map_err(to_py_err)
    }

    #[pyo3(signature = (data, scale=None))]
    fn median_absolute_deviation(
        &self,
        data: PyReadonlyArray1<f64>,
        scale: Option<f64>,
    ) -> PyResult<f64> {
        median_absolute_deviation_fn(data.as_slice()?, scale).map_err(to_py_err)
    }

    fn derivative<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        spacing: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = derivative_fn(data.as_slice()?, spacing).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn second_derivative<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        spacing: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = second_derivative_fn(data.as_slice()?, spacing).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn definite_integral(&self, data: PyReadonlyArray1<f64>, spacing: f64) -> PyResult<f64> {
        definite_integral_fn(data.as_slice()?, spacing).map_err(to_py_err)
    }

    fn cumulative_integral<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        spacing: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = cumulative_integral_fn(data.as_slice()?, spacing).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    #[pyo3(signature = (data, unbiased=true))]
    fn variance(&self, data: PyReadonlyArray1<f64>, unbiased: bool) -> PyResult<f64> {
        variance_fn(data.as_slice()?, unbiased).map_err(to_py_err)
    }

    #[pyo3(signature = (data, unbiased=true))]
    fn standard_deviation(&self, data: PyReadonlyArray1<f64>, unbiased: bool) -> PyResult<f64> {
        std_dev_fn(data.as_slice()?, unbiased).map_err(to_py_err)
    }

    #[pyo3(signature = (data, unbiased=true))]
    fn coefficient_of_variation(
        &self,
        data: PyReadonlyArray1<f64>,
        unbiased: bool,
    ) -> PyResult<f64> {
        coefficient_of_variation_fn(data.as_slice()?, unbiased).map_err(to_py_err)
    }

    fn median(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        median_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn percentile(&self, data: PyReadonlyArray1<f64>, percentile: f64) -> PyResult<f64> {
        percentile_fn(data.as_slice()?, percentile).map_err(to_py_err)
    }

    fn interquartile_range(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        interquartile_range_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn skewness(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        skewness_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn kurtosis(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        kurtosis_fn(data.as_slice()?).map_err(to_py_err)
    }

    fn central_moment(&self, data: PyReadonlyArray1<f64>, order: u32) -> PyResult<f64> {
        central_moment_fn(data.as_slice()?, order).map_err(to_py_err)
    }

    fn dot(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
        dot_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
    }

    fn hadamard_product<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = hadamard_product_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn norm(&self, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
        norm_fn(data.as_slice()?).map_err(to_py_err)
    }

    #[pyo3(signature = (data, p=2.0))]
    fn lp_norm(&self, data: PyReadonlyArray1<f64>, p: f64) -> PyResult<f64> {
        lp_norm_fn(data.as_slice()?, p).map_err(to_py_err)
    }

    fn normalize<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = normalize_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn cosine_similarity(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        cosine_similarity_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
    }

    fn angle_between(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
        angle_between_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
    }

    fn euclidean_distance(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        euclidean_distance_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
    }

    fn manhattan_distance(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        manhattan_distance_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
    }

    fn chebyshev_distance(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<f64> {
        chebyshev_distance_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
    }

    #[pyo3(signature = (values, shape, spacings))]
    fn gradient<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let gradient =
            gradient_fn(values.as_slice()?, &shape, spacings.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, gradient))
    }

    #[pyo3(signature = (values, shape, spacings))]
    fn normalized_gradient<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let normalized = normalized_gradient_fn(values.as_slice()?, &shape, spacings.as_slice()?)
            .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, normalized))
    }

    #[pyo3(signature = (values, shape, spacings))]
    fn gradient_magnitude<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let magnitudes = gradient_magnitude_fn(values.as_slice()?, &shape, spacings.as_slice()?)
            .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, magnitudes))
    }

    #[pyo3(signature = (values, shape, spacings, direction))]
    fn directional_derivative<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
        direction: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let derivative = directional_derivative_fn(
            values.as_slice()?,
            &shape,
            spacings.as_slice()?,
            direction.as_slice()?,
        )
        .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, derivative))
    }

    #[pyo3(signature = (field, shape, spacings))]
    fn divergence<'py>(
        &self,
        py: Python<'py>,
        field: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let divergence =
            divergence_fn(field.as_slice()?, &shape, spacings.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, divergence))
    }

    #[pyo3(signature = (field, shape, spacings))]
    fn curl<'py>(
        &self,
        py: Python<'py>,
        field: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let curl = curl_fn(field.as_slice()?, &shape, spacings.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, curl))
    }

    #[pyo3(signature = (values, shape, spacings))]
    fn laplacian<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let laplacian =
            laplacian_fn(values.as_slice()?, &shape, spacings.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, laplacian))
    }

    #[pyo3(signature = (field, shape, spacings))]
    fn vector_laplacian<'py>(
        &self,
        py: Python<'py>,
        field: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let vector_laplacian = vector_laplacian_fn(field.as_slice()?, &shape, spacings.as_slice()?)
            .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, vector_laplacian))
    }

    #[pyo3(signature = (values, shape, spacings))]
    fn hessian<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hessian =
            hessian_fn(values.as_slice()?, &shape, spacings.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, hessian))
    }

    #[pyo3(signature = (field, shape, spacings))]
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        field: PyReadonlyArray1<f64>,
        shape: Vec<usize>,
        spacings: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let jacobian =
            jacobian_fn(field.as_slice()?, &shape, spacings.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, jacobian))
    }

    fn vector_add<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = vector_add_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn vector_subtract<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = vector_subtract_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn scalar_multiply<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        scalar: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = scalar_multiply_fn(data.as_slice()?, scalar).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn cross_product<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = cross_product_fn(a.as_slice()?, b.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn projection<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        onto: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = projection_fn(a.as_slice()?, onto.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn cumulative_sum<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = cumulative_sum_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn cumulative_product<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = cumulative_product_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn cumulative_max<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = cumulative_max_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn cumulative_min<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = cumulative_min_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn first_difference<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = first_difference_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn moving_average<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        window_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = moving_average_fn(data.as_slice()?, window_size).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn z_scores<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = z_scores_fn(data.as_slice()?).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    #[pyo3(signature = (x, y, unbiased=false))]
    fn covariance(
        &self,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        unbiased: bool,
    ) -> PyResult<f64> {
        covariance_fn(x.as_slice()?, y.as_slice()?, unbiased).map_err(to_py_err)
    }

    fn correlation(&self, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        correlation_fn(x.as_slice()?, y.as_slice()?).map_err(to_py_err)
    }

    fn factorial(&self, n: u64) -> PyResult<u128> {
        factorial(n).map_err(to_py_err)
    }

    fn permutation(&self, n: u64, k: u64) -> PyResult<u128> {
        permutation(n, k).map_err(to_py_err)
    }

    fn combination(&self, n: u64, k: u64) -> PyResult<u128> {
        combination(n, k).map_err(to_py_err)
    }

    fn falling_factorial(&self, n: u64, k: u64) -> PyResult<u128> {
        falling_factorial_fn(n, k).map_err(to_py_err)
    }

    fn rising_factorial(&self, n: u64, k: u64) -> PyResult<u128> {
        rising_factorial_fn(n, k).map_err(to_py_err)
    }

    fn binomial_probability(&self, n: u64, k: u64, p: f64) -> PyResult<f64> {
        binomial_probability_fn(n, k, p).map_err(to_py_err)
    }

    fn catalan_number(&self, n: u64) -> PyResult<u128> {
        catalan_number_fn(n).map_err(to_py_err)
    }

    fn stirling_number_second(&self, n: u64, k: u64) -> PyResult<u128> {
        stirling_number_second_fn(n, k).map_err(to_py_err)
    }

    fn stirling_number_first(&self, n: u64, k: u64) -> PyResult<u128> {
        stirling_number_first_fn(n, k).map_err(to_py_err)
    }

    fn bell_number(&self, n: u64) -> PyResult<u128> {
        bell_number_fn(n).map_err(to_py_err)
    }

    fn lah_number(&self, n: u64, k: u64) -> PyResult<u128> {
        lah_number_fn(n, k).map_err(to_py_err)
    }

    fn multinomial(&self, counts: Vec<u64>) -> PyResult<u128> {
        multinomial_fn(&counts).map_err(to_py_err)
    }
}

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "core")?;
    module.add_class::<PyCore>()?;
    parent.add_class::<PyCore>()?;
    parent.add_submodule(&module)?;
    parent.setattr("core", module)?;
    Ok(())
}
