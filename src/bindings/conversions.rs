// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! NumPy conversion utilities

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

/// Convert Python list to Rust Vec<f64>
#[allow(dead_code)]
pub fn py_list_to_vec(list: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    list.extract::<Vec<f64>>()
}

/// Convert Rust Vec<f64> to Python NumPy array
#[allow(dead_code)]
pub fn vec_to_numpy(py: Python<'_>, vec: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    PyArray1::from_vec(py, vec)
}

/// Convert 2D Rust Vec to Python NumPy array
#[allow(dead_code)]
pub fn vec2d_to_numpy(py: Python<'_>, vec: Vec<Vec<f64>>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let rows = vec.len();
    if rows == 0 {
        return Ok(PyArray2::zeros(py, (0, 0), false));
    }
    let _cols = vec[0].len();
    let flat: Vec<f64> = vec.into_iter().flatten().collect();
    Ok(PyArray2::from_vec2(py, &[flat])?)
}
