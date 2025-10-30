// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for numerical methods module

use crate::common::error::DervflowError;
use crate::numerical::integration::{
    IntegrationConfig, IntegrationResult, adaptive_gauss_legendre, adaptive_simpsons,
    gauss_legendre,
};
use crate::numerical::linalgops::{
    MatrixNorm, cholesky_decomposition, correlate_samples, eigen_decomposition,
    is_positive_definite, lu_decomposition, matrix_condition_number, matrix_determinant,
    matrix_exponential, matrix_inverse, matrix_multiply, matrix_norm, matrix_power, matrix_rank,
    matrix_trace, nearest_positive_definite, pseudo_inverse, qr_decomposition, solve_least_squares,
    solve_linear_system, svd_decomposition,
};
use crate::numerical::optimization::{
    BFGS, GradientDescent, NelderMead, OptimizationConfig, OptimizationResult,
};
use crate::numerical::random::{HaltonSequence, RandomGenerator, SobolSequence, ThreadLocalRng};
use crate::numerical::root_finding::{
    RootFindingConfig, RootFindingResult, bisection, brent, newton_raphson, secant,
};
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Mutex;

/// Convert Dervflow error to Python exception
fn to_py_err(err: DervflowError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn numpy_to_dmatrix(array: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
    let view = array.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(view[(i, j)]);
        }
    }
    DMatrix::from_row_slice(rows, cols, &data)
}

fn numpy_to_dvector(array: &PyReadonlyArray1<f64>) -> DVector<f64> {
    let view = array.as_array();
    DVector::from_iterator(view.len(), view.iter().copied())
}

fn dmatrix_to_py<'py>(
    py: Python<'py>,
    matrix: DMatrix<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(matrix[(i, j)]);
        }
    }
    let array = ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array))
}

fn dvector_to_py<'py>(py: Python<'py>, vector: DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    let data = vector.as_slice().to_vec();
    PyArray1::from_vec(py, data)
}

fn capture_callable_error() -> Rc<RefCell<Option<PyErr>>> {
    Rc::new(RefCell::new(None))
}

fn handle_callable_error(err_cell: Rc<RefCell<Option<PyErr>>>) -> PyResult<()> {
    if let Some(err) = err_cell.borrow_mut().take() {
        Err(err)
    } else {
        Ok(())
    }
}

#[pyclass(name = "IntegrationResult")]
pub struct PyIntegrationResult {
    value: f64,
    error_estimate: f64,
    function_evaluations: usize,
    converged: bool,
}

impl From<IntegrationResult> for PyIntegrationResult {
    fn from(result: IntegrationResult) -> Self {
        Self {
            value: result.value,
            error_estimate: result.error_estimate,
            function_evaluations: result.function_evaluations,
            converged: result.converged,
        }
    }
}

#[pymethods]
impl PyIntegrationResult {
    #[getter]
    fn value(&self) -> f64 {
        self.value
    }

    #[getter]
    fn error_estimate(&self) -> f64 {
        self.error_estimate
    }

    #[getter]
    fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "IntegrationResult(value={:.6}, error_estimate={:.3e}, evaluations={}, converged={})",
            self.value, self.error_estimate, self.function_evaluations, self.converged
        )
    }
}

fn integration_config(
    tolerance: Option<f64>,
    relative_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> IntegrationConfig {
    let mut config = IntegrationConfig::default();
    if let Some(tol) = tolerance {
        config.tolerance = tol;
    }
    if let Some(rel_tol) = relative_tolerance {
        config.relative_tolerance = rel_tol;
    }
    if let Some(max_iter) = max_iterations {
        config.max_iterations = max_iter;
    }
    config
}

fn root_config(
    tolerance: Option<f64>,
    relative_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> RootFindingConfig {
    let mut config = RootFindingConfig::default();
    if let Some(tol) = tolerance {
        config.tolerance = tol;
    }
    if let Some(rel_tol) = relative_tolerance {
        config.relative_tolerance = rel_tol;
    }
    if let Some(max_iter) = max_iterations {
        config.max_iterations = max_iter;
    }
    config
}

fn optimization_config(
    max_iterations: Option<usize>,
    f_tol: Option<f64>,
    g_tol: Option<f64>,
    x_tol: Option<f64>,
) -> OptimizationConfig {
    let mut config = OptimizationConfig::default();
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = f_tol {
        config.f_tol = value;
    }
    if let Some(value) = g_tol {
        config.g_tol = value;
    }
    if let Some(value) = x_tol {
        config.x_tol = value;
    }
    config
}

#[pyclass(name = "AdaptiveSimpsonsIntegrator")]
pub struct PyAdaptiveSimpsonsIntegrator;

#[pymethods]
impl PyAdaptiveSimpsonsIntegrator {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, a, b, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn integrate(
        &self,
        func: Py<PyAny>,
        a: f64,
        b: f64,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyIntegrationResult> {
        let err_cell = capture_callable_error();
        let func_obj = func;
        let closure_err = err_cell.clone();
        let integrand = move |x: f64| -> f64 {
            Python::attach(|py| {
                let callable = func_obj.bind(py);
                match callable.call1((x,)) {
                    Ok(value) => value.extract::<f64>().unwrap_or_else(|err| {
                        *closure_err.borrow_mut() = Some(err);
                        f64::NAN
                    }),
                    Err(err) => {
                        *closure_err.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            })
        };

        let config = integration_config(tolerance, relative_tolerance, max_iterations);
        match adaptive_simpsons(integrand, a, b, &config) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "GaussLegendreIntegrator")]
pub struct PyGaussLegendreIntegrator;

#[pymethods]
impl PyGaussLegendreIntegrator {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, a, b, n_points, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn integrate(
        &self,
        func: Py<PyAny>,
        a: f64,
        b: f64,
        n_points: usize,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyIntegrationResult> {
        let err_cell = capture_callable_error();
        let func_obj = func;
        let closure_err = err_cell.clone();
        let integrand = move |x: f64| -> f64 {
            Python::attach(|py| {
                let callable = func_obj.bind(py);
                match callable.call1((x,)) {
                    Ok(value) => value.extract::<f64>().unwrap_or_else(|err| {
                        *closure_err.borrow_mut() = Some(err);
                        f64::NAN
                    }),
                    Err(err) => {
                        *closure_err.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            })
        };

        let _config = integration_config(tolerance, relative_tolerance, max_iterations);
        match gauss_legendre(integrand, a, b, n_points) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "AdaptiveGaussLegendreIntegrator")]
pub struct PyAdaptiveGaussLegendreIntegrator;

#[pymethods]
impl PyAdaptiveGaussLegendreIntegrator {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, a, b, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn integrate(
        &self,
        func: Py<PyAny>,
        a: f64,
        b: f64,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyIntegrationResult> {
        let err_cell = capture_callable_error();
        let func_obj = func;
        let closure_err = err_cell.clone();
        let integrand = move |x: f64| -> f64 {
            Python::attach(|py| {
                let callable = func_obj.bind(py);
                match callable.call1((x,)) {
                    Ok(value) => value.extract::<f64>().unwrap_or_else(|err| {
                        *closure_err.borrow_mut() = Some(err);
                        f64::NAN
                    }),
                    Err(err) => {
                        *closure_err.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            })
        };

        let config = integration_config(tolerance, relative_tolerance, max_iterations);
        match adaptive_gauss_legendre(integrand, a, b, &config) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "RootFindingResult")]
pub struct PyRootFindingResult {
    root: f64,
    iterations: usize,
    error: f64,
    converged: bool,
}

impl From<RootFindingResult> for PyRootFindingResult {
    fn from(result: RootFindingResult) -> Self {
        Self {
            root: result.root,
            iterations: result.iterations,
            error: result.error,
            converged: result.converged,
        }
    }
}

#[pymethods]
impl PyRootFindingResult {
    #[getter]
    fn root(&self) -> f64 {
        self.root
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    #[getter]
    fn error(&self) -> f64 {
        self.error
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "RootFindingResult(root={:.6}, iterations={}, error={:.3e}, converged={})",
            self.root, self.iterations, self.error, self.converged
        )
    }
}

fn build_scalar_callable(
    func: Py<PyAny>,
    err_cell: Rc<RefCell<Option<PyErr>>>,
) -> impl Fn(f64) -> f64 {
    move |x: f64| {
        Python::attach(|py| {
            let callable = func.bind(py);
            match callable.call1((x,)) {
                Ok(value) => value.extract::<f64>().unwrap_or_else(|err| {
                    *err_cell.borrow_mut() = Some(err);
                    f64::NAN
                }),
                Err(err) => {
                    *err_cell.borrow_mut() = Some(err);
                    f64::NAN
                }
            }
        })
    }
}

#[pyclass(name = "NewtonRaphsonSolver")]
pub struct PyNewtonRaphsonSolver;

#[pymethods]
impl PyNewtonRaphsonSolver {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, derivative, initial_guess, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn solve(
        &self,
        func: Py<PyAny>,
        derivative: Py<PyAny>,
        initial_guess: f64,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyRootFindingResult> {
        let err_cell = capture_callable_error();
        let f_callable = build_scalar_callable(func, err_cell.clone());
        let df_callable = build_scalar_callable(derivative, err_cell.clone());
        let config = root_config(tolerance, relative_tolerance, max_iterations);
        match newton_raphson(f_callable, df_callable, initial_guess, &config) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "BrentSolver")]
pub struct PyBrentSolver;

#[pymethods]
impl PyBrentSolver {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, a, b, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn solve(
        &self,
        func: Py<PyAny>,
        a: f64,
        b: f64,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyRootFindingResult> {
        let err_cell = capture_callable_error();
        let callable = build_scalar_callable(func, err_cell.clone());
        let config = root_config(tolerance, relative_tolerance, max_iterations);
        match brent(callable, a, b, &config) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "BisectionSolver")]
pub struct PyBisectionSolver;

#[pymethods]
impl PyBisectionSolver {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, a, b, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn solve(
        &self,
        func: Py<PyAny>,
        a: f64,
        b: f64,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyRootFindingResult> {
        let err_cell = capture_callable_error();
        let callable = build_scalar_callable(func, err_cell.clone());
        let config = root_config(tolerance, relative_tolerance, max_iterations);
        match bisection(callable, a, b, &config) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "SecantSolver")]
pub struct PySecantSolver;

#[pymethods]
impl PySecantSolver {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (func, x0, x1, tolerance=None, relative_tolerance=None, max_iterations=None))]
    fn solve(
        &self,
        func: Py<PyAny>,
        x0: f64,
        x1: f64,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
        max_iterations: Option<usize>,
    ) -> PyResult<PyRootFindingResult> {
        let err_cell = capture_callable_error();
        let callable = build_scalar_callable(func, err_cell.clone());
        let config = root_config(tolerance, relative_tolerance, max_iterations);
        match secant(callable, x0, x1, &config) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "OptimizationResult")]
pub struct PyOptimizationResult {
    x: Vec<f64>,
    f_val: f64,
    iterations: usize,
    converged: bool,
    gradient_norm: Option<f64>,
}

impl From<OptimizationResult> for PyOptimizationResult {
    fn from(result: OptimizationResult) -> Self {
        Self {
            x: result.x,
            f_val: result.f_val,
            iterations: result.iterations,
            converged: result.converged,
            gradient_norm: result.gradient_norm,
        }
    }
}

#[pymethods]
impl PyOptimizationResult {
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.x.clone())
    }

    #[getter]
    fn f_val(&self) -> f64 {
        self.f_val
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    #[getter]
    fn gradient_norm(&self) -> Option<f64> {
        self.gradient_norm
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimizationResult(x_len={}, f_val={:.6}, iterations={}, converged={}, gradient_norm={:?})",
            self.x.len(),
            self.f_val,
            self.iterations,
            self.converged,
            self.gradient_norm
        )
    }
}

fn build_vector_callable(
    func: Py<PyAny>,
    err_cell: Rc<RefCell<Option<PyErr>>>,
) -> impl Fn(&[f64]) -> f64 {
    move |x: &[f64]| {
        Python::attach(|py| {
            let array = PyArray1::from_vec(py, x.to_vec());
            let callable = func.bind(py);
            match callable.call1((array,)) {
                Ok(value) => value.extract::<f64>().unwrap_or_else(|err| {
                    *err_cell.borrow_mut() = Some(err);
                    f64::NAN
                }),
                Err(err) => {
                    *err_cell.borrow_mut() = Some(err);
                    f64::NAN
                }
            }
        })
    }
}

fn build_gradient_callable(
    func: Py<PyAny>,
    err_cell: Rc<RefCell<Option<PyErr>>>,
) -> impl Fn(&[f64]) -> Vec<f64> {
    move |x: &[f64]| {
        Python::attach(|py| {
            let array = PyArray1::from_vec(py, x.to_vec());
            let callable = func.bind(py);
            match callable.call1((array,)) {
                Ok(value) => value.extract::<Vec<f64>>().unwrap_or_else(|err| {
                    *err_cell.borrow_mut() = Some(err);
                    vec![f64::NAN; x.len()]
                }),
                Err(err) => {
                    *err_cell.borrow_mut() = Some(err);
                    vec![f64::NAN; x.len()]
                }
            }
        })
    }
}

fn extract_initial_point(x0: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    x0.extract::<Vec<f64>>()
}

#[pyclass(name = "GradientDescentOptimizer")]
pub struct PyGradientDescentOptimizer {
    config: OptimizationConfig,
}

#[pymethods]
impl PyGradientDescentOptimizer {
    #[new]
    #[pyo3(signature = (max_iterations=None, f_tol=None, g_tol=None, x_tol=None))]
    fn new(
        max_iterations: Option<usize>,
        f_tol: Option<f64>,
        g_tol: Option<f64>,
        x_tol: Option<f64>,
    ) -> Self {
        Self {
            config: optimization_config(max_iterations, f_tol, g_tol, x_tol),
        }
    }

    fn optimize(
        &self,
        objective: Py<PyAny>,
        gradient: Py<PyAny>,
        x0: Bound<'_, PyAny>,
    ) -> PyResult<PyOptimizationResult> {
        let err_cell = capture_callable_error();
        let objective_callable = build_vector_callable(objective, err_cell.clone());
        let gradient_callable = build_gradient_callable(gradient, err_cell.clone());
        let x0_vec = extract_initial_point(&x0)?;
        let optimizer = GradientDescent::new(self.config.clone());
        match optimizer.optimize(objective_callable, gradient_callable, &x0_vec) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "BFGSOptimizer")]
pub struct PyBFGSOptimizer {
    config: OptimizationConfig,
}

#[pymethods]
impl PyBFGSOptimizer {
    #[new]
    #[pyo3(signature = (max_iterations=None, f_tol=None, g_tol=None, x_tol=None))]
    fn new(
        max_iterations: Option<usize>,
        f_tol: Option<f64>,
        g_tol: Option<f64>,
        x_tol: Option<f64>,
    ) -> Self {
        Self {
            config: optimization_config(max_iterations, f_tol, g_tol, x_tol),
        }
    }

    fn optimize(
        &self,
        objective: Py<PyAny>,
        gradient: Py<PyAny>,
        x0: Bound<'_, PyAny>,
    ) -> PyResult<PyOptimizationResult> {
        let err_cell = capture_callable_error();
        let objective_callable = build_vector_callable(objective, err_cell.clone());
        let gradient_callable = build_gradient_callable(gradient, err_cell.clone());
        let x0_vec = extract_initial_point(&x0)?;
        let optimizer = BFGS::new(self.config.clone());
        match optimizer.optimize(objective_callable, gradient_callable, &x0_vec) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "NelderMeadOptimizer")]
pub struct PyNelderMeadOptimizer {
    config: OptimizationConfig,
}

#[pymethods]
impl PyNelderMeadOptimizer {
    #[new]
    #[pyo3(signature = (max_iterations=None, f_tol=None, x_tol=None))]
    fn new(max_iterations: Option<usize>, f_tol: Option<f64>, x_tol: Option<f64>) -> Self {
        Self {
            config: optimization_config(max_iterations, f_tol, None, x_tol),
        }
    }

    fn optimize(
        &self,
        objective: Py<PyAny>,
        x0: Bound<'_, PyAny>,
    ) -> PyResult<PyOptimizationResult> {
        let err_cell = capture_callable_error();
        let objective_callable = build_vector_callable(objective, err_cell.clone());
        let x0_vec = extract_initial_point(&x0)?;
        let optimizer = NelderMead::new(self.config.clone());
        match optimizer.optimize(objective_callable, &x0_vec) {
            Ok(result) => {
                handle_callable_error(err_cell)?;
                Ok(result.into())
            }
            Err(err) => {
                if let Err(py_err) = handle_callable_error(err_cell.clone()) {
                    return Err(py_err);
                }
                Err(to_py_err(err))
            }
        }
    }
}

#[pyclass(name = "LinearAlgebra")]
pub struct PyLinearAlgebra;

#[pymethods]
impl PyLinearAlgebra {
    #[new]
    fn new() -> Self {
        Self
    }

    fn cholesky<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = numpy_to_dmatrix(&matrix);
        let l = cholesky_decomposition(&mat).map_err(to_py_err)?;
        dmatrix_to_py(py, l)
    }

    fn correlate_samples<'py>(
        &self,
        py: Python<'py>,
        correlation: PyReadonlyArray2<f64>,
        samples: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let corr = numpy_to_dmatrix(&correlation);
        let sam = numpy_to_dmatrix(&samples);
        let result = correlate_samples(&corr, &sam).map_err(to_py_err)?;
        dmatrix_to_py(py, result)
    }

    fn matrix_multiply<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat_a = numpy_to_dmatrix(&a);
        let mat_b = numpy_to_dmatrix(&b);
        let result = matrix_multiply(&mat_a, &mat_b).map_err(to_py_err)?;
        dmatrix_to_py(py, result)
    }

    #[pyo3(signature = (matrix, check_condition=true))]
    fn matrix_inverse<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
        check_condition: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = numpy_to_dmatrix(&matrix);
        let inv = matrix_inverse(&mat, check_condition).map_err(to_py_err)?;
        dmatrix_to_py(py, inv)
    }

    fn determinant(&self, matrix: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let mat = numpy_to_dmatrix(&matrix);
        matrix_determinant(&mat).map_err(to_py_err)
    }

    fn trace(&self, matrix: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let mat = numpy_to_dmatrix(&matrix);
        matrix_trace(&mat).map_err(to_py_err)
    }

    fn matrix_power<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
        power: u32,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = numpy_to_dmatrix(&matrix);
        let result = matrix_power(&mat, power).map_err(to_py_err)?;
        dmatrix_to_py(py, result)
    }

    fn solve_linear_system<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mat_a = numpy_to_dmatrix(&a);
        let vec_b = numpy_to_dvector(&b);
        let solution = solve_linear_system(&mat_a, &vec_b).map_err(to_py_err)?;
        Ok(dvector_to_py(py, solution))
    }

    fn solve_least_squares<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mat_a = numpy_to_dmatrix(&a);
        let vec_b = numpy_to_dvector(&b);
        let solution = solve_least_squares(&mat_a, &vec_b).map_err(to_py_err)?;
        Ok(dvector_to_py(py, solution))
    }

    fn eigen_decomposition<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mat = numpy_to_dmatrix(&matrix);
        let (values, vectors) = eigen_decomposition(&mat).map_err(to_py_err)?;
        Ok((dvector_to_py(py, values), dmatrix_to_py(py, vectors)?))
    }

    fn qr_decomposition<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mat = numpy_to_dmatrix(&matrix);
        let (q, r) = qr_decomposition(&mat).map_err(to_py_err)?;
        Ok((dmatrix_to_py(py, q)?, dmatrix_to_py(py, r)?))
    }

    fn svd<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let mat = numpy_to_dmatrix(&matrix);
        let (u, singular, v_t) = svd_decomposition(&mat).map_err(to_py_err)?;
        Ok((
            dmatrix_to_py(py, u)?,
            dvector_to_py(py, singular),
            dmatrix_to_py(py, v_t)?,
        ))
    }

    fn lu_decomposition<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let mat = numpy_to_dmatrix(&matrix);
        let (l, u, p) = lu_decomposition(&mat).map_err(to_py_err)?;
        Ok((
            dmatrix_to_py(py, l)?,
            dmatrix_to_py(py, u)?,
            dmatrix_to_py(py, p)?,
        ))
    }

    fn matrix_exponential<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = numpy_to_dmatrix(&matrix);
        let exp = matrix_exponential(&mat).map_err(to_py_err)?;
        dmatrix_to_py(py, exp)
    }

    #[pyo3(signature = (matrix, ord="fro"))]
    fn matrix_norm(&self, matrix: PyReadonlyArray2<f64>, ord: &str) -> PyResult<f64> {
        let mat = numpy_to_dmatrix(&matrix);
        let norm = MatrixNorm::from_str(ord).map_err(to_py_err)?;
        matrix_norm(&mat, norm).map_err(to_py_err)
    }

    #[pyo3(signature = (matrix, ord="spectral"))]
    fn condition_number(&self, matrix: PyReadonlyArray2<f64>, ord: &str) -> PyResult<f64> {
        let mat = numpy_to_dmatrix(&matrix);
        let norm = MatrixNorm::from_str(ord).map_err(to_py_err)?;
        matrix_condition_number(&mat, norm).map_err(to_py_err)
    }

    fn is_positive_definite(&self, matrix: PyReadonlyArray2<f64>) -> bool {
        let mat = numpy_to_dmatrix(&matrix);
        is_positive_definite(&mat)
    }

    fn nearest_positive_definite<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = numpy_to_dmatrix(&matrix);
        let result = nearest_positive_definite(&mat).map_err(to_py_err)?;
        dmatrix_to_py(py, result)
    }

    #[pyo3(signature = (matrix, tol=None))]
    fn pseudo_inverse<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
        tol: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = numpy_to_dmatrix(&matrix);
        let result = pseudo_inverse(&mat, tol).map_err(to_py_err)?;
        dmatrix_to_py(py, result)
    }

    #[pyo3(signature = (matrix, tol=None))]
    fn matrix_rank(&self, matrix: PyReadonlyArray2<f64>, tol: Option<f64>) -> PyResult<usize> {
        let mat = numpy_to_dmatrix(&matrix);
        matrix_rank(&mat, tol).map_err(to_py_err)
    }
}

#[pyclass(name = "RandomGenerator")]
pub struct PyRandomGenerator {
    inner: Mutex<RandomGenerator>,
}

#[pymethods]
impl PyRandomGenerator {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        let generator = match seed {
            Some(s) => RandomGenerator::new(s),
            None => RandomGenerator::from_entropy(),
        };
        Self {
            inner: Mutex::new(generator),
        }
    }

    fn standard_normal(&self) -> f64 {
        self.inner
            .lock()
            .expect("RandomGenerator mutex poisoned")
            .standard_normal()
    }

    fn normal(&self, mean: f64, std_dev: f64) -> PyResult<f64> {
        if std_dev <= 0.0 {
            return Err(PyValueError::new_err("standard deviation must be positive"));
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("RandomGenerator mutex poisoned"))?;
        Ok(guard.normal(mean, std_dev))
    }

    fn standard_normal_vec<'py>(&self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
        let samples = self
            .inner
            .lock()
            .expect("RandomGenerator mutex poisoned")
            .standard_normal_vec(n);
        PyArray1::from_vec(py, samples)
    }

    fn uniform(&self) -> f64 {
        self.inner
            .lock()
            .expect("RandomGenerator mutex poisoned")
            .uniform()
    }

    fn uniform_range(&self, a: f64, b: f64) -> f64 {
        self.inner
            .lock()
            .expect("RandomGenerator mutex poisoned")
            .uniform_range(a, b)
    }
}

#[pyclass(name = "ThreadLocalRandom")]
pub struct PyThreadLocalRandom;

#[pymethods]
impl PyThreadLocalRandom {
    #[new]
    fn new() -> Self {
        Self
    }

    fn standard_normal(&self) -> f64 {
        ThreadLocalRng::standard_normal()
    }

    fn normal(&self, mean: f64, std_dev: f64) -> PyResult<f64> {
        if std_dev <= 0.0 {
            return Err(PyValueError::new_err("standard deviation must be positive"));
        }
        Ok(ThreadLocalRng::normal(mean, std_dev))
    }

    fn uniform(&self) -> f64 {
        ThreadLocalRng::uniform()
    }

    fn seed(&self, seed: u64) {
        ThreadLocalRng::seed(seed)
    }
}

#[pyclass(name = "SobolSequence")]
pub struct PySobolSequence {
    inner: Mutex<SobolSequence>,
}

#[pymethods]
impl PySobolSequence {
    #[new]
    fn new(dimension: usize) -> Self {
        Self {
            inner: Mutex::new(SobolSequence::new(dimension)),
        }
    }

    fn next_point<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let point = self
            .inner
            .lock()
            .expect("SobolSequence mutex poisoned")
            .next_point();
        PyArray1::from_vec(py, point)
    }

    fn generate<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let points = self
            .inner
            .lock()
            .expect("SobolSequence mutex poisoned")
            .generate(n);
        let rows = points.len();
        let cols = if rows > 0 { points[0].len() } else { 0 };
        let flat: Vec<f64> = points.into_iter().flatten().collect();
        let array = ndarray::Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, array))
    }

    fn reset(&self) {
        if let Ok(mut seq) = self.inner.lock() {
            seq.reset();
        }
    }
}

#[pyclass(name = "HaltonSequence")]
pub struct PyHaltonSequence {
    inner: Mutex<HaltonSequence>,
}

#[pymethods]
impl PyHaltonSequence {
    #[new]
    fn new(dimension: usize) -> Self {
        Self {
            inner: Mutex::new(HaltonSequence::new(dimension)),
        }
    }

    fn next_point<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let point = self
            .inner
            .lock()
            .expect("HaltonSequence mutex poisoned")
            .next_point();
        PyArray1::from_vec(py, point)
    }

    fn generate<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let points = self
            .inner
            .lock()
            .expect("HaltonSequence mutex poisoned")
            .generate(n);
        let rows = points.len();
        let cols = if rows > 0 { points[0].len() } else { 0 };
        let flat: Vec<f64> = points.into_iter().flatten().collect();
        let array = ndarray::Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, array))
    }

    fn reset(&self) {
        if let Ok(mut seq) = self.inner.lock() {
            seq.reset();
        }
    }
}

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "numerical")?;

    module.add_class::<PyAdaptiveSimpsonsIntegrator>()?;
    module.add_class::<PyGaussLegendreIntegrator>()?;
    module.add_class::<PyAdaptiveGaussLegendreIntegrator>()?;
    module.add_class::<PyIntegrationResult>()?;

    module.add_class::<PyNewtonRaphsonSolver>()?;
    module.add_class::<PyBrentSolver>()?;
    module.add_class::<PyBisectionSolver>()?;
    module.add_class::<PySecantSolver>()?;
    module.add_class::<PyRootFindingResult>()?;

    module.add_class::<PyGradientDescentOptimizer>()?;
    module.add_class::<PyBFGSOptimizer>()?;
    module.add_class::<PyNelderMeadOptimizer>()?;
    module.add_class::<PyOptimizationResult>()?;

    module.add_class::<PyLinearAlgebra>()?;

    module.add_class::<PyRandomGenerator>()?;
    module.add_class::<PyThreadLocalRandom>()?;
    module.add_class::<PySobolSequence>()?;
    module.add_class::<PyHaltonSequence>()?;

    parent.add_class::<PyAdaptiveSimpsonsIntegrator>()?;
    parent.add_class::<PyGaussLegendreIntegrator>()?;
    parent.add_class::<PyAdaptiveGaussLegendreIntegrator>()?;
    parent.add_class::<PyIntegrationResult>()?;

    parent.add_class::<PyNewtonRaphsonSolver>()?;
    parent.add_class::<PyBrentSolver>()?;
    parent.add_class::<PyBisectionSolver>()?;
    parent.add_class::<PySecantSolver>()?;
    parent.add_class::<PyRootFindingResult>()?;

    parent.add_class::<PyGradientDescentOptimizer>()?;
    parent.add_class::<PyBFGSOptimizer>()?;
    parent.add_class::<PyNelderMeadOptimizer>()?;
    parent.add_class::<PyOptimizationResult>()?;

    parent.add_class::<PyLinearAlgebra>()?;

    parent.add_class::<PyRandomGenerator>()?;
    parent.add_class::<PyThreadLocalRandom>()?;
    parent.add_class::<PySobolSequence>()?;
    parent.add_class::<PyHaltonSequence>()?;

    parent.add_submodule(&module)?;
    parent.setattr("numerical", module)?;
    Ok(())
}
