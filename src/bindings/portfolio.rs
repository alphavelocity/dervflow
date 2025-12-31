// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for portfolio module

use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::portfolio::black_litterman::{BlackLittermanModel, InvestorViews};
use crate::portfolio::efficient_frontier::EfficientFrontier;
use crate::portfolio::factor_model::FactorModel;
use crate::portfolio::mean_variance::{
    MeanVarianceOptimizer, OptimizationResult, OptimizationTarget, calculate_portfolio_return,
    calculate_portfolio_volatility,
};
use crate::portfolio::risk_parity::RiskParityOptimizer;
use crate::risk::portfolio_risk::PortfolioSummary;

/// Portfolio optimizer using mean-variance optimization
#[pyclass(name = "PortfolioOptimizer")]
pub struct PyPortfolioOptimizer {
    optimizer: MeanVarianceOptimizer,
    expected_returns: Vec<f64>,
    covariance: DMatrix<f64>,
}

#[pymethods]
impl PyPortfolioOptimizer {
    /// Create a new portfolio optimizer
    ///
    /// Parameters
    /// ----------
    /// expected_returns : np.ndarray
    ///     Expected returns for each asset (1D array)
    /// covariance : np.ndarray
    ///     Covariance matrix (2D array, n_assets x n_assets)
    ///
    /// Returns
    /// -------
    /// PortfolioOptimizer
    ///     Portfolio optimizer instance
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> from dervflow.portfolio import PortfolioOptimizer
    /// >>> returns = np.array([0.10, 0.12, 0.08])
    /// >>> cov = np.array([[0.04, 0.01, 0.005],
    /// ...                 [0.01, 0.09, 0.01],
    /// ...                 [0.005, 0.01, 0.0225]])
    /// >>> optimizer = PortfolioOptimizer(returns, cov)
    #[new]
    #[pyo3(signature = (returns, covariance=None))]
    fn new(
        returns: &Bound<'_, PyAny>,
        covariance: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        // Check if returns is 1D or 2D
        let ndim = returns
            .getattr("ndim")
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "First argument must be a numpy array",
                )
            })?
            .extract::<usize>()?;
        let is_1d = ndim == 1;

        let (expected_returns, cov_matrix) = if is_1d {
            // 1D array: expected returns provided directly
            let returns_array: PyReadonlyArray1<f64> = returns.extract()?;
            let exp_returns = returns_array.as_slice()?.to_vec();

            // Covariance must be provided
            let cov = covariance.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Covariance matrix must be provided when returns is 1D",
                )
            })?;

            let cov_array = cov.as_array();
            let (rows, cols) = (cov_array.shape()[0], cov_array.shape()[1]);
            let cov_data: Vec<f64> = cov_array.iter().copied().collect();
            let cov_matrix = DMatrix::from_row_slice(rows, cols, &cov_data);

            (exp_returns, cov_matrix)
        } else {
            // 2D array: historical returns, calculate mean and covariance
            let returns_array: PyReadonlyArray2<f64> = returns.extract()?;
            let returns_2d = returns_array.as_array();
            let (n_periods, n_assets) = (returns_2d.shape()[0], returns_2d.shape()[1]);
            if n_assets == 0 {
                return Err(PyValueError::new_err(
                    "Historical returns must contain at least one asset",
                ));
            }
            if n_periods < 2 {
                return Err(PyValueError::new_err(
                    "Historical returns must contain at least two periods",
                ));
            }

            let n_periods_f = n_periods as f64;

            // Calculate mean returns
            let mut exp_returns = vec![0.0; n_assets];
            for i in 0..n_periods {
                for j in 0..n_assets {
                    let value = returns_2d[[i, j]];
                    if !value.is_finite() {
                        return Err(PyValueError::new_err(
                            "Historical returns contain non-finite values",
                        ));
                    }
                    exp_returns[j] += value;
                }
            }
            for value in &mut exp_returns {
                *value /= n_periods_f;
            }

            // Calculate covariance matrix (symmetric, centered)
            let mut cov_matrix = DMatrix::zeros(n_assets, n_assets);
            let denom = (n_periods - 1) as f64;
            let mut deviations = vec![0.0; n_assets];
            for k in 0..n_periods {
                for i in 0..n_assets {
                    deviations[i] = returns_2d[[k, i]] - exp_returns[i];
                }
                for i in 0..n_assets {
                    let di = deviations[i];
                    for j in 0..=i {
                        cov_matrix[(i, j)] += di * deviations[j];
                    }
                }
            }

            for i in 0..n_assets {
                for j in 0..=i {
                    let cov = cov_matrix[(i, j)] / denom;
                    cov_matrix[(i, j)] = cov;
                    if i != j {
                        cov_matrix[(j, i)] = cov;
                    }
                }
            }

            (exp_returns, cov_matrix)
        };

        let optimizer = MeanVarianceOptimizer::new(expected_returns.clone(), cov_matrix.clone())
            .map_err(PyErr::from)?;

        Ok(Self {
            optimizer,
            expected_returns,
            covariance: cov_matrix,
        })
    }

    /// Return the expected returns vector used by the optimizer.
    #[getter]
    fn expected_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_vec(py, self.expected_returns.clone()).unbind()
    }

    /// Return the covariance matrix backing the optimizer.
    #[getter]
    fn covariance_matrix(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let (rows, cols) = (self.covariance.nrows(), self.covariance.ncols());
        let data: Vec<f64> = self.covariance.iter().copied().collect();
        let array = Array2::from_shape_vec((rows, cols), data)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyArray2::from_owned_array(py, array).unbind())
    }

    /// Number of assets tracked by the optimizer.
    #[getter]
    fn num_assets(&self) -> usize {
        self.expected_returns.len()
    }

    /// Optimize portfolio for a given target
    ///
    /// Parameters
    /// ----------
    /// target_return : float, optional
    ///     Target return for minimum variance optimization
    /// target_risk : float, optional
    ///     Target risk (volatility) for maximum return optimization
    /// risk_free_rate : float, optional
    ///     Risk-free rate for Sharpe ratio maximization
    /// min_weights : np.ndarray, optional
    ///     Minimum weight for each asset (default: 0)
    /// max_weights : np.ndarray, optional
    ///     Maximum weight for each asset (default: 1)
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing:
    ///     - weights: Optimal portfolio weights
    ///     - expected_return: Expected portfolio return
    ///     - volatility: Portfolio volatility
    ///     - sharpe_ratio: Sharpe ratio (if risk_free_rate provided)
    ///     - status: Optimization status
    ///
    /// Examples
    /// --------
    /// >>> # Minimize variance for target return
    /// >>> result = optimizer.optimize(target_return=0.10)
    /// >>> print(result['weights'])
    ///
    /// >>> # Maximize Sharpe ratio
    /// >>> result = optimizer.optimize(risk_free_rate=0.03)
    /// >>> print(result['sharpe_ratio'])
    #[pyo3(signature = (target_return=None, target_risk=None, risk_free_rate=None, min_weights=None, max_weights=None))]
    fn optimize(
        &self,
        py: Python<'_>,
        target_return: Option<f64>,
        target_risk: Option<f64>,
        risk_free_rate: Option<f64>,
        min_weights: Option<PyReadonlyArray1<f64>>,
        max_weights: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        // Determine optimization target
        let target = if let Some(ret) = target_return {
            OptimizationTarget::MinimizeVariance { target_return: ret }
        } else if let Some(risk) = target_risk {
            OptimizationTarget::MaximizeReturn {
                target_variance: risk * risk,
            }
        } else if let Some(rf) = risk_free_rate {
            OptimizationTarget::MaximizeSharpeRatio { risk_free_rate: rf }
        } else {
            OptimizationTarget::MinimumVariance
        };

        // Extract weight bounds
        let min_w = min_weights.map(|arr| arr.as_slice().unwrap().to_vec());
        let max_w = max_weights.map(|arr| arr.as_slice().unwrap().to_vec());

        // Optimize
        let result = self
            .optimizer
            .optimize(target, min_w.as_deref(), max_w.as_deref())
            .map_err(PyErr::from)?;

        // Convert result to Python dict
        Ok(optimization_result_to_dict(py, &result))
    }

    /// Generate efficient frontier
    ///
    /// Parameters
    /// ----------
    /// num_points : int
    ///     Number of points to generate along the frontier
    /// min_weights : np.ndarray, optional
    ///     Minimum weight for each asset
    /// max_weights : np.ndarray, optional
    ///     Maximum weight for each asset
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     List of optimization results for each frontier point
    ///
    /// Examples
    /// --------
    /// >>> frontier = optimizer.efficient_frontier(num_points=20)
    /// >>> returns = [p['expected_return'] for p in frontier]
    /// >>> risks = [p['volatility'] for p in frontier]
    #[pyo3(signature = (num_points, min_weights=None, max_weights=None))]
    fn efficient_frontier(
        &self,
        py: Python<'_>,
        num_points: usize,
        min_weights: Option<PyReadonlyArray1<f64>>,
        max_weights: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        let frontier =
            EfficientFrontier::new(self.expected_returns.clone(), self.covariance.clone())
                .map_err(PyErr::from)?;

        let min_w = min_weights.map(|arr| arr.as_slice().unwrap().to_vec());
        let max_w = max_weights.map(|arr| arr.as_slice().unwrap().to_vec());

        let results = frontier
            .generate(num_points, min_w.as_deref(), max_w.as_deref())
            .map_err(PyErr::from)?;

        // Convert to list of dicts
        let py_list = PyList::empty(py);
        for result in results {
            py_list.append(optimization_result_to_dict(py, &result))?;
        }

        Ok(py_list.into())
    }

    /// Calculate portfolio return for given weights
    ///
    /// Parameters
    /// ----------
    /// weights : np.ndarray
    ///     Portfolio weights
    ///
    /// Returns
    /// -------
    /// float
    ///     Expected portfolio return
    #[pyo3(signature = (weights))]
    fn portfolio_return(&self, weights: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let w = weights.as_slice()?.to_vec();
        calculate_portfolio_return(&w, &self.expected_returns).map_err(PyErr::from)
    }

    /// Calculate portfolio volatility for given weights
    ///
    /// Parameters
    /// ----------
    /// weights : np.ndarray
    ///     Portfolio weights
    ///
    /// Returns
    /// -------
    /// float
    ///     Portfolio volatility (standard deviation)
    #[pyo3(signature = (weights))]
    fn portfolio_volatility(&self, weights: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let w = weights.as_slice()?.to_vec();
        calculate_portfolio_volatility(&w, &self.covariance).map_err(PyErr::from)
    }

    /// Calculate the Sharpe ratio for given weights and a risk-free rate.
    ///
    /// Parameters
    /// ----------
    /// weights : np.ndarray
    ///     Portfolio weights
    /// risk_free_rate : float
    ///     Risk-free rate used to compute the Sharpe ratio
    ///
    /// Returns
    /// -------
    /// float
    ///     Sharpe ratio value
    #[pyo3(signature = (weights, risk_free_rate))]
    fn sharpe_ratio(&self, weights: PyReadonlyArray1<f64>, risk_free_rate: f64) -> PyResult<f64> {
        let w = weights.as_slice()?.to_vec();
        self.optimizer
            .sharpe_ratio(&w, risk_free_rate)
            .map_err(PyErr::from)
    }

    /// Calculate percentage risk contributions for a set of weights.
    #[pyo3(signature = (weights))]
    fn risk_contributions(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let w = weights.as_slice()?.to_vec();
        let contributions = self.optimizer.risk_contributions(&w).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, contributions).unbind())
    }

    /// Parametric (variance-covariance) Value at Risk for the portfolio.
    #[pyo3(signature = (weights, confidence_level=0.95))]
    fn value_at_risk(
        &self,
        weights: PyReadonlyArray1<f64>,
        confidence_level: f64,
    ) -> PyResult<f64> {
        let w = weights.as_slice()?.to_vec();
        self.optimizer
            .value_at_risk(&w, confidence_level)
            .map_err(PyErr::from)
    }

    /// Parametric Conditional Value at Risk (Expected Shortfall) for the portfolio.
    #[pyo3(signature = (weights, confidence_level=0.95))]
    fn conditional_value_at_risk(
        &self,
        weights: PyReadonlyArray1<f64>,
        confidence_level: f64,
    ) -> PyResult<f64> {
        let w = weights.as_slice()?.to_vec();
        self.optimizer
            .conditional_value_at_risk(&w, confidence_level)
            .map_err(PyErr::from)
    }

    /// Comprehensive portfolio risk summary for supplied weights.
    #[pyo3(signature = (weights, risk_free_rate=None))]
    fn portfolio_summary(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        risk_free_rate: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let w = weights.as_slice()?.to_vec();
        let summary = self
            .optimizer
            .portfolio_summary(&w, risk_free_rate)
            .map_err(PyErr::from)?;
        portfolio_summary_to_dict(py, summary)
    }
}

/// Ordinary least squares multi-factor regression model.
#[pyclass(name = "FactorModel")]
pub struct PyFactorModel {
    model: FactorModel,
    factor_names: Vec<String>,
}

#[pymethods]
impl PyFactorModel {
    /// Fit a factor model from historical asset and factor returns.
    ///
    /// Parameters
    /// ----------
    /// asset_returns : np.ndarray
    ///     Matrix of asset returns with shape (observations, assets)
    /// factor_returns : np.ndarray
    ///     Matrix of factor returns with shape (observations, factors)
    /// include_intercept : bool, optional
    ///     Whether to fit an intercept term (default True)
    /// factor_names : list[str], optional
    ///     Optional names for the factors. If omitted, generic names are assigned.
    #[new]
    #[pyo3(signature = (asset_returns, factor_returns, include_intercept=true, factor_names=None))]
    fn new(
        asset_returns: PyReadonlyArray2<f64>,
        factor_returns: PyReadonlyArray2<f64>,
        include_intercept: bool,
        factor_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let assets = array2_to_dmatrix(&asset_returns)?;
        let factors = array2_to_dmatrix(&factor_returns)?;
        let model = FactorModel::fit(assets, factors, include_intercept).map_err(PyErr::from)?;

        let names = factor_names.unwrap_or_else(|| {
            (0..model.n_factors())
                .map(|idx| format!("factor_{}", idx))
                .collect::<Vec<_>>()
        });

        if names.len() != model.n_factors() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "factor_names must contain exactly {} entries",
                model.n_factors()
            )));
        }

        Ok(Self {
            model,
            factor_names: names,
        })
    }

    /// Whether the model contains an intercept term.
    #[getter]
    fn include_intercept(&self) -> bool {
        self.model.include_intercept()
    }

    /// Number of assets in the regression.
    #[getter]
    fn n_assets(&self) -> usize {
        self.model.n_assets()
    }

    /// Number of factors in the regression.
    #[getter]
    fn n_factors(&self) -> usize {
        self.model.n_factors()
    }

    /// Number of observations used in the regression.
    #[getter]
    fn n_observations(&self) -> usize {
        self.model.n_observations()
    }

    /// Names associated with each factor exposure.
    #[getter]
    fn factor_names(&self) -> Vec<String> {
        self.factor_names.clone()
    }

    /// Matrix of factor exposures (assets x factors).
    fn factor_exposures(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        dmatrix_to_pyarray(py, self.model.loadings())
    }

    /// Regression intercepts for each asset.
    fn alphas(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_slice(py, self.model.alphas()).unbind())
    }

    /// Coefficient of determination (R²) for each asset regression.
    fn r_squared(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_slice(py, self.model.r_squared()).unbind())
    }

    /// Residual variance for each asset.
    fn residual_variance(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_slice(py, self.model.residual_variances()).unbind())
    }

    /// Residual standard deviation for each asset.
    fn residual_volatility(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.model.residual_volatilities()).unbind())
    }

    /// Expected asset returns implied by the factor premia.
    #[pyo3(signature = (factor_premia, risk_free_rate=0.0))]
    fn expected_returns(
        &self,
        py: Python<'_>,
        factor_premia: PyReadonlyArray1<f64>,
        risk_free_rate: f64,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let premia = factor_premia.as_slice()?.to_vec();
        let expected = self
            .model
            .expected_returns(&premia, Some(risk_free_rate))
            .map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, expected).unbind())
    }

    /// Portfolio factor exposure for the supplied asset weights.
    fn portfolio_factor_exposure(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let w = weights.as_slice()?.to_vec();
        let exposure = self
            .model
            .portfolio_factor_exposure(&w)
            .map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, exposure).unbind())
    }

    /// Factor attribution (beta * premium) for the supplied weights.
    fn factor_attribution(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
        factor_premia: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let w = weights.as_slice()?.to_vec();
        let premia = factor_premia.as_slice()?.to_vec();
        let attribution = self
            .model
            .factor_attribution(&w, &premia)
            .map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, attribution).unbind())
    }

    /// Expected portfolio return from the factor model for given weights.
    #[pyo3(signature = (weights, factor_premia, risk_free_rate=0.0))]
    fn portfolio_expected_return(
        &self,
        weights: PyReadonlyArray1<f64>,
        factor_premia: PyReadonlyArray1<f64>,
        risk_free_rate: f64,
    ) -> PyResult<f64> {
        let w = weights.as_slice()?.to_vec();
        let premia = factor_premia.as_slice()?.to_vec();
        self.model
            .portfolio_expected_return(&w, &premia, Some(risk_free_rate))
            .map_err(PyErr::from)
    }
}

/// Risk parity portfolio optimizer
#[pyclass(name = "RiskParityOptimizer")]
pub struct PyRiskParityOptimizer {
    optimizer: RiskParityOptimizer,
}

#[pymethods]
impl PyRiskParityOptimizer {
    /// Create a new risk parity optimizer
    ///
    /// Parameters
    /// ----------
    /// covariance : np.ndarray
    ///     Covariance matrix (2D array, n_assets x n_assets)
    ///
    /// Returns
    /// -------
    /// RiskParityOptimizer
    ///     Risk parity optimizer instance
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> from dervflow.portfolio import RiskParityOptimizer
    /// >>> cov = np.array([[0.04, 0.01, 0.005],
    /// ...                 [0.01, 0.09, 0.01],
    /// ...                 [0.005, 0.01, 0.0225]])
    /// >>> optimizer = RiskParityOptimizer(cov)
    #[new]
    fn new(covariance: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let cov_array = covariance.as_array();
        let (rows, cols) = (cov_array.shape()[0], cov_array.shape()[1]);

        let cov_data: Vec<f64> = cov_array.iter().copied().collect();
        let covariance = DMatrix::from_row_slice(rows, cols, &cov_data);

        let optimizer = RiskParityOptimizer::new(covariance).map_err(PyErr::from)?;

        Ok(Self { optimizer })
    }

    /// Optimize portfolio using risk parity approach
    ///
    /// Parameters
    /// ----------
    /// target_risk_contributions : np.ndarray, optional
    ///     Target risk contributions for each asset (must sum to 1)
    ///     Default: equal risk contributions
    /// max_iterations : int, optional
    ///     Maximum number of iterations (default: 1000)
    /// tolerance : float, optional
    ///     Convergence tolerance (default: 1e-8)
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     Optimal portfolio weights
    ///
    /// Examples
    /// --------
    /// >>> # Equal risk contribution
    /// >>> weights = optimizer.optimize()
    ///
    /// >>> # Custom risk contributions
    /// >>> target_rc = np.array([0.5, 0.3, 0.2])
    /// >>> weights = optimizer.optimize(target_risk_contributions=target_rc)
    #[pyo3(signature = (target_risk_contributions=None, max_iterations=None, tolerance=None))]
    fn optimize(
        &self,
        py: Python<'_>,
        target_risk_contributions: Option<PyReadonlyArray1<f64>>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let target_rc = target_risk_contributions.map(|arr| arr.as_slice().unwrap().to_vec());

        let weights = self
            .optimizer
            .optimize(target_rc.as_deref(), max_iterations, tolerance)
            .map_err(PyErr::from)?;

        Ok(PyArray1::from_vec(py, weights).unbind())
    }

    /// Calculate risk contributions for given weights
    ///
    /// Parameters
    /// ----------
    /// weights : np.ndarray
    ///     Portfolio weights
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     Risk contributions for each asset (sum to 1)
    #[pyo3(signature = (weights))]
    fn risk_contributions(
        &self,
        py: Python<'_>,
        weights: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let w = weights.as_slice()?.to_vec();
        let rc = self
            .optimizer
            .calculate_risk_contributions(&w)
            .map_err(PyErr::from)?;

        Ok(PyArray1::from_vec(py, rc).unbind())
    }
}

/// Investor views used by the Black-Litterman model
#[pyclass(name = "InvestorViews", module = "dervflow.portfolio")]
pub struct PyInvestorViews {
    views: InvestorViews,
}

#[pymethods]
impl PyInvestorViews {
    /// Create a new set of investor views.
    ///
    /// Parameters
    /// ----------
    /// pick_matrix : np.ndarray
    ///     Matrix describing exposures of each view to the assets.
    /// view_returns : np.ndarray
    ///     Expected excess return for each view.
    #[new]
    fn new(
        pick_matrix: PyReadonlyArray2<f64>,
        view_returns: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let pick = array2_to_dmatrix(&pick_matrix)?;
        let views = array1_to_dvector(&view_returns)?;

        let investor_views = InvestorViews::new(pick, views).map_err(PyErr::from)?;
        Ok(Self {
            views: investor_views,
        })
    }

    /// Return a copy of the views with an explicit uncertainty matrix.
    ///
    /// Parameters
    /// ----------
    /// uncertainty : np.ndarray
    ///     Positive-definite matrix describing view uncertainty.
    fn with_uncertainty(&self, uncertainty: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let matrix = array2_to_dmatrix(&uncertainty)?;
        let updated = self
            .views
            .clone()
            .with_uncertainty(matrix)
            .map_err(PyErr::from)?;

        Ok(Self { views: updated })
    }

    /// Return the pick matrix used for the views.
    fn pick_matrix(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        dmatrix_to_pyarray(py, &self.views.pick_matrix)
    }

    /// Return the view return vector.
    fn view_returns(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        dvector_to_pyarray(py, &self.views.view_returns)
    }
}

/// Black-Litterman portfolio construction model
#[pyclass(name = "BlackLittermanModel", module = "dervflow.portfolio")]
pub struct PyBlackLittermanModel {
    model: BlackLittermanModel,
}

#[pymethods]
impl PyBlackLittermanModel {
    /// Create a new Black-Litterman model instance.
    ///
    /// Parameters
    /// ----------
    /// market_weights : np.ndarray
    ///     Market-capitalisation weights summing to one.
    /// covariance : np.ndarray
    ///     Prior covariance matrix of excess returns.
    /// risk_aversion : float
    ///     Risk aversion parameter used to infer equilibrium returns.
    /// tau : float
    ///     Scalar describing the uncertainty in the prior covariance.
    #[new]
    fn new(
        market_weights: PyReadonlyArray1<f64>,
        covariance: PyReadonlyArray2<f64>,
        risk_aversion: f64,
        tau: f64,
    ) -> PyResult<Self> {
        let weights = market_weights.as_slice()?.to_vec();
        let cov = array2_to_dmatrix(&covariance)?;

        let model =
            BlackLittermanModel::new(weights, cov, risk_aversion, tau).map_err(PyErr::from)?;

        Ok(Self { model })
    }

    /// Return the implied equilibrium returns (π).
    fn equilibrium_returns(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let equilibrium = self.model.implied_equilibrium_returns();
        dvector_to_pyarray(py, &equilibrium)
    }

    /// Compute the posterior distribution incorporating optional investor views.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing equilibrium returns, posterior returns,
    ///     posterior covariance matrix, and optimal weights.
    #[pyo3(signature = (views=None))]
    fn posterior(&self, py: Python<'_>, views: Option<&PyInvestorViews>) -> PyResult<Py<PyAny>> {
        let result = self
            .model
            .posterior(views.map(|v| &v.views))
            .map_err(PyErr::from)?;

        let dict = PyDict::new(py);
        dict.set_item(
            "equilibrium_returns",
            dvector_to_pyarray(py, &result.equilibrium_returns)?,
        )?;
        dict.set_item(
            "posterior_returns",
            dvector_to_pyarray(py, &result.posterior_returns)?,
        )?;
        dict.set_item(
            "posterior_covariance",
            dmatrix_to_pyarray(py, &result.posterior_covariance)?,
        )?;
        dict.set_item(
            "optimal_weights",
            dvector_to_pyarray(py, &result.optimal_weights)?,
        )?;

        Ok(dict.into())
    }
}

fn array1_to_dvector(array: &PyReadonlyArray1<f64>) -> PyResult<DVector<f64>> {
    Ok(DVector::from_row_slice(array.as_slice()?))
}

fn array2_to_dmatrix(array: &PyReadonlyArray2<f64>) -> PyResult<DMatrix<f64>> {
    let arr = array.as_array();
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
    let data: Vec<f64> = arr.iter().copied().collect();
    Ok(DMatrix::from_row_slice(rows, cols, &data))
}

fn dvector_to_pyarray(py: Python<'_>, vector: &DVector<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Ok(PyArray1::from_vec(py, vector.iter().copied().collect()).unbind())
}

fn dmatrix_to_pyarray(py: Python<'_>, matrix: &DMatrix<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let array = Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(i, j)| matrix[(i, j)]);
    Ok(array.into_pyarray(py).unbind())
}

fn portfolio_summary_to_dict(py: Python<'_>, summary: PortfolioSummary) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("expected_return", summary.expected_return)?;
    dict.set_item("variance", summary.variance)?;
    dict.set_item("volatility", summary.volatility)?;
    dict.set_item("sharpe_ratio", summary.sharpe_ratio)?;
    dict.set_item("diversification_ratio", summary.diversification_ratio)?;
    dict.set_item("weight_concentration", summary.weight_concentration)?;
    dict.set_item("risk_concentration", summary.risk_concentration)?;

    let contributions = PyDict::new(py);
    contributions.set_item(
        "marginal",
        PyArray1::from_vec(py, summary.marginal_risk).unbind(),
    )?;
    contributions.set_item(
        "component",
        PyArray1::from_vec(py, summary.component_risk).unbind(),
    )?;
    contributions.set_item(
        "percentage",
        PyArray1::from_vec(py, summary.percentage_risk).unbind(),
    )?;
    dict.set_item("risk_contributions", contributions)?;

    Ok(dict.into())
}

/// Convert OptimizationResult to Python dict
fn optimization_result_to_dict(py: Python<'_>, result: &OptimizationResult) -> Py<PyAny> {
    let dict = PyDict::new(py);
    dict.set_item("weights", PyArray1::from_vec(py, result.weights.clone()))
        .unwrap();
    dict.set_item("expected_return", result.expected_return)
        .unwrap();
    dict.set_item("volatility", result.volatility).unwrap();
    dict.set_item("sharpe_ratio", result.sharpe_ratio).unwrap();
    dict.set_item("status", result.status.clone()).unwrap();
    dict.into()
}

/// Register portfolio module with Python
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyPortfolioOptimizer>()?;
    parent.add_class::<PyFactorModel>()?;
    parent.add_class::<PyRiskParityOptimizer>()?;
    parent.add_class::<PyBlackLittermanModel>()?;
    parent.add_class::<PyInvestorViews>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_registration() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let module = PyModule::new(py, "test_portfolio").unwrap();
            assert!(register_module(&module).is_ok());
        });
    }
}
