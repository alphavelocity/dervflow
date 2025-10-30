// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Multi-factor regression models for portfolio analysis.
//!
//! This module implements ordinary least squares factor models that map
//! historical asset returns to a set of systematic factor returns. The
//! resulting exposures can be used to attribute performance, construct
//! portfolios with targeted factor tilts, or estimate expected returns from
//! factor risk premia. The routines are written entirely in Rust so they can be
//! reused by both the native APIs and the Python bindings without duplicating
//! logic.

use crate::common::error::{DervflowError, Result};
use nalgebra::{DMatrix, DVector};

/// Result of fitting an ordinary least squares factor model.
#[derive(Debug, Clone)]
pub struct FactorModel {
    include_intercept: bool,
    loadings: DMatrix<f64>,
    alphas: Vec<f64>,
    residual_variances: Vec<f64>,
    r_squared: Vec<f64>,
    n_observations: usize,
}

impl FactorModel {
    /// Fit a factor model to historical asset and factor returns.
    ///
    /// # Arguments
    /// * `asset_returns` - Matrix of shape (observations x assets)
    /// * `factor_returns` - Matrix of shape (observations x factors)
    /// * `include_intercept` - If true, estimate a regression intercept
    pub fn fit(
        asset_returns: DMatrix<f64>,
        factor_returns: DMatrix<f64>,
        include_intercept: bool,
    ) -> Result<Self> {
        let (n_obs, n_assets) = (asset_returns.nrows(), asset_returns.ncols());
        let (factor_obs, n_factors) = (factor_returns.nrows(), factor_returns.ncols());

        if n_obs == 0 {
            return Err(DervflowError::InvalidInput(
                "Asset returns must contain at least one observation".to_string(),
            ));
        }
        if factor_obs == 0 {
            return Err(DervflowError::InvalidInput(
                "Factor returns must contain at least one observation".to_string(),
            ));
        }
        if n_obs != factor_obs {
            return Err(DervflowError::InvalidInput(format!(
                "Asset returns have {} rows but factor returns have {} rows",
                n_obs, factor_obs
            )));
        }
        if n_factors == 0 {
            return Err(DervflowError::InvalidInput(
                "Factor matrix must have at least one column".to_string(),
            ));
        }
        if n_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "Asset matrix must have at least one column".to_string(),
            ));
        }

        let predictor_cols = n_factors + if include_intercept { 1 } else { 0 };
        if n_obs <= predictor_cols {
            return Err(DervflowError::InvalidInput(format!(
                "Need strictly more observations ({}) than parameters ({})",
                n_obs, predictor_cols
            )));
        }

        let design = if include_intercept {
            let mut matrix = DMatrix::zeros(n_obs, predictor_cols);
            for i in 0..n_obs {
                matrix[(i, 0)] = 1.0;
                for j in 0..n_factors {
                    matrix[(i, j + 1)] = factor_returns[(i, j)];
                }
            }
            matrix
        } else {
            factor_returns.clone()
        };

        let mut loadings = DMatrix::zeros(n_assets, n_factors);
        let mut alphas = vec![0.0; n_assets];
        let mut residual_variances = vec![0.0; n_assets];
        let mut r_squared = vec![0.0; n_assets];

        let svd = design.clone().svd(true, true);
        if svd.rank(1e-12) < predictor_cols {
            return Err(DervflowError::NumericalError(
                "Design matrix is rank deficient; factors are linearly dependent".to_string(),
            ));
        }

        for asset_idx in 0..n_assets {
            let returns = asset_returns.column(asset_idx).into_owned();
            let rhs = DMatrix::from_column_slice(n_obs, 1, returns.as_slice());
            let coeff_matrix = svd.solve(&rhs, 1e-12).map_err(|_| {
                DervflowError::NumericalError(
                    "Failed to solve regression system for asset".to_string(),
                )
            })?;
            if coeff_matrix.nrows() != predictor_cols {
                return Err(DervflowError::NumericalError(
                    "Unexpected coefficient dimension from factor regression".to_string(),
                ));
            }
            let coefficients = coeff_matrix.column(0).into_owned();

            if include_intercept {
                alphas[asset_idx] = coefficients[0];
                for j in 0..n_factors {
                    loadings[(asset_idx, j)] = coefficients[j + 1];
                }
            } else {
                alphas[asset_idx] = 0.0;
                for j in 0..n_factors {
                    loadings[(asset_idx, j)] = coefficients[j];
                }
            }

            let fitted = &design * &coefficients;
            let residuals = &returns - fitted;
            let sse: f64 = residuals.iter().map(|r| r * r).sum();
            let mean = returns.iter().sum::<f64>() / n_obs as f64;
            let tss: f64 = returns
                .iter()
                .map(|r| {
                    let diff = r - mean;
                    diff * diff
                })
                .sum();

            let dof = (n_obs - predictor_cols) as f64;
            residual_variances[asset_idx] = (sse / dof).max(0.0);
            if tss > 1e-12 {
                let r2 = 1.0 - (sse / tss);
                r_squared[asset_idx] = r2.clamp(0.0, 1.0);
            } else {
                r_squared[asset_idx] = 1.0;
            }
        }

        Ok(Self {
            include_intercept,
            loadings,
            alphas,
            residual_variances,
            r_squared,
            n_observations: n_obs,
        })
    }

    /// Whether the regression included an intercept term.
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Number of assets the model was fit on.
    pub fn n_assets(&self) -> usize {
        self.loadings.nrows()
    }

    /// Number of systematic factors.
    pub fn n_factors(&self) -> usize {
        self.loadings.ncols()
    }

    /// Number of observations used in the regression.
    pub fn n_observations(&self) -> usize {
        self.n_observations
    }

    /// Factor loadings matrix (assets x factors).
    pub fn loadings(&self) -> &DMatrix<f64> {
        &self.loadings
    }

    /// Regression intercepts for each asset.
    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Regression R² statistics for each asset.
    pub fn r_squared(&self) -> &[f64] {
        &self.r_squared
    }

    /// Residual variances for each asset.
    pub fn residual_variances(&self) -> &[f64] {
        &self.residual_variances
    }

    /// Residual standard deviations for each asset.
    pub fn residual_volatilities(&self) -> Vec<f64> {
        self.residual_variances
            .iter()
            .map(|var| var.max(0.0).sqrt())
            .collect()
    }

    /// Expected returns implied by the factor premia and regression intercepts.
    pub fn expected_returns(
        &self,
        factor_premia: &[f64],
        risk_free_rate: Option<f64>,
    ) -> Result<Vec<f64>> {
        if factor_premia.len() != self.n_factors() {
            return Err(DervflowError::InvalidInput(format!(
                "Factor premia length ({}) does not match number of factors ({})",
                factor_premia.len(),
                self.n_factors()
            )));
        }

        let rf = risk_free_rate.unwrap_or(0.0);
        let mut expected = Vec::with_capacity(self.n_assets());
        for asset in 0..self.n_assets() {
            let mut contribution = 0.0;
            for (beta, premium) in self.loadings.row(asset).iter().zip(factor_premia.iter()) {
                contribution += beta * premium;
            }
            expected.push(rf + self.alphas[asset] + contribution);
        }
        Ok(expected)
    }

    /// Factor exposure of a portfolio specified by asset weights.
    pub fn portfolio_factor_exposure(&self, weights: &[f64]) -> Result<Vec<f64>> {
        if weights.len() != self.n_assets() {
            return Err(DervflowError::InvalidInput(format!(
                "Weights length ({}) does not match number of assets ({})",
                weights.len(),
                self.n_assets()
            )));
        }

        let w = DVector::from_column_slice(weights);
        let exposure = self.loadings.transpose() * w;
        Ok(exposure.iter().copied().collect())
    }

    /// Factor contribution to expected return for the supplied weights.
    pub fn factor_attribution(&self, weights: &[f64], factor_premia: &[f64]) -> Result<Vec<f64>> {
        let exposure = self.portfolio_factor_exposure(weights)?;
        if exposure.len() != factor_premia.len() {
            return Err(DervflowError::InvalidInput(format!(
                "Factor premia length ({}) does not match exposure length ({})",
                factor_premia.len(),
                exposure.len()
            )));
        }

        Ok(exposure
            .iter()
            .zip(factor_premia.iter())
            .map(|(beta, premium)| beta * premium)
            .collect())
    }

    /// Expected portfolio return given weights and factor premia.
    pub fn portfolio_expected_return(
        &self,
        weights: &[f64],
        factor_premia: &[f64],
        risk_free_rate: Option<f64>,
    ) -> Result<f64> {
        let asset_expected = self.expected_returns(factor_premia, risk_free_rate)?;
        if asset_expected.len() != weights.len() {
            return Err(DervflowError::InvalidInput(format!(
                "Weights length ({}) does not match number of assets ({})",
                weights.len(),
                asset_expected.len()
            )));
        }

        Ok(weights
            .iter()
            .zip(asset_expected.iter())
            .map(|(w, r)| w * r)
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_matrix(data: &[&[f64]]) -> DMatrix<f64> {
        let rows = data.len();
        let cols = data.first().map(|row| row.len()).unwrap_or(0);
        let flattened: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        DMatrix::from_row_slice(rows, cols, &flattened)
    }

    #[test]
    fn test_factor_model_fit_simple() {
        let factor_returns = build_matrix(&[
            &[0.01, 0.02],
            &[0.00, -0.01],
            &[0.02, 0.01],
            &[-0.01, 0.00],
            &[0.03, 0.02],
        ]);
        let asset_returns = build_matrix(&[
            &[0.02, 0.02],
            &[-0.003, -0.006],
            &[0.023, 0.018],
            &[-0.006, -0.004],
            &[0.036, 0.03],
        ]);

        let model = FactorModel::fit(asset_returns, factor_returns, true).unwrap();
        assert_eq!(model.n_assets(), 2);
        assert_eq!(model.n_factors(), 2);
        assert!(model.include_intercept());

        let loadings = model.loadings();
        assert_relative_eq(loadings[(0, 0)], 0.8, 1e-2);
        assert_relative_eq(loadings[(0, 1)], 0.5, 1e-2);
        assert_relative_eq(loadings[(1, 0)], 0.5, 1e-2);
        assert_relative_eq(loadings[(1, 1)], 0.7, 1e-2);

        let alphas = model.alphas();
        assert_relative_eq(alphas[0], 0.002, 2e-3);
        assert_relative_eq(alphas[1], 0.001, 2e-3);

        for &r2 in model.r_squared() {
            assert!(r2 > 0.95);
        }
    }

    #[test]
    fn test_expected_returns_and_portfolio_metrics() {
        let factor_returns = build_matrix(&[&[0.01], &[0.02], &[0.0], &[-0.01], &[0.03]]);
        let asset_returns = build_matrix(&[&[0.012], &[0.025], &[0.001], &[-0.009], &[0.032]]);

        let model = FactorModel::fit(asset_returns, factor_returns.clone(), true).unwrap();
        let premia = [0.015];
        let expected = model.expected_returns(&premia, Some(0.01)).unwrap();
        assert_eq!(expected.len(), 1);
        assert!(expected[0] > 0.02 && expected[0] < 0.03);

        let weights = [1.0];
        let exposure = model.portfolio_factor_exposure(&weights).unwrap();
        assert_eq!(exposure.len(), 1);
        let attribution = model.factor_attribution(&weights, &premia).unwrap();
        assert_eq!(attribution.len(), 1);

        let portfolio_ret = model
            .portfolio_expected_return(&weights, &premia, Some(0.01))
            .unwrap();
        assert!((portfolio_ret - expected[0]).abs() < 1e-12);
    }

    fn assert_relative_eq(a: f64, b: f64, tol: f64) {
        let diff = (a - b).abs();
        let scale = b.abs().max(1.0);
        assert!(diff <= tol * scale, "{} not within {} of {}", a, tol, b);
    }
}
