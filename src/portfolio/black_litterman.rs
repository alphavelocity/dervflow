// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Black-Litterman portfolio construction model
//!
//! The Black-Litterman framework combines market equilibrium returns with
//! investor views to produce a posterior return distribution. This module
//! implements the core Black-Litterman equations, including support for
//! arbitrary view matrices and optional view uncertainty specifications.

use crate::common::error::{DervflowError, Result};
use nalgebra::{DMatrix, DVector};

/// Container for investor views used in the Black-Litterman model
#[derive(Debug, Clone)]
pub struct InvestorViews {
    /// Pick matrix (P) selecting asset exposures for each view
    pub pick_matrix: DMatrix<f64>,
    /// View return vector (Q)
    pub view_returns: DVector<f64>,
    /// Optional view uncertainty matrix (Ω). If absent, it will be derived
    /// from the pick matrix and prior covariance.
    pub view_uncertainty: Option<DMatrix<f64>>,
}

impl InvestorViews {
    /// Create a new set of investor views
    pub fn new(pick_matrix: DMatrix<f64>, view_returns: DVector<f64>) -> Result<Self> {
        if pick_matrix.nrows() == 0 {
            return Err(DervflowError::InvalidInput(
                "Pick matrix must have at least one view".to_string(),
            ));
        }
        if pick_matrix.ncols() == 0 {
            return Err(DervflowError::InvalidInput(
                "Pick matrix must have at least one asset".to_string(),
            ));
        }
        if pick_matrix.nrows() != view_returns.len() {
            return Err(DervflowError::InvalidInput(
                "Number of view returns must match number of rows in pick matrix".to_string(),
            ));
        }

        Ok(Self {
            pick_matrix,
            view_returns,
            view_uncertainty: None,
        })
    }

    /// Provide an explicit view uncertainty matrix (Ω)
    pub fn with_uncertainty(mut self, uncertainty: DMatrix<f64>) -> Result<Self> {
        let n = self.pick_matrix.nrows();

        if uncertainty.nrows() != n || uncertainty.ncols() != n {
            return Err(DervflowError::InvalidInput(
                "View uncertainty matrix must be square with dimension equal to number of views"
                    .to_string(),
            ));
        }

        // Basic validation: diagonal elements must be positive
        for i in 0..n {
            if uncertainty[(i, i)] <= 0.0 {
                return Err(DervflowError::InvalidInput(
                    "View uncertainty must have positive diagonal entries".to_string(),
                ));
            }
        }

        self.view_uncertainty = Some(uncertainty);
        Ok(self)
    }

    fn build_uncertainty(&self, tau_cov: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        if let Some(omega) = &self.view_uncertainty {
            return Ok(omega.clone());
        }

        let projected = &self.pick_matrix * tau_cov * self.pick_matrix.transpose();
        let n = projected.nrows();
        let mut omega = DMatrix::zeros(n, n);

        for i in 0..n {
            let variance = projected[(i, i)].abs();
            if variance <= 0.0 {
                return Err(DervflowError::NumericalError(
                    "Derived view uncertainty is non-positive".to_string(),
                ));
            }
            omega[(i, i)] = variance;
        }

        Ok(omega)
    }
}

/// Result of a Black-Litterman update
#[derive(Debug, Clone)]
pub struct BlackLittermanResult {
    /// Implied equilibrium returns (π)
    pub equilibrium_returns: DVector<f64>,
    /// Posterior expected returns incorporating investor views (μ_bl)
    pub posterior_returns: DVector<f64>,
    /// Posterior covariance matrix (Σ_bl)
    pub posterior_covariance: DMatrix<f64>,
    /// Optimal portfolio weights derived from the posterior distribution
    pub optimal_weights: DVector<f64>,
}

impl BlackLittermanResult {
    /// Return optimal weights as a slice for convenience
    pub fn weights(&self) -> &[f64] {
        self.optimal_weights.as_slice()
    }
}

/// Black-Litterman model configuration
pub struct BlackLittermanModel {
    market_weights: DVector<f64>,
    covariance: DMatrix<f64>,
    risk_aversion: f64,
    tau: f64,
}

impl BlackLittermanModel {
    /// Create a new Black-Litterman model instance
    ///
    /// # Arguments
    /// * `market_weights` - Market capitalisation weights summing to one
    /// * `covariance` - Prior covariance matrix of excess returns
    /// * `risk_aversion` - Risk aversion (λ) used to infer equilibrium returns
    /// * `tau` - Scalar reflecting the uncertainty in the prior covariance
    pub fn new(
        market_weights: Vec<f64>,
        covariance: DMatrix<f64>,
        risk_aversion: f64,
        tau: f64,
    ) -> Result<Self> {
        if market_weights.is_empty() {
            return Err(DervflowError::InvalidInput(
                "Market weights vector cannot be empty".to_string(),
            ));
        }

        if covariance.nrows() != covariance.ncols() {
            return Err(DervflowError::InvalidInput(
                "Covariance matrix must be square".to_string(),
            ));
        }

        if covariance.nrows() != market_weights.len() {
            return Err(DervflowError::InvalidInput(
                "Covariance matrix dimension must match number of assets".to_string(),
            ));
        }

        if risk_aversion <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Risk aversion must be positive".to_string(),
            ));
        }

        if tau <= 0.0 {
            return Err(DervflowError::InvalidInput(
                "Tau must be positive".to_string(),
            ));
        }

        let weight_sum: f64 = market_weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-6 {
            return Err(DervflowError::InvalidInput(
                "Market weights must sum to one".to_string(),
            ));
        }

        Ok(Self {
            market_weights: DVector::from_vec(market_weights),
            covariance,
            risk_aversion,
            tau,
        })
    }

    /// Compute the implied equilibrium returns (π = λ Σ w)
    pub fn implied_equilibrium_returns(&self) -> DVector<f64> {
        &self.covariance * (&self.market_weights * self.risk_aversion)
    }

    fn tau_covariance(&self) -> DMatrix<f64> {
        &self.covariance * self.tau
    }

    /// Perform the Black-Litterman update with optional investor views
    pub fn posterior(&self, views: Option<&InvestorViews>) -> Result<BlackLittermanResult> {
        let equilibrium_returns = self.implied_equilibrium_returns();

        // If there are no views, posterior equals prior
        if views.is_none() {
            let posterior_covariance = self.covariance.clone();
            let optimal_weights =
                Self::mean_variance_weights(&posterior_covariance, &equilibrium_returns)?;

            return Ok(BlackLittermanResult {
                equilibrium_returns: equilibrium_returns.clone(),
                posterior_returns: equilibrium_returns,
                posterior_covariance,
                optimal_weights,
            });
        }

        let views = views.unwrap();
        if views.pick_matrix.ncols() != self.market_weights.len() {
            return Err(DervflowError::InvalidInput(
                "Pick matrix column count must equal number of assets".to_string(),
            ));
        }

        let tau_cov = self.tau_covariance();
        let tau_cov_inv = tau_cov.clone().try_inverse().ok_or_else(|| {
            DervflowError::NumericalError("Prior covariance is singular".to_string())
        })?;

        let omega = views.build_uncertainty(&tau_cov)?;
        let omega_inv = omega.clone().try_inverse().ok_or_else(|| {
            DervflowError::NumericalError("View uncertainty matrix is singular".to_string())
        })?;

        let p_t = views.pick_matrix.transpose();
        let m = tau_cov_inv.clone() + &p_t * &omega_inv * &views.pick_matrix;
        let m_inv = m.clone().try_inverse().ok_or_else(|| {
            DervflowError::NumericalError(
                "Failed to invert Black-Litterman system matrix".to_string(),
            )
        })?;

        let right_hand =
            tau_cov_inv * equilibrium_returns.clone() + &p_t * &omega_inv * &views.view_returns;
        let posterior_returns = m_inv.clone() * right_hand;
        let posterior_covariance = &self.covariance + &m_inv;

        let optimal_weights =
            Self::mean_variance_weights(&posterior_covariance, &posterior_returns)?;

        Ok(BlackLittermanResult {
            equilibrium_returns: equilibrium_returns.clone(),
            posterior_returns,
            posterior_covariance,
            optimal_weights,
        })
    }

    fn mean_variance_weights(
        covariance: &DMatrix<f64>,
        expected_returns: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let cov_inv = covariance.clone().try_inverse().ok_or_else(|| {
            DervflowError::NumericalError("Posterior covariance is singular".to_string())
        })?;

        let mut weights = cov_inv * expected_returns;
        let sum: f64 = weights.iter().sum();

        if sum.abs() < 1e-12 {
            return Err(DervflowError::NumericalError(
                "Failed to normalise optimal weights".to_string(),
            ));
        }

        weights.iter_mut().for_each(|w| *w /= sum);
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    fn sample_covariance() -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[0.04, 0.006, 0.006, 0.09])
    }

    #[test]
    fn test_equilibrium_returns() {
        let model =
            BlackLittermanModel::new(vec![0.6, 0.4], sample_covariance(), 3.0, 0.05).unwrap();
        let equilibrium = model.implied_equilibrium_returns();

        assert_abs_diff_eq!(equilibrium[0], 0.0792, epsilon = 1e-6);
        assert_abs_diff_eq!(equilibrium[1], 0.1188, epsilon = 1e-6);
    }

    #[test]
    fn test_posterior_without_views_matches_prior() {
        let model =
            BlackLittermanModel::new(vec![0.6, 0.4], sample_covariance(), 3.0, 0.05).unwrap();
        let result = model.posterior(None).unwrap();

        assert_relative_eq!(
            result.equilibrium_returns[0],
            result.posterior_returns[0],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            result.equilibrium_returns[1],
            result.posterior_returns[1],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            result.posterior_covariance[(0, 1)],
            sample_covariance()[(0, 1)],
            epsilon = 1e-12
        );

        let weights_sum: f64 = result.optimal_weights.iter().sum();
        assert_abs_diff_eq!(weights_sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_black_litterman_with_views() {
        let model =
            BlackLittermanModel::new(vec![0.6, 0.4], sample_covariance(), 3.0, 0.05).unwrap();

        let pick_matrix = DMatrix::from_row_slice(1, 2, &[1.0, -1.0]);
        let view_returns = DVector::from_vec(vec![0.02]);
        let views = InvestorViews::new(pick_matrix, view_returns).unwrap();

        let result = model.posterior(Some(&views)).unwrap();

        assert_abs_diff_eq!(result.posterior_returns[0], 0.08778644, epsilon = 1e-6);
        assert_abs_diff_eq!(result.posterior_returns[1], 0.09758644, epsilon = 1e-6);

        assert_abs_diff_eq!(
            result.posterior_covariance[(0, 0)],
            0.04175508,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            result.posterior_covariance[(1, 1)],
            0.09300508,
            epsilon = 1e-6
        );

        assert_abs_diff_eq!(result.optimal_weights[0], 0.68350558, epsilon = 1e-6);
        assert_abs_diff_eq!(result.optimal_weights[1], 0.31649442, epsilon = 1e-6);
    }
}
