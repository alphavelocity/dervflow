// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Portfolio constraint handling
//!
//! Provides various constraint types for portfolio optimization:
//! - Box constraints (min/max weights)
//! - Sector exposure limits
//! - Turnover constraints
//! - Cardinality constraints

use crate::common::error::{DervflowError, Result};
use std::collections::HashMap;

/// Portfolio constraints for optimization
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    /// Minimum weight for each asset
    pub min_weights: Vec<f64>,
    /// Maximum weight for each asset
    pub max_weights: Vec<f64>,
    /// Sector exposure limits (sector name -> max exposure)
    pub sector_limits: Option<HashMap<String, f64>>,
    /// Maximum portfolio turnover (sum of absolute weight changes)
    pub turnover_limit: Option<f64>,
    /// Maximum number of assets to hold (cardinality constraint)
    pub max_assets: Option<usize>,
    /// Current portfolio weights (for turnover calculation)
    pub current_weights: Option<Vec<f64>>,
    /// Asset to sector mapping (asset index -> sector name)
    pub asset_sectors: Option<HashMap<usize, String>>,
}

impl PortfolioConstraints {
    /// Create a new PortfolioConstraints with default values
    ///
    /// # Arguments
    /// * `n_assets` - Number of assets in the portfolio
    ///
    /// # Returns
    /// Default constraints with min_weight=0, max_weight=1 for all assets
    pub fn new(n_assets: usize) -> Self {
        Self {
            min_weights: vec![0.0; n_assets],
            max_weights: vec![1.0; n_assets],
            sector_limits: None,
            turnover_limit: None,
            max_assets: None,
            current_weights: None,
            asset_sectors: None,
        }
    }

    /// Set box constraints (min and max weights for each asset)
    pub fn with_box_constraints(
        mut self,
        min_weights: Vec<f64>,
        max_weights: Vec<f64>,
    ) -> Result<Self> {
        if min_weights.len() != self.min_weights.len() {
            return Err(DervflowError::InvalidInput(format!(
                "min_weights length ({}) does not match number of assets ({})",
                min_weights.len(),
                self.min_weights.len()
            )));
        }

        if max_weights.len() != self.max_weights.len() {
            return Err(DervflowError::InvalidInput(format!(
                "max_weights length ({}) does not match number of assets ({})",
                max_weights.len(),
                self.max_weights.len()
            )));
        }

        // Validate that min <= max for each asset
        for i in 0..min_weights.len() {
            if min_weights[i] > max_weights[i] {
                return Err(DervflowError::InvalidInput(format!(
                    "min_weight ({}) exceeds max_weight ({}) for asset {}",
                    min_weights[i], max_weights[i], i
                )));
            }
        }

        self.min_weights = min_weights;
        self.max_weights = max_weights;
        Ok(self)
    }

    /// Set sector exposure limits
    ///
    /// # Arguments
    /// * `sector_limits` - Map of sector name to maximum exposure
    /// * `asset_sectors` - Map of asset index to sector name
    pub fn with_sector_limits(
        mut self,
        sector_limits: HashMap<String, f64>,
        asset_sectors: HashMap<usize, String>,
    ) -> Result<Self> {
        // Validate that all sector limits are between 0 and 1
        for (sector, limit) in &sector_limits {
            if *limit < 0.0 || *limit > 1.0 {
                return Err(DervflowError::InvalidInput(format!(
                    "Sector limit for {} must be between 0 and 1, got {}",
                    sector, limit
                )));
            }
        }

        // Validate that all assets have a sector assignment
        for i in 0..self.min_weights.len() {
            if !asset_sectors.contains_key(&i) {
                return Err(DervflowError::InvalidInput(format!(
                    "Asset {} does not have a sector assignment",
                    i
                )));
            }
        }

        self.sector_limits = Some(sector_limits);
        self.asset_sectors = Some(asset_sectors);
        Ok(self)
    }

    /// Set turnover constraint
    ///
    /// # Arguments
    /// * `turnover_limit` - Maximum allowed turnover (sum of absolute weight changes)
    /// * `current_weights` - Current portfolio weights
    pub fn with_turnover_limit(
        mut self,
        turnover_limit: f64,
        current_weights: Vec<f64>,
    ) -> Result<Self> {
        if turnover_limit < 0.0 {
            return Err(DervflowError::InvalidInput(format!(
                "Turnover limit must be non-negative, got {}",
                turnover_limit
            )));
        }

        if current_weights.len() != self.min_weights.len() {
            return Err(DervflowError::InvalidInput(format!(
                "current_weights length ({}) does not match number of assets ({})",
                current_weights.len(),
                self.min_weights.len()
            )));
        }

        self.turnover_limit = Some(turnover_limit);
        self.current_weights = Some(current_weights);
        Ok(self)
    }

    /// Set cardinality constraint (maximum number of assets to hold)
    pub fn with_max_assets(mut self, max_assets: usize) -> Result<Self> {
        if max_assets == 0 {
            return Err(DervflowError::InvalidInput(
                "max_assets must be at least 1".to_string(),
            ));
        }

        if max_assets > self.min_weights.len() {
            return Err(DervflowError::InvalidInput(format!(
                "max_assets ({}) exceeds number of assets ({})",
                max_assets,
                self.min_weights.len()
            )));
        }

        self.max_assets = Some(max_assets);
        Ok(self)
    }

    /// Validate that proposed weights satisfy all constraints
    ///
    /// # Arguments
    /// * `weights` - Proposed portfolio weights
    ///
    /// # Returns
    /// Ok(()) if all constraints are satisfied, Err otherwise
    pub fn validate(&self, weights: &[f64]) -> Result<()> {
        let n_assets = self.min_weights.len();

        if weights.len() != n_assets {
            return Err(DervflowError::InvalidInput(format!(
                "weights length ({}) does not match number of assets ({})",
                weights.len(),
                n_assets
            )));
        }

        // Check box constraints
        for (i, &weight) in weights.iter().enumerate().take(n_assets) {
            if weight < self.min_weights[i] - 1e-6 {
                return Err(DervflowError::OptimizationInfeasible(format!(
                    "Weight for asset {} ({:.6}) is below minimum ({:.6})",
                    i, weight, self.min_weights[i]
                )));
            }
            if weight > self.max_weights[i] + 1e-6 {
                return Err(DervflowError::OptimizationInfeasible(format!(
                    "Weight for asset {} ({:.6}) exceeds maximum ({:.6})",
                    i, weight, self.max_weights[i]
                )));
            }
        }

        // Check budget constraint (weights sum to 1)
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-4 {
            return Err(DervflowError::OptimizationInfeasible(format!(
                "Weights sum to {:.6}, expected 1.0",
                sum
            )));
        }

        // Check sector limits
        if let (Some(sector_limits), Some(asset_sectors)) =
            (&self.sector_limits, &self.asset_sectors)
        {
            let mut sector_exposures: HashMap<String, f64> = HashMap::new();

            for (i, &weight) in weights.iter().enumerate() {
                if let Some(sector) = asset_sectors.get(&i) {
                    *sector_exposures.entry(sector.clone()).or_insert(0.0) += weight;
                }
            }

            for (sector, exposure) in sector_exposures {
                if let Some(&limit) = sector_limits.get(&sector)
                    && exposure > limit + 1e-6
                {
                    return Err(DervflowError::OptimizationInfeasible(format!(
                        "Sector {} exposure ({:.6}) exceeds limit ({:.6})",
                        sector, exposure, limit
                    )));
                }
            }
        }

        // Check turnover limit
        if let (Some(turnover_limit), Some(current_weights)) =
            (self.turnover_limit, &self.current_weights)
        {
            let turnover: f64 = weights
                .iter()
                .zip(current_weights.iter())
                .map(|(new, old)| (new - old).abs())
                .sum();

            if turnover > turnover_limit + 1e-6 {
                return Err(DervflowError::OptimizationInfeasible(format!(
                    "Turnover ({:.6}) exceeds limit ({:.6})",
                    turnover, turnover_limit
                )));
            }
        }

        // Check cardinality constraint
        if let Some(max_assets) = self.max_assets {
            let n_held = weights.iter().filter(|&&w| w > 1e-6).count();
            if n_held > max_assets {
                return Err(DervflowError::OptimizationInfeasible(format!(
                    "Number of assets held ({}) exceeds maximum ({})",
                    n_held, max_assets
                )));
            }
        }

        Ok(())
    }

    /// Calculate turnover from current weights to new weights
    pub fn calculate_turnover(&self, new_weights: &[f64]) -> Result<f64> {
        if let Some(current_weights) = &self.current_weights {
            if new_weights.len() != current_weights.len() {
                return Err(DervflowError::InvalidInput(
                    "Weight vectors have different lengths".to_string(),
                ));
            }

            let turnover = new_weights
                .iter()
                .zip(current_weights.iter())
                .map(|(new, old)| (new - old).abs())
                .sum();

            Ok(turnover)
        } else {
            Err(DervflowError::InvalidInput(
                "Current weights not set".to_string(),
            ))
        }
    }

    /// Calculate sector exposures from weights
    pub fn calculate_sector_exposures(&self, weights: &[f64]) -> Result<HashMap<String, f64>> {
        if let Some(asset_sectors) = &self.asset_sectors {
            let mut sector_exposures: HashMap<String, f64> = HashMap::new();

            for (i, &weight) in weights.iter().enumerate() {
                if let Some(sector) = asset_sectors.get(&i) {
                    *sector_exposures.entry(sector.clone()).or_insert(0.0) += weight;
                }
            }

            Ok(sector_exposures)
        } else {
            Err(DervflowError::InvalidInput(
                "Asset sectors not set".to_string(),
            ))
        }
    }

    /// Count number of assets with non-zero weights
    pub fn count_active_assets(weights: &[f64], threshold: f64) -> usize {
        weights.iter().filter(|&&w| w > threshold).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraints_creation() {
        let constraints = PortfolioConstraints::new(3);
        assert_eq!(constraints.min_weights.len(), 3);
        assert_eq!(constraints.max_weights.len(), 3);
        assert!(constraints.sector_limits.is_none());
        assert!(constraints.turnover_limit.is_none());
    }

    #[test]
    fn test_box_constraints() {
        let constraints = PortfolioConstraints::new(3)
            .with_box_constraints(vec![0.1, 0.1, 0.1], vec![0.5, 0.5, 0.5]);
        assert!(constraints.is_ok());

        let constraints = constraints.unwrap();
        assert_eq!(constraints.min_weights, vec![0.1, 0.1, 0.1]);
        assert_eq!(constraints.max_weights, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_invalid_box_constraints() {
        let constraints = PortfolioConstraints::new(3)
            .with_box_constraints(vec![0.6, 0.1, 0.1], vec![0.5, 0.5, 0.5]);
        assert!(constraints.is_err());
    }

    #[test]
    fn test_sector_limits() {
        let mut sector_limits = HashMap::new();
        sector_limits.insert("Tech".to_string(), 0.4);
        sector_limits.insert("Finance".to_string(), 0.3);

        let mut asset_sectors = HashMap::new();
        asset_sectors.insert(0, "Tech".to_string());
        asset_sectors.insert(1, "Tech".to_string());
        asset_sectors.insert(2, "Finance".to_string());

        let constraints =
            PortfolioConstraints::new(3).with_sector_limits(sector_limits, asset_sectors);
        assert!(constraints.is_ok());
    }

    #[test]
    fn test_turnover_limit() {
        let current_weights = vec![0.3, 0.4, 0.3];
        let constraints = PortfolioConstraints::new(3).with_turnover_limit(0.2, current_weights);
        assert!(constraints.is_ok());
    }

    #[test]
    fn test_max_assets() {
        let constraints = PortfolioConstraints::new(5).with_max_assets(3);
        assert!(constraints.is_ok());

        let constraints = PortfolioConstraints::new(3).with_max_assets(0);
        assert!(constraints.is_err());
    }

    #[test]
    fn test_validate_weights() {
        let constraints = PortfolioConstraints::new(3);
        let weights = vec![0.3, 0.4, 0.3];
        assert!(constraints.validate(&weights).is_ok());

        // Test weights that don't sum to 1
        let bad_weights = vec![0.3, 0.3, 0.3];
        assert!(constraints.validate(&bad_weights).is_err());
    }

    #[test]
    fn test_validate_box_constraints() {
        let constraints = PortfolioConstraints::new(3)
            .with_box_constraints(vec![0.2, 0.2, 0.2], vec![0.5, 0.5, 0.5])
            .unwrap();

        let valid_weights = vec![0.3, 0.4, 0.3];
        assert!(constraints.validate(&valid_weights).is_ok());

        let invalid_weights = vec![0.1, 0.4, 0.5]; // First weight below minimum
        assert!(constraints.validate(&invalid_weights).is_err());
    }

    #[test]
    fn test_validate_sector_limits() {
        let mut sector_limits = HashMap::new();
        sector_limits.insert("Tech".to_string(), 0.5);

        let mut asset_sectors = HashMap::new();
        asset_sectors.insert(0, "Tech".to_string());
        asset_sectors.insert(1, "Tech".to_string());
        asset_sectors.insert(2, "Finance".to_string());

        let constraints = PortfolioConstraints::new(3)
            .with_sector_limits(sector_limits, asset_sectors)
            .unwrap();

        let valid_weights = vec![0.2, 0.2, 0.6]; // Tech = 0.4, within limit
        assert!(constraints.validate(&valid_weights).is_ok());

        let invalid_weights = vec![0.3, 0.3, 0.4]; // Tech = 0.6, exceeds limit
        assert!(constraints.validate(&invalid_weights).is_err());
    }

    #[test]
    fn test_calculate_turnover() {
        let current_weights = vec![0.3, 0.4, 0.3];
        let constraints = PortfolioConstraints::new(3)
            .with_turnover_limit(0.5, current_weights)
            .unwrap();

        let new_weights = vec![0.4, 0.3, 0.3];
        let turnover = constraints.calculate_turnover(&new_weights).unwrap();
        assert!((turnover - 0.2).abs() < 1e-6); // |0.4-0.3| + |0.3-0.4| + |0.3-0.3| = 0.2
    }

    #[test]
    fn test_calculate_sector_exposures() {
        let mut asset_sectors = HashMap::new();
        asset_sectors.insert(0, "Tech".to_string());
        asset_sectors.insert(1, "Tech".to_string());
        asset_sectors.insert(2, "Finance".to_string());

        let constraints = PortfolioConstraints::new(3)
            .with_sector_limits(HashMap::new(), asset_sectors)
            .unwrap();

        let weights = vec![0.2, 0.3, 0.5];
        let exposures = constraints.calculate_sector_exposures(&weights).unwrap();

        assert!((exposures["Tech"] - 0.5).abs() < 1e-6);
        assert!((exposures["Finance"] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_count_active_assets() {
        let weights = vec![0.3, 0.0001, 0.6999, 0.0];
        let count = PortfolioConstraints::count_active_assets(&weights, 0.001);
        assert_eq!(count, 2); // Only first and third weights are above threshold
    }
}
