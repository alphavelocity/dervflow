// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Quasi-random sequences (Sobol, Halton)
//!
//! Quasi-random sequence utilities for Monte Carlo simulation
//!
//! This module provides low-discrepancy sequence generators that can be used
//! to reduce variance in Monte Carlo simulations. Both Sobol and Halton
//! sequences are supported and can be transformed into standard normal draws
//! via inverse transform sampling.

use crate::common::error::{DervflowError, Result};
use crate::numerical::random::{HaltonSequence, SobolSequence};
use ndarray::{Array2, ArrayView2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};

/// Available quasi-random sequence methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuasiRandomMethod {
    /// Sobol low-discrepancy sequence
    Sobol,
    /// Halton low-discrepancy sequence
    Halton,
}

/// Quasi-random sequence generator
///
/// The generator maintains the state of the underlying low-discrepancy
/// sequence and can efficiently produce uniform samples in `[0, 1)` or
/// transform them into standard normal draws.
pub struct QuasiRandomGenerator {
    method: QuasiRandomMethod,
    dimension: usize,
    sobol: Option<SobolSequence>,
    halton: Option<HaltonSequence>,
    normal: Normal,
}

impl QuasiRandomGenerator {
    /// Create a new quasi-random generator
    ///
    /// # Arguments
    /// * `method` - Low-discrepancy sequence to use
    /// * `dimension` - Dimensionality of the sequence (number of factors)
    pub fn new(method: QuasiRandomMethod, dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(DervflowError::InvalidInput(
                "Dimension must be at least 1".to_string(),
            ));
        }

        if matches!(method, QuasiRandomMethod::Sobol) && dimension > 40 {
            return Err(DervflowError::InvalidInput(
                "Sobol sequence supports dimensions up to 40".to_string(),
            ));
        }

        let sobol = match method {
            QuasiRandomMethod::Sobol => Some(SobolSequence::new(dimension)),
            _ => None,
        };

        let halton = match method {
            QuasiRandomMethod::Halton => Some(HaltonSequence::new(dimension)),
            _ => None,
        };

        Ok(Self {
            method,
            dimension,
            sobol,
            halton,
            normal: Normal::new(0.0, 1.0).map_err(|e| {
                DervflowError::NumericalError(format!(
                    "Failed to initialise normal distribution: {}",
                    e
                ))
            })?,
        })
    }

    /// Reset the underlying sequence back to the origin
    pub fn reset(&mut self) {
        match self.method {
            QuasiRandomMethod::Sobol => {
                self.sobol = Some(SobolSequence::new(self.dimension));
            }
            QuasiRandomMethod::Halton => {
                self.halton = Some(HaltonSequence::new(self.dimension));
            }
        }
    }

    /// Skip the first `n` points of the sequence
    pub fn skip(&mut self, n: usize) {
        for _ in 0..n {
            let _ = self.next_uniform();
        }
    }

    /// Generate the next quasi-random point with uniform marginals
    pub fn next_uniform(&mut self) -> Vec<f64> {
        match self.method {
            QuasiRandomMethod::Sobol => self
                .sobol
                .as_mut()
                .expect("Sobol sequence not initialised")
                .next_point(),
            QuasiRandomMethod::Halton => self
                .halton
                .as_mut()
                .expect("Halton sequence not initialised")
                .next_point(),
        }
    }

    /// Generate the next quasi-random point transformed to standard normal
    pub fn next_normal(&mut self) -> Vec<f64> {
        self.next_uniform()
            .into_iter()
            .map(|u| {
                // Guard against numerical issues at the boundaries
                let clipped = u.clamp(1e-12, 1.0 - 1e-12);
                self.normal.inverse_cdf(clipped)
            })
            .collect()
    }

    /// Generate a matrix of uniform quasi-random points
    pub fn generate_uniform(&mut self, n: usize) -> Result<Array2<f64>> {
        if n == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of samples must be positive".to_string(),
            ));
        }

        let mut data = Array2::<f64>::zeros((n, self.dimension));

        for mut row in data.axis_iter_mut(Axis(0)) {
            let point = self.next_uniform();
            for (col_idx, value) in point.into_iter().enumerate() {
                row[col_idx] = value;
            }
        }

        Ok(data)
    }

    /// Generate a matrix of standard normal quasi-random draws
    pub fn generate_normal(&mut self, n: usize) -> Result<Array2<f64>> {
        if n == 0 {
            return Err(DervflowError::InvalidInput(
                "Number of samples must be positive".to_string(),
            ));
        }

        let mut data = Array2::<f64>::zeros((n, self.dimension));

        for mut row in data.axis_iter_mut(Axis(0)) {
            let point = self.next_normal();
            for (col_idx, value) in point.into_iter().enumerate() {
                row[col_idx] = value;
            }
        }

        Ok(data)
    }

    /// Convenience method to transform an existing matrix of uniforms to normals
    pub fn uniforms_to_normals(&self, uniforms: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut normals = Array2::<f64>::zeros(uniforms.raw_dim());

        for ((i, j), value) in uniforms.indexed_iter() {
            let clipped = value.clamp(1e-12, 1.0 - 1e-12);
            normals[(i, j)] = self.normal.inverse_cdf(clipped);
        }

        normals
    }

    /// Get the dimension of the quasi-random sequence
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the method used by the generator
    pub fn method(&self) -> QuasiRandomMethod {
        self.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_sobol_deterministic() {
        let mut gen1 = QuasiRandomGenerator::new(QuasiRandomMethod::Sobol, 2).unwrap();
        let mut gen2 = QuasiRandomGenerator::new(QuasiRandomMethod::Sobol, 2).unwrap();

        let first = gen1.next_uniform();
        let first_again = gen2.next_uniform();
        assert_relative_eq!(first[0], first_again[0], epsilon = 1e-12);
        assert_relative_eq!(first[1], first_again[1], epsilon = 1e-12);

        let second = gen1.next_uniform();
        let second_again = gen2.next_uniform();
        assert_relative_eq!(second[0], second_again[0], epsilon = 1e-12);
        assert_relative_eq!(second[1], second_again[1], epsilon = 1e-12);
    }

    #[test]
    fn test_halton_skip() {
        let mut generator = QuasiRandomGenerator::new(QuasiRandomMethod::Halton, 3).unwrap();
        let _first = generator.next_uniform();
        let second = generator.next_uniform();

        generator.reset();
        generator.skip(1);
        let skipped = generator.next_uniform();

        assert_relative_eq!(second[0], skipped[0], epsilon = 1e-12);
        assert_relative_eq!(second[1], skipped[1], epsilon = 1e-12);
        assert_relative_eq!(second[2], skipped[2], epsilon = 1e-12);
    }

    #[test]
    fn test_normal_stat() {
        let mut generator = QuasiRandomGenerator::new(QuasiRandomMethod::Sobol, 1).unwrap();
        let samples = generator.generate_normal(10_000).unwrap();

        let col = samples.column(0);
        let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
        let variance: f64 =
            col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (col.len() as f64 - 1.0);

        assert_abs_diff_eq!(mean, 0.0, epsilon = 5e-3);
        assert_abs_diff_eq!(variance, 1.0, epsilon = 5e-2);
    }
}
