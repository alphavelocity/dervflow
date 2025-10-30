// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Numerical methods foundation
//!
//! Provides core numerical algorithms including:
//! - Integration (adaptive quadrature, Gauss-Legendre)
//! - Root finding (Newton-Raphson, Brent's method)
//! - Optimization (gradient descent, BFGS, Nelder-Mead)
//! - Linear algebra operations
//! - Random number generation
//!
//! The foundational statistical and vector helpers now live in the top-level
//! [`crate::core`] module and can be imported directly from there when needed.

pub mod integration;
pub mod linalgops;
pub mod optimization;
pub mod random;
pub mod root_finding;

#[cfg(test)]
mod tests;
