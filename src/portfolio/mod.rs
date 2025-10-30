// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Portfolio optimization module
//!
//! Provides portfolio construction and optimization:
//! - Mean-variance optimization
//! - Black-Litterman model
//! - Risk parity allocation
//! - Constraint handling
//! - Efficient frontier calculation

pub mod black_litterman;
pub mod constraints;
pub mod efficient_frontier;
pub mod factor_model;
pub mod mean_variance;
pub mod risk_parity;
