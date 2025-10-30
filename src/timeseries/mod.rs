// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Time series analysis module
//!
//! Provides statistical analysis of financial time series:
//! - Return calculations
//! - Statistical moments and measures
//! - Correlation analysis
//! - Time series models (ARMA, GARCH)
//! - Statistical tests (stationarity, normality)

pub mod correlation;
pub mod models;
pub mod returns;
pub mod stat;
pub mod tests;
