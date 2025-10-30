// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Risk analytics module
//!
//! Provides risk measurement and analysis tools:
//! - Greeks calculation (first, second, and third order)
//! - Value at Risk (VaR) using multiple methods
//! - Portfolio-level risk metrics

pub mod greeks;
pub mod metrics;
pub mod portfolio_risk;
pub mod var;
