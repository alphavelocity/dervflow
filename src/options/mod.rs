// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Options pricing module
//!
//! Provides various option pricing models:
//! - Analytical models (Black-Scholes, Black-76, Garman-Kohlhagen)
//! - Tree methods (binomial, trinomial)
//! - Monte Carlo simulation
//! - Exotic options (Asian, barrier, lookback, digital)
//! - Volatility surface construction and implied volatility

pub mod analytical;
pub mod exotic;
pub mod monte_carlo;
pub mod tree;
pub mod volatility;
