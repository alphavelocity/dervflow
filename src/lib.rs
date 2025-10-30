// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! dervflow - High-performance quantitative finance library
//!
//! This library provides production-grade implementations of quantitative finance
//! algorithms including options pricing, risk analytics, portfolio optimization,
//! yield curve construction, time series analysis, and Monte Carlo simulation.
//!
//! The library is built with Rust for performance and exposed to Python via PyO3.
//!
//! # Features
//!
//! - **Options Pricing**: Black-Scholes, binomial trees, Monte Carlo, exotic options
//! - **Risk Analytics**: Greeks calculation, Value at Risk (VaR), portfolio risk metrics
//! - **Portfolio Optimization**: Mean-variance optimization, efficient frontier
//! - **Yield Curves**: Bootstrapping, interpolation, bond analytics
//! - **Time Series**: Statistical analysis, GARCH models, stationarity tests
//! - **Monte Carlo**: Stochastic process simulation, correlated paths
//! - **Numerical Methods**: Integration, root finding, optimization
//!
//! # Performance
//!
//! dervflow leverages Rust's performance characteristics to deliver 10x+ speedups
//! over pure Python implementations through:
//! - SIMD vectorization for mathematical operations
//! - Parallel processing with Rayon
//! - Zero-cost abstractions and efficient memory management
//!
//! # Examples
//!
//! ## Black-Scholes Option Pricing
//!
//! ```rust
//! use dervflow::options::analytical::{black_scholes_price};
//! use dervflow::common::types::{OptionParams, OptionType};
//!
//! let params = OptionParams::new(
//!     100.0,  // spot
//!     100.0,  // strike
//!     0.05,   // rate
//!     0.02,   // dividend
//!     0.2,    // volatility
//!     1.0,    // time to maturity
//!     OptionType::Call,
//! );
//!
//! let price = black_scholes_price(&params).unwrap();
//! println!("Option price: {:.2}", price);
//! ```
//!
//! ## Greeks Calculation
//!
//! ```rust
//! use dervflow::options::analytical::{black_scholes_greeks};
//! use dervflow::common::types::{OptionParams, OptionType};
//!
//! let params = OptionParams::new(
//!     100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call
//! );
//!
//! let greeks = black_scholes_greeks(&params).unwrap();
//! println!("Delta: {:.4}", greeks.delta);
//! println!("Gamma: {:.4}", greeks.gamma);
//! ```
//!
//! # Module Organization
//!
//! - [`common`]: Common types, traits, and error handling
//! - [`numerical`]: Foundational numerical methods (integration, optimization, root finding)
//! - [`options`]: Options pricing models and volatility analysis
//! - [`risk`]: Risk analytics including Greeks and VaR
//! - [`portfolio`]: Portfolio optimization and allocation
//! - [`yield_curve`]: Yield curve construction and bond analytics
//! - [`timeseries`]: Time series analysis and statistical models
//! - [`monte_carlo`]: Monte Carlo simulation and stochastic processes

#[cfg(feature = "python")]
use pyo3::prelude::*;

// Module declarations
pub mod common;

#[cfg(feature = "core")]
pub mod core;

#[cfg(feature = "monte_carlo")]
pub mod monte_carlo;

#[cfg(feature = "numerical")]
pub mod numerical;

#[cfg(feature = "options")]
pub mod options;

#[cfg(feature = "portfolio")]
pub mod portfolio;

#[cfg(feature = "risk")]
pub mod risk;

#[cfg(feature = "timeseries")]
pub mod timeseries;

#[cfg(feature = "yield_curve")]
pub mod yield_curve;

// Python bindings module
#[cfg(feature = "python")]
mod bindings;

/// dervflow Python module
///
/// High-performance quantitative finance library providing:
/// - Options pricing (Black-Scholes, binomial trees, Monte Carlo)
/// - Risk analytics (Greeks, VaR, portfolio risk)
/// - Portfolio optimization (mean-variance, efficient frontier)
/// - Yield curve construction and bond analytics
/// - Time series analysis (returns, stat, GARCH)
/// - Monte Carlo simulation (GBM, jump-diffusion, stochastic volatility)
#[cfg(feature = "python")]
#[pymodule]
fn _dervflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Set module version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register submodules
    #[cfg(feature = "core")]
    bindings::core::register_module(m)?;

    #[cfg(feature = "numerical")]
    bindings::numerical::register_module(m)?;

    #[cfg(feature = "options")]
    bindings::options::register_module(m)?;

    #[cfg(feature = "risk")]
    bindings::risk::register_module(m)?;

    #[cfg(feature = "risk")]
    bindings::utils::register_module(m)?;

    #[cfg(feature = "portfolio")]
    bindings::portfolio::register_module(m)?;

    #[cfg(feature = "yield_curve")]
    bindings::yield_curve::register_module(m)?;

    #[cfg(feature = "timeseries")]
    bindings::timeseries::register_module(m)?;

    #[cfg(feature = "monte_carlo")]
    bindings::monte_carlo::register_module(m)?;

    Ok(())
}
