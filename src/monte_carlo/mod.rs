// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Monte Carlo simulation module
//!
//! Provides stochastic process simulation:
//! - Diffusion processes (GBM, OU, CIR, Vasicek)
//! - Jump processes (Merton, Kou)
//! - Stochastic volatility models (Heston, SABR)
//! - Path generation engine
//! - Correlated multi-asset simulation
//! - Quasi-random sequences

pub mod correlation;
pub mod path_generator;
pub mod processes;
pub mod quasi_random;
