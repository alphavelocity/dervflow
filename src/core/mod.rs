// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Foundational mathematical utilities.
//!
//! This crate-level module provides building blocks for statistical analysis,
//! vector algebra, numerical series, and combinatorics. The implementation is
//! intentionally modular so higher-level quantitative routines can depend on a
//! consistent, well-tested core without coupling to the numerical engines.

pub mod calc;
pub mod combinatorics;
pub mod series;
pub mod stat;
mod validation;
pub mod vectors;
