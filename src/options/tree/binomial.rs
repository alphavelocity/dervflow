// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Binomial tree option pricing models
//!
//! Implements Cox-Ross-Rubinstein (CRR) and Jarrow-Rudd (JR) binomial tree methods
//! for pricing European and American options.

use crate::common::error::{DervflowError, Result};
use crate::common::types::{ExerciseStyle, OptionParams, OptionType};

/// Type of binomial tree parameterization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinomialTreeType {
    /// Cox-Ross-Rubinstein parameterization
    CoxRossRubinstein,
    /// Jarrow-Rudd parameterization
    JarrowRudd,
}

/// Price an option using a binomial tree model
///
/// # Arguments
///
/// * `params` - Option parameters (spot, strike, rate, dividend, volatility, time, option_type)
/// * `steps` - Number of time steps in the tree
/// * `style` - Exercise style (European or American)
/// * `tree_type` - Type of binomial tree (CRR or JR)
///
/// # Returns
///
/// Option price
///
/// # Errors
///
/// Returns an error if:
/// - Parameters are invalid
/// - Number of steps is zero
/// - Numerical computation fails
///
/// # Examples
///
/// ```
/// use dervflow::options::tree::binomial::{binomial_tree_price, BinomialTreeType};
/// use dervflow::common::types::{OptionParams, OptionType, ExerciseStyle};
///
/// let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
/// let price = binomial_tree_price(&params, 100, ExerciseStyle::European, BinomialTreeType::CoxRossRubinstein).unwrap();
/// ```
pub fn binomial_tree_price(
    params: &OptionParams,
    steps: usize,
    style: ExerciseStyle,
    tree_type: BinomialTreeType,
) -> Result<f64> {
    // Validate parameters
    params.validate().map_err(DervflowError::InvalidInput)?;

    if steps == 0 {
        return Err(DervflowError::InvalidInput(
            "Number of steps must be greater than zero".to_string(),
        ));
    }

    match tree_type {
        BinomialTreeType::CoxRossRubinstein => crr_binomial_tree(params, steps, style),
        BinomialTreeType::JarrowRudd => jr_binomial_tree(params, steps, style),
    }
}

/// Cox-Ross-Rubinstein binomial tree implementation
fn crr_binomial_tree(params: &OptionParams, steps: usize, style: ExerciseStyle) -> Result<f64> {
    let dt = params.time_to_maturity / steps as f64;
    let discount = (-params.rate * dt).exp();

    // CRR parameterization
    let u = (params.volatility * dt.sqrt()).exp();
    let d = 1.0 / u;
    let a = ((params.rate - params.dividend) * dt).exp();

    // Risk-neutral probability
    let p = (a - d) / (u - d);

    // Validate probability is in valid range
    if !(0.0..=1.0).contains(&p) {
        return Err(DervflowError::NumericalError(format!(
            "Risk-neutral probability {} is outside valid range [0, 1]. Check parameters.",
            p
        )));
    }

    // Build terminal stock prices and option values
    let mut option_values = vec![0.0; steps + 1];

    // Calculate option values at maturity
    for (i, option_value) in option_values.iter_mut().enumerate().take(steps + 1) {
        let stock_price = params.spot * u.powi(i as i32) * d.powi((steps - i) as i32);
        *option_value = match params.option_type {
            OptionType::Call => (stock_price - params.strike).max(0.0),
            OptionType::Put => (params.strike - stock_price).max(0.0),
        };
    }

    // Backward induction through the tree
    for step in (0..steps).rev() {
        for i in 0..=step {
            // Calculate continuation value (discounted expected value)
            let continuation_value =
                discount * (p * option_values[i + 1] + (1.0 - p) * option_values[i]);

            // For American options, check early exercise
            if style == ExerciseStyle::American {
                let stock_price = params.spot * u.powi(i as i32) * d.powi((step - i) as i32);
                let exercise_value = match params.option_type {
                    OptionType::Call => (stock_price - params.strike).max(0.0),
                    OptionType::Put => (params.strike - stock_price).max(0.0),
                };
                option_values[i] = continuation_value.max(exercise_value);
            } else {
                option_values[i] = continuation_value;
            }
        }
    }

    Ok(option_values[0])
}

/// Jarrow-Rudd binomial tree implementation
fn jr_binomial_tree(params: &OptionParams, steps: usize, style: ExerciseStyle) -> Result<f64> {
    let dt = params.time_to_maturity / steps as f64;
    let discount = (-params.rate * dt).exp();

    // Jarrow-Rudd parameterization (matches drift)
    let nu = params.rate - params.dividend - 0.5 * params.volatility * params.volatility;
    let u = (nu * dt + params.volatility * dt.sqrt()).exp();
    let d = (nu * dt - params.volatility * dt.sqrt()).exp();

    // Risk-neutral probability (0.5 for JR)
    let p = 0.5;

    // Build terminal stock prices and option values
    let mut option_values = vec![0.0; steps + 1];

    // Calculate option values at maturity
    for (i, option_value) in option_values.iter_mut().enumerate().take(steps + 1) {
        let stock_price = params.spot * u.powi(i as i32) * d.powi((steps - i) as i32);
        *option_value = match params.option_type {
            OptionType::Call => (stock_price - params.strike).max(0.0),
            OptionType::Put => (params.strike - stock_price).max(0.0),
        };
    }

    // Backward induction through the tree
    for step in (0..steps).rev() {
        for i in 0..=step {
            // Calculate continuation value (discounted expected value)
            let continuation_value =
                discount * (p * option_values[i + 1] + (1.0 - p) * option_values[i]);

            // For American options, check early exercise
            if style == ExerciseStyle::American {
                let stock_price = params.spot * u.powi(i as i32) * d.powi((step - i) as i32);
                let exercise_value = match params.option_type {
                    OptionType::Call => (stock_price - params.strike).max(0.0),
                    OptionType::Put => (params.strike - stock_price).max(0.0),
                };
                option_values[i] = continuation_value.max(exercise_value);
            } else {
                option_values[i] = continuation_value;
            }
        }
    }

    Ok(option_values[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_crr_european_call() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        // Should be close to Black-Scholes price (~10.45)
        assert!(price > 10.0 && price < 11.0);
    }

    #[test]
    fn test_crr_european_put() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);
        let price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        // Should be close to Black-Scholes price (~5.57)
        assert!(price > 5.0 && price < 6.0);
    }

    #[test]
    fn test_crr_american_put() {
        // American put should be worth more than European put
        let params = OptionParams::new(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let european_price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        let american_price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::American,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        assert!(american_price >= european_price);
    }

    #[test]
    fn test_jr_european_call() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::JarrowRudd,
        )
        .unwrap();

        // Should be close to Black-Scholes price (~10.45)
        assert!(price > 10.0 && price < 11.0);
    }

    #[test]
    fn test_jr_american_put() {
        let params = OptionParams::new(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let european_price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::JarrowRudd,
        )
        .unwrap();

        let american_price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::American,
            BinomialTreeType::JarrowRudd,
        )
        .unwrap();

        assert!(american_price >= european_price);
    }

    #[test]
    fn test_invalid_steps() {
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let result = binomial_tree_price(
            &params,
            0,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_params() {
        let params = OptionParams::new(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let result = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_convergence_to_bs() {
        // Test that binomial tree converges to Black-Scholes as steps increase
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        let price_50 = binomial_tree_price(
            &params,
            50,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        let price_200 = binomial_tree_price(
            &params,
            200,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        // Prices should be getting closer to each other as steps increase
        // Both should be close to BS price (~10.45)
        assert_relative_eq!(price_50, price_200, epsilon = 0.1);
    }

    #[test]
    fn test_crr_vs_jr_european() {
        // CRR and JR should give similar results for European options
        let params = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

        let crr_price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        let jr_price = binomial_tree_price(
            &params,
            100,
            ExerciseStyle::European,
            BinomialTreeType::JarrowRudd,
        )
        .unwrap();

        assert_relative_eq!(crr_price, jr_price, epsilon = 0.5);
    }

    #[test]
    fn test_put_call_parity_european() {
        // Test put-call parity for European options: C - P = S - K*e^(-rT)
        let params_call = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
        let params_put = OptionParams::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Put);

        let call_price = binomial_tree_price(
            &params_call,
            200,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        let put_price = binomial_tree_price(
            &params_put,
            200,
            ExerciseStyle::European,
            BinomialTreeType::CoxRossRubinstein,
        )
        .unwrap();

        let parity_diff = call_price - put_price;
        let expected_diff = params_call.spot
            - params_call.strike * (-params_call.rate * params_call.time_to_maturity).exp();

        assert_relative_eq!(parity_diff, expected_diff, epsilon = 0.1);
    }
}
