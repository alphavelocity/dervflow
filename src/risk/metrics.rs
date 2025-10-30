// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Performance and risk analytics utilities.
//!
//! The functions in this module implement common portfolio analytics used in
//! quantitative finance including Sharpe and Sortino ratios, tracking error,
//! downside deviation, and drawdown analysis.  They are written in pure Rust so
//! they can be consumed both from the Rust crate and via the PyO3 bindings.

use std::f64::EPSILON;

use crate::common::error::{DervflowError, Result};

fn ensure_non_empty(name: &str, data: &[f64]) -> Result<()> {
    if data.is_empty() {
        return Err(DervflowError::InvalidInput(format!(
            "{name} must contain at least one observation"
        )));
    }
    if !data.iter().all(|value| value.is_finite()) {
        return Err(DervflowError::InvalidInput(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn ensure_periods(periods_per_year: usize) -> Result<f64> {
    if periods_per_year == 0 {
        return Err(DervflowError::InvalidInput(
            "periods_per_year must be positive".to_string(),
        ));
    }
    Ok(periods_per_year as f64)
}

fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64], ddof: usize) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let denominator = (data.len().saturating_sub(ddof)).max(1) as f64;
    let mu = mean(data);
    data.iter()
        .map(|value| {
            let diff = value - mu;
            diff * diff
        })
        .sum::<f64>()
        / denominator
}

fn standard_deviation(data: &[f64], ddof: usize) -> f64 {
    variance(data, ddof).sqrt()
}

fn linear_quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    if data.len() == 1 {
        return data[0];
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return *sorted.last().unwrap();
    }

    let position = (sorted.len() - 1) as f64 * q;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;

    if lower_index == upper_index {
        return sorted[lower_index];
    }

    let weight = position - lower_index as f64;
    sorted[lower_index] * (1.0 - weight) + sorted[upper_index] * weight
}

/// Compound periodic returns to an annualised rate (CAGR).
pub fn annualize_returns(returns: &[f64], periods_per_year: usize) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    let periods = ensure_periods(periods_per_year)?;

    if returns.iter().any(|value| *value <= -1.0) {
        return Err(DervflowError::InvalidInput(
            "returns must be greater than -100% to compute CAGR".to_string(),
        ));
    }

    let growth = returns.iter().fold(1.0, |acc, value| acc * (1.0 + value));
    let n_periods = returns.len() as f64;
    Ok(growth.powf(periods / n_periods) - 1.0)
}

/// Annualise a volatility measure represented as a scalar standard deviation.
pub fn annualize_volatility_scalar(volatility: f64, periods_per_year: usize) -> Result<f64> {
    let periods = ensure_periods(periods_per_year)?;
    if !volatility.is_finite() {
        return Err(DervflowError::InvalidInput(
            "volatility must be finite".to_string(),
        ));
    }
    if volatility < 0.0 {
        return Err(DervflowError::InvalidInput(
            "volatility must be non-negative".to_string(),
        ));
    }

    Ok(volatility * periods.sqrt())
}

/// Annualise volatility estimated from a series of periodic returns.
pub fn annualize_volatility(returns: &[f64], periods_per_year: usize) -> Result<f64> {
    ensure_non_empty("volatility", returns)?;
    let periods = ensure_periods(periods_per_year)?;
    let ddof = if returns.len() > 1 { 1 } else { 0 };
    Ok(standard_deviation(returns, ddof) * periods.sqrt())
}

/// Calculate the Sharpe ratio from periodic returns and risk-free rates.
pub fn sharpe_ratio(returns: &[f64], risk_free: &[f64], periods_per_year: usize) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if returns.len() != risk_free.len() {
        return Err(DervflowError::InvalidInput(
            "returns and risk_free must have matching lengths".to_string(),
        ));
    }
    let periods = ensure_periods(periods_per_year)?;

    let excess: Vec<f64> = returns
        .iter()
        .zip(risk_free.iter())
        .map(|(r, rf)| r - rf)
        .collect();
    let mean_excess = mean(&excess);
    let ddof = if excess.len() > 1 { 1 } else { 0 };
    let std_excess = standard_deviation(&excess, ddof);

    if std_excess <= 0.0 {
        return Ok(0.0);
    }

    Ok((mean_excess / std_excess) * periods.sqrt())
}

/// Compute the tracking error between portfolio and benchmark returns.
pub fn tracking_error(returns: &[f64], benchmark: &[f64], periods_per_year: usize) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if returns.len() != benchmark.len() {
        return Err(DervflowError::InvalidInput(
            "returns and benchmark_returns must have matching lengths".to_string(),
        ));
    }
    let periods = ensure_periods(periods_per_year)?;
    let diff: Vec<f64> = returns
        .iter()
        .zip(benchmark.iter())
        .map(|(r, b)| r - b)
        .collect();
    let ddof = if diff.len() > 1 { 1 } else { 0 };
    Ok(standard_deviation(&diff, ddof) * periods.sqrt())
}

/// Compute the information ratio between portfolio and benchmark returns.
pub fn information_ratio(
    returns: &[f64],
    benchmark: &[f64],
    periods_per_year: usize,
) -> Result<f64> {
    let periods = ensure_periods(periods_per_year)?;
    let te = tracking_error(returns, benchmark, periods_per_year)?;
    let diff: Vec<f64> = returns
        .iter()
        .zip(benchmark.iter())
        .map(|(r, b)| r - b)
        .collect();
    let mean_diff = mean(&diff) * periods;

    if te <= EPSILON {
        return Ok(if mean_diff > 0.0 {
            f64::INFINITY
        } else if mean_diff < 0.0 {
            f64::NEG_INFINITY
        } else {
            0.0
        });
    }

    Ok(mean_diff / te)
}

/// Compute portfolio beta relative to a benchmark.
pub fn beta(returns: &[f64], benchmark: &[f64]) -> Result<f64> {
    if returns.len() != benchmark.len() {
        return Err(DervflowError::InvalidInput(
            "returns and benchmark_returns must have matching lengths".to_string(),
        ));
    }
    if returns.len() < 2 {
        return Err(DervflowError::InvalidInput(
            "at least two observations are required to compute beta".to_string(),
        ));
    }

    let var_bench = standard_deviation(benchmark, 1).powi(2);
    if var_bench <= 0.0 {
        return Err(DervflowError::InvalidInput(
            "benchmark_returns variance must be positive".to_string(),
        ));
    }

    let mean_r = mean(returns);
    let mean_b = mean(benchmark);
    let mut cov = 0.0;
    for (r, b) in returns.iter().zip(benchmark.iter()) {
        cov += (r - mean_r) * (b - mean_b);
    }
    cov /= (returns.len() - 1) as f64;
    Ok(cov / var_bench)
}

/// Compute Jensen's alpha for a portfolio relative to a benchmark and risk-free rate.
pub fn alpha(
    returns: &[f64],
    benchmark: &[f64],
    risk_free: &[f64],
    periods_per_year: usize,
) -> Result<f64> {
    if returns.len() != benchmark.len() || returns.len() != risk_free.len() {
        return Err(DervflowError::InvalidInput(
            "returns, benchmark_returns, and risk_free must have matching lengths".to_string(),
        ));
    }
    let periods = ensure_periods(periods_per_year)?;

    let beta_value = beta(returns, benchmark)?;
    let excess_port: Vec<f64> = returns
        .iter()
        .zip(risk_free.iter())
        .map(|(r, rf)| r - rf)
        .collect();
    let excess_bench: Vec<f64> = benchmark
        .iter()
        .zip(risk_free.iter())
        .map(|(b, rf)| b - rf)
        .collect();

    let mean_portfolio = mean(&excess_port) * periods;
    let mean_benchmark = mean(&excess_bench) * periods;

    Ok(mean_portfolio - beta_value * mean_benchmark)
}

/// Compute the annualised downside deviation relative to a target return.
pub fn downside_deviation(
    returns: &[f64],
    target_return: f64,
    periods_per_year: usize,
) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    let periods = ensure_periods(periods_per_year)?;
    let downside: Vec<f64> = returns
        .iter()
        .map(|value| (value - target_return).min(0.0))
        .collect();

    if downside.iter().all(|v| *v == 0.0) {
        return Ok(0.0);
    }

    let mean_square = downside.iter().map(|v| v * v).sum::<f64>() / downside.len() as f64;
    Ok(mean_square.sqrt() * periods.sqrt())
}

/// Compute the Sortino ratio from periodic returns.
pub fn sortino_ratio(
    returns: &[f64],
    risk_free: &[f64],
    target_return: f64,
    periods_per_year: usize,
) -> Result<f64> {
    if returns.len() != risk_free.len() {
        return Err(DervflowError::InvalidInput(
            "returns and risk_free must have matching lengths".to_string(),
        ));
    }
    let periods = ensure_periods(periods_per_year)?;
    let excess: Vec<f64> = returns
        .iter()
        .zip(risk_free.iter())
        .map(|(r, rf)| r - rf)
        .collect();
    let mean_excess = mean(&excess);

    let downside = downside_deviation(returns, target_return, periods_per_year)?;
    if downside == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(mean_excess * periods / downside)
}

/// Compute the Treynor ratio for a portfolio.
pub fn treynor_ratio(
    returns: &[f64],
    benchmark: &[f64],
    risk_free: &[f64],
    periods_per_year: usize,
) -> Result<f64> {
    if returns.len() != benchmark.len() || returns.len() != risk_free.len() {
        return Err(DervflowError::InvalidInput(
            "returns, benchmark_returns, and risk_free must have matching lengths".to_string(),
        ));
    }
    let periods = ensure_periods(periods_per_year)?;

    let excess: Vec<f64> = returns
        .iter()
        .zip(risk_free.iter())
        .map(|(r, rf)| r - rf)
        .collect();
    let mean_excess = mean(&excess) * periods;

    let beta_value = beta(returns, benchmark)?;
    if beta_value.abs() <= EPSILON {
        return Ok(if mean_excess > 0.0 {
            f64::INFINITY
        } else {
            0.0
        });
    }

    Ok(mean_excess / beta_value)
}

/// Compute the Omega ratio for a series of returns relative to a threshold.
pub fn omega_ratio(returns: &[f64], threshold: f64) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if !threshold.is_finite() {
        return Err(DervflowError::InvalidInput(
            "threshold must be finite".to_string(),
        ));
    }

    let mut gain_sum = 0.0;
    let mut loss_sum = 0.0;
    for value in returns {
        let gain = (*value - threshold).max(0.0);
        let loss = (threshold - *value).max(0.0);
        gain_sum += gain;
        loss_sum += loss;
    }

    let expected_gain = gain_sum / returns.len() as f64;
    let expected_loss = loss_sum / returns.len() as f64;

    if expected_loss <= EPSILON {
        return Ok(if expected_gain > 0.0 {
            f64::INFINITY
        } else {
            0.0
        });
    }

    Ok(expected_gain / expected_loss)
}

/// Compute the sample skewness of a return series.
pub fn skewness(returns: &[f64]) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if returns.len() < 3 {
        return Err(DervflowError::InvalidInput(
            "at least three observations are required to compute skewness".to_string(),
        ));
    }

    let mean_value = mean(returns);
    let std_dev = standard_deviation(returns, 1);
    if std_dev <= 0.0 {
        return Ok(0.0);
    }

    let n = returns.len() as f64;
    let mut third_moment = 0.0;
    for value in returns {
        let diff = *value - mean_value;
        third_moment += diff.powi(3);
    }

    let coefficient = n / ((n - 1.0) * (n - 2.0));
    Ok(coefficient * third_moment / std_dev.powi(3))
}

/// Compute the sample excess kurtosis of a return series.
pub fn excess_kurtosis(returns: &[f64]) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if returns.len() < 4 {
        return Err(DervflowError::InvalidInput(
            "at least four observations are required to compute excess kurtosis".to_string(),
        ));
    }

    let mean_value = mean(returns);
    let std_dev = standard_deviation(returns, 1);
    if std_dev <= 0.0 {
        return Ok(0.0);
    }

    let n = returns.len() as f64;
    let mut fourth_moment = 0.0;
    for value in returns {
        let diff = *value - mean_value;
        fourth_moment += diff.powi(4);
    }

    let factor = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    let adjustment = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
    Ok(factor * fourth_moment / std_dev.powi(4) - adjustment)
}

/// Ratio of average gains to average losses in absolute terms.
pub fn gain_loss_ratio(returns: &[f64]) -> Result<f64> {
    ensure_non_empty("returns", returns)?;

    let mut positive_sum = 0.0;
    let mut positive_count = 0.0;
    let mut negative_sum = 0.0;
    let mut negative_count = 0.0;

    for value in returns {
        if *value > 0.0 {
            positive_sum += *value;
            positive_count += 1.0;
        } else if *value < 0.0 {
            negative_sum += value.abs();
            negative_count += 1.0;
        }
    }

    if positive_count == 0.0 {
        return Ok(0.0);
    }
    if negative_count == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok((positive_sum / positive_count) / (negative_sum / negative_count))
}

/// Ratio of upper tail magnitude to lower tail magnitude at a percentile level.
pub fn tail_ratio(returns: &[f64], percentile: f64) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if !(0.5..1.0).contains(&percentile) {
        return Err(DervflowError::InvalidInput(
            "percentile must be between 0.5 and 1.0".to_string(),
        ));
    }

    let lower = linear_quantile(returns, 1.0 - percentile);
    let upper = linear_quantile(returns, percentile);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(DervflowError::NumericalError(
            "failed to compute quantiles for tail ratio".to_string(),
        ));
    }

    if lower >= 0.0 {
        return Ok(f64::INFINITY);
    }
    if upper <= 0.0 {
        return Ok(0.0);
    }

    Ok(upper / lower.abs())
}

/// Compute the upside potential ratio relative to a threshold.
pub fn upside_potential_ratio(returns: &[f64], threshold: f64) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    if !threshold.is_finite() {
        return Err(DervflowError::InvalidInput(
            "threshold must be finite".to_string(),
        ));
    }

    let mut upside_sum = 0.0;
    let mut downside_square_sum = 0.0;
    for value in returns {
        let diff = *value - threshold;
        if diff > 0.0 {
            upside_sum += diff;
        } else if diff < 0.0 {
            downside_square_sum += diff.powi(2);
        }
    }

    let count = returns.len() as f64;
    let upside_expectation = upside_sum / count;
    let downside_expectation = (downside_square_sum / count).sqrt();

    if downside_expectation <= EPSILON {
        return Ok(if upside_expectation > 0.0 {
            f64::INFINITY
        } else {
            0.0
        });
    }

    Ok(upside_expectation / downside_expectation)
}

fn ensure_capture_preconditions(returns: &[f64], benchmark: &[f64]) -> Result<()> {
    if returns.len() != benchmark.len() {
        return Err(DervflowError::InvalidInput(
            "returns and benchmark_returns must have matching lengths".to_string(),
        ));
    }
    if returns
        .iter()
        .chain(benchmark.iter())
        .any(|value| *value <= -1.0)
    {
        return Err(DervflowError::InvalidInput(
            "returns and benchmark_returns must be greater than -100% to compute capture ratios"
                .to_string(),
        ));
    }
    Ok(())
}

/// Generic capture ratio helper.
pub fn capture_ratio(
    returns: &[f64],
    benchmark: &[f64],
    condition: impl Fn(f64) -> bool,
    condition_name: &str,
    periods_per_year: usize,
) -> Result<f64> {
    ensure_non_empty("returns", returns)?;
    ensure_capture_preconditions(returns, benchmark)?;
    let periods = ensure_periods(periods_per_year)?;

    let mut subset_returns = Vec::new();
    let mut subset_benchmark = Vec::new();

    for (r, b) in returns.iter().zip(benchmark.iter()) {
        if condition(*b) {
            subset_returns.push(*r);
            subset_benchmark.push(*b);
        }
    }

    if subset_benchmark.is_empty() {
        return Err(DervflowError::InvalidInput(format!(
            "benchmark_returns must contain at least one {condition_name} observation"
        )));
    }

    let periods_count = subset_returns.len() as f64;
    let portfolio_growth = subset_returns
        .iter()
        .fold(1.0, |acc, value| acc * (1.0 + value));
    let benchmark_growth = subset_benchmark
        .iter()
        .fold(1.0, |acc, value| acc * (1.0 + value));

    let portfolio_cagr = portfolio_growth.powf(periods / periods_count) - 1.0;
    let benchmark_cagr = benchmark_growth.powf(periods / periods_count) - 1.0;

    if benchmark_cagr.abs() <= EPSILON {
        return Ok(if portfolio_cagr > 0.0 {
            f64::INFINITY
        } else {
            0.0
        });
    }

    Ok(portfolio_cagr / benchmark_cagr)
}

/// Upside capture ratio.
pub fn upside_capture_ratio(
    returns: &[f64],
    benchmark: &[f64],
    periods_per_year: usize,
) -> Result<f64> {
    capture_ratio(
        returns,
        benchmark,
        |v| v > 0.0,
        "positive",
        periods_per_year,
    )
}

/// Downside capture ratio.
pub fn downside_capture_ratio(
    returns: &[f64],
    benchmark: &[f64],
    periods_per_year: usize,
) -> Result<f64> {
    capture_ratio(
        returns,
        benchmark,
        |v| v < 0.0,
        "negative",
        periods_per_year,
    )
}

/// Compute the running drawdown series for a price path.
pub fn drawdown_series(prices: &[f64]) -> Result<Vec<f64>> {
    ensure_non_empty("prices", prices)?;
    if prices.iter().any(|price| *price <= 0.0) {
        return Err(DervflowError::InvalidInput(
            "prices must be strictly positive to compute drawdowns".to_string(),
        ));
    }

    let mut running_max = f64::NEG_INFINITY;
    let mut drawdowns = Vec::with_capacity(prices.len());

    for price in prices {
        running_max = running_max.max(*price);
        drawdowns.push(price / running_max - 1.0);
    }

    Ok(drawdowns)
}

/// Compute the maximum drawdown of a price series.
pub fn max_drawdown(prices: &[f64]) -> Result<f64> {
    let drawdowns = drawdown_series(prices)?;
    Ok(drawdowns.into_iter().fold(0.0, |acc, value| acc.min(value)))
}

/// Compute the pain index – the mean magnitude of drawdowns.
pub fn pain_index(prices: &[f64]) -> Result<f64> {
    let drawdowns = drawdown_series(prices)?;
    let total = drawdowns.iter().map(|value| value.abs()).sum::<f64>();
    Ok(total / drawdowns.len() as f64)
}

/// Compute the ulcer index – the root mean square of drawdowns.
pub fn ulcer_index(prices: &[f64]) -> Result<f64> {
    let drawdowns = drawdown_series(prices)?;
    let mean_square =
        drawdowns.iter().map(|value| value.powi(2)).sum::<f64>() / drawdowns.len() as f64;
    Ok(mean_square.sqrt())
}

/// Compute the Calmar ratio given an annual return and maximum drawdown.
pub fn calmar_ratio(annual_return: f64, max_drawdown_value: f64) -> Result<f64> {
    if !annual_return.is_finite() {
        return Err(DervflowError::InvalidInput(
            "annual_return must be finite".to_string(),
        ));
    }
    if !max_drawdown_value.is_finite() {
        return Err(DervflowError::InvalidInput(
            "max_drawdown must be finite".to_string(),
        ));
    }

    let magnitude = max_drawdown_value.abs();
    if magnitude <= EPSILON {
        return Ok(if annual_return > 0.0 {
            f64::INFINITY
        } else {
            0.0
        });
    }

    Ok(annual_return / magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annualize_returns() {
        let returns = [0.01, -0.02, 0.015];
        let expected = returns.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r));
        let expected = expected.powf(252.0 / returns.len() as f64) - 1.0;
        let result = annualize_returns(&returns, 252).unwrap();
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = [0.01, 0.02, -0.005, 0.015];
        let risk_free = [0.001; 4];
        let excess: Vec<f64> = returns
            .iter()
            .zip(risk_free.iter())
            .map(|(r, rf)| r - rf)
            .collect();
        let expected = mean(&excess) / standard_deviation(&excess, 1) * 252.0f64.sqrt();
        let result = sharpe_ratio(&returns, &risk_free, 252).unwrap();
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_tracking_and_information_ratio() {
        let returns = [0.01, 0.015, -0.005, 0.02];
        let benchmark = [0.008, 0.012, -0.004, 0.018];
        let diff: Vec<f64> = returns
            .iter()
            .zip(benchmark.iter())
            .map(|(r, b)| r - b)
            .collect();
        let te_expected = standard_deviation(&diff, 1) * 12.0f64.sqrt();
        let te = tracking_error(&returns, &benchmark, 12).unwrap();
        assert!((te - te_expected).abs() < 1e-12);

        let mean_diff = mean(&diff) * 12.0;
        let ir_expected = mean_diff / te_expected;
        let ir = information_ratio(&returns, &benchmark, 12).unwrap();
        assert!((ir - ir_expected).abs() < 1e-12);
    }

    #[test]
    fn test_information_ratio_zero_tracking_error_signs() {
        let returns = [0.02, 0.02, 0.02];
        let benchmark = [0.01, 0.01, 0.01];
        let ir = information_ratio(&returns, &benchmark, 12).unwrap();
        assert_eq!(ir, f64::INFINITY);

        let returns_under = [0.01, 0.01, 0.01];
        let benchmark_over = [0.02, 0.02, 0.02];
        let ir_under = information_ratio(&returns_under, &benchmark_over, 12).unwrap();
        assert_eq!(ir_under, f64::NEG_INFINITY);

        let returns_equal = [0.015, 0.015, 0.015];
        let benchmark_equal = [0.015, 0.015, 0.015];
        let ir_equal = information_ratio(&returns_equal, &benchmark_equal, 12).unwrap();
        assert_eq!(ir_equal, 0.0);
    }

    #[test]
    fn test_beta_and_alpha() {
        let returns = [0.01, 0.015, -0.005, 0.02];
        let benchmark = [0.008, 0.012, -0.004, 0.018];
        let risk_free = [0.0005; 4];
        let beta_value = beta(&returns, &benchmark).unwrap();
        let cov = returns
            .iter()
            .zip(benchmark.iter())
            .map(|(r, b)| (r - mean(&returns)) * (b - mean(&benchmark)))
            .sum::<f64>()
            / (returns.len() - 1) as f64;
        let var = standard_deviation(&benchmark, 1).powi(2);
        assert!((beta_value - cov / var).abs() < 1e-12);

        let alpha_value = alpha(&returns, &benchmark, &risk_free, 252).unwrap();
        let excess_port: Vec<f64> = returns
            .iter()
            .zip(risk_free.iter())
            .map(|(r, rf)| r - rf)
            .collect();
        let excess_bench: Vec<f64> = benchmark
            .iter()
            .zip(risk_free.iter())
            .map(|(b, rf)| b - rf)
            .collect();
        let expected = mean(&excess_port) * 252.0 - beta_value * mean(&excess_bench) * 252.0;
        assert!((alpha_value - expected).abs() < 1e-12);
    }

    #[test]
    fn test_downside_and_sortino() {
        let returns = [0.01, -0.02, 0.015, 0.005];
        let risk_free = [0.0; 4];
        let target = 0.0;
        let downside = downside_deviation(&returns, target, 252).unwrap();
        assert!(downside > 0.0);

        let sortino = sortino_ratio(&returns, &risk_free, target, 252).unwrap();
        assert!(sortino.is_finite());
    }

    #[test]
    fn test_treynor_and_omega() {
        let returns = [0.012, 0.018, -0.004, 0.022, 0.01];
        let benchmark = [0.01, 0.015, -0.003, 0.02, 0.008];
        let risk_free = [0.001; 5];
        let treynor = treynor_ratio(&returns, &benchmark, &risk_free, 12).unwrap();
        assert!(treynor.is_finite());

        let omega = omega_ratio(&returns, 0.0).unwrap();
        assert!(omega.is_finite());
    }

    #[test]
    fn test_moments_and_tail_metrics() {
        let returns = [
            0.01, 0.015, -0.005, 0.02, -0.012, 0.03, -0.008, 0.011, 0.007, -0.009,
        ];

        let skew = skewness(&returns).unwrap();
        assert!((skew - 0.2402101883280366).abs() < 1e-12);

        let kurt = excess_kurtosis(&returns).unwrap();
        assert!((kurt + 0.986431458590427).abs() < 1e-12);

        let gain_loss = gain_loss_ratio(&returns).unwrap();
        assert!((gain_loss - 1.8235294117647058).abs() < 1e-12);

        let tail = tail_ratio(&returns, 0.95).unwrap();
        assert!((tail - 2.394366197183098).abs() < 1e-12);

        let upr = upside_potential_ratio(&returns, 0.0).unwrap();
        assert!((upr - 1.6596561688271803).abs() < 1e-12);
    }

    #[test]
    fn test_drawdown_indices() {
        let prices = [100.0, 98.0, 101.0, 99.5, 104.0, 102.0];
        let pain = pain_index(&prices).unwrap();
        assert!((pain - 0.009013709063214026).abs() < 1e-12);

        let ulcer = ulcer_index(&prices).unwrap();
        assert!((ulcer - 0.012847756589664857).abs() < 1e-12);
    }

    #[test]
    fn test_capture_drawdown_calmar() {
        let returns = [0.02, -0.01, 0.015, 0.005, 0.018];
        let benchmark = [0.015, -0.008, 0.012, 0.004, 0.014];
        let upside = upside_capture_ratio(&returns, &benchmark, 12).unwrap();
        let downside = downside_capture_ratio(&returns, &benchmark, 12).unwrap();
        assert!(upside.is_finite());
        assert!(downside.is_finite());

        let prices = [100.0, 102.0, 101.0, 99.0, 105.0];
        let drawdowns = drawdown_series(&prices).unwrap();
        assert_eq!(drawdowns.len(), prices.len());
        let max_dd = max_drawdown(&prices).unwrap();
        assert!(max_dd <= 0.0);

        let calmar = calmar_ratio(0.12, max_dd.abs()).unwrap();
        assert!(calmar.is_finite());
    }
}
