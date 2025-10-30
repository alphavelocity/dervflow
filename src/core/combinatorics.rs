// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::common::error::{DervflowError, Result};

pub fn factorial(n: u64) -> Result<u128> {
    let mut result = 1u128;
    for i in 2..=n as u128 {
        result = result
            .checked_mul(i)
            .ok_or_else(|| DervflowError::NumericalError("Factorial overflow".to_string()))?;
    }
    Ok(result)
}

pub fn permutation(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for permutations".to_string(),
        ));
    }

    let mut result = 1u128;
    for i in 0..k {
        let factor = (n - i) as u128;
        result = result
            .checked_mul(factor)
            .ok_or_else(|| DervflowError::NumericalError("Permutation overflow".to_string()))?;
    }
    Ok(result)
}

pub fn combination(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for combinations".to_string(),
        ));
    }

    let k = k.min(n - k);
    if k == 0 {
        return Ok(1);
    }

    let mut result = 1u128;
    for i in 1..=k {
        let factor = (n - k + i) as u128;
        result = result
            .checked_mul(factor)
            .ok_or_else(|| DervflowError::NumericalError("Combination overflow".to_string()))?;
        result /= i as u128;
    }

    Ok(result)
}

pub fn falling_factorial(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for falling factorial".to_string(),
        ));
    }

    let mut result = 1u128;
    for i in 0..k {
        let factor = (n - i) as u128;
        result = result.checked_mul(factor).ok_or_else(|| {
            DervflowError::NumericalError("Falling factorial overflow".to_string())
        })?;
    }
    Ok(result)
}

pub fn rising_factorial(n: u64, k: u64) -> Result<u128> {
    let mut result = 1u128;
    for i in 0..k {
        let addend = n
            .checked_add(i)
            .ok_or_else(|| DervflowError::NumericalError("Rising factorial overflow".to_string()))?;
        let factor = addend as u128;
        result = result
            .checked_mul(factor)
            .ok_or_else(|| DervflowError::NumericalError("Rising factorial overflow".to_string()))?;
    }
    Ok(result)
}

pub fn binomial_probability(n: u64, k: u64, p: f64) -> Result<f64> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for binomial probability".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&p) {
        return Err(DervflowError::InvalidInput(
            "Probability must be between 0 and 1".to_string(),
        ));
    }

    let comb = combination(n, k)? as f64;
    let success = p.powi(k as i32);
    let failure = (1.0 - p).powi((n - k) as i32);

    Ok(comb * success * failure)
}

pub fn multinomial(counts: &[u64]) -> Result<u128> {
    if counts.is_empty() {
        return Err(DervflowError::InvalidInput(
            "Counts cannot be empty for multinomial coefficient".to_string(),
        ));
    }

    let mut total: u64 = 0;
    let mut result = 1u128;

    for &count in counts {
        total = total.checked_add(count).ok_or_else(|| {
            DervflowError::NumericalError("Multinomial total overflow".to_string())
        })?;

        if count == 0 {
            continue;
        }

        let comb = combination(total, count)?;
        result = result.checked_mul(comb).ok_or_else(|| {
            DervflowError::NumericalError("Multinomial coefficient overflow".to_string())
        })?;
    }

    Ok(result)
}

pub fn catalan_number(n: u64) -> Result<u128> {
    let doubled = n
        .checked_mul(2)
        .ok_or_else(|| DervflowError::NumericalError("Catalan input too large".to_string()))?;

    let comb = combination(doubled, n)?;
    let divisor = (n + 1) as u128;

    Ok(comb / divisor)
}

pub fn stirling_number_second(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for Stirling numbers".to_string(),
        ));
    }

    if k == 0 {
        return Ok(if n == 0 { 1 } else { 0 });
    }

    let k_usize = k as usize;
    let mut previous = vec![0u128; k_usize + 1];
    previous[0] = 1;

    for i in 1..=n as usize {
        let mut current = vec![0u128; k_usize + 1];
        let upper = k_usize.min(i);
        for j in 1..=upper {
            let term1 = previous[j].checked_mul(j as u128).ok_or_else(|| {
                DervflowError::NumericalError("Stirling number multiplication overflow".to_string())
            })?;
            let term2 = previous[j - 1];
            current[j] = term1.checked_add(term2).ok_or_else(|| {
                DervflowError::NumericalError("Stirling number addition overflow".to_string())
            })?;
        }
        previous = current;
    }

    Ok(previous[k_usize])
}

pub fn stirling_number_first(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for Stirling numbers".to_string(),
        ));
    }

    if k == 0 {
        return Ok(if n == 0 { 1 } else { 0 });
    }

    let k_usize = k as usize;
    let mut previous = vec![0u128; k_usize + 1];
    previous[0] = 1;

    for i in 1..=n as usize {
        let mut current = vec![0u128; k_usize + 1];
        let upper = k_usize.min(i);
        for j in 1..=upper {
            let term1 = previous[j - 1];
            let term2 = previous[j].checked_mul((i - 1) as u128).ok_or_else(|| {
                DervflowError::NumericalError("Stirling number multiplication overflow".to_string())
            })?;
            current[j] = term1.checked_add(term2).ok_or_else(|| {
                DervflowError::NumericalError("Stirling number addition overflow".to_string())
            })?;
        }
        previous = current;
    }

    Ok(previous[k_usize])
}

pub fn bell_number(n: u64) -> Result<u128> {
    if n == 0 {
        return Ok(1);
    }

    let mut previous = vec![0u128; 1];
    previous[0] = 1;

    for i in 1..=n as usize {
        let mut current = vec![0u128; i + 1];
        current[0] = previous[i - 1];
        for j in 1..=i {
            current[j] = current[j - 1]
                .checked_add(previous[j - 1])
                .ok_or_else(|| DervflowError::NumericalError("Bell number overflow".to_string()))?;
        }
        previous = current;
    }

    Ok(previous[0])
}

pub fn lah_number(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Err(DervflowError::InvalidInput(
            "k cannot be greater than n for Lah numbers".to_string(),
        ));
    }

    if k == 0 {
        return Ok(if n == 0 { 1 } else { 0 });
    }

    let combination_term = combination(n - 1, k - 1)?;
    let factorial_n = factorial(n)?;
    let factorial_k = factorial(k)?;

    let numerator = combination_term.checked_mul(factorial_n).ok_or_else(|| {
        DervflowError::NumericalError("Lah number multiplication overflow".to_string())
    })?;

    numerator
        .checked_div(factorial_k)
        .ok_or_else(|| DervflowError::NumericalError("Lah number division overflow".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_combinatorics_basics() {
        assert_eq!(factorial(5).unwrap(), 120);
        assert_eq!(permutation(5, 2).unwrap(), 20);
        assert_eq!(combination(5, 2).unwrap(), 10);
    }

    #[test]
    fn test_factorial_variants() {
        assert_eq!(falling_factorial(5, 3).unwrap(), 60);
        assert_eq!(rising_factorial(3, 3).unwrap(), 60);
    }

    #[test]
    fn test_binomial_probability() {
        let prob = binomial_probability(10, 3, 0.5).unwrap();
        assert_abs_diff_eq!(prob, 0.117_187_5, epsilon = 1e-9);
    }

    #[test]
    fn test_multinomial_and_catalan() {
        assert_eq!(multinomial(&[2, 1, 1]).unwrap(), 12);
        assert_eq!(multinomial(&[3, 0, 2]).unwrap(), 10);
        assert_eq!(catalan_number(5).unwrap(), 42);
        assert_eq!(catalan_number(0).unwrap(), 1);
        assert!(multinomial(&[]).is_err());
    }

    #[test]
    fn test_stirling_numbers() {
        assert_eq!(stirling_number_second(0, 0).unwrap(), 1);
        assert_eq!(stirling_number_second(5, 1).unwrap(), 1);
        assert_eq!(stirling_number_second(5, 2).unwrap(), 15);
        assert_eq!(stirling_number_second(6, 3).unwrap(), 90);
        assert_eq!(stirling_number_second(5, 0).unwrap(), 0);
        assert!(stirling_number_second(3, 4).is_err());
    }

    #[test]
    fn test_additional_combinatorics_sequences() {
        assert_eq!(stirling_number_first(5, 2).unwrap(), 50);
        assert_eq!(stirling_number_first(7, 3).unwrap(), 1624);
        assert_eq!(stirling_number_first(5, 0).unwrap(), 0);
        assert_eq!(bell_number(0).unwrap(), 1);
        assert_eq!(bell_number(5).unwrap(), 52);
        assert_eq!(lah_number(0, 0).unwrap(), 1);
        assert_eq!(lah_number(5, 2).unwrap(), 240);
        assert_eq!(lah_number(6, 3).unwrap(), 1200);
        assert!(lah_number(4, 6).is_err());
    }
}
