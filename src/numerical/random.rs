// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Random number generation utilities
//!
//! Provides RNG infrastructure for Monte Carlo simulations and stochastic processes.
//! Includes normal distribution sampling, quasi-random sequences, and thread-local RNG.

use rand::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, StandardNormal};
use std::cell::RefCell;

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(new_entropy_rng());
}

/// Create a `StdRng` backed by system entropy
fn new_entropy_rng() -> StdRng {
    let mut entropy = rand::rng();
    StdRng::from_rng(&mut entropy)
}

/// Random number generator with normal distribution sampling
pub struct RandomGenerator {
    rng: StdRng,
}

impl RandomGenerator {
    /// Create a new random generator with a seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create a new random generator from entropy
    pub fn from_entropy() -> Self {
        Self {
            rng: new_entropy_rng(),
        }
    }

    /// Generate a standard normal random variable using Box-Muller transform
    pub fn standard_normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.rng)
    }

    /// Generate a normal random variable with specified mean and standard deviation
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let normal = Normal::new(mean, std_dev).unwrap();
        normal.sample(&mut self.rng)
    }

    /// Generate a vector of standard normal random variables
    pub fn standard_normal_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.standard_normal()).collect()
    }

    /// Generate a uniform random variable in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        self.rng.random()
    }

    /// Generate a uniform random variable in [a, b)
    pub fn uniform_range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }
}

/// Thread-local random number generator for parallel operations
pub struct ThreadLocalRng;

impl ThreadLocalRng {
    /// Generate a standard normal random variable using thread-local RNG
    pub fn standard_normal() -> f64 {
        THREAD_RNG.with(|rng| StandardNormal.sample(&mut *rng.borrow_mut()))
    }

    /// Generate a normal random variable with specified mean and standard deviation
    pub fn normal(mean: f64, std_dev: f64) -> f64 {
        THREAD_RNG.with(|rng| {
            let normal = Normal::new(mean, std_dev).unwrap();
            normal.sample(&mut *rng.borrow_mut())
        })
    }

    /// Generate a uniform random variable in [0, 1)
    pub fn uniform() -> f64 {
        THREAD_RNG.with(|rng| rng.borrow_mut().random())
    }

    /// Seed the thread-local RNG
    pub fn seed(seed: u64) {
        THREAD_RNG.with(|rng| {
            *rng.borrow_mut() = StdRng::seed_from_u64(seed);
        });
    }
}

/// Sobol quasi-random sequence generator
pub struct SobolSequence {
    dimension: usize,
    index: u64,
    direction_numbers: Vec<Vec<u32>>,
}

impl SobolSequence {
    /// Create a new Sobol sequence generator for the given dimension
    pub fn new(dimension: usize) -> Self {
        assert!(
            dimension > 0 && dimension <= 40,
            "Dimension must be between 1 and 40"
        );

        let direction_numbers = Self::initialize_direction_numbers(dimension);

        Self {
            dimension,
            index: 0,
            direction_numbers,
        }
    }

    /// Generate the next point in the Sobol sequence
    pub fn next_point(&mut self) -> Vec<f64> {
        self.index += 1;
        let mut point = vec![0.0; self.dimension];

        for (d, point_d) in point.iter_mut().enumerate().take(self.dimension) {
            let mut value = 0u32;
            let mut i = self.index;
            let mut j = 0;

            while i > 0 {
                if i & 1 == 1 {
                    value ^= self.direction_numbers[d][j];
                }
                i >>= 1;
                j += 1;
            }

            *point_d = value as f64 / (1u64 << 32) as f64;
        }

        point
    }

    /// Generate n points from the Sobol sequence
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    /// Reset the sequence to the beginning
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Initialize direction numbers for Sobol sequence
    fn initialize_direction_numbers(dimension: usize) -> Vec<Vec<u32>> {
        let mut direction_numbers = vec![vec![0u32; 32]; dimension];

        // First dimension uses simple binary fractions
        for i in 0..32 {
            direction_numbers[0][i] = 1u32 << (31 - i);
        }

        // For higher dimensions, use primitive polynomials
        // This is a simplified version - production code would use full tables
        for direction_number in direction_numbers.iter_mut().take(dimension).skip(1) {
            for i in 0..32 {
                if i == 0 {
                    direction_number[i] = 1u32 << 31;
                } else {
                    direction_number[i] = direction_number[i - 1] ^ (direction_number[i - 1] >> 1);
                }
            }
        }

        direction_numbers
    }
}

/// Halton quasi-random sequence generator
pub struct HaltonSequence {
    dimension: usize,
    index: u64,
    bases: Vec<u64>,
}

impl HaltonSequence {
    /// Create a new Halton sequence generator for the given dimension
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "Dimension must be positive");

        // Use first n primes as bases
        let bases = Self::first_n_primes(dimension);

        Self {
            dimension,
            index: 0,
            bases,
        }
    }

    /// Generate the next point in the Halton sequence
    pub fn next_point(&mut self) -> Vec<f64> {
        self.index += 1;
        let mut point = vec![0.0; self.dimension];

        for (d, point_d) in point.iter_mut().enumerate().take(self.dimension) {
            *point_d = Self::van_der_corput(self.index, self.bases[d]);
        }

        point
    }

    /// Generate n points from the Halton sequence
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    /// Reset the sequence to the beginning
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Van der Corput sequence in base b
    fn van_der_corput(mut n: u64, base: u64) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;

        while n > 0 {
            result += f * (n % base) as f64;
            n /= base;
            f /= base as f64;
        }

        result
    }

    /// Generate first n prime numbers
    fn first_n_primes(n: usize) -> Vec<u64> {
        let mut primes = Vec::with_capacity(n);
        let mut candidate = 2u64;

        while primes.len() < n {
            if Self::is_prime(candidate) {
                primes.push(candidate);
            }
            candidate += 1;
        }

        primes
    }

    /// Check if a number is prime
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_random_generator_standard_normal() {
        let mut rng = RandomGenerator::new(42);
        let samples: Vec<f64> = (0..10000).map(|_| rng.standard_normal()).collect();

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert_relative_eq!(mean, 0.0, epsilon = 0.05);
        assert_relative_eq!(variance, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_random_generator_normal() {
        let mut rng = RandomGenerator::new(42);
        let mean = 5.0;
        let std_dev = 2.0;
        let samples: Vec<f64> = (0..10000).map(|_| rng.normal(mean, std_dev)).collect();

        let sample_mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let sample_variance: f64 = samples
            .iter()
            .map(|x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        assert_relative_eq!(sample_mean, mean, epsilon = 0.1);
        assert_relative_eq!(sample_variance, std_dev.powi(2), epsilon = 0.2);
    }

    #[test]
    fn test_random_generator_uniform() {
        let mut rng = RandomGenerator::new(42);
        let samples: Vec<f64> = (0..1000).map(|_| rng.uniform()).collect();

        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_thread_local_rng() {
        ThreadLocalRng::seed(42);
        let sample = ThreadLocalRng::standard_normal();
        assert!(sample.is_finite());
    }

    #[test]
    fn test_sobol_sequence() {
        let mut sobol = SobolSequence::new(2);
        let points = sobol.generate(100);

        assert_eq!(points.len(), 100);
        assert!(points.iter().all(|p| p.len() == 2));
        assert!(
            points
                .iter()
                .all(|p| p.iter().all(|&x| (0.0..=1.0).contains(&x)))
        );
    }

    #[test]
    fn test_halton_sequence() {
        let mut halton = HaltonSequence::new(2);
        let points = halton.generate(100);

        assert_eq!(points.len(), 100);
        assert!(points.iter().all(|p| p.len() == 2));
        assert!(
            points
                .iter()
                .all(|p| p.iter().all(|&x| (0.0..=1.0).contains(&x)))
        );
    }

    #[test]
    fn test_sobol_reset() {
        let mut sobol = SobolSequence::new(1);
        let first = sobol.next_point();
        sobol.next_point();
        sobol.reset();
        let after_reset = sobol.next_point();

        assert_eq!(first, after_reset);
    }

    #[test]
    fn test_halton_reset() {
        let mut halton = HaltonSequence::new(1);
        let first = halton.next_point();
        halton.next_point();
        halton.reset();
        let after_reset = halton.next_point();

        assert_eq!(first, after_reset);
    }
}
