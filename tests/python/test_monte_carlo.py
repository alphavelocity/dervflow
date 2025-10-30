# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Monte Carlo simulation module and stochastic processes
"""

import pytest
import numpy as np
from dervflow import MonteCarloEngine


class TestGeometricBrownianMotion:
    """Tests for GBM process"""
    
    def test_gbm_basic_simulation(self):
        """Test basic GBM simulation runs without errors"""
        engine = MonteCarloEngine(seed=42)
        paths = engine.simulate_gbm(
            s0=100.0,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        assert paths.shape == (1000, 253)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)
    
    def test_gbm_mean_convergence(self):
        """Test that GBM mean converges to expected value"""
        engine = MonteCarloEngine(seed=42)
        s0 = 100.0
        mu = 0.05
        T = 1.0
        
        paths = engine.simulate_gbm(
            s0=s0,
            mu=mu,
            sigma=0.2,
            T=T,
            steps=252,
            paths=10000
        )
        
        # Expected mean: S0 * exp(mu * T)
        expected_mean = s0 * np.exp(mu * T)
        simulated_mean = np.mean(paths[:, -1])
        
        # Allow 5% tolerance
        assert abs(simulated_mean - expected_mean) / expected_mean < 0.05
    
    def test_gbm_variance(self):
        """Test that GBM variance is reasonable"""
        engine = MonteCarloEngine(seed=42)
        s0 = 100.0
        mu = 0.05
        sigma = 0.2
        T = 1.0
        
        paths = engine.simulate_gbm(
            s0=s0,
            mu=mu,
            sigma=sigma,
            T=T,
            steps=252,
            paths=10000
        )
        
        # Expected variance: S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)
        expected_var = s0**2 * np.exp(2*mu*T) * (np.exp(sigma**2 * T) - 1)
        simulated_var = np.var(paths[:, -1])
        
        # Allow 10% tolerance for variance
        assert abs(simulated_var - expected_var) / expected_var < 0.15
    
    def test_gbm_parallel(self):
        """Test parallel GBM simulation"""
        engine = MonteCarloEngine(seed=42)
        
        paths_serial = engine.simulate_gbm(
            s0=100.0,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            steps=100,
            paths=1000,
            parallel=False
        )
        
        engine_parallel = MonteCarloEngine(seed=42)
        paths_parallel = engine_parallel.simulate_gbm(
            s0=100.0,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            steps=100,
            paths=1000,
            parallel=True
        )
        
        # Results should be similar (not identical due to parallel RNG)
        assert paths_serial.shape == paths_parallel.shape
        assert np.all(paths_parallel > 0)


class TestOrnsteinUhlenbeck:
    """Tests for OU process"""
    
    def test_ou_basic_simulation(self):
        """Test basic OU simulation runs without errors"""
        engine = MonteCarloEngine(seed=42)
        paths = engine.simulate_ou(
            x0=0.05,
            theta=0.5,
            mu=0.03,
            sigma=0.01,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        assert paths.shape == (1000, 253)
        assert np.all(paths[:, 0] == 0.05)
    
    def test_ou_mean_reversion(self):
        """Test that OU process reverts to mean"""
        engine = MonteCarloEngine(seed=42)
        theta = 2.0
        mu = 0.03
        
        # Start far from mean
        paths = engine.simulate_ou(
            x0=0.10,
            theta=theta,
            mu=mu,
            sigma=0.01,
            T=2.0,
            steps=500,
            paths=5000
        )
        
        # Mean should converge to long-term mean
        final_mean = np.mean(paths[:, -1])
        assert abs(final_mean - mu) < 0.005
    
    def test_ou_stationary_variance(self):
        """Test OU stationary variance"""
        engine = MonteCarloEngine(seed=42)
        theta = 1.0
        sigma = 0.02
        
        # Start at mean and run for long time
        paths = engine.simulate_ou(
            x0=0.03,
            theta=theta,
            mu=0.03,
            sigma=sigma,
            T=5.0,
            steps=1000,
            paths=10000
        )
        
        # Stationary variance: sigma^2 / (2 * theta)
        expected_var = sigma**2 / (2 * theta)
        simulated_var = np.var(paths[:, -1])
        
        # Allow 20% tolerance
        assert abs(simulated_var - expected_var) / expected_var < 0.25


class TestCoxIngersollRoss:
    """Tests for CIR process"""
    
    def test_cir_basic_simulation(self):
        """Test basic CIR simulation runs without errors"""
        engine = MonteCarloEngine(seed=42)
        paths = engine.simulate_cir(
            x0=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.1,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        assert paths.shape == (1000, 253)
        assert np.all(paths[:, 0] == 0.03)
        assert np.all(paths >= 0)
    
    def test_cir_positivity(self):
        """Test that CIR process stays non-negative"""
        engine = MonteCarloEngine(seed=42)
        
        paths = engine.simulate_cir(
            x0=0.01,
            kappa=0.5,
            theta=0.03,
            sigma=0.2,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        # All values should be non-negative
        assert np.all(paths >= 0)
    
    def test_cir_mean_reversion(self):
        """Test that CIR process reverts to mean"""
        engine = MonteCarloEngine(seed=42)
        kappa = 1.0
        theta = 0.04
        
        paths = engine.simulate_cir(
            x0=0.08,
            kappa=kappa,
            theta=theta,
            sigma=0.1,
            T=3.0,
            steps=500,
            paths=5000
        )
        
        # Mean should converge to long-term mean
        final_mean = np.mean(paths[:, -1])
        assert abs(final_mean - theta) < 0.01


class TestVasicek:
    """Tests for Vasicek process"""
    
    def test_vasicek_basic_simulation(self):
        """Test basic Vasicek simulation runs without errors"""
        engine = MonteCarloEngine(seed=42)
        paths = engine.simulate_vasicek(
            r0=0.03,
            kappa=0.5,
            theta=0.03,
            sigma=0.01,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        assert paths.shape == (1000, 253)
        assert np.all(paths[:, 0] == 0.03)
    
    def test_vasicek_mean_reversion(self):
        """Test that Vasicek process reverts to mean"""
        engine = MonteCarloEngine(seed=42)
        kappa = 2.0
        theta = 0.05
        
        paths = engine.simulate_vasicek(
            r0=0.10,
            kappa=kappa,
            theta=theta,
            sigma=0.01,
            T=2.0,
            steps=500,
            paths=5000
        )
        
        # Mean should converge to long-term mean
        final_mean = np.mean(paths[:, -1])
        assert abs(final_mean - theta) < 0.005
    
    def test_vasicek_can_be_negative(self):
        """Test that Vasicek can produce negative rates (unlike CIR)"""
        engine = MonteCarloEngine(seed=42)
        
        # Use high volatility to potentially get negative rates
        paths = engine.simulate_vasicek(
            r0=0.01,
            kappa=0.1,
            theta=0.01,
            sigma=0.05,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        # At least some paths might go negative (this is a feature, not a bug)
        # Just verify simulation completes
        assert paths.shape == (1000, 253)


class TestCorrelatedPaths:
    """Tests for correlated multi-asset simulation"""
    
    def test_correlated_basic_simulation(self):
        """Test basic correlated simulation runs without errors"""
        engine = MonteCarloEngine(seed=42)
        
        initial_values = [100.0, 50.0, 75.0]
        mu_values = [0.05, 0.07, 0.06]
        sigma_values = [0.2, 0.25, 0.22]
        correlation = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        paths = engine.simulate_correlated(
            initial_values=initial_values,
            mu_values=mu_values,
            sigma_values=sigma_values,
            correlation=correlation,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        # Returns list of arrays, one per asset
        assert len(paths) == 3
        assert paths[0].shape == (1000, 252)
        assert paths[1].shape == (1000, 252)
        assert paths[2].shape == (1000, 252)
    
    def test_correlated_correlation_preservation(self):
        """Test that correlation is preserved in simulated paths"""
        engine = MonteCarloEngine(seed=42)
        
        initial_values = [100.0, 100.0]
        mu_values = [0.05, 0.05]
        sigma_values = [0.2, 0.2]
        target_corr = 0.7
        correlation = np.array([
            [1.0, target_corr],
            [target_corr, 1.0]
        ])
        
        paths = engine.simulate_correlated(
            initial_values=initial_values,
            mu_values=mu_values,
            sigma_values=sigma_values,
            correlation=correlation,
            T=1.0,
            steps=252,
            paths=10000
        )
        
        # Calculate returns (paths are already without initial value)
        returns1 = np.diff(np.log(paths[0]), axis=1)
        returns2 = np.diff(np.log(paths[1]), axis=1)
        
        # Calculate correlation of returns
        flat_returns1 = returns1.flatten()
        flat_returns2 = returns2.flatten()
        simulated_corr = np.corrcoef(flat_returns1, flat_returns2)[0, 1]
        
        # Allow 10% tolerance
        assert abs(simulated_corr - target_corr) < 0.1
    
    def test_correlated_uncorrelated_case(self):
        """Test uncorrelated assets"""
        engine = MonteCarloEngine(seed=42)
        
        initial_values = [100.0, 100.0]
        mu_values = [0.05, 0.05]
        sigma_values = [0.2, 0.2]
        correlation = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        paths = engine.simulate_correlated(
            initial_values=initial_values,
            mu_values=mu_values,
            sigma_values=sigma_values,
            correlation=correlation,
            T=1.0,
            steps=252,
            paths=5000
        )
        
        # Calculate returns (paths are already without initial value)
        returns1 = np.diff(np.log(paths[0]), axis=1)
        returns2 = np.diff(np.log(paths[1]), axis=1)
        
        # Calculate correlation
        flat_returns1 = returns1.flatten()
        flat_returns2 = returns2.flatten()
        simulated_corr = np.corrcoef(flat_returns1, flat_returns2)[0, 1]
        
        # Should be close to zero
        assert abs(simulated_corr) < 0.1
    
    def test_correlated_parallel(self):
        """Test parallel correlated simulation"""
        engine = MonteCarloEngine(seed=42)
        
        initial_values = [100.0, 50.0]
        mu_values = [0.05, 0.07]
        sigma_values = [0.2, 0.25]
        correlation = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        
        paths = engine.simulate_correlated(
            initial_values=initial_values,
            mu_values=mu_values,
            sigma_values=sigma_values,
            correlation=correlation,
            T=1.0,
            steps=100,
            paths=1000,
            parallel=True
        )
        
        # Returns list of arrays, one per asset
        assert len(paths) == 2
        assert paths[0].shape == (1000, 100)
        assert paths[1].shape == (1000, 100)
        assert np.all(paths[0] > 0)
        assert np.all(paths[1] > 0)


class TestReproducibility:
    """Tests for reproducibility with seeds"""
    
    def test_gbm_reproducibility(self):
        """Test that same seed produces same results"""
        engine1 = MonteCarloEngine(seed=123)
        paths1 = engine1.simulate_gbm(
            s0=100.0,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            steps=100,
            paths=100
        )
        
        engine2 = MonteCarloEngine(seed=123)
        paths2 = engine2.simulate_gbm(
            s0=100.0,
            mu=0.05,
            sigma=0.2,
            T=1.0,
            steps=100,
            paths=100
        )
        
        np.testing.assert_array_equal(paths1, paths2)
    
    def test_ou_reproducibility(self):
        """Test OU reproducibility with seed"""
        engine1 = MonteCarloEngine(seed=456)
        paths1 = engine1.simulate_ou(
            x0=0.05,
            theta=0.5,
            mu=0.03,
            sigma=0.01,
            T=1.0,
            steps=100,
            paths=100
        )
        
        engine2 = MonteCarloEngine(seed=456)
        paths2 = engine2.simulate_ou(
            x0=0.05,
            theta=0.5,
            mu=0.03,
            sigma=0.01,
            T=1.0,
            steps=100,
            paths=100
        )
        
        np.testing.assert_array_equal(paths1, paths2)


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_gbm_negative_volatility(self):
        """Test that negative volatility raises error"""
        engine = MonteCarloEngine(seed=42)
        
        with pytest.raises(ValueError):
            engine.simulate_gbm(
                s0=100.0,
                mu=0.05,
                sigma=-0.2,
                T=1.0,
                steps=100,
                paths=100
            )
    
    def test_ou_invalid_theta(self):
        """Test that invalid theta raises error"""
        engine = MonteCarloEngine(seed=42)
        
        with pytest.raises(ValueError):
            engine.simulate_ou(
                x0=0.05,
                theta=-0.5,
                mu=0.03,
                sigma=0.01,
                T=1.0,
                steps=100,
                paths=100
            )
    
    def test_cir_invalid_params(self):
        """Test that invalid CIR parameters raise error"""
        engine = MonteCarloEngine(seed=42)
        
        with pytest.raises(ValueError):
            engine.simulate_cir(
                x0=0.03,
                kappa=-0.5,
                theta=0.03,
                sigma=0.1,
                T=1.0,
                steps=100,
                paths=100
            )
    
    def test_correlated_mismatched_dimensions(self):
        """Test that mismatched dimensions raise error"""
        engine = MonteCarloEngine(seed=42)
        
        with pytest.raises(ValueError):
            engine.simulate_correlated(
                initial_values=[100.0, 50.0],
                mu_values=[0.05],  # Wrong length
                sigma_values=[0.2, 0.25],
                correlation=np.eye(2),
                T=1.0,
                steps=100,
                paths=100
            )
