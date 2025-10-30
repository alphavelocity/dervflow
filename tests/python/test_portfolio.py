# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for portfolio optimization module
"""

import pytest
import numpy as np
from dervflow.portfolio import (
    PortfolioOptimizer,
    RiskParityOptimizer,
    BlackLittermanModel,
    InvestorViews,
    FactorModel,
)


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        expected_returns = np.array([0.10, 0.12, 0.08])
        covariance = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.0225]
        ])
        return expected_returns, covariance
    
    def test_optimizer_creation(self, sample_data):
        """Test optimizer can be created"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        assert optimizer.n_assets == 3
        assert len(optimizer.expected_returns) == 3
        assert optimizer.covariance.shape == (3, 3)
    
    def test_minimum_variance_portfolio(self, sample_data):
        """Test minimum variance portfolio optimization"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        result = optimizer.optimize()
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'status' in result
        
        weights = result['weights']
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights >= -1e-6)  # Non-negative weights
    
    def test_target_return_optimization(self, sample_data):
        """Test optimization with target return"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        target_return = 0.10
        result = optimizer.optimize(target_return=target_return)
        
        assert np.abs(result['expected_return'] - target_return) < 1e-4
        assert np.abs(result['weights'].sum() - 1.0) < 1e-6
    
    def test_sharpe_ratio_optimization(self, sample_data):
        """Test Sharpe ratio maximization"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        risk_free_rate = 0.03
        result = optimizer.optimize(risk_free_rate=risk_free_rate)
        
        assert 'sharpe_ratio' in result
        assert result['sharpe_ratio'] is not None
        assert result['sharpe_ratio'] > 0.0
        assert np.abs(result['weights'].sum() - 1.0) < 1e-6
    
    def test_box_constraints(self, sample_data):
        """Test optimization with box constraints"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        min_weights = np.array([0.1, 0.1, 0.1])
        max_weights = np.array([0.5, 0.5, 0.5])
        
        result = optimizer.optimize(
            min_weights=min_weights,
            max_weights=max_weights
        )
        
        weights = result['weights']
        assert np.all(weights >= min_weights - 1e-6)
        assert np.all(weights <= max_weights + 1e-6)
    
    def test_efficient_frontier(self, sample_data):
        """Test efficient frontier generation"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        num_points = 10
        frontier = optimizer.efficient_frontier(num_points=num_points)
        
        assert len(frontier) > 0
        assert len(frontier) <= num_points
        
        # Check that returns are increasing along the frontier
        for i in range(1, len(frontier)):
            assert frontier[i]['expected_return'] >= frontier[i-1]['expected_return'] - 1e-6
    
    def test_portfolio_return_calculation(self, sample_data):
        """Test portfolio return calculation"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        weights = np.array([0.3, 0.4, 0.3])
        portfolio_return = optimizer.portfolio_return(weights)
        
        expected = np.dot(weights, returns)
        assert np.abs(portfolio_return - expected) < 1e-10
    
    def test_portfolio_volatility_calculation(self, sample_data):
        """Test portfolio volatility calculation"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)
        
        weights = np.array([0.3, 0.4, 0.3])
        volatility = optimizer.portfolio_volatility(weights)
        
        assert volatility > 0.0
        assert volatility < 0.5  # Reasonable range
    
    def test_sharpe_ratio_calculation(self, sample_data):
        """Test Sharpe ratio calculation"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)

        weights = np.array([0.3, 0.4, 0.3])
        risk_free_rate = 0.03
        sharpe = optimizer.sharpe_ratio(weights, risk_free_rate)

        ret = optimizer.portfolio_return(weights)
        vol = optimizer.portfolio_volatility(weights)
        expected_sharpe = (ret - risk_free_rate) / vol

        assert np.abs(sharpe - expected_sharpe) < 1e-10

    def test_risk_contributions_sum_to_one(self, sample_data):
        """Risk contributions should sum to one"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)

        weights = np.array([0.4, 0.3, 0.3])
        contributions = optimizer.risk_contributions(weights)

        assert contributions.shape == (3,)
        assert np.all(contributions >= 0.0)
        assert np.abs(contributions.sum() - 1.0) < 1e-10

    def test_parametric_var_and_cvar(self, sample_data):
        """Test parametric VaR and CVaR calculations"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)

        weights = np.array([0.5, 0.3, 0.2])
        var = optimizer.value_at_risk(weights, confidence_level=0.95)
        cvar = optimizer.conditional_value_at_risk(weights, confidence_level=0.95)

        assert var >= 0.0
        assert cvar >= var

    def test_portfolio_summary_contents(self, sample_data):
        """Summary should contain key risk metrics"""
        returns, cov = sample_data
        optimizer = PortfolioOptimizer(returns, cov)

        weights = np.array([0.3, 0.4, 0.3])
        summary = optimizer.portfolio_summary(weights, risk_free_rate=0.02)

        expected_keys = {
            'expected_return',
            'variance',
            'volatility',
            'sharpe_ratio',
            'diversification_ratio',
            'weight_concentration',
            'risk_concentration',
            'risk_contributions',
        }
        assert expected_keys.issubset(summary.keys())
        assert summary['risk_contributions']['percentage'].shape == (3,)


class TestRiskParityOptimizer:
    """Tests for RiskParityOptimizer class"""
    
    @pytest.fixture
    def sample_covariance(self):
        """Create sample covariance matrix"""
        return np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.0225]
        ])
    
    def test_optimizer_creation(self, sample_covariance):
        """Test optimizer can be created"""
        optimizer = RiskParityOptimizer(sample_covariance)
        assert optimizer.n_assets == 3
        assert optimizer.covariance.shape == (3, 3)
    
    def test_equal_risk_contribution(self, sample_covariance):
        """Test equal risk contribution optimization"""
        optimizer = RiskParityOptimizer(sample_covariance)
        
        weights = optimizer.optimize()
        
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights > 0.0)
        
        # Check risk contributions are approximately equal
        risk_contributions = optimizer.risk_contributions(weights)
        target = 1.0 / 3.0
        for rc in risk_contributions:
            assert np.abs(rc - target) < 1e-4
    
    def test_custom_risk_contributions(self, sample_covariance):
        """Test optimization with custom risk contributions"""
        optimizer = RiskParityOptimizer(sample_covariance)
        
        target_rc = np.array([0.5, 0.3, 0.2])
        weights = optimizer.optimize(target_risk_contributions=target_rc)
        
        assert np.abs(weights.sum() - 1.0) < 1e-6
        
        # Check risk contributions match targets
        risk_contributions = optimizer.risk_contributions(weights)
        for i in range(3):
            assert np.abs(risk_contributions[i] - target_rc[i]) < 1e-4
    
    def test_risk_contributions_sum_to_one(self, sample_covariance):
        """Test that risk contributions sum to 1"""
        optimizer = RiskParityOptimizer(sample_covariance)
        
        weights = np.array([0.3, 0.4, 0.3])
        risk_contributions = optimizer.risk_contributions(weights)
        
        assert len(risk_contributions) == 3
        assert np.abs(risk_contributions.sum() - 1.0) < 1e-6
        assert np.all(risk_contributions >= 0.0)
    
    def test_convergence_parameters(self, sample_covariance):
        """Test optimization with custom convergence parameters"""
        optimizer = RiskParityOptimizer(sample_covariance)
        
        weights = optimizer.optimize(max_iterations=100, tolerance=1e-6)
        
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-6


class TestPortfolioIntegration:
    """Integration tests for portfolio optimization"""
    
    def test_efficient_frontier_properties(self):
        """Test efficient frontier has expected properties"""
        returns = np.array([0.10, 0.12, 0.08])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.0225]
        ])
        
        optimizer = PortfolioOptimizer(returns, cov)
        frontier = optimizer.efficient_frontier(num_points=15)
        
        # Extract returns and risks
        frontier_returns = [p['expected_return'] for p in frontier]
        frontier_risks = [p['volatility'] for p in frontier]
        
        # Returns should be increasing
        for i in range(1, len(frontier_returns)):
            assert frontier_returns[i] >= frontier_returns[i-1] - 1e-6
        
        # All points should be feasible
        for point in frontier:
            assert np.abs(point['weights'].sum() - 1.0) < 1e-6
            assert np.all(point['weights'] >= -1e-6)
    
    def test_minimum_variance_on_frontier(self):
        """Test that minimum variance portfolio is on the frontier"""
        returns = np.array([0.10, 0.12, 0.08])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.0225]
        ])
        
        optimizer = PortfolioOptimizer(returns, cov)
        
        # Get minimum variance portfolio
        min_var = optimizer.optimize()
        
        # Get frontier
        frontier = optimizer.efficient_frontier(num_points=20)
        
        # Minimum variance portfolio should have lowest risk on frontier
        frontier_risks = [p['volatility'] for p in frontier]
        min_frontier_risk = min(frontier_risks)
        
        assert min_var['volatility'] <= min_frontier_risk + 1e-4
    
    def test_risk_parity_vs_equal_weights(self):
        """Test that risk parity differs from equal weights"""
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.0225]
        ])
        
        optimizer = RiskParityOptimizer(cov)
        rp_weights = optimizer.optimize()
        
        equal_weights = np.array([1/3, 1/3, 1/3])
        
        # Risk parity weights should differ from equal weights
        # (because assets have different volatilities)
        assert not np.allclose(rp_weights, equal_weights, atol=1e-2)
        
        # Risk contributions should be more equal for risk parity
        rp_rc = optimizer.risk_contributions(rp_weights)
        eq_rc = optimizer.risk_contributions(equal_weights)
        
        rp_rc_std = np.std(rp_rc)
        eq_rc_std = np.std(eq_rc)
        
        assert rp_rc_std < eq_rc_std


class TestBlackLittermanModel:
    """Tests for the Black-Litterman portfolio model"""

    @staticmethod
    def _sample_covariance() -> np.ndarray:
        return np.array([[0.04, 0.006], [0.006, 0.09]])

    def test_equilibrium_returns_and_prior_posterior(self):
        """Equilibrium returns should match the posterior without views"""

        model = BlackLittermanModel(
            np.array([0.6, 0.4]),
            self._sample_covariance(),
            risk_aversion=3.0,
            tau=0.05,
        )

        equilibrium = model.equilibrium_returns()
        np.testing.assert_allclose(equilibrium, [0.0792, 0.1188], rtol=1e-6)

        posterior = model.posterior()
        np.testing.assert_allclose(
            posterior["equilibrium_returns"], equilibrium, atol=1e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            posterior["posterior_returns"], equilibrium, atol=1e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            posterior["posterior_covariance"],
            self._sample_covariance(),
            atol=1e-12,
            rtol=0.0,
        )
        assert abs(posterior["optimal_weights"].sum() - 1.0) < 1e-12

    def test_posterior_with_investor_views(self):
        """Investor views should tilt returns and optimal weights"""

        model = BlackLittermanModel(
            np.array([0.6, 0.4]),
            self._sample_covariance(),
            risk_aversion=3.0,
            tau=0.05,
        )

        views = InvestorViews(np.array([[1.0, -1.0]]), np.array([0.02]))
        posterior = model.posterior(views)

        np.testing.assert_allclose(
            posterior["posterior_returns"],
            [0.08778644, 0.09758644],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            posterior["posterior_covariance"],
            [[0.0417550847, 0.0069050847], [0.0069050847, 0.0930050847]],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            posterior["optimal_weights"],
            [0.68350558, 0.31649442],
            rtol=1e-6,
        )

        # Explicit uncertainty matrices should be accepted and change the tilt
        adjusted_views = views.with_uncertainty(np.array([[0.005]]))
        posterior_adjusted = model.posterior(adjusted_views)
        assert not np.allclose(
            posterior_adjusted["posterior_returns"],
            posterior["posterior_returns"],
        )


class TestFactorModel:
    """Tests for the multi-factor regression model"""

    @pytest.fixture
    def factor_dataset(self):
        """Simple synthetic factor/asset return dataset."""
        factor_returns = np.array(
            [
                [0.01, 0.02],
                [0.00, -0.01],
                [0.02, 0.01],
                [-0.01, 0.00],
                [0.03, 0.02],
            ]
        )
        asset_returns = np.array(
            [
                [0.020, 0.020],
                [-0.003, -0.006],
                [0.023, 0.018],
                [-0.006, -0.004],
                [0.036, 0.030],
            ]
        )
        return asset_returns, factor_returns

    def test_factor_model_estimation(self, factor_dataset):
        """Factor loadings and intercepts should match expectations."""
        asset_returns, factor_returns = factor_dataset
        model = FactorModel(
            asset_returns,
            factor_returns,
            include_intercept=True,
            factor_names=["market", "value"],
        )

        exposures = model.factor_exposures()
        assert exposures.shape == (2, 2)
        assert model.factor_names == ["market", "value"]
        assert model.include_intercept is True
        assert model.n_assets == 2
        assert model.n_factors == 2

        # Loadings should be close to the generating parameters
        np.testing.assert_allclose(exposures[0, 0], 0.8, rtol=0.05)
        np.testing.assert_allclose(exposures[0, 1], 0.5, rtol=0.05)
        np.testing.assert_allclose(exposures[1, 0], 0.5, rtol=0.05)
        np.testing.assert_allclose(exposures[1, 1], 0.7, rtol=0.05)

        alphas = model.alphas()
        assert alphas.shape == (2,)
        assert alphas[0] > 0.0
        assert alphas[1] > 0.0
        r_squared = model.r_squared()
        assert np.all((r_squared >= 0.0) & (r_squared <= 1.0))

    def test_expected_returns_and_attribution(self, factor_dataset):
        """Expected returns and factor attribution should align."""
        asset_returns, factor_returns = factor_dataset
        model = FactorModel(asset_returns, factor_returns)

        premia = np.array([0.015, 0.005])
        expected = model.expected_returns(premia, risk_free_rate=0.01)
        assert expected.shape == (2,)
        assert np.all(expected > 0.0)

        weights = np.array([0.6, 0.4])
        exposure = model.portfolio_factor_exposure(weights)
        assert exposure.shape == (2,)
        attribution = model.factor_attribution(weights, premia)
        assert attribution.shape == (2,)
        np.testing.assert_allclose(attribution.sum(), np.dot(exposure, premia), rtol=1e-6)

        portfolio_return = model.portfolio_expected_return(weights, premia, risk_free_rate=0.01)
        np.testing.assert_allclose(portfolio_return, float(np.dot(weights, expected)), rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
