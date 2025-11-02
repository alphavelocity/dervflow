# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for risk metrics including VaR calculations
"""

import math
from statistics import NormalDist

import numpy as np
import pytest

from dervflow import RiskMetrics


class TestVaRCalculations:
    """Test Value at Risk calculations"""

    def test_historical_var_basic(self):
        """Test basic historical VaR calculation"""
        rm = RiskMetrics()

        # Create simple returns data
        returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

        result = rm.var(returns, confidence_level=0.95, method="historical")

        assert "var" in result
        assert "confidence_level" in result
        assert "method" in result
        assert result["var"] > 0.0
        assert result["confidence_level"] == 0.95
        assert result["method"] == "historical"

    def test_parametric_var_basic(self):
        """Test parametric VaR calculation"""
        rm = RiskMetrics()

        # Create normally distributed returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        result = rm.var(returns, confidence_level=0.95, method="parametric")

        assert result["var"] > 0.0
        assert result["method"] == "parametric"

    def test_cornish_fisher_var(self):
        """Test Cornish-Fisher VaR calculation"""
        rm = RiskMetrics()

        # Create returns with some skewness
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        result = rm.var(returns, confidence_level=0.95, method="cornish_fisher")

        assert result["var"] > 0.0
        assert result["method"] == "cornish_fisher"

    def test_monte_carlo_var(self):
        """Test Monte Carlo VaR calculation"""
        rm = RiskMetrics()

        result = rm.var(
            returns=None,
            confidence_level=0.95,
            method="monte_carlo",
            mean=0.0,
            std_dev=0.02,
            num_simulations=10000,
            seed=42,
        )

        assert result["var"] > 0.0
        assert result["method"] == "monte_carlo"
        # For normal distribution with mean 0 and std 0.02, 95% VaR should be around 1.645 * 0.02
        assert abs(result["var"] - 0.0329) < 0.005

    def test_ewma_var(self):
        """Test EWMA (RiskMetrics) VaR calculation"""
        rm = RiskMetrics()

        returns = np.array([0.01, -0.015, 0.02, -0.005, 0.012])
        confidence = 0.975
        decay = 0.93

        result = rm.var(returns, confidence_level=confidence, method="ewma", decay=decay)
        assert result["method"].lower() == "ewma"

        variance = returns[0] ** 2
        for value in returns[1:]:
            variance = decay * variance + (1 - decay) * (value ** 2)
        sigma = math.sqrt(variance)
        alpha = 1 - confidence
        z = NormalDist().inv_cdf(alpha)
        expected = -z * sigma

        assert math.isclose(result["var"], expected)

    def test_ewma_var_aliases(self):
        """EWMA method should accept RiskMetrics aliases"""
        rm = RiskMetrics()
        returns = np.array([0.01, -0.015, 0.02, -0.005, 0.012])

        baseline = rm.var(returns, method="ewma", decay=0.94)
        alias = rm.var(returns, method="risk metrics", decay=0.94)

        assert math.isclose(baseline["var"], alias["var"])

    def test_ewma_var_invalid_decay(self):
        """EWMA should validate decay factor bounds"""
        rm = RiskMetrics()
        returns = np.array([0.01, -0.02, 0.015])

        with pytest.raises(Exception):
            rm.var(returns, method="ewma", decay=1.0)
        with pytest.raises(Exception):
            rm.var(returns, method="ewma", decay=-0.05)

    def test_var_comparison_methods(self):
        """Compare VaR across different methods"""
        rm = RiskMetrics()

        # Create normally distributed returns
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 5000)

        hist_result = rm.var(returns, confidence_level=0.95, method="historical")
        param_result = rm.var(returns, confidence_level=0.95, method="parametric")
        cf_result = rm.var(returns, confidence_level=0.95, method="cornish_fisher")

        # All methods should give similar results for normal distribution
        assert abs(hist_result["var"] - param_result["var"]) < 0.01
        assert abs(param_result["var"] - cf_result["var"]) < 0.01

    def test_var_different_confidence_levels(self):
        """Test VaR at different confidence levels"""
        rm = RiskMetrics()

        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 1000)

        var_90 = rm.var(returns, confidence_level=0.90, method="parametric")
        var_95 = rm.var(returns, confidence_level=0.95, method="parametric")
        var_99 = rm.var(returns, confidence_level=0.99, method="parametric")

        # Higher confidence level should give higher VaR
        assert var_90["var"] < var_95["var"]
        assert var_95["var"] < var_99["var"]

    def test_var_invalid_inputs(self):
        """Test VaR with invalid inputs"""
        rm = RiskMetrics()

        returns = np.array([0.01, -0.01, 0.02])

        # Invalid confidence level
        with pytest.raises(Exception):
            rm.var(returns, confidence_level=0.0, method="historical")

        with pytest.raises(Exception):
            rm.var(returns, confidence_level=1.0, method="historical")

        # Invalid method
        with pytest.raises(Exception):
            rm.var(returns, confidence_level=0.95, method="invalid_method")

        # Missing parameters for Monte Carlo
        with pytest.raises(Exception):
            rm.var(returns=None, confidence_level=0.95, method="monte_carlo")


class TestCVaRCalculations:
    """Test Conditional Value at Risk calculations"""

    def test_historical_cvar_basic(self):
        """Test basic historical CVaR calculation"""
        rm = RiskMetrics()

        returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

        result = rm.cvar(returns, confidence_level=0.95, method="historical")

        assert "cvar" in result
        assert result["cvar"] > 0.0

    def test_cvar_greater_than_var(self):
        """Test that CVaR is at least as large as VaR"""
        rm = RiskMetrics()

        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 1000)

        var_result = rm.var(returns, confidence_level=0.95, method="historical")
        cvar_result = rm.cvar(returns, confidence_level=0.95, method="historical")

        # CVaR should be >= VaR
        assert cvar_result["cvar"] >= var_result["var"]

    def test_monte_carlo_cvar(self):
        """Test Monte Carlo CVaR calculation"""
        rm = RiskMetrics()

        result = rm.cvar(
            returns=None,
            confidence_level=0.95,
            method="monte_carlo",
            mean=0.0,
            std_dev=0.02,
            num_simulations=10000,
            seed=42,
        )

        assert result["cvar"] > 0.0
        assert result["method"] == "monte_carlo"

    def test_ewma_cvar(self):
        """Test EWMA CVaR aligns with closed-form expected shortfall"""
        rm = RiskMetrics()
        returns = np.array([0.01, -0.015, 0.02, -0.005, 0.012])
        confidence = 0.975
        decay = 0.93

        result = rm.cvar(returns, confidence_level=confidence, method="ewma", decay=decay)

        variance = returns[0] ** 2
        for value in returns[1:]:
            variance = decay * variance + (1 - decay) * (value**2)
        sigma = math.sqrt(variance)
        alpha = 1 - confidence
        z = NormalDist().inv_cdf(alpha)
        pdf = math.exp(-0.5 * (z**2)) / math.sqrt(2 * math.pi)
        expected = sigma * (pdf / alpha)

        assert math.isclose(result["cvar"], expected, rel_tol=1e-8, abs_tol=1e-12)

    def test_ewma_cvar_aliases_and_validation(self):
        """EWMA CVaR should accept aliases and validate decay"""
        rm = RiskMetrics()
        returns = np.array([0.01, -0.015, 0.02, -0.005, 0.012])

        baseline = rm.cvar(returns, method="ewma", decay=0.9)
        alias = rm.cvar(returns, method="risk metrics", decay=0.9)

        assert math.isclose(baseline["cvar"], alias["cvar"])

        with pytest.raises(Exception):
            rm.cvar(returns, method="ewma", decay=1.0)
        with pytest.raises(Exception):
            rm.cvar(returns, method="ewma", decay=-0.05)


class TestRiskMetrics:
    """Test other risk metrics"""

    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        rm = RiskMetrics()

        # Create returns with a clear drawdown
        returns = np.array([0.01, 0.02, -0.05, -0.03, 0.01, 0.02])

        mdd = rm.max_drawdown(returns)

        assert mdd > 0.0
        assert mdd <= 1.0  # Drawdown should be between 0 and 1

    def test_max_drawdown_no_drawdown(self):
        """Test maximum drawdown with only positive returns"""
        rm = RiskMetrics()

        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])

        mdd = rm.max_drawdown(returns)

        # Should be very small or zero
        assert mdd >= 0.0
        assert mdd < 0.01

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        rm = RiskMetrics()

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        sortino = rm.sortino_ratio(returns, risk_free_rate=0.0)

        # Sortino ratio should be a reasonable number
        assert not np.isnan(sortino)
        assert not np.isinf(sortino) or sortino > 0

    def test_sortino_ratio_with_target(self):
        """Test Sortino ratio with custom target return"""
        rm = RiskMetrics()

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        sortino1 = rm.sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0)
        sortino2 = rm.sortino_ratio(returns, risk_free_rate=0.0, target_return=0.001)

        # Different target returns should give different Sortino ratios
        assert sortino1 != sortino2

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation"""
        rm = RiskMetrics()

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        calmar = rm.calmar_ratio(returns, periods_per_year=252)

        # Calmar ratio should be a reasonable number
        assert not np.isnan(calmar)

    def test_calmar_ratio_positive_returns(self):
        """Test Calmar ratio with mostly positive returns"""
        rm = RiskMetrics()

        # Create returns with small drawdown
        returns = np.array([0.01, 0.02, -0.005, 0.01, 0.015, 0.02])

        calmar = rm.calmar_ratio(returns, periods_per_year=252)

        # Should be positive for positive average returns
        assert calmar > 0.0


class TestVaREdgeCases:
    """Test edge cases and error handling"""

    def test_empty_returns(self):
        """Test with empty returns array"""
        rm = RiskMetrics()

        returns = np.array([])

        with pytest.raises(Exception):
            rm.var(returns, confidence_level=0.95, method="historical")

    def test_single_return(self):
        """Test with single return value"""
        rm = RiskMetrics()

        returns = np.array([0.01])

        result = rm.var(returns, confidence_level=0.95, method="historical")
        # With a single positive return, VaR will be negative (no loss)
        assert isinstance(result["var"], float)

    def test_all_same_returns(self):
        """Test with all identical returns"""
        rm = RiskMetrics()

        returns = np.full(100, 0.01)

        result = rm.var(returns, confidence_level=0.95, method="parametric")
        # With constant positive returns, VaR will be negative (no loss)
        # The parametric method uses std dev which is 0, so VaR = -(mean + 0)
        assert isinstance(result["var"], float)

    def test_extreme_returns(self):
        """Test with extreme return values"""
        rm = RiskMetrics()

        returns = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])

        result = rm.var(returns, confidence_level=0.95, method="historical")
        assert result["var"] > 0.0

    def test_monte_carlo_reproducibility(self):
        """Test that Monte Carlo VaR is reproducible with same seed"""
        rm = RiskMetrics()

        result1 = rm.var(
            returns=None,
            confidence_level=0.95,
            method="monte_carlo",
            mean=0.0,
            std_dev=0.02,
            num_simulations=1000,
            seed=42,
        )

        result2 = rm.var(
            returns=None,
            confidence_level=0.95,
            method="monte_carlo",
            mean=0.0,
            std_dev=0.02,
            num_simulations=1000,
            seed=42,
        )

        # Results should be identical with same seed
        assert result1["var"] == result2["var"]


class TestPortfolioRiskMetrics:
    """Tests for portfolio-level risk utilities"""

    def setup_method(self):
        self.rm = RiskMetrics()
        self.weights = np.array([0.4, 0.6])
        self.covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        self.expected_returns = np.array([0.10, 0.12])

    def test_portfolio_metrics_basic(self):
        metrics = self.rm.portfolio_metrics(
            self.weights,
            self.covariance,
            expected_returns=self.expected_returns,
            risk_free_rate=0.02,
        )

        assert metrics["volatility"] > 0.0
        assert metrics["variance"] > 0.0
        assert metrics["expected_return"] is not None

        risk_contributions = metrics["risk_contributions"]
        assert set(risk_contributions.keys()) == {"marginal", "component", "percentage"}
        np.testing.assert_allclose(np.sum(risk_contributions["percentage"]), 1.0, atol=1e-10)

    def test_portfolio_metrics_without_returns(self):
        metrics = self.rm.portfolio_metrics(self.weights, self.covariance)
        assert metrics["expected_return"] is None
        assert metrics["sharpe_ratio"] is None

    def test_parametric_var_and_cvar(self):
        var_value = self.rm.portfolio_var_parametric(
            self.weights,
            self.covariance,
            confidence_level=0.95,
            expected_returns=self.expected_returns,
        )
        cvar_value = self.rm.portfolio_cvar_parametric(
            self.weights,
            self.covariance,
            confidence_level=0.95,
            expected_returns=self.expected_returns,
        )

        assert var_value >= 0.0
        assert cvar_value >= var_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
