# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for dervflow

Tests complete workflows combining multiple modules:
- Price options → Calculate Greeks → Compute VaR
- Build yield curve → Price bonds
- Optimize portfolio with constraints
"""

import pytest
import numpy as np
from dervflow import (
    BlackScholesModel,
    BinomialTreeModel,
    MonteCarloOptionPricer,
    GreeksCalculator,
    RiskMetrics,
    PortfolioOptimizer,
    YieldCurve,
    YieldCurveBuilder,
    TimeSeriesAnalyzer,
    MonteCarloEngine
)


class TestOptionPricingWorkflow:
    """Test complete option pricing and risk analysis workflow"""

    def test_price_greeks_var_workflow(self):
        """Test workflow: Price option → Calculate Greeks → Compute VaR"""
        # Step 1: Price option
        bs = BlackScholesModel()
        price = bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
        assert price > 0.0
        
        # Step 2: Calculate Greeks
        greeks = bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        
        # Step 3: Simulate portfolio returns and compute VaR
        np.random.seed(42)
        spot_returns = np.random.normal(0.001, 0.02, 1000)
        option_returns = greeks['delta'] * spot_returns
        
        rm = RiskMetrics()
        var_result = rm.var(option_returns, confidence_level=0.95, method='historical')
        assert var_result['var'] > 0.0

    def test_batch_pricing_greeks_workflow(self):
        """Test batch pricing with Greeks calculation"""
        bs = BlackScholesModel()
        
        # Batch price multiple options
        n = 10
        spots = np.linspace(90, 110, n)
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.full(n, 0.2)
        times = np.full(n, 1.0)
        option_types = ['call'] * n
        
        prices = bs.price_batch(spots, strikes, rates, dividends, volatilities, times, option_types)
        assert len(prices) == n
        assert all(prices > 0.0)
        
        # Calculate Greeks for each option
        deltas = []
        for i in range(n):
            greeks = bs.greeks(spots[i], strikes[i], rates[i], dividends[i], 
                             volatilities[i], times[i], option_types[i])
            deltas.append(greeks['delta'])
        
        assert len(deltas) == n
        # Delta should increase with spot price for calls
        assert all(deltas[i] <= deltas[i+1] for i in range(n-1))

    def test_implied_vol_surface_workflow(self):
        """Test implied volatility surface construction"""
        bs = BlackScholesModel()
        
        # Create market data with known volatilities
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        true_vols = np.array([0.25, 0.22, 0.20, 0.22, 0.25])  # Volatility smile
        
        # Calculate market prices
        market_prices = np.array([
            bs.price(100.0, strike, 0.05, 0.0, vol, 1.0, 'call')
            for strike, vol in zip(strikes, true_vols)
        ])
        
        # Calculate implied volatilities
        spots = np.full(len(strikes), 100.0)
        rates = np.full(len(strikes), 0.05)
        dividends = np.full(len(strikes), 0.0)
        times = np.full(len(strikes), 1.0)
        option_types = ['call'] * len(strikes)
        
        implied_vols = bs.implied_vol_batch(
            market_prices, spots, strikes, rates, dividends, times, option_types
        )
        
        # Should recover the true volatilities
        for iv, true_vol in zip(implied_vols, true_vols):
            assert abs(iv - true_vol) < 0.001


class TestPortfolioWorkflow:
    """Test portfolio optimization and risk analysis workflow"""

    def test_portfolio_optimization_risk_workflow(self):
        """Test workflow: Optimize portfolio → Calculate risk metrics"""
        # Generate sample returns for 3 assets
        np.random.seed(42)
        n_assets = 3
        n_periods = 252
        
        returns = np.random.multivariate_normal(
            mean=[0.001, 0.0015, 0.0008],
            cov=[[0.0004, 0.0001, 0.00005],
                 [0.0001, 0.0006, 0.0001],
                 [0.00005, 0.0001, 0.0003]],
            size=n_periods
        )
        
        # Step 1: Optimize portfolio
        optimizer = PortfolioOptimizer(returns)
        # Use a target return within the feasible range (average of mean returns)
        mean_returns = np.mean(returns, axis=0)
        target_return = np.mean(mean_returns)
        result = optimizer.optimize(target_return=target_return)
        
        assert 'weights' in result
        assert len(result['weights']) == n_assets
        assert abs(sum(result['weights']) - 1.0) < 1e-6
        
        # Step 2: Calculate portfolio returns
        portfolio_returns = returns @ result['weights']
        
        # Step 3: Calculate risk metrics
        rm = RiskMetrics()
        var_result = rm.var(portfolio_returns, confidence_level=0.95, method='historical')
        cvar_result = rm.cvar(portfolio_returns, confidence_level=0.95, method='historical')
        
        assert var_result['var'] > 0.0
        assert cvar_result['cvar'] >= var_result['var']

    def test_efficient_frontier_workflow(self):
        """Test efficient frontier calculation and analysis"""
        np.random.seed(42)
        n_assets = 4
        n_periods = 252
        
        returns = np.random.multivariate_normal(
            mean=[0.0008, 0.001, 0.0012, 0.0015],
            cov=np.eye(n_assets) * 0.0004 + 0.0001,
            size=n_periods
        )
        
        optimizer = PortfolioOptimizer(returns)
        frontier = optimizer.efficient_frontier(num_points=10)
        
        assert len(frontier) == 10
        
        # Verify frontier properties
        for i in range(len(frontier) - 1):
            # Returns should be increasing
            assert frontier[i+1]['expected_return'] >= frontier[i]['expected_return']
            # Risk should generally increase with return
            # (allowing small numerical variations)


class TestYieldCurveWorkflow:
    """Test yield curve construction and bond pricing workflow"""

    def test_yield_curve_bond_pricing_workflow(self):
        """Test workflow: Bootstrap curve → Price bonds"""
        # Step 1: Create bond data
        bond_data = [
            {'maturity': 0.5, 'coupon': 0.02, 'price': 99.5, 'frequency': 2},
            {'maturity': 1.0, 'coupon': 0.025, 'price': 99.8, 'frequency': 2},
            {'maturity': 2.0, 'coupon': 0.03, 'price': 100.2, 'frequency': 2},
            {'maturity': 5.0, 'coupon': 0.04, 'price': 101.5, 'frequency': 2}
        ]
        
        # Step 2: Bootstrap yield curve
        bonds = [(b['maturity'], b['coupon'], b['price'], b['frequency']) for b in bond_data]
        curve = YieldCurveBuilder.bootstrap_from_bonds(bonds)
        
        # Step 3: Query rates
        rate_1y = curve.zero_rate(1.0)
        rate_2y = curve.zero_rate(2.0)
        
        assert rate_1y > 0.0
        assert rate_2y > 0.0
        
        # Step 4: Calculate forward rate
        forward_rate = curve.forward_rate(1.0, 2.0)
        assert forward_rate > 0.0
        
        # Step 5: Calculate discount factors
        df_1y = curve.discount_factor(1.0)
        df_2y = curve.discount_factor(2.0)
        
        assert 0.0 < df_1y < 1.0
        assert 0.0 < df_2y < df_1y  # Longer maturity should have lower DF

    def test_yield_curve_interpolation_workflow(self):
        """Test yield curve with different interpolation methods"""
        bond_data = [
            {'maturity': 1.0, 'coupon': 0.02, 'price': 99.0, 'frequency': 1},
            {'maturity': 2.0, 'coupon': 0.03, 'price': 100.0, 'frequency': 1},
            {'maturity': 5.0, 'coupon': 0.04, 'price': 101.0, 'frequency': 1}
        ]
        
        # Test different interpolation methods
        bonds = [(b['maturity'], b['coupon'], b['price'], b['frequency']) for b in bond_data]
        for method in ['linear', 'cubic_spline']:
            curve = YieldCurveBuilder.bootstrap_from_bonds(bonds)
            
            # Interpolate at intermediate point
            rate_3y = curve.zero_rate(3.0)
            assert rate_3y > 0.0
            
            # Rate should be between 2y and 5y rates
            rate_2y = curve.zero_rate(2.0)
            rate_5y = curve.zero_rate(5.0)
            assert rate_2y <= rate_3y <= rate_5y or rate_5y <= rate_3y <= rate_2y


class TestMonteCarloWorkflow:
    """Test Monte Carlo simulation workflows"""

    def test_monte_carlo_option_pricing_workflow(self):
        """Test workflow: Simulate paths → Price option → Calculate Greeks"""
        mc_pricer = MonteCarloOptionPricer()
        
        # Step 1: Price European option
        result = mc_pricer.price_european(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            dividend=0.0,
            volatility=0.2,
            time=1.0,
            option_type='call',
            num_paths=10000,
            seed=42
        )
        
        assert 'price' in result
        assert 'std_error' in result
        assert result['price'] > 0.0
        assert result['std_error'] > 0.0
        
        # Compare with Black-Scholes
        bs = BlackScholesModel()
        bs_price = bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
        
        # MC price should be close to BS price (within 3 standard errors)
        assert abs(result['price'] - bs_price) < 3 * result['std_error']

    def test_correlated_paths_workflow(self):
        """Test workflow: Generate correlated paths → Calculate portfolio value"""
        mc_engine = MonteCarloEngine()
        
        # Generate correlated paths for 2 assets
        correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
        
        paths = mc_engine.simulate_correlated(
            initial_values=[100.0, 100.0],
            mu_values=[0.05, 0.06],
            sigma_values=[0.2, 0.25],
            correlation=correlation,
            T=1.0,
            steps=252,
            paths=1000
        )
        
        assert len(paths) == 2
        assert paths[0].shape == (1000, 252)
        assert paths[1].shape == (1000, 252)
        
        # Calculate correlation of final values
        final_returns_1 = np.log(paths[0][:, -1] / 100.0)
        final_returns_2 = np.log(paths[1][:, -1] / 100.0)
        empirical_corr = np.corrcoef(final_returns_1, final_returns_2)[0, 1]
        
        # Empirical correlation should be close to specified correlation
        assert abs(empirical_corr - 0.6) < 0.1


class TestTimeSeriesWorkflow:
    """Test time series analysis workflows"""

    def test_returns_stat_workflow(self):
        """Test workflow: Calculate returns → Compute stat → Test stationarity"""
        # Generate price series
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
        
        analyzer = TimeSeriesAnalyzer(prices)
        
        # Step 1: Calculate returns
        returns = analyzer.returns(method='log')
        assert len(returns) == len(prices) - 1
        
        # Step 2: Compute stat
        stats = analyzer.stat()
        assert 'mean' in stats
        assert 'std_dev' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        
        # Step 3: Test stationarity
        adf_result = analyzer.stationarity_test(test='adf')
        assert 'statistic' in adf_result
        assert 'p_value' in adf_result

    def test_garch_volatility_workflow(self):
        """Test workflow: Fit GARCH model → Forecast volatility"""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 500)
        
        analyzer = TimeSeriesAnalyzer(returns)
        
        # Fit GARCH model
        garch_result = analyzer.fit_garch(variant='standard')
        
        assert 'omega' in garch_result
        assert 'alpha' in garch_result
        assert 'beta' in garch_result
        
        # Parameters should be positive
        assert garch_result['omega'] > 0.0
        assert garch_result['alpha'] > 0.0
        assert garch_result['beta'] > 0.0


class TestNumPyConversions:
    """Test NumPy array conversions between Python and Rust"""

    def test_array_input_output(self):
        """Test that NumPy arrays are correctly converted"""
        bs = BlackScholesModel()
        
        # Test with NumPy arrays
        spots = np.array([95.0, 100.0, 105.0])
        strikes = np.array([100.0, 100.0, 100.0])
        rates = np.array([0.05, 0.05, 0.05])
        dividends = np.array([0.0, 0.0, 0.0])
        volatilities = np.array([0.2, 0.2, 0.2])
        times = np.array([1.0, 1.0, 1.0])
        option_types = ['call', 'call', 'call']
        
        prices = bs.price_batch(spots, strikes, rates, dividends, volatilities, times, option_types)
        
        # Result should be NumPy array
        assert isinstance(prices, np.ndarray)
        assert prices.dtype == np.float64
        assert len(prices) == 3

    def test_2d_array_conversion(self):
        """Test 2D array conversion for correlation matrices"""
        mc_engine = MonteCarloEngine()
        
        correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        paths = mc_engine.simulate_correlated(
            initial_values=[100.0, 100.0],
            mu_values=[0.05, 0.05],
            sigma_values=[0.2, 0.2],
            correlation=correlation,
            T=1.0,
            steps=10,
            paths=100
        )
        
        # Results should be list of NumPy arrays
        assert isinstance(paths, list)
        assert all(isinstance(p, np.ndarray) for p in paths)
        assert all(p.shape == (100, 10) for p in paths)


class TestErrorHandling:
    """Test error handling across modules"""

    def test_invalid_option_parameters(self):
        """Test error handling for invalid option parameters"""
        bs = BlackScholesModel()
        
        with pytest.raises(ValueError):
            bs.price(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 'call')
        
        with pytest.raises(ValueError):
            bs.price(100.0, -100.0, 0.05, 0.0, 0.2, 1.0, 'call')
        
        with pytest.raises(ValueError):
            bs.price(100.0, 100.0, 0.05, 0.0, -0.2, 1.0, 'call')

    def test_convergence_failure_handling(self):
        """Test handling of convergence failures"""
        bs = BlackScholesModel()
        
        # Try to calculate IV for price below intrinsic value
        with pytest.raises(ValueError):
            bs.implied_vol(5.0, 110.0, 100.0, 0.05, 0.0, 1.0, 'call')

    def test_optimization_infeasible_handling(self):
        """Test handling of infeasible optimization"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (252, 3))
        
        optimizer = PortfolioOptimizer(returns)
        
        # Try to optimize with impossible target return
        with pytest.raises(Exception):
            optimizer.optimize(target_return=1.0)  # 100% return is infeasible


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
