# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Greeks calculation module

Tests cover:
- Numerical Greeks calculation using finite differences
- Extended Greeks (second and third order)
- Portfolio Greeks aggregation
- Comparison with analytical Greeks
"""

import numpy as np
import pytest

from dervflow import BlackScholesModel, GreeksCalculator


class TestNumericalGreeks:
    """Test numerical Greeks calculation using finite differences"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calc = GreeksCalculator()
        self.bs = BlackScholesModel()

    def test_delta_call_atm(self):
        """Test delta for at-the-money call option"""
        greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        # Delta should be around 0.5-0.6 for ATM call
        assert 0.5 < greeks["delta"] < 0.7

    def test_delta_put_atm(self):
        """Test delta for at-the-money put option"""
        greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        # Delta should be around -0.4 to -0.5 for ATM put
        assert -0.6 < greeks["delta"] < 0.0

    def test_gamma_positive(self):
        """Test that gamma is always positive"""
        call_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert call_greeks["gamma"] > 0.0
        assert put_greeks["gamma"] > 0.0

    def test_vega_positive(self):
        """Test that vega is always positive"""
        call_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert call_greeks["vega"] > 0.0
        assert put_greeks["vega"] > 0.0

    def test_theta_negative(self):
        """Test that theta is typically negative (time decay)"""
        call_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert call_greeks["theta"] < 0.0
        assert put_greeks["theta"] < 0.0

    def test_rho_call_positive(self):
        """Test that call rho is positive"""
        greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert greeks["rho"] > 0.0

    def test_rho_put_negative(self):
        """Test that put rho is negative"""
        greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert greeks["rho"] < 0.0

    def test_numerical_vs_analytical_delta(self):
        """Test numerical delta matches analytical delta"""
        numerical = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        analytical = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Should be within 1% of analytical
        rel_error = abs(numerical["delta"] - analytical["delta"]) / abs(analytical["delta"])
        assert rel_error < 0.01

    def test_numerical_vs_analytical_gamma(self):
        """Test numerical gamma matches analytical gamma"""
        numerical = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        analytical = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Gamma is second derivative, so allow 5% error
        rel_error = abs(numerical["gamma"] - analytical["gamma"]) / abs(analytical["gamma"])
        assert rel_error < 0.05

    def test_numerical_vs_analytical_vega(self):
        """Test numerical vega matches analytical vega"""
        numerical = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        analytical = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Should be within 1% of analytical
        rel_error = abs(numerical["vega"] - analytical["vega"]) / abs(analytical["vega"])
        assert rel_error < 0.01

    def test_numerical_vs_analytical_theta(self):
        """Test numerical theta matches analytical theta"""
        numerical = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        analytical = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Theta can have larger relative error due to numerical differentiation
        # Check absolute difference instead for small values
        abs_diff = abs(numerical["theta"] - analytical["theta"])
        assert (
            abs_diff < 0.5
            or abs(numerical["theta"] - analytical["theta"]) / abs(analytical["theta"]) < 0.10
        )

    def test_numerical_vs_analytical_rho(self):
        """Test numerical rho matches analytical rho"""
        numerical = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        analytical = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Should be within 1% of analytical
        rel_error = abs(numerical["rho"] - analytical["rho"]) / abs(analytical["rho"])
        assert rel_error < 0.01

    def test_greeks_itm_call(self):
        """Test Greeks for in-the-money call option"""
        greeks = self.calc.calculate(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Delta should be high for ITM call
        assert greeks["delta"] > 0.7
        # Gamma should be positive
        assert greeks["gamma"] > 0.0
        # Vega should be positive
        assert greeks["vega"] > 0.0

    def test_greeks_otm_put(self):
        """Test Greeks for out-of-the-money put option"""
        greeks = self.calc.calculate(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")

        # Delta should be small negative for OTM put
        assert -0.3 < greeks["delta"] < 0.0
        # Gamma should be positive
        assert greeks["gamma"] > 0.0

    def test_greeks_with_dividend(self):
        """Test Greeks calculation with dividend yield"""
        greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, "call")

        # All Greeks should be finite
        assert np.isfinite(greeks["delta"])
        assert np.isfinite(greeks["gamma"])
        assert np.isfinite(greeks["vega"])
        assert np.isfinite(greeks["theta"])
        assert np.isfinite(greeks["rho"])

    def test_custom_bump_sizes(self):
        """Test Greeks calculation with custom bump sizes"""
        calc_small = GreeksCalculator(spot_bump=0.001, vol_bump=0.001)
        greeks = calc_small.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Should still produce reasonable results
        assert 0.5 < greeks["delta"] < 0.7
        assert greeks["gamma"] > 0.0

    def test_invalid_option_type(self):
        """Test error handling for invalid option type"""
        with pytest.raises(ValueError, match="Invalid option type"):
            self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "invalid")


class TestExtendedGreeks:
    """Test extended Greeks including second and third order"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calc = GreeksCalculator()

    def test_extended_greeks_call_atm(self):
        """Test extended Greeks for at-the-money call option"""
        extended = self.calc.calculate_extended(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # First-order Greeks
        assert 0.5 < extended["delta"] < 0.7
        assert extended["gamma"] > 0.0
        assert extended["vega"] > 0.0

        # Second-order Greeks
        assert np.isfinite(extended["vanna"])
        assert extended["volga"] > 0.0  # Volga should be positive for ATM

        # Third-order Greeks
        assert np.isfinite(extended["speed"])
        assert np.isfinite(extended["zomma"])
        assert np.isfinite(extended["color"])
        assert np.isfinite(extended["ultima"])

    def test_extended_greeks_put_atm(self):
        """Test extended Greeks for at-the-money put option"""
        extended = self.calc.calculate_extended(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")

        # First-order Greeks
        assert extended["delta"] < 0.0
        assert extended["gamma"] > 0.0
        assert extended["vega"] > 0.0

        # Second and third order Greeks should be finite
        assert np.isfinite(extended["vanna"])
        assert np.isfinite(extended["volga"])
        assert np.isfinite(extended["speed"])
        assert np.isfinite(extended["zomma"])
        assert np.isfinite(extended["color"])
        assert np.isfinite(extended["ultima"])

    def test_vanna_non_zero(self):
        """Test that vanna is non-zero for ATM options"""
        extended = self.calc.calculate_extended(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert abs(extended["vanna"]) > 0.0

    def test_volga_positive_atm(self):
        """Test that volga is positive for ATM options"""
        extended = self.calc.calculate_extended(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert extended["volga"] > 0.0

    def test_extended_greeks_itm_call(self):
        """Test extended Greeks for in-the-money call option"""
        extended = self.calc.calculate_extended(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        # Delta should be high for ITM call
        assert extended["delta"] > 0.7

        # All higher-order Greeks should be calculable
        assert np.isfinite(extended["vanna"])
        assert np.isfinite(extended["volga"])
        assert np.isfinite(extended["speed"])
        assert np.isfinite(extended["zomma"])
        assert np.isfinite(extended["color"])
        assert np.isfinite(extended["ultima"])


class TestPortfolioGreeks:
    """Test portfolio Greeks aggregation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calc = GreeksCalculator()

    def test_portfolio_greeks_two_positions(self):
        """Test portfolio Greeks with two positions"""
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0, 105.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "put"]
        quantities = np.array([100.0, 50.0])

        portfolio = self.calc.portfolio_greeks(
            spots, strikes, rates, dividends, volatilities, times, option_types, quantities
        )

        # Portfolio Greeks should be finite
        assert np.isfinite(portfolio["delta"])
        assert np.isfinite(portfolio["gamma"])
        assert np.isfinite(portfolio["vega"])
        assert np.isfinite(portfolio["theta"])
        assert np.isfinite(portfolio["rho"])

    def test_portfolio_greeks_with_short_positions(self):
        """Test portfolio Greeks with short positions"""
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "call"]
        quantities = np.array([100.0, -50.0])  # Long 100, short 50

        portfolio = self.calc.portfolio_greeks(
            spots, strikes, rates, dividends, volatilities, times, option_types, quantities
        )

        # Net delta should be positive but less than 100 calls
        single_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        expected_delta = single_greeks["delta"] * 100.0 - single_greeks["delta"] * 50.0

        assert abs(portfolio["delta"] - expected_delta) < 1.0

    def test_portfolio_greeks_hedged_position(self):
        """Test portfolio Greeks for hedged position"""
        # Long ATM call + short OTM call (call spread)
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0, 105.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "call"]
        quantities = np.array([100.0, -100.0])

        portfolio = self.calc.portfolio_greeks(
            spots, strikes, rates, dividends, volatilities, times, option_types, quantities
        )

        # Delta should be reduced due to hedging
        long_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert abs(portfolio["delta"]) < abs(long_greeks["delta"] * 100.0)

    def test_portfolio_greeks_large_portfolio(self):
        """Test portfolio Greeks with many positions"""
        n = 20
        spots = np.full(n, 100.0)
        strikes = np.linspace(90.0, 110.0, n)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.full(n, 0.2)
        times = np.full(n, 1.0)
        option_types = ["call"] * (n // 2) + ["put"] * (n // 2)
        quantities = np.random.uniform(-100, 100, n)

        portfolio = self.calc.portfolio_greeks(
            spots, strikes, rates, dividends, volatilities, times, option_types, quantities
        )

        # All Greeks should be finite
        assert np.isfinite(portfolio["delta"])
        assert np.isfinite(portfolio["gamma"])
        assert np.isfinite(portfolio["vega"])
        assert np.isfinite(portfolio["theta"])
        assert np.isfinite(portfolio["rho"])

    def test_portfolio_extended_greeks(self):
        """Test portfolio extended Greeks"""
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0, 105.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "put"]
        quantities = np.array([100.0, 50.0])

        portfolio = self.calc.portfolio_extended_greeks(
            spots, strikes, rates, dividends, volatilities, times, option_types, quantities
        )

        # All Greeks should be finite
        assert np.isfinite(portfolio["delta"])
        assert np.isfinite(portfolio["gamma"])
        assert np.isfinite(portfolio["vega"])
        assert np.isfinite(portfolio["vanna"])
        assert np.isfinite(portfolio["volga"])
        assert np.isfinite(portfolio["speed"])
        assert np.isfinite(portfolio["zomma"])
        assert np.isfinite(portfolio["color"])
        assert np.isfinite(portfolio["ultima"])

    def test_portfolio_greeks_mismatched_lengths(self):
        """Test error handling for mismatched array lengths"""
        spots = np.array([100.0, 100.0])
        strikes = np.array([100.0])  # Wrong length
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "call"]
        quantities = np.array([100.0, 50.0])

        with pytest.raises(ValueError, match="same length"):
            self.calc.portfolio_greeks(
                spots, strikes, rates, dividends, volatilities, times, option_types, quantities
            )

    def test_empty_portfolio(self):
        """Test portfolio Greeks with empty arrays"""
        spots = np.array([])
        strikes = np.array([])
        rates = np.array([])
        dividends = np.array([])
        volatilities = np.array([])
        times = np.array([])
        option_types = []
        quantities = np.array([])

        portfolio = self.calc.portfolio_greeks(
            spots, strikes, rates, dividends, volatilities, times, option_types, quantities
        )

        # Empty portfolio should have zero Greeks
        assert portfolio["delta"] == 0.0
        assert portfolio["gamma"] == 0.0
        assert portfolio["vega"] == 0.0
        assert portfolio["theta"] == 0.0
        assert portfolio["rho"] == 0.0


class TestGreeksRelationships:
    """Test mathematical relationships between Greeks"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calc = GreeksCalculator()

    def test_gamma_symmetry_call_put(self):
        """Test that gamma is the same for call and put"""
        call_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")

        # Gamma should be identical (within numerical precision)
        assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 0.001

    def test_vega_symmetry_call_put(self):
        """Test that vega is the same for call and put"""
        call_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")

        # Vega should be identical (within numerical precision)
        assert abs(call_greeks["vega"] - put_greeks["vega"]) < 0.001

    def test_delta_increases_with_spot(self):
        """Test that call delta increases with spot price"""
        delta_90 = self.calc.calculate(90.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["delta"]
        delta_100 = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["delta"]
        delta_110 = self.calc.calculate(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["delta"]

        assert delta_90 < delta_100 < delta_110

    def test_gamma_peaks_at_atm(self):
        """Test that gamma is highest near at-the-money options"""
        # Use analytical Greeks for this test since numerical can have errors
        bs = BlackScholesModel()
        gamma_80 = bs.greeks(80.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["gamma"]
        gamma_100 = bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["gamma"]
        gamma_120 = bs.greeks(120.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["gamma"]

        # Gamma should be higher near ATM than far OTM or ITM
        assert gamma_100 > gamma_80
        assert gamma_100 > gamma_120

    def test_vega_peaks_at_atm(self):
        """Test that vega is highest for at-the-money options"""
        vega_90 = self.calc.calculate(90.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["vega"]
        vega_100 = self.calc.calculate(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["vega"]
        vega_110 = self.calc.calculate(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")["vega"]

        # ATM vega should be highest
        assert vega_100 > vega_90
        assert vega_100 > vega_110


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
