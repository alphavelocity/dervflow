# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Property-based tests for dervflow

Tests mathematical invariants and relationships that should hold for all valid inputs:
- Put-call parity
- Greeks relationships
- Arbitrage-free conditions
- Monotonicity properties
"""

import numpy as np
import pytest

from dervflow import BinomialTreeModel, BlackScholesModel, MonteCarloOptionPricer


class TestPutCallParity:
    """Test put-call parity relationship"""

    def test_put_call_parity_no_dividend(self):
        """Test put-call parity: C - P = S - K*e^(-r*T)"""
        bs = BlackScholesModel()

        # Test across various parameters
        test_cases = [
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (110.0, 100.0, 0.05, 0.0, 0.3, 0.5),
            (90.0, 100.0, 0.03, 0.0, 0.15, 2.0),
            (100.0, 120.0, 0.04, 0.0, 0.25, 1.5),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            call_price = bs.price(spot, strike, rate, dividend, vol, time, "call")
            put_price = bs.price(spot, strike, rate, dividend, vol, time, "put")

            lhs = call_price - put_price
            rhs = spot * np.exp(-dividend * time) - strike * np.exp(-rate * time)

            assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated for params: {test_cases}"

    def test_put_call_parity_with_dividend(self):
        """Test put-call parity with dividend: C - P = S*e^(-q*T) - K*e^(-r*T)"""
        bs = BlackScholesModel()

        test_cases = [
            (100.0, 100.0, 0.05, 0.02, 0.2, 1.0),
            (110.0, 100.0, 0.05, 0.03, 0.3, 0.5),
            (90.0, 100.0, 0.03, 0.01, 0.15, 2.0),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            call_price = bs.price(spot, strike, rate, dividend, vol, time, "call")
            put_price = bs.price(spot, strike, rate, dividend, vol, time, "put")

            lhs = call_price - put_price
            rhs = spot * np.exp(-dividend * time) - strike * np.exp(-rate * time)

            assert abs(lhs - rhs) < 1e-10

    def test_put_call_parity_various_strikes(self):
        """Test put-call parity across strike prices"""
        bs = BlackScholesModel()

        spot, rate, dividend, vol, time = 100.0, 0.05, 0.02, 0.2, 1.0
        strikes = np.linspace(70, 130, 20)

        for strike in strikes:
            call_price = bs.price(spot, strike, rate, dividend, vol, time, "call")
            put_price = bs.price(spot, strike, rate, dividend, vol, time, "put")

            lhs = call_price - put_price
            rhs = spot * np.exp(-dividend * time) - strike * np.exp(-rate * time)

            assert abs(lhs - rhs) < 1e-9

    def test_put_call_parity_various_maturities(self):
        """Test put-call parity across maturities"""
        bs = BlackScholesModel()

        spot, strike, rate, dividend, vol = 100.0, 100.0, 0.05, 0.02, 0.2
        maturities = np.linspace(0.1, 5.0, 20)

        for time in maturities:
            call_price = bs.price(spot, strike, rate, dividend, vol, time, "call")
            put_price = bs.price(spot, strike, rate, dividend, vol, time, "put")

            lhs = call_price - put_price
            rhs = spot * np.exp(-dividend * time) - strike * np.exp(-rate * time)

            assert abs(lhs - rhs) < 1e-9


class TestGreeksRelationships:
    """Test relationships between Greeks"""

    def test_delta_put_call_relationship(self):
        """Test Delta_call - Delta_put = e^(-q*T)"""
        bs = BlackScholesModel()

        test_cases = [
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (110.0, 100.0, 0.05, 0.02, 0.3, 0.5),
            (90.0, 100.0, 0.03, 0.01, 0.15, 2.0),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            call_greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "call")
            put_greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "put")

            delta_diff = call_greeks["delta"] - put_greeks["delta"]
            expected = np.exp(-dividend * time)

            assert abs(delta_diff - expected) < 1e-10

    def test_gamma_symmetry(self):
        """Test that Gamma is the same for call and put"""
        bs = BlackScholesModel()

        test_cases = [
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (110.0, 100.0, 0.05, 0.02, 0.3, 0.5),
            (90.0, 100.0, 0.03, 0.01, 0.15, 2.0),
            (100.0, 120.0, 0.04, 0.0, 0.25, 1.5),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            call_greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "call")
            put_greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "put")

            assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 1e-10

    def test_vega_symmetry(self):
        """Test that Vega is the same for call and put"""
        bs = BlackScholesModel()

        test_cases = [
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (110.0, 100.0, 0.05, 0.02, 0.3, 0.5),
            (90.0, 100.0, 0.03, 0.01, 0.15, 2.0),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            call_greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "call")
            put_greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "put")

            assert abs(call_greeks["vega"] - put_greeks["vega"]) < 1e-10

    def test_gamma_always_positive(self):
        """Test that Gamma is always positive"""
        bs = BlackScholesModel()

        # Test across wide range of parameters
        spots = [80, 90, 100, 110, 120]
        strikes = [90, 100, 110]
        vols = [0.1, 0.2, 0.3, 0.4]
        times = [0.25, 0.5, 1.0, 2.0]

        for spot in spots:
            for strike in strikes:
                for vol in vols:
                    for time in times:
                        call_greeks = bs.greeks(spot, strike, 0.05, 0.0, vol, time, "call")
                        put_greeks = bs.greeks(spot, strike, 0.05, 0.0, vol, time, "put")

                        assert call_greeks["gamma"] > 0.0
                        assert put_greeks["gamma"] > 0.0

    def test_vega_always_positive(self):
        """Test that Vega is always positive"""
        bs = BlackScholesModel()

        spots = [80, 90, 100, 110, 120]
        strikes = [90, 100, 110]
        times = [0.25, 0.5, 1.0, 2.0]

        for spot in spots:
            for strike in strikes:
                for time in times:
                    call_greeks = bs.greeks(spot, strike, 0.05, 0.0, 0.2, time, "call")
                    put_greeks = bs.greeks(spot, strike, 0.05, 0.0, 0.2, time, "put")

                    assert call_greeks["vega"] > 0.0
                    assert put_greeks["vega"] > 0.0

    def test_call_delta_bounds(self):
        """Test that call delta is between 0 and 1"""
        bs = BlackScholesModel()

        spots = np.linspace(50, 150, 30)
        strike, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0

        for spot in spots:
            greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "call")
            assert 0.0 <= greeks["delta"] <= 1.0

    def test_put_delta_bounds(self):
        """Test that put delta is between -1 and 0"""
        bs = BlackScholesModel()

        spots = np.linspace(50, 150, 30)
        strike, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0

        for spot in spots:
            greeks = bs.greeks(spot, strike, rate, dividend, vol, time, "put")
            assert -1.0 <= greeks["delta"] <= 0.0


class TestMonotonicityProperties:
    """Test monotonicity properties of option prices"""

    def test_call_price_increases_with_spot(self):
        """Test that call price increases with spot price"""
        bs = BlackScholesModel()

        spots = np.linspace(80, 120, 20)
        strike, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0

        prices = [bs.price(spot, strike, rate, dividend, vol, time, "call") for spot in spots]

        # Prices should be monotonically increasing
        for i in range(len(prices) - 1):
            assert prices[i + 1] >= prices[i]

    def test_put_price_decreases_with_spot(self):
        """Test that put price decreases with spot price"""
        bs = BlackScholesModel()

        spots = np.linspace(80, 120, 20)
        strike, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0

        prices = [bs.price(spot, strike, rate, dividend, vol, time, "put") for spot in spots]

        # Prices should be monotonically decreasing
        for i in range(len(prices) - 1):
            assert prices[i + 1] <= prices[i]

    def test_call_price_decreases_with_strike(self):
        """Test that call price decreases with strike price"""
        bs = BlackScholesModel()

        spot, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0
        strikes = np.linspace(80, 120, 20)

        prices = [bs.price(spot, strike, rate, dividend, vol, time, "call") for strike in strikes]

        # Prices should be monotonically decreasing
        for i in range(len(prices) - 1):
            assert prices[i + 1] <= prices[i]

    def test_put_price_increases_with_strike(self):
        """Test that put price increases with strike price"""
        bs = BlackScholesModel()

        spot, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0
        strikes = np.linspace(80, 120, 20)

        prices = [bs.price(spot, strike, rate, dividend, vol, time, "put") for strike in strikes]

        # Prices should be monotonically increasing
        for i in range(len(prices) - 1):
            assert prices[i + 1] >= prices[i]

    def test_option_price_increases_with_volatility(self):
        """Test that option price increases with volatility"""
        bs = BlackScholesModel()

        spot, strike, rate, dividend, time = 100.0, 100.0, 0.05, 0.0, 1.0
        vols = np.linspace(0.1, 0.5, 20)

        call_prices = [bs.price(spot, strike, rate, dividend, vol, time, "call") for vol in vols]
        put_prices = [bs.price(spot, strike, rate, dividend, vol, time, "put") for vol in vols]

        # Both call and put prices should increase with volatility
        for i in range(len(vols) - 1):
            assert call_prices[i + 1] >= call_prices[i]
            assert put_prices[i + 1] >= put_prices[i]

    def test_option_price_increases_with_time(self):
        """Test that option price increases with time to maturity"""
        bs = BlackScholesModel()

        spot, strike, rate, dividend, vol = 100.0, 100.0, 0.05, 0.0, 0.2
        times = np.linspace(0.1, 3.0, 20)

        call_prices = [bs.price(spot, strike, rate, dividend, vol, time, "call") for time in times]
        put_prices = [bs.price(spot, strike, rate, dividend, vol, time, "put") for time in times]

        # Prices should generally increase with time (European options)
        for i in range(len(times) - 1):
            assert call_prices[i + 1] >= call_prices[i]
            # Put prices may not always increase due to interest rate effects


class TestArbitrageFreeConditions:
    """Test arbitrage-free conditions"""

    def test_call_price_above_intrinsic_value(self):
        """Test that call price >= max(S - K, 0)"""
        bs = BlackScholesModel()

        test_cases = [
            (110.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (120.0, 100.0, 0.05, 0.0, 0.3, 0.5),
            (105.0, 100.0, 0.03, 0.0, 0.15, 2.0),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            price = bs.price(spot, strike, rate, dividend, vol, time, "call")
            intrinsic = max(spot - strike, 0.0)

            assert price >= intrinsic - 1e-10

    def test_put_price_above_intrinsic_value(self):
        """Test that put price >= max(K - S, 0)"""
        bs = BlackScholesModel()

        test_cases = [
            (90.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (80.0, 100.0, 0.05, 0.0, 0.3, 0.5),
            (95.0, 100.0, 0.03, 0.0, 0.15, 2.0),
        ]

        for spot, strike, rate, dividend, vol, time in test_cases:
            price = bs.price(spot, strike, rate, dividend, vol, time, "put")
            # For European puts, lower bound is max(K*e^(-rT) - S, 0)
            import math

            discounted_strike = strike * math.exp(-rate * time)
            intrinsic = max(discounted_strike - spot, 0.0)

            assert price >= intrinsic - 1e-10

    def test_call_price_below_spot(self):
        """Test that call price <= S (no arbitrage upper bound)"""
        bs = BlackScholesModel()

        spots = np.linspace(80, 120, 20)
        strike, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0

        for spot in spots:
            price = bs.price(spot, strike, rate, dividend, vol, time, "call")
            assert price <= spot + 1e-10

    def test_put_price_below_strike(self):
        """Test that put price <= K*e^(-r*T) (no arbitrage upper bound)"""
        bs = BlackScholesModel()

        spot, strike, rate, dividend, vol, time = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0

        price = bs.price(spot, strike, rate, dividend, vol, time, "put")
        upper_bound = strike * np.exp(-rate * time)

        assert price <= upper_bound + 1e-10

    def test_american_option_value_bounds(self):
        """Test that American option >= European option"""
        tree = BinomialTreeModel()
        bs = BlackScholesModel()

        test_cases = [
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call"),
            (100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put"),
            (110.0, 100.0, 0.05, 0.02, 0.3, 0.5, "call"),
        ]

        for spot, strike, rate, dividend, vol, time, opt_type in test_cases:
            european_price = tree.price(
                spot, strike, rate, dividend, vol, time, 100, "european", opt_type, "crr"
            )
            american_price = tree.price(
                spot, strike, rate, dividend, vol, time, 100, "american", opt_type, "crr"
            )

            assert american_price >= european_price - 1e-10


class TestConvexityProperties:
    """Test convexity properties of option prices"""

    def test_call_price_convex_in_strike(self):
        """Test that call price is convex in strike"""
        bs = BlackScholesModel()

        spot, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0
        strikes = np.linspace(80, 120, 21)

        prices = [bs.price(spot, strike, rate, dividend, vol, time, "call") for strike in strikes]

        # Test convexity: f(x) + f(z) >= 2*f(y) where x < y < z and y = (x+z)/2
        for i in range(len(strikes) - 2):
            if abs(strikes[i + 1] - (strikes[i] + strikes[i + 2]) / 2) < 1e-6:
                assert prices[i] + prices[i + 2] >= 2 * prices[i + 1] - 1e-6

    def test_butterfly_spread_non_negative(self):
        """Test that butterfly spread has non-negative value"""
        bs = BlackScholesModel()

        spot, rate, dividend, vol, time = 100.0, 0.05, 0.0, 0.2, 1.0

        # Butterfly: long 1 call at K1, short 2 calls at K2, long 1 call at K3
        k1, k2, k3 = 90.0, 100.0, 110.0

        c1 = bs.price(spot, k1, rate, dividend, vol, time, "call")
        c2 = bs.price(spot, k2, rate, dividend, vol, time, "call")
        c3 = bs.price(spot, k3, rate, dividend, vol, time, "call")

        butterfly_value = c1 - 2 * c2 + c3

        # Butterfly should have non-negative value
        assert butterfly_value >= -1e-10


class TestImpliedVolatilityProperties:
    """Test properties of implied volatility"""

    def test_iv_recovers_market_price(self):
        """Test that pricing with IV recovers the market price"""
        bs = BlackScholesModel()

        test_cases = [
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call"),
            (110.0, 100.0, 0.05, 0.0, 0.25, 0.5, "call"),
            (90.0, 100.0, 0.03, 0.0, 0.3, 2.0, "put"),
        ]

        for spot, strike, rate, dividend, true_vol, time, opt_type in test_cases:
            # Calculate market price with known volatility
            market_price = bs.price(spot, strike, rate, dividend, true_vol, time, opt_type)

            # Calculate implied volatility
            iv = bs.implied_vol(market_price, spot, strike, rate, dividend, time, opt_type)

            # Price with IV should match market price
            recovered_price = bs.price(spot, strike, rate, dividend, iv, time, opt_type)

            assert abs(recovered_price - market_price) < 1e-6

    def test_iv_monotonic_in_price(self):
        """Test that implied volatility increases with option price"""
        bs = BlackScholesModel()

        spot, strike, rate, dividend, time = 100.0, 100.0, 0.05, 0.0, 1.0

        # Generate prices with increasing volatilities
        vols = np.linspace(0.15, 0.35, 10)
        prices = [bs.price(spot, strike, rate, dividend, vol, time, "call") for vol in vols]

        # Calculate IVs
        ivs = [
            bs.implied_vol(price, spot, strike, rate, dividend, time, "call") for price in prices
        ]

        # IVs should be monotonically increasing
        for i in range(len(ivs) - 1):
            assert ivs[i + 1] >= ivs[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
