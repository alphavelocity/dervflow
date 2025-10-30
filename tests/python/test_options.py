# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for options pricing module

Tests cover:
- Black-Scholes pricing against published values
- Put-call parity
- Greeks relationships
- Batch pricing correctness
"""

import numpy as np
import pytest

from dervflow import BinomialTreeModel, BlackScholesModel, MonteCarloOptionPricer


class TestBlackScholesPricing:
    """Test Black-Scholes option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.bs = BlackScholesModel()

    def test_atm_call_price(self):
        """Test at-the-money call option price against published values"""
        # Standard Black-Scholes test case
        price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        # Expected price approximately 10.45 from standard BS tables
        assert abs(price - 10.45) < 0.1

    def test_atm_put_price(self):
        """Test at-the-money put option price against published values"""
        price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        # Expected price approximately 5.57 from standard BS tables
        assert abs(price - 5.57) < 0.1

    def test_itm_call_price(self):
        """Test in-the-money call option"""
        price = self.bs.price(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        # Price should be greater than intrinsic value
        assert price > 10.0
        assert price < 20.0

    def test_otm_put_price(self):
        """Test out-of-the-money put option"""
        price = self.bs.price(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        # Price should be small but positive
        assert price > 0.0
        assert price < 5.0

    def test_call_with_dividend(self):
        """Test that dividend reduces call price"""
        price_with_div = self.bs.price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, "call")
        price_no_div = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert price_with_div < price_no_div

    def test_put_with_dividend(self):
        """Test that dividend increases put price"""
        price_with_div = self.bs.price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, "put")
        price_no_div = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert price_with_div > price_no_div

    def test_zero_time_call(self):
        """Test call option at expiry equals intrinsic value"""
        price = self.bs.price(110.0, 100.0, 0.05, 0.0, 0.2, 0.0, "call")
        assert abs(price - 10.0) < 1e-10

    def test_zero_time_put(self):
        """Test put option at expiry equals intrinsic value"""
        price = self.bs.price(90.0, 100.0, 0.05, 0.0, 0.2, 0.0, "put")
        assert abs(price - 10.0) < 1e-10

    def test_zero_volatility(self):
        """Test option with zero volatility"""
        price = self.bs.price(110.0, 100.0, 0.05, 0.0, 0.0, 1.0, "call")
        # Should be discounted forward intrinsic value
        forward = 110.0 * np.exp(0.05 * 1.0)
        intrinsic = max(forward - 100.0, 0.0)
        expected = intrinsic * np.exp(-0.05 * 1.0)
        assert abs(price - expected) < 1e-10

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)"""
        s, k, r, q, sigma, t = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0

        call_price = self.bs.price(s, k, r, q, sigma, t, "call")
        put_price = self.bs.price(s, k, r, q, sigma, t, "put")

        lhs = call_price - put_price
        rhs = s * np.exp(-q * t) - k * np.exp(-r * t)

        assert abs(lhs - rhs) < 1e-10

    def test_invalid_spot(self):
        """Test error handling for negative spot price"""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            self.bs.price(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

    def test_invalid_strike(self):
        """Test error handling for negative strike price"""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            self.bs.price(100.0, -100.0, 0.05, 0.0, 0.2, 1.0, "call")

    def test_invalid_volatility(self):
        """Test error handling for negative volatility"""
        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            self.bs.price(100.0, 100.0, 0.05, 0.0, -0.2, 1.0, "call")

    def test_invalid_time(self):
        """Test error handling for negative time"""
        with pytest.raises(ValueError, match="Time to maturity must be non-negative"):
            self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, -1.0, "call")

    def test_invalid_option_type(self):
        """Test error handling for invalid option type"""
        with pytest.raises(ValueError, match="Invalid option type"):
            self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "invalid")


class TestBlackScholesGreeks:
    """Test Black-Scholes Greeks calculations"""

    def setup_method(self):
        """Set up test fixtures"""
        self.bs = BlackScholesModel()

    def test_call_delta_range(self):
        """Test that call delta is between 0 and 1"""
        greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert 0.0 < greeks["delta"] < 1.0

    def test_put_delta_range(self):
        """Test that put delta is between -1 and 0"""
        greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert -1.0 < greeks["delta"] < 0.0

    def test_gamma_positive(self):
        """Test that gamma is always positive"""
        call_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert call_greeks["gamma"] > 0.0
        assert put_greeks["gamma"] > 0.0

    def test_vega_positive(self):
        """Test that vega is always positive"""
        call_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert call_greeks["vega"] > 0.0
        assert put_greeks["vega"] > 0.0

    def test_call_rho_positive(self):
        """Test that call rho is positive"""
        greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert greeks["rho"] > 0.0

    def test_put_rho_negative(self):
        """Test that put rho is negative"""
        greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert greeks["rho"] < 0.0

    def test_gamma_symmetry(self):
        """Test that gamma is the same for call and put"""
        call_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 1e-10

    def test_vega_symmetry(self):
        """Test that vega is the same for call and put"""
        call_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        put_greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert abs(call_greeks["vega"] - put_greeks["vega"]) < 1e-10

    def test_delta_put_call_parity(self):
        """Test delta relationship: Delta_call - Delta_put = exp(-q*t)"""
        q, t = 0.02, 1.0
        call_greeks = self.bs.greeks(100.0, 100.0, 0.05, q, 0.2, t, "call")
        put_greeks = self.bs.greeks(100.0, 100.0, 0.05, q, 0.2, t, "put")

        expected_diff = np.exp(-q * t)
        actual_diff = call_greeks["delta"] - put_greeks["delta"]

        assert abs(actual_diff - expected_diff) < 1e-10

    def test_atm_call_delta(self):
        """Test that ATM call delta is around 0.5-0.6"""
        greeks = self.bs.greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert 0.5 < greeks["delta"] < 0.7

    def test_itm_call_delta(self):
        """Test that ITM call delta is higher"""
        greeks = self.bs.greeks(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert greeks["delta"] > 0.7

    def test_otm_put_delta(self):
        """Test that OTM put delta is small negative"""
        greeks = self.bs.greeks(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert -0.3 < greeks["delta"] < 0.0


class TestBatchPricing:
    """Test batch pricing functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.bs = BlackScholesModel()

    def test_batch_vs_single(self):
        """Test that batch pricing matches single pricing"""
        spots = np.array([95.0, 100.0, 105.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0, 100.0])
        rates = np.array([0.05, 0.05, 0.05, 0.05])
        dividends = np.array([0.0, 0.0, 0.0, 0.0])
        volatilities = np.array([0.2, 0.2, 0.2, 0.2])
        times = np.array([1.0, 1.0, 1.0, 1.0])
        option_types = ["call", "call", "call", "call"]

        batch_prices = self.bs.price_batch(
            spots, strikes, rates, dividends, volatilities, times, option_types
        )

        for i in range(len(spots)):
            single_price = self.bs.price(
                spots[i],
                strikes[i],
                rates[i],
                dividends[i],
                volatilities[i],
                times[i],
                option_types[i],
            )
            assert abs(batch_prices[i] - single_price) < 1e-10

    def test_batch_mixed_types(self):
        """Test batch pricing with mixed call and put options"""
        n = 10
        spots = np.full(n, 100.0)
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.full(n, 0.2)
        times = np.full(n, 1.0)
        option_types = ["call"] * 5 + ["put"] * 5

        prices = self.bs.price_batch(
            spots, strikes, rates, dividends, volatilities, times, option_types
        )

        assert len(prices) == n
        # Call prices should be higher than put prices for same parameters
        assert all(prices[:5] > prices[5:])

    def test_batch_large_size(self):
        """Test batch pricing with large number of options"""
        n = 1000
        spots = np.random.uniform(80, 120, n)
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.random.uniform(0.15, 0.25, n)
        times = np.random.uniform(0.5, 2.0, n)
        option_types = ["call"] * (n // 2) + ["put"] * (n // 2)

        prices = self.bs.price_batch(
            spots, strikes, rates, dividends, volatilities, times, option_types
        )

        assert len(prices) == n
        assert all(prices > 0.0)

    def test_batch_mismatched_lengths(self):
        """Test error handling for mismatched array lengths"""
        spots = np.array([100.0, 105.0])
        strikes = np.array([100.0])  # Wrong length
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "call"]

        with pytest.raises(ValueError, match="same length"):
            self.bs.price_batch(spots, strikes, rates, dividends, volatilities, times, option_types)

    def test_batch_invalid_option_type(self):
        """Test error handling for invalid option type in batch"""
        spots = np.array([100.0, 105.0])
        strikes = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        option_types = ["call", "invalid"]

        with pytest.raises(ValueError, match="Invalid option type"):
            self.bs.price_batch(spots, strikes, rates, dividends, volatilities, times, option_types)


class TestImpliedVolatility:
    """Test implied volatility calculations"""

    def setup_method(self):
        """Set up test fixtures"""
        self.bs = BlackScholesModel()

    def test_iv_atm_call(self):
        """Test implied volatility for at-the-money call option"""
        # Calculate market price with known volatility
        true_vol = 0.2
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        # Calculate implied volatility
        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 1.0, "call")

        # Should recover the true volatility
        assert abs(iv - true_vol) < 0.001

    def test_iv_atm_put(self):
        """Test implied volatility for at-the-money put option"""
        true_vol = 0.25
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.0, true_vol, 1.0, "put")

        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 1.0, "put")

        assert abs(iv - true_vol) < 0.001

    def test_iv_itm_call(self):
        """Test implied volatility for in-the-money call option"""
        true_vol = 0.3
        market_price = self.bs.price(110.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 110.0, 100.0, 0.05, 0.0, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_otm_call(self):
        """Test implied volatility for out-of-the-money call option"""
        true_vol = 0.35
        market_price = self.bs.price(90.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 90.0, 100.0, 0.05, 0.0, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_deep_itm(self):
        """Test implied volatility for deep in-the-money option"""
        true_vol = 0.4
        market_price = self.bs.price(150.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 150.0, 100.0, 0.05, 0.0, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_deep_otm(self):
        """Test implied volatility for deep out-of-the-money option"""
        # Use less extreme parameters for better numerical stability
        true_vol = 0.4
        market_price = self.bs.price(70.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 70.0, 100.0, 0.05, 0.0, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_high_volatility(self):
        """Test implied volatility for high volatility scenario"""
        true_vol = 0.8
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_low_volatility(self):
        """Test implied volatility for low volatility scenario"""
        true_vol = 0.05
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.0, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_with_dividend(self):
        """Test implied volatility with dividend yield"""
        true_vol = 0.22
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.02, true_vol, 1.0, "call")

        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.02, 1.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_short_maturity(self):
        """Test implied volatility for short maturity option"""
        true_vol = 0.25
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.0, true_vol, 0.1, "call")

        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 0.1, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_long_maturity(self):
        """Test implied volatility for long maturity option"""
        true_vol = 0.28
        market_price = self.bs.price(100.0, 100.0, 0.05, 0.0, true_vol, 5.0, "call")

        iv = self.bs.implied_vol(market_price, 100.0, 100.0, 0.05, 0.0, 5.0, "call")

        assert abs(iv - true_vol) < 0.001

    def test_iv_various_strikes(self):
        """Test implied volatility across various strike prices"""
        true_vol = 0.25
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

        for strike in strikes:
            market_price = self.bs.price(100.0, strike, 0.05, 0.0, true_vol, 1.0, "call")
            iv = self.bs.implied_vol(market_price, 100.0, strike, 0.05, 0.0, 1.0, "call")
            assert abs(iv - true_vol) < 0.001

    def test_iv_invalid_negative_price(self):
        """Test error handling for negative market price"""
        with pytest.raises(ValueError, match="Market price must be positive"):
            self.bs.implied_vol(-10.0, 100.0, 100.0, 0.05, 0.0, 1.0, "call")

    def test_iv_invalid_zero_price(self):
        """Test error handling for zero market price"""
        with pytest.raises(ValueError, match="Market price must be positive"):
            self.bs.implied_vol(0.0, 100.0, 100.0, 0.05, 0.0, 1.0, "call")

    def test_iv_below_intrinsic_value(self):
        """Test error handling for price below intrinsic value"""
        # ITM call with intrinsic value of 10.0
        with pytest.raises(ValueError, match="(below intrinsic value|not bracketed)"):
            self.bs.implied_vol(5.0, 110.0, 100.0, 0.05, 0.0, 1.0, "call")

    def test_iv_at_expiry(self):
        """Test error handling for option at expiry"""
        with pytest.raises(ValueError, match="at expiry"):
            self.bs.implied_vol(10.0, 100.0, 100.0, 0.05, 0.0, 0.0, "call")

    def test_iv_batch_vs_single(self):
        """Test that batch IV calculation matches single IV calculation"""
        true_vols = [0.15, 0.20, 0.25, 0.30]
        spots = np.array([95.0, 100.0, 105.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0, 100.0])
        rates = np.array([0.05, 0.05, 0.05, 0.05])
        dividends = np.array([0.0, 0.0, 0.0, 0.0])
        times = np.array([1.0, 1.0, 1.0, 1.0])
        option_types = ["call", "call", "call", "call"]

        # Calculate market prices
        market_prices = np.array(
            [
                self.bs.price(
                    spots[i],
                    strikes[i],
                    rates[i],
                    dividends[i],
                    true_vols[i],
                    times[i],
                    option_types[i],
                )
                for i in range(len(spots))
            ]
        )

        # Batch IV calculation
        batch_ivs = self.bs.implied_vol_batch(
            market_prices, spots, strikes, rates, dividends, times, option_types
        )

        # Compare with single calculations
        for i in range(len(spots)):
            single_iv = self.bs.implied_vol(
                market_prices[i],
                spots[i],
                strikes[i],
                rates[i],
                dividends[i],
                times[i],
                option_types[i],
            )
            assert abs(batch_ivs[i] - single_iv) < 1e-10

    def test_iv_batch_mixed_types(self):
        """Test batch IV calculation with mixed call and put options"""
        true_vol = 0.25
        n = 6
        spots = np.full(n, 100.0)
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        times = np.full(n, 1.0)
        option_types = ["call"] * 3 + ["put"] * 3

        # Calculate market prices
        market_prices = np.array(
            [
                self.bs.price(
                    spots[i],
                    strikes[i],
                    rates[i],
                    dividends[i],
                    true_vol,
                    times[i],
                    option_types[i],
                )
                for i in range(n)
            ]
        )

        # Calculate IVs
        ivs = self.bs.implied_vol_batch(
            market_prices, spots, strikes, rates, dividends, times, option_types
        )

        # All should recover the true volatility
        for iv in ivs:
            assert abs(iv - true_vol) < 0.001

    def test_iv_batch_large_size(self):
        """Test batch IV calculation with large number of options"""
        n = 50  # Reduced size for more stable test
        np.random.seed(42)

        # Use more reasonable ranges for better numerical stability
        true_vols = np.random.uniform(0.18, 0.32, n)
        spots = np.random.uniform(90, 110, n)  # Closer to ATM
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        times = np.random.uniform(0.5, 2.0, n)
        option_types = ["call"] * (n // 2) + ["put"] * (n // 2)

        # Calculate market prices
        market_prices = np.array(
            [
                self.bs.price(
                    spots[i],
                    strikes[i],
                    rates[i],
                    dividends[i],
                    true_vols[i],
                    times[i],
                    option_types[i],
                )
                for i in range(n)
            ]
        )

        # Calculate IVs
        ivs = self.bs.implied_vol_batch(
            market_prices, spots, strikes, rates, dividends, times, option_types
        )

        # Check that IVs match true volatilities
        for i in range(n):
            assert abs(ivs[i] - true_vols[i]) < 0.001

    def test_iv_batch_mismatched_lengths(self):
        """Test error handling for mismatched array lengths in batch IV"""
        market_prices = np.array([10.0, 12.0])
        spots = np.array([100.0])  # Wrong length
        strikes = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        times = np.array([1.0, 1.0])
        option_types = ["call", "call"]

        with pytest.raises(ValueError, match="same length"):
            self.bs.implied_vol_batch(
                market_prices, spots, strikes, rates, dividends, times, option_types
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestBinomialTreePricing:
    """Test binomial tree option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tree = BinomialTreeModel()
        self.bs = BlackScholesModel()

    def test_european_call_crr(self):
        """Test European call pricing with CRR tree"""
        price = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr")
        assert price > 0.0
        # Should be close to Black-Scholes
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert abs(price - bs_price) < 0.5

    def test_european_put_crr(self):
        """Test European put pricing with CRR tree"""
        price = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "crr")
        assert price > 0.0
        # Should be close to Black-Scholes
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert abs(price - bs_price) < 0.5

    def test_european_call_jr(self):
        """Test European call pricing with JR tree"""
        price = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "jr")
        assert price > 0.0
        # Should be close to Black-Scholes
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert abs(price - bs_price) < 0.5

    def test_european_put_jr(self):
        """Test European put pricing with JR tree"""
        price = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "jr")
        assert price > 0.0
        # Should be close to Black-Scholes
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert abs(price - bs_price) < 0.5

    def test_convergence_to_black_scholes_50_steps(self):
        """Test convergence to Black-Scholes with 50 steps"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        tree_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 50, "european", "call", "crr"
        )
        # With 50 steps, should be within 1.0 of BS price
        assert abs(tree_price - bs_price) < 1.0

    def test_convergence_to_black_scholes_100_steps(self):
        """Test convergence to Black-Scholes with 100 steps"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        tree_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr"
        )
        # With 100 steps, should be within 0.5 of BS price
        assert abs(tree_price - bs_price) < 0.5

    def test_convergence_to_black_scholes_200_steps(self):
        """Test convergence to Black-Scholes with 200 steps"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        tree_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 200, "european", "call", "crr"
        )
        # With 200 steps, should be within 0.3 of BS price
        assert abs(tree_price - bs_price) < 0.3

    def test_convergence_to_black_scholes_500_steps(self):
        """Test convergence to Black-Scholes with 500 steps"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        tree_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, "european", "call", "crr"
        )
        # With 500 steps, should be very close to BS price
        assert abs(tree_price - bs_price) < 0.2

    def test_american_call_no_dividend(self):
        """Test that American call equals European call with no dividend"""
        # Without dividends, American call should not be exercised early
        european_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr"
        )
        american_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "american", "call", "crr"
        )
        # Should be very close (within numerical precision)
        assert abs(american_price - european_price) < 0.01

    def test_american_put_early_exercise(self):
        """Test American put early exercise premium"""
        # Deep ITM put should have early exercise value
        european_price = self.tree.price(
            100.0, 120.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "crr"
        )
        american_price = self.tree.price(
            100.0, 120.0, 0.05, 0.0, 0.2, 1.0, 100, "american", "put", "crr"
        )
        # American should be worth more due to early exercise
        assert american_price > european_price
        # Early exercise premium should be meaningful
        assert american_price - european_price > 0.1

    def test_american_put_greater_than_european(self):
        """Test that American put is always worth at least as much as European"""
        # Test various strikes
        for strike in [90.0, 100.0, 110.0, 120.0]:
            european_price = self.tree.price(
                100.0, strike, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "crr"
            )
            american_price = self.tree.price(
                100.0, strike, 0.05, 0.0, 0.2, 1.0, 100, "american", "put", "crr"
            )
            assert american_price >= european_price - 1e-10  # Allow for small numerical errors

    def test_american_call_with_dividend(self):
        """Test American call with dividend may have early exercise value"""
        # With high dividend, American call may be exercised early
        european_price = self.tree.price(
            110.0, 100.0, 0.05, 0.05, 0.2, 1.0, 100, "european", "call", "crr"
        )
        american_price = self.tree.price(
            110.0, 100.0, 0.05, 0.05, 0.2, 1.0, 100, "american", "call", "crr"
        )
        # American should be worth at least as much as European
        assert american_price >= european_price - 1e-10

    def test_crr_vs_jr_european_call(self):
        """Test CRR vs JR tree types for European call"""
        crr_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr"
        )
        jr_price = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "jr")
        # Both should converge to similar values
        assert abs(crr_price - jr_price) < 0.5

    def test_crr_vs_jr_european_put(self):
        """Test CRR vs JR tree types for European put"""
        crr_price = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "crr"
        )
        jr_price = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "jr")
        # Both should converge to similar values
        assert abs(crr_price - jr_price) < 0.5

    def test_crr_vs_jr_american_put(self):
        """Test CRR vs JR tree types for American put"""
        crr_price = self.tree.price(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 100, "american", "put", "crr"
        )
        jr_price = self.tree.price(100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 100, "american", "put", "jr")
        # Both should produce similar results
        assert abs(crr_price - jr_price) < 0.5

    def test_itm_call_price(self):
        """Test in-the-money call option"""
        price = self.tree.price(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr")
        # Price should be greater than intrinsic value
        intrinsic = 10.0
        assert price > intrinsic

    def test_otm_put_price(self):
        """Test out-of-the-money put option"""
        price = self.tree.price(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "crr")
        # Price should be small but positive
        assert price > 0.0
        assert price < 5.0

    def test_near_expiry_call(self):
        """Test call option near expiry approaches intrinsic value"""
        # Use very small time instead of zero to avoid numerical issues
        price = self.tree.price(110.0, 100.0, 0.05, 0.0, 0.2, 0.001, 10, "european", "call", "crr")
        # Near expiry, price should be close to intrinsic value
        intrinsic = 10.0
        assert abs(price - intrinsic) < 0.1

    def test_near_expiry_put(self):
        """Test put option near expiry approaches intrinsic value"""
        # Use very small time instead of zero to avoid numerical issues
        price = self.tree.price(90.0, 100.0, 0.05, 0.0, 0.2, 0.001, 10, "european", "put", "crr")
        # Near expiry, price should be close to intrinsic value
        intrinsic = 10.0
        assert abs(price - intrinsic) < 0.1

    def test_call_with_dividend(self):
        """Test that dividend reduces call price"""
        price_with_div = self.tree.price(
            100.0, 100.0, 0.05, 0.02, 0.2, 1.0, 100, "european", "call", "crr"
        )
        price_no_div = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr"
        )
        assert price_with_div < price_no_div

    def test_put_with_dividend(self):
        """Test that dividend increases put price"""
        price_with_div = self.tree.price(
            100.0, 100.0, 0.05, 0.02, 0.2, 1.0, 100, "european", "put", "crr"
        )
        price_no_div = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "put", "crr"
        )
        assert price_with_div > price_no_div

    def test_default_tree_type(self):
        """Test default tree type parameter"""
        # Without specifying tree_type (should default to 'crr')
        price_default = self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call")
        price_crr = self.tree.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr"
        )
        assert abs(price_default - price_crr) < 1e-10

    def test_invalid_steps_zero(self):
        """Test error handling for zero steps"""
        with pytest.raises(ValueError, match="Number of steps must be greater than zero"):
            self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0, "european", "call", "crr")

    def test_invalid_option_type(self):
        """Test error handling for invalid option type"""
        with pytest.raises(ValueError, match="Invalid option type"):
            self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "invalid", "crr")

    def test_invalid_exercise_style(self):
        """Test error handling for invalid exercise style"""
        with pytest.raises(ValueError, match="Invalid exercise style"):
            self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "invalid", "call", "crr")

    def test_invalid_tree_type(self):
        """Test error handling for invalid tree type"""
        with pytest.raises(ValueError, match="Invalid tree type"):
            self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "invalid")

    def test_invalid_spot(self):
        """Test error handling for negative spot price"""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            self.tree.price(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr")

    def test_invalid_strike(self):
        """Test error handling for negative strike price"""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            self.tree.price(100.0, -100.0, 0.05, 0.0, 0.2, 1.0, 100, "european", "call", "crr")

    def test_invalid_volatility(self):
        """Test error handling for negative volatility"""
        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            self.tree.price(100.0, 100.0, 0.05, 0.0, -0.2, 1.0, 100, "european", "call", "crr")

    def test_invalid_time(self):
        """Test error handling for negative time"""
        with pytest.raises(ValueError, match="Time to maturity must be non-negative"):
            self.tree.price(100.0, 100.0, 0.05, 0.0, 0.2, -1.0, 100, "european", "call", "crr")


class TestBinomialTreeBatchPricing:
    """Test binomial tree batch pricing functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tree = BinomialTreeModel()

    def test_batch_vs_single_european(self):
        """Test that batch pricing matches single pricing for European options"""
        spots = np.array([95.0, 100.0, 105.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0, 100.0])
        rates = np.array([0.05, 0.05, 0.05, 0.05])
        dividends = np.array([0.0, 0.0, 0.0, 0.0])
        volatilities = np.array([0.2, 0.2, 0.2, 0.2])
        times = np.array([1.0, 1.0, 1.0, 1.0])
        styles = ["european", "european", "european", "european"]
        option_types = ["call", "call", "call", "call"]

        batch_prices = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types, "crr"
        )

        for i in range(len(spots)):
            single_price = self.tree.price(
                spots[i],
                strikes[i],
                rates[i],
                dividends[i],
                volatilities[i],
                times[i],
                100,
                styles[i],
                option_types[i],
                "crr",
            )
            assert abs(batch_prices[i] - single_price) < 1e-10

    def test_batch_vs_single_american(self):
        """Test that batch pricing matches single pricing for American options"""
        spots = np.array([95.0, 100.0, 105.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0, 100.0])
        rates = np.array([0.05, 0.05, 0.05, 0.05])
        dividends = np.array([0.0, 0.0, 0.0, 0.0])
        volatilities = np.array([0.2, 0.2, 0.2, 0.2])
        times = np.array([1.0, 1.0, 1.0, 1.0])
        styles = ["american", "american", "american", "american"]
        option_types = ["put", "put", "put", "put"]

        batch_prices = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types, "crr"
        )

        for i in range(len(spots)):
            single_price = self.tree.price(
                spots[i],
                strikes[i],
                rates[i],
                dividends[i],
                volatilities[i],
                times[i],
                100,
                styles[i],
                option_types[i],
                "crr",
            )
            assert abs(batch_prices[i] - single_price) < 1e-10

    def test_batch_mixed_types(self):
        """Test batch pricing with mixed call and put options"""
        n = 10
        spots = np.full(n, 100.0)
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.full(n, 0.2)
        times = np.full(n, 1.0)
        styles = ["european"] * n
        option_types = ["call"] * 5 + ["put"] * 5

        prices = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types, "crr"
        )

        assert len(prices) == n
        # Call prices should be higher than put prices for same parameters
        assert all(prices[:5] > prices[5:])

    def test_batch_mixed_styles(self):
        """Test batch pricing with mixed European and American options"""
        n = 6
        spots = np.full(n, 100.0)
        strikes = np.full(n, 110.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.full(n, 0.2)
        times = np.full(n, 1.0)
        styles = ["european"] * 3 + ["american"] * 3
        option_types = ["put"] * n

        prices = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types, "crr"
        )

        assert len(prices) == n
        # American puts should be worth at least as much as European puts
        for i in range(3):
            assert prices[i + 3] >= prices[i] - 1e-10

    def test_batch_jr_tree(self):
        """Test batch pricing with JR tree type"""
        spots = np.array([95.0, 100.0, 105.0])
        strikes = np.array([100.0, 100.0, 100.0])
        rates = np.array([0.05, 0.05, 0.05])
        dividends = np.array([0.0, 0.0, 0.0])
        volatilities = np.array([0.2, 0.2, 0.2])
        times = np.array([1.0, 1.0, 1.0])
        styles = ["european", "european", "european"]
        option_types = ["call", "call", "call"]

        prices = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types, "jr"
        )

        assert len(prices) == 3
        assert all(prices > 0.0)

    def test_batch_large_size(self):
        """Test batch pricing with large number of options"""
        n = 100
        spots = np.random.uniform(80, 120, n)
        strikes = np.full(n, 100.0)
        rates = np.full(n, 0.05)
        dividends = np.full(n, 0.0)
        volatilities = np.random.uniform(0.15, 0.25, n)
        times = np.random.uniform(0.5, 2.0, n)
        styles = ["european"] * (n // 2) + ["american"] * (n // 2)
        option_types = ["call"] * (n // 2) + ["put"] * (n // 2)

        prices = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 50, styles, option_types, "crr"
        )

        assert len(prices) == n
        assert all(prices > 0.0)

    def test_batch_mismatched_lengths(self):
        """Test error handling for mismatched array lengths"""
        spots = np.array([100.0, 105.0])
        strikes = np.array([100.0])  # Wrong length
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        styles = ["european", "european"]
        option_types = ["call", "call"]

        with pytest.raises(ValueError, match="same length"):
            self.tree.price_batch(
                spots,
                strikes,
                rates,
                dividends,
                volatilities,
                times,
                100,
                styles,
                option_types,
                "crr",
            )

    def test_batch_invalid_option_type(self):
        """Test error handling for invalid option type in batch"""
        spots = np.array([100.0, 105.0])
        strikes = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        styles = ["european", "european"]
        option_types = ["call", "invalid"]

        with pytest.raises(ValueError, match="Invalid option type"):
            self.tree.price_batch(
                spots,
                strikes,
                rates,
                dividends,
                volatilities,
                times,
                100,
                styles,
                option_types,
                "crr",
            )

    def test_batch_invalid_exercise_style(self):
        """Test error handling for invalid exercise style in batch"""
        spots = np.array([100.0, 105.0])
        strikes = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        styles = ["european", "invalid"]
        option_types = ["call", "call"]

        with pytest.raises(ValueError, match="Invalid exercise style"):
            self.tree.price_batch(
                spots,
                strikes,
                rates,
                dividends,
                volatilities,
                times,
                100,
                styles,
                option_types,
                "crr",
            )

    def test_batch_default_tree_type(self):
        """Test batch pricing with default tree type"""
        spots = np.array([100.0, 105.0])
        strikes = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        dividends = np.array([0.0, 0.0])
        volatilities = np.array([0.2, 0.2])
        times = np.array([1.0, 1.0])
        styles = ["european", "european"]
        option_types = ["call", "call"]

        # Without specifying tree_type (should default to 'crr')
        prices_default = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types
        )
        prices_crr = self.tree.price_batch(
            spots, strikes, rates, dividends, volatilities, times, 100, styles, option_types, "crr"
        )

        assert np.allclose(prices_default, prices_crr)


class TestMonteCarloEuropeanPricing:
    """Test Monte Carlo European option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mc = MonteCarloOptionPricer()
        self.bs = BlackScholesModel()

    def test_european_call_convergence(self):
        """Test that MC price converges to Black-Scholes for European call"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        result = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=50000, use_antithetic=True, seed=42
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        # MC price should be within 3 standard errors of BS price (99.7% confidence)
        assert abs(mc_price - bs_price) < 3 * std_error

    def test_european_put_convergence(self):
        """Test that MC price converges to Black-Scholes for European put"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        result = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=50000, use_antithetic=True, seed=42
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        assert abs(mc_price - bs_price) < 3 * std_error

    def test_itm_call_convergence(self):
        """Test MC pricing for in-the-money call"""
        bs_price = self.bs.price(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        result = self.mc.price_european(
            110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=50000, use_antithetic=True, seed=42
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        assert abs(mc_price - bs_price) < 3 * std_error

    def test_otm_put_convergence(self):
        """Test MC pricing for out-of-the-money put"""
        bs_price = self.bs.price(110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        result = self.mc.price_european(
            110.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=50000, use_antithetic=True, seed=42
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        assert abs(mc_price - bs_price) < 3 * std_error

    def test_antithetic_variance_reduction(self):
        """Test that antithetic variates reduce standard error"""
        # Without antithetic variates
        result_no_av = self.mc.price_european(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_paths=10000,
            use_antithetic=False,
            seed=42,
        )
        # With antithetic variates
        result_with_av = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, use_antithetic=True, seed=42
        )

        # Antithetic variates should reduce standard error
        assert result_with_av["std_error"] < result_no_av["std_error"]

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed"""
        result1 = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=42
        )
        result2 = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=42
        )

        assert abs(result1["price"] - result2["price"]) < 1e-10
        assert abs(result1["std_error"] - result2["std_error"]) < 1e-10

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        result1 = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=42
        )
        result2 = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=123
        )

        # Prices should be different but close (both converging to same value)
        assert abs(result1["price"] - result2["price"]) > 0.0
        assert abs(result1["price"] - result2["price"]) < 1.0

    def test_parallel_vs_sequential(self):
        """Test that parallel and sequential execution produce similar results"""
        result_parallel = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=42, parallel=True
        )
        result_sequential = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=42, parallel=False
        )

        # Results should converge to similar values (within combined standard errors)
        combined_error = result_parallel["std_error"] + result_sequential["std_error"]
        assert abs(result_parallel["price"] - result_sequential["price"]) < 3 * combined_error

    def test_zero_time_equals_intrinsic(self):
        """Test that option at expiry equals intrinsic value"""
        result = self.mc.price_european(
            110.0, 100.0, 0.05, 0.0, 0.2, 0.0, "call", num_paths=10000, seed=42
        )
        intrinsic = 10.0
        assert abs(result["price"] - intrinsic) < 1e-10
        assert result["std_error"] == 0.0

    def test_with_dividend(self):
        """Test MC pricing with dividend yield"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, "call")
        result = self.mc.price_european(
            100.0,
            100.0,
            0.05,
            0.02,
            0.2,
            1.0,
            "call",
            num_paths=50000,
            use_antithetic=True,
            seed=42,
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        assert abs(mc_price - bs_price) < 3 * std_error

    def test_high_volatility(self):
        """Test MC pricing with high volatility"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.5, 1.0, "call")
        result = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.5, 1.0, "call", num_paths=50000, use_antithetic=True, seed=42
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        assert abs(mc_price - bs_price) < 3 * std_error

    def test_low_volatility(self):
        """Test MC pricing with low volatility"""
        bs_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.05, 1.0, "call")
        result = self.mc.price_european(
            100.0,
            100.0,
            0.05,
            0.0,
            0.05,
            1.0,
            "call",
            num_paths=50000,
            use_antithetic=True,
            seed=42,
        )
        mc_price = result["price"]
        std_error = result["std_error"]

        assert abs(mc_price - bs_price) < 3 * std_error

    def test_invalid_num_paths(self):
        """Test error handling for zero paths"""
        with pytest.raises(ValueError, match="Number of paths must be positive"):
            self.mc.price_european(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=0)

    def test_invalid_spot(self):
        """Test error handling for negative spot price"""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            self.mc.price_european(-100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

    def test_invalid_option_type(self):
        """Test error handling for invalid option type"""
        with pytest.raises(ValueError, match="Invalid option type"):
            self.mc.price_european(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "invalid")


class TestMonteCarloAmericanPricing:
    """Test Monte Carlo American option pricing using Longstaff-Schwartz"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mc = MonteCarloOptionPricer()
        self.bs = BlackScholesModel()
        self.tree = BinomialTreeModel()

    def test_american_put_early_exercise_premium(self):
        """Test that American put has early exercise premium"""
        # European put price
        euro_result = self.mc.price_european(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, seed=42
        )
        # American put price
        amer_result = self.mc.price_american(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=50, seed=42
        )

        # American should be worth more due to early exercise
        assert amer_result["price"] > euro_result["price"]

    def test_american_call_no_dividend_equals_european(self):
        """Test that American call equals European call with no dividend"""
        # Without dividends, American call should not be exercised early
        euro_result = self.mc.price_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, seed=42
        )
        amer_result = self.mc.price_american(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_paths=10000, num_steps=50, seed=42
        )

        # Should be very close (within standard errors)
        combined_error = euro_result["std_error"] + amer_result["std_error"]
        assert abs(amer_result["price"] - euro_result["price"]) < 3 * combined_error

    def test_american_put_vs_tree(self):
        """Test American put MC price against binomial tree"""
        tree_price = self.tree.price(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 100, "american", "put", "crr"
        )
        mc_result = self.mc.price_american(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=20000, num_steps=50, seed=42
        )

        # MC and tree should produce similar results
        assert abs(mc_result["price"] - tree_price) < 3 * mc_result["std_error"]

    def test_deep_itm_put_early_exercise(self):
        """Test deep in-the-money put has significant early exercise value"""
        euro_result = self.mc.price_european(
            80.0, 120.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, seed=42
        )
        amer_result = self.mc.price_american(
            80.0, 120.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=50, seed=42
        )

        # Early exercise premium should be meaningful
        assert amer_result["price"] - euro_result["price"] > 1.0

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed"""
        result1 = self.mc.price_american(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=50, seed=42
        )
        result2 = self.mc.price_american(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=50, seed=42
        )

        assert abs(result1["price"] - result2["price"]) < 1e-10
        assert abs(result1["std_error"] - result2["std_error"]) < 1e-10

    def test_parallel_vs_sequential(self):
        """Test that parallel and sequential execution produce similar results"""
        result_parallel = self.mc.price_american(
            100.0,
            110.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            num_paths=5000,
            num_steps=50,
            seed=42,
            parallel=True,
        )
        result_sequential = self.mc.price_american(
            100.0,
            110.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            num_paths=5000,
            num_steps=50,
            seed=42,
            parallel=False,
        )

        # Results should converge to similar values (within combined standard errors)
        combined_error = result_parallel["std_error"] + result_sequential["std_error"]
        assert abs(result_parallel["price"] - result_sequential["price"]) < 3 * combined_error

    def test_more_steps_better_accuracy(self):
        """Test that more time steps improve accuracy"""
        tree_price = self.tree.price(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, 100, "american", "put", "crr"
        )

        # Fewer steps
        result_few = self.mc.price_american(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=20, seed=42
        )
        # More steps
        result_many = self.mc.price_american(
            100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=100, seed=42
        )

        # More steps should be closer to tree price
        error_few = abs(result_few["price"] - tree_price)
        error_many = abs(result_many["price"] - tree_price)
        assert error_many <= error_few + 0.5  # Allow some tolerance

    def test_zero_time_equals_intrinsic(self):
        """Test that option at expiry equals intrinsic value"""
        result = self.mc.price_american(
            90.0, 100.0, 0.05, 0.0, 0.2, 0.0, "put", num_paths=10000, num_steps=50, seed=42
        )
        intrinsic = 10.0
        assert abs(result["price"] - intrinsic) < 1e-10
        assert result["std_error"] == 0.0

    def test_with_dividend(self):
        """Test American option pricing with dividend"""
        result = self.mc.price_american(
            100.0, 110.0, 0.05, 0.02, 0.2, 1.0, "put", num_paths=10000, num_steps=50, seed=42
        )
        # Should produce a reasonable price
        assert result["price"] > 0.0
        assert result["price"] < 20.0

    def test_invalid_num_paths(self):
        """Test error handling for zero paths"""
        with pytest.raises(ValueError, match="Number of paths must be positive"):
            self.mc.price_american(
                100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=0, num_steps=50
            )

    def test_invalid_num_steps(self):
        """Test error handling for zero steps"""
        with pytest.raises(ValueError, match="Number of steps must be positive"):
            self.mc.price_american(
                100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=0
            )

    def test_invalid_spot(self):
        """Test error handling for negative spot price"""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            self.mc.price_american(
                -100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "put", num_paths=10000, num_steps=50
            )

    def test_invalid_option_type(self):
        """Test error handling for invalid option type"""
        with pytest.raises(ValueError, match="Invalid option type"):
            self.mc.price_american(
                100.0, 110.0, 0.05, 0.0, 0.2, 1.0, "invalid", num_paths=10000, num_steps=50
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Exotic Options Tests
# ============================================================================


class TestAsianOptions:
    """Test Asian option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.asian = pytest.importorskip("dervflow").AsianOption()
        self.bs = BlackScholesModel()

    def test_arithmetic_asian_call_fixed_strike(self):
        """Test arithmetic average Asian call with fixed strike"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        # Asian option should be cheaper than vanilla due to averaging
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert 0.0 < price < vanilla_price

    def test_arithmetic_asian_put_fixed_strike(self):
        """Test arithmetic average Asian put with fixed strike"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        # Asian option should be cheaper than vanilla due to averaging
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert 0.0 < price < vanilla_price

    def test_arithmetic_asian_call_floating_strike(self):
        """Test arithmetic average Asian call with floating strike"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=False,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0

    def test_geometric_asian_call_fixed_strike(self):
        """Test geometric average Asian call with fixed strike"""
        price = self.asian.price_geometric(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_observations=12, fixed_strike=True
        )
        # Geometric average is always less than or equal to arithmetic average
        # So geometric Asian should be cheaper
        assert price > 0.0

    def test_geometric_asian_put_fixed_strike(self):
        """Test geometric average Asian put with fixed strike"""
        price = self.asian.price_geometric(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put", num_observations=12, fixed_strike=True
        )
        assert price > 0.0

    def test_geometric_cheaper_than_arithmetic(self):
        """Test that geometric Asian is cheaper than arithmetic Asian"""
        arith_price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=20000,
            seed=42,
        )
        geom_price = self.asian.price_geometric(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", num_observations=12, fixed_strike=True
        )
        # Geometric average is always <= arithmetic average
        assert geom_price <= arith_price

    def test_more_observations_cheaper(self):
        """Test that more observations reduce option value"""
        price_few = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=4,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        price_many = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=52,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        # More averaging reduces volatility, making option cheaper
        assert price_many < price_few

    def test_arithmetic_stats_returns_price_and_error(self):
        """Detailed API should expose price and Monte Carlo error"""
        result = self.asian.price_arithmetic_stats(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=20000,
            seed=123,
            use_antithetic=True,
        )

        assert set(result.keys()) == {"price", "std_error"}
        assert result["price"] > 0.0
        assert result["std_error"] >= 0.0

    def test_arithmetic_stats_control_variate_requires_fixed(self):
        """Control variate is not available for floating-strike contracts"""
        with pytest.raises(ValueError):
            self.asian.price_arithmetic_stats(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                num_observations=12,
                fixed_strike=False,
                num_paths=1000,
                seed=123,
                use_control_variate=True,
            )

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed"""
        price1 = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        price2 = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert abs(price1 - price2) < 1e-10

    def test_invalid_num_observations(self):
        """Test error handling for zero observations"""
        with pytest.raises(ValueError, match="Number of observations must be positive"):
            self.asian.price_arithmetic(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                num_observations=0,
                fixed_strike=True,
                num_paths=10000,
            )


class TestBarrierOptions:
    """Test barrier option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.barrier = pytest.importorskip("dervflow").BarrierOption()
        self.bs = BlackScholesModel()

    def test_down_and_out_call(self):
        """Test down-and-out call option"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        # Barrier option should be cheaper than vanilla
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert 0.0 < price < vanilla_price

    def test_up_and_out_call(self):
        """Test up-and-out call option"""
        # Use strike below barrier to ensure option has value
        price = self.barrier.price(
            100.0,
            95.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=120.0,
            barrier_type="up-and-out",
            rebate=0.0,
        )
        # Barrier option should be non-negative and cheaper than vanilla
        vanilla_price = self.bs.price(100.0, 95.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert price >= 0.0
        assert price <= vanilla_price

    def test_down_and_out_put(self):
        """Test down-and-out put option"""
        # Use strike above barrier to ensure option has value
        price = self.barrier.price(
            100.0,
            105.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        # Barrier option should be non-negative and cheaper than vanilla
        vanilla_price = self.bs.price(100.0, 105.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert price >= 0.0
        assert price <= vanilla_price

    def test_up_and_out_put(self):
        """Test up-and-out put option"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            barrier=110.0,
            barrier_type="up-and-out",
            rebate=0.0,
        )
        # Barrier option should be cheaper than vanilla
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert 0.0 < price < vanilla_price

    def test_down_and_in_call(self):
        """Test down-and-in call option"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-in",
            rebate=0.0,
        )
        assert price > 0.0

    def test_up_and_in_call(self):
        """Test up-and-in call option"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=120.0,
            barrier_type="up-and-in",
            rebate=0.0,
        )
        assert price > 0.0

    def test_knock_in_plus_knock_out_equals_vanilla(self):
        """Test that knock-in + knock-out = vanilla option"""
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")

        knock_out = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        knock_in = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-in",
            rebate=0.0,
        )

        # Knock-in + knock-out should equal vanilla
        assert abs((knock_in + knock_out) - vanilla_price) < 0.01

    def test_barrier_with_rebate(self):
        """Test barrier option with rebate payment"""
        price_no_rebate = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        price_with_rebate = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=5.0,
        )
        # Rebate adds value
        assert price_with_rebate > price_no_rebate

    def test_barrier_closer_to_spot_cheaper(self):
        """Test that barrier closer to spot makes option cheaper"""
        price_far = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=80.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        price_close = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=95.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        # Closer barrier means higher knock-out probability
        assert price_close < price_far

    def test_invalid_barrier_up_below_spot(self):
        """Test error handling for up barrier below spot"""
        with pytest.raises(ValueError, match="Up barrier must be above spot"):
            self.barrier.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                barrier=90.0,
                barrier_type="up-and-out",
                rebate=0.0,
            )

    def test_invalid_barrier_down_above_spot(self):
        """Test error handling for down barrier above spot"""
        with pytest.raises(ValueError, match="Down barrier must be below spot"):
            self.barrier.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                barrier=110.0,
                barrier_type="down-and-out",
                rebate=0.0,
            )

    def test_invalid_negative_rebate(self):
        """Test error handling for negative rebate"""
        with pytest.raises(ValueError, match="Rebate must be non-negative"):
            self.barrier.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                barrier=90.0,
                barrier_type="down-and-out",
                rebate=-5.0,
            )


class TestLookbackOptions:
    """Test lookback option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.lookback = pytest.importorskip("dervflow").LookbackOption()
        self.bs = BlackScholesModel()

    def test_fixed_strike_lookback_call(self):
        """Test fixed strike lookback call option"""
        price = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        # Lookback option should be more expensive than vanilla
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        assert price > vanilla_price

    def test_fixed_strike_lookback_put(self):
        """Test fixed strike lookback put option"""
        price = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put", lookback_type="fixed", current_extremum=None
        )
        # Lookback option should be more expensive than vanilla
        vanilla_price = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        assert price > vanilla_price

    def test_floating_strike_lookback_call(self):
        """Test floating strike lookback call option"""
        price = self.lookback.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            lookback_type="floating",
            current_extremum=95.0,
        )
        # Floating strike lookback should have positive value
        assert price > 0.0

    def test_floating_strike_lookback_put(self):
        """Test floating strike lookback put option"""
        price = self.lookback.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            lookback_type="floating",
            current_extremum=105.0,
        )
        # Floating strike lookback should have positive value
        assert price > 0.0

    def test_lookback_more_expensive_than_vanilla(self):
        """Test that lookback options are more expensive than vanilla"""
        vanilla_call = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
        lookback_call = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        assert lookback_call > vanilla_call

        vanilla_put = self.bs.price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put")
        lookback_put = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "put", lookback_type="fixed", current_extremum=None
        )
        assert lookback_put > vanilla_put

    def test_higher_volatility_increases_value(self):
        """Test that higher volatility increases lookback value"""
        price_low_vol = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.15, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        price_high_vol = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.30, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        assert price_high_vol > price_low_vol

    def test_floating_strike_with_extremum(self):
        """Test floating strike with current extremum"""
        # Test that floating strike lookback with extremum produces valid prices
        price = self.lookback.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            lookback_type="floating",
            current_extremum=95.0,
        )
        assert price > 0.0

        # Test put with extremum
        price_put = self.lookback.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            lookback_type="floating",
            current_extremum=105.0,
        )
        assert price_put > 0.0


class TestDigitalOptions:
    """Test digital/binary option pricing"""

    def setup_method(self):
        """Set up test fixtures"""
        self.digital = pytest.importorskip("dervflow").DigitalOption()
        self.bs = BlackScholesModel()

    def test_cash_or_nothing_call(self):
        """Test cash-or-nothing call option"""
        price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # Price should be between 0 and discounted payout
        discounted_payout = 10.0 * np.exp(-0.05 * 1.0)
        assert 0.0 < price < discounted_payout

    def test_cash_or_nothing_put(self):
        """Test cash-or-nothing put option"""
        price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # Price should be between 0 and discounted payout
        discounted_payout = 10.0 * np.exp(-0.05 * 1.0)
        assert 0.0 < price < discounted_payout

    def test_asset_or_nothing_call(self):
        """Test asset-or-nothing call option"""
        price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="asset-or-nothing",
            cash_payout=1.0,
        )
        # Price should be positive
        assert price > 0.0

    def test_asset_or_nothing_put(self):
        """Test asset-or-nothing put option"""
        price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            digital_type="asset-or-nothing",
            cash_payout=1.0,
        )
        # Price should be positive
        assert price > 0.0

    def test_digital_call_plus_put_equals_discounted_payout(self):
        """Test that digital call + put = discounted payout"""
        payout = 10.0
        call_price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=payout,
        )
        put_price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            digital_type="cash-or-nothing",
            cash_payout=payout,
        )

        discounted_payout = payout * np.exp(-0.05 * 1.0)
        # Call + put should equal discounted payout
        assert abs((call_price + put_price) - discounted_payout) < 0.01

    def test_itm_digital_call_higher_value(self):
        """Test that ITM digital call has higher value"""
        atm_price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        itm_price = self.digital.price(
            110.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # ITM digital should be worth more
        assert itm_price > atm_price

    def test_otm_digital_put_lower_value(self):
        """Test that OTM digital put has lower value"""
        atm_price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        otm_price = self.digital.price(
            110.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # OTM digital should be worth less
        assert otm_price < atm_price

    def test_zero_time_itm_equals_payout(self):
        """Test that ITM digital at expiry equals payout"""
        price = self.digital.price(
            110.0,
            100.0,
            0.05,
            0.0,
            0.2,
            0.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # At expiry, ITM digital pays full amount
        assert abs(price - 10.0) < 1e-10

    def test_zero_time_otm_equals_zero(self):
        """Test that OTM digital at expiry equals zero"""
        price = self.digital.price(
            90.0,
            100.0,
            0.05,
            0.0,
            0.2,
            0.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # At expiry, OTM digital pays nothing
        assert abs(price - 0.0) < 1e-10

    def test_higher_payout_increases_value(self):
        """Test that higher payout increases digital value"""
        price_low = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=5.0,
        )
        price_high = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=15.0,
        )
        # Higher payout should give higher value
        assert price_high > price_low
        # Ratio should be approximately 3:1
        assert abs((price_high / price_low) - 3.0) < 0.1

    def test_invalid_negative_payout(self):
        """Test error handling for negative payout"""
        with pytest.raises(ValueError, match="Cash payout must be non-negative"):
            self.digital.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                digital_type="cash-or-nothing",
                cash_payout=-10.0,
            )


# ============================================================================
# Comprehensive Edge Case Tests for Exotic Options
# ============================================================================


class TestAsianOptionsEdgeCases:
    """Comprehensive edge case tests for Asian options"""

    def setup_method(self):
        """Set up test fixtures"""
        from dervflow import AsianOption

        self.asian = AsianOption()

    def test_very_high_volatility(self):
        """Test Asian option with very high volatility"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            1.0,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0
        assert price < 200.0  # Should still be bounded

    def test_very_low_volatility(self):
        """Test Asian option with very low volatility"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.01,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0

    def test_deep_itm_call(self):
        """Test deep in-the-money Asian call"""
        price = self.asian.price_arithmetic(
            150.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 40.0  # Should have significant intrinsic value

    def test_deep_otm_put(self):
        """Test deep out-of-the-money Asian put"""
        price = self.asian.price_arithmetic(
            150.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "put",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price >= 0.0
        assert price < 5.0  # Should be very small

    def test_single_observation(self):
        """Test Asian option with single observation (should be like vanilla)"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=1,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0

    def test_many_observations(self):
        """Test Asian option with many observations"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            num_observations=252,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0

    def test_short_maturity(self):
        """Test Asian option with very short maturity"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            0.01,
            "call",
            num_observations=2,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price >= 0.0

    def test_long_maturity(self):
        """Test Asian option with long maturity"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            5.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0

    def test_high_dividend_yield(self):
        """Test Asian option with high dividend yield"""
        price = self.asian.price_arithmetic(
            100.0,
            100.0,
            0.05,
            0.10,
            0.2,
            1.0,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        assert price > 0.0

    def test_geometric_vs_arithmetic_consistency(self):
        """Test that geometric is always <= arithmetic for same parameters"""
        for strike in [90.0, 100.0, 110.0]:
            for vol in [0.1, 0.2, 0.3]:
                geom = self.asian.price_geometric(
                    100.0,
                    strike,
                    0.05,
                    0.0,
                    vol,
                    1.0,
                    "call",
                    num_observations=12,
                    fixed_strike=True,
                )
                arith = self.asian.price_arithmetic(
                    100.0,
                    strike,
                    0.05,
                    0.0,
                    vol,
                    1.0,
                    "call",
                    num_observations=12,
                    fixed_strike=True,
                    num_paths=20000,
                    seed=42,
                )
                assert geom <= arith + 0.5  # Allow small MC error


class TestBarrierOptionsEdgeCases:
    """Comprehensive edge case tests for barrier options"""

    def setup_method(self):
        """Set up test fixtures"""
        from dervflow import BarrierOption

        self.barrier = BarrierOption()

    def test_barrier_very_close_to_spot(self):
        """Test barrier very close to spot price"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=100.1,
            barrier_type="up-and-out",
            rebate=0.0,
        )
        assert price >= 0.0

    def test_barrier_very_far_from_spot(self):
        """Test barrier very far from spot price"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=200.0,
            barrier_type="up-and-out",
            rebate=0.0,
        )
        # When barrier is very far, option should be close to vanilla
        assert price >= 0.0

    def test_high_rebate(self):
        """Test barrier option with high rebate"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=50.0,
        )
        assert price > 0.0

    def test_all_barrier_types_positive(self):
        """Test that all barrier types produce non-negative prices"""
        barrier_types = ["up-and-out", "down-and-out", "up-and-in", "down-and-in"]
        for btype in barrier_types:
            if "up" in btype:
                barrier = 120.0
            else:
                barrier = 80.0

            price = self.barrier.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                barrier=barrier,
                barrier_type=btype,
                rebate=0.0,
            )
            assert price >= 0.0, f"Negative price for {btype}"

    def test_barrier_put_call_symmetry(self):
        """Test barrier options for both calls and puts"""
        for opt_type in ["call", "put"]:
            price = self.barrier.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                opt_type,
                barrier=90.0,
                barrier_type="down-and-out",
                rebate=0.0,
            )
            assert price >= 0.0

    def test_extreme_volatility_barrier(self):
        """Test barrier option with extreme volatility"""
        price_low = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.01,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        price_high = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.8,
            1.0,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        assert price_low >= 0.0
        assert price_high >= 0.0

    def test_short_maturity_barrier(self):
        """Test barrier option with very short maturity"""
        price = self.barrier.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            0.01,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        assert price >= 0.0


class TestLookbackOptionsEdgeCases:
    """Comprehensive edge case tests for lookback options"""

    def setup_method(self):
        """Set up test fixtures"""
        from dervflow import LookbackOption

        self.lookback = LookbackOption()

    def test_extreme_volatility_lookback(self):
        """Test lookback with extreme volatility"""
        price_low = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.05, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        price_high = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.8, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        assert price_low > 0.0
        assert price_high > price_low  # Higher vol should increase value

    def test_lookback_all_types(self):
        """Test all combinations of lookback types and option types"""
        for lookback_type in ["fixed", "floating"]:
            for option_type in ["call", "put"]:
                price = self.lookback.price(
                    100.0,
                    100.0,
                    0.05,
                    0.0,
                    0.2,
                    1.0,
                    option_type,
                    lookback_type=lookback_type,
                    current_extremum=None,
                )
                assert price > 0.0, f"Invalid price for {lookback_type} {option_type}"

    def test_floating_with_various_extrema(self):
        """Test floating strike with various current extrema"""
        extrema = [80.0, 90.0, 95.0, 100.0, 105.0]
        prices = []
        for ext in extrema:
            price = self.lookback.price(
                100.0,
                100.0,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                lookback_type="floating",
                current_extremum=ext,
            )
            prices.append(price)
            assert price > 0.0

        # Prices should vary with extremum
        assert len(set(prices)) > 1

    def test_deep_itm_lookback(self):
        """Test deep in-the-money lookback"""
        price = self.lookback.price(
            150.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call", lookback_type="fixed", current_extremum=None
        )
        assert price > 40.0  # Should have significant value

    def test_short_maturity_lookback(self):
        """Test lookback with very short maturity"""
        price = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 0.01, "call", lookback_type="fixed", current_extremum=None
        )
        assert price >= 0.0

    def test_long_maturity_lookback(self):
        """Test lookback with long maturity"""
        price = self.lookback.price(
            100.0, 100.0, 0.05, 0.0, 0.2, 5.0, "call", lookback_type="fixed", current_extremum=None
        )
        assert price > 0.0


class TestDigitalOptionsEdgeCases:
    """Comprehensive edge case tests for digital options"""

    def setup_method(self):
        """Set up test fixtures"""
        from dervflow import DigitalOption

        self.digital = DigitalOption()

    def test_extreme_payouts(self):
        """Test digital options with extreme payouts"""
        price_small = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=0.01,
        )
        price_large = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=1000.0,
        )
        assert price_small > 0.0
        assert price_large > price_small
        # Should scale linearly with payout
        assert abs((price_large / price_small) - 100000.0) < 1000.0

    def test_deep_itm_digital(self):
        """Test deep in-the-money digital"""
        price = self.digital.price(
            150.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        discounted_payout = 10.0 * np.exp(-0.05 * 1.0)
        # Should be close to discounted payout
        assert price > 0.8 * discounted_payout

    def test_deep_otm_digital(self):
        """Test deep out-of-the-money digital"""
        price = self.digital.price(
            50.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        # Should be very small
        assert price >= 0.0
        assert price < 1.0

    def test_extreme_volatility_digital(self):
        """Test digital with extreme volatility"""
        price_low = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.01,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        price_high = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.8,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        assert price_low > 0.0
        assert price_high > 0.0

    def test_both_digital_types(self):
        """Test both cash-or-nothing and asset-or-nothing"""
        cash_price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        asset_price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            "call",
            digital_type="asset-or-nothing",
            cash_payout=1.0,
        )
        assert cash_price > 0.0
        assert asset_price > 0.0

    def test_short_maturity_digital(self):
        """Test digital with very short maturity"""
        price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            0.01,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        assert price >= 0.0

    def test_long_maturity_digital(self):
        """Test digital with long maturity"""
        price = self.digital.price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            5.0,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        assert price > 0.0

    def test_various_strikes_digital(self):
        """Test digital across various strikes"""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices = []
        for strike in strikes:
            price = self.digital.price(
                100.0,
                strike,
                0.05,
                0.0,
                0.2,
                1.0,
                "call",
                digital_type="cash-or-nothing",
                cash_payout=10.0,
            )
            prices.append(price)
            assert price >= 0.0

        # Prices should decrease as strike increases for calls
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1] - 0.1  # Allow small numerical errors


class TestExoticOptionsIntegration:
    """Integration tests across all exotic option types"""

    def setup_method(self):
        """Set up test fixtures"""
        from dervflow import (
            AsianOption,
            BarrierOption,
            BlackScholesModel,
            DigitalOption,
            LookbackOption,
        )

        self.asian = AsianOption()
        self.barrier = BarrierOption()
        self.lookback = LookbackOption()
        self.digital = DigitalOption()
        self.bs = BlackScholesModel()

    def test_all_exotic_types_produce_valid_prices(self):
        """Test that all exotic types produce valid prices for same parameters"""
        spot, strike, rate, div, vol, time = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0

        asian_price = self.asian.price_arithmetic(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )

        barrier_price = self.barrier.price(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )

        lookback_price = self.lookback.price(
            spot, strike, rate, div, vol, time, "call", lookback_type="fixed", current_extremum=None
        )

        digital_price = self.digital.price(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )

        vanilla_price = self.bs.price(spot, strike, rate, div, vol, time, "call")

        # All should be positive
        assert asian_price > 0.0
        assert barrier_price > 0.0
        assert lookback_price > 0.0
        assert digital_price > 0.0

        # Asian should be cheaper than vanilla
        assert asian_price < vanilla_price

        # Barrier should be cheaper than vanilla
        assert barrier_price < vanilla_price

        # Lookback should be more expensive than vanilla
        assert lookback_price > vanilla_price

    def test_consistency_across_option_types(self):
        """Test consistency between calls and puts for all exotic types"""
        spot, strike, rate, div, vol, time = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0

        # Test each exotic type has valid prices for both calls and puts
        asian_call = self.asian.price_arithmetic(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "call",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )
        asian_put = self.asian.price_arithmetic(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "put",
            num_observations=12,
            fixed_strike=True,
            num_paths=10000,
            seed=42,
        )

        barrier_call = self.barrier.price(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "call",
            barrier=90.0,
            barrier_type="down-and-out",
            rebate=0.0,
        )
        barrier_put = self.barrier.price(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "put",
            barrier=110.0,
            barrier_type="up-and-out",
            rebate=0.0,
        )

        lookback_call = self.lookback.price(
            spot, strike, rate, div, vol, time, "call", lookback_type="fixed", current_extremum=None
        )
        lookback_put = self.lookback.price(
            spot, strike, rate, div, vol, time, "put", lookback_type="fixed", current_extremum=None
        )

        digital_call = self.digital.price(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "call",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )
        digital_put = self.digital.price(
            spot,
            strike,
            rate,
            div,
            vol,
            time,
            "put",
            digital_type="cash-or-nothing",
            cash_payout=10.0,
        )

        # All should be positive
        assert all(
            p > 0.0
            for p in [
                asian_call,
                asian_put,
                barrier_call,
                barrier_put,
                lookback_call,
                lookback_put,
                digital_call,
                digital_put,
            ]
        )
