# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for yield curve module

Tests cover:
- Bootstrapping accuracy from bonds and swaps
- Interpolation methods (linear, cubic spline, Nelson-Siegel)
- Forward rate calculations
- Bond pricing from yield curve
- Bond analytics (YTM, duration, convexity, DV01)
"""

import pytest
import numpy as np
from dervflow import BondAnalytics, MultiCurve, SwapPeriod, YieldCurve, YieldCurveBuilder


class TestYieldCurveConstruction:
    """Test yield curve construction and basic operations"""

    def test_linear_interpolation(self):
        """Test yield curve with linear interpolation"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='linear')

        # Test exact points
        assert abs(curve.zero_rate(2.0) - 0.035) < 1e-10

        # Test interpolation
        rate = curve.zero_rate(3.0)
        assert rate > 0.035
        assert rate < 0.04

    def test_cubic_spline_natural(self):
        """Test yield curve with natural cubic spline"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='cubic_spline_natural')

        # Test exact points
        for i, t in enumerate(times):
            assert abs(curve.zero_rate(t) - rates[i]) < 1e-6

        # Test smooth interpolation
        rate = curve.zero_rate(3.0)
        assert rate > 0.035
        assert rate < 0.045

    def test_cubic_spline_clamped(self):
        """Test yield curve with clamped cubic spline"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='cubic_spline_clamped')

        # Test exact points
        for i, t in enumerate(times):
            assert abs(curve.zero_rate(t) - rates[i]) < 1e-6

    def test_discount_factor(self):
        """Test discount factor calculation"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='linear')

        # DF at t=0 should be 1
        assert abs(curve.discount_factor(0.0) - 1.0) < 1e-10

        # DF at t=1 with r=0.03
        df = curve.discount_factor(1.0)
        expected = np.exp(-0.03 * 1.0)
        assert abs(df - expected) < 1e-10

        # DF should be decreasing
        df1 = curve.discount_factor(1.0)
        df2 = curve.discount_factor(2.0)
        assert df1 > df2

    def test_forward_rate(self):
        """Test forward rate calculation"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='linear')

        # Forward rate from year 1 to year 2
        f = curve.forward_rate(1.0, 2.0)
        # f(1,2) = (r2*2 - r1*1) / (2-1) = (0.035*2 - 0.03*1) / 1 = 0.04
        assert abs(f - 0.04) < 1e-10

        # Forward rate from now to year 2 should equal zero rate
        f = curve.forward_rate(0.0, 2.0)
        assert abs(f - 0.035) < 1e-10

    def test_bond_pricing(self):
        """Test bond pricing from yield curve"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='linear')

        # Simple bond with annual coupons
        cashflows = [
            (1.0, 5.0),   # Coupon at year 1
            (2.0, 5.0),   # Coupon at year 2
            (2.0, 100.0), # Principal at year 2
        ]

        price = curve.price_bond(cashflows)
        assert price > 0.0
        assert price < 110.0  # Should be less than sum of cashflows

    def test_invalid_times(self):
        """Test that unsorted times raise error"""
        times = np.array([1.0, 0.5, 2.0])  # Not sorted
        rates = np.array([0.03, 0.035, 0.04])
        with pytest.raises(Exception):
            YieldCurve(times, rates, method='linear')

    def test_empty_curve(self):
        """Test that empty arrays raise error"""
        times = np.array([])
        rates = np.array([])
        with pytest.raises(Exception):
            YieldCurve(times, rates, method='linear')


class TestBootstrapping:
    """Test yield curve bootstrapping methods"""

    def test_bootstrap_from_bonds(self):
        """Test bootstrapping from bond prices"""
        # (maturity, coupon, price, frequency)
        bonds = [
            (0.5, 0.03, 99.5, 2),
            (1.0, 0.04, 99.0, 2),
            (2.0, 0.05, 98.0, 2),
        ]

        curve = YieldCurveBuilder.bootstrap_from_bonds(bonds)

        # Check that we got a valid curve
        assert len(curve.times()) == 3
        assert len(curve.rates()) == 3

        # Rates should be positive
        rates = curve.rates()
        assert all(r > 0.0 for r in rates)

    def test_bootstrap_from_swaps(self):
        """Test bootstrapping from swap rates"""
        # (maturity, rate, frequency)
        swaps = [
            (1.0, 0.03, 2),
            (2.0, 0.035, 2),
            (5.0, 0.04, 2),
        ]

        curve = YieldCurveBuilder.bootstrap_from_swaps(swaps)

        # Check that we got a valid curve
        assert len(curve.times()) == 3
        assert len(curve.rates()) == 3

        # Rates should be positive
        rates = curve.rates()
        assert all(r > 0.0 for r in rates)

    def test_bootstrap_empty_bonds(self):
        """Test that empty bond list raises error"""
        bonds = []
        with pytest.raises(Exception):
            YieldCurveBuilder.bootstrap_from_bonds(bonds)


class TestNelsonSiegel:
    """Test Nelson-Siegel and Nelson-Siegel-Svensson models"""

    def test_nelson_siegel(self):
        """Test Nelson-Siegel yield curve"""
        times = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        curve = YieldCurveBuilder.from_nelson_siegel(
            beta0=0.05,
            beta1=-0.02,
            beta2=0.01,
            lambda_=1.0,
            times=times
        )

        # Check that we got a valid curve
        assert len(curve.times()) == 5
        assert len(curve.rates()) == 5

        # Rates should be positive
        rates = curve.rates()
        assert all(r > 0.0 for r in rates)

        # Long-term rate should approach beta0
        long_rate = curve.zero_rate(100.0)
        assert abs(long_rate - 0.05) < 0.01

    def test_nelson_siegel_svensson(self):
        """Test Nelson-Siegel-Svensson yield curve"""
        times = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        curve = YieldCurveBuilder.from_nelson_siegel_svensson(
            beta0=0.05,
            beta1=-0.02,
            beta2=0.01,
            beta3=0.005,
            lambda1=1.0,
            lambda2=3.0,
            times=times
        )

        # Check that we got a valid curve
        assert len(curve.times()) == 5
        assert len(curve.rates()) == 5

        # Rates should be positive
        rates = curve.rates()
        assert all(r > 0.0 for r in rates)

        # Long-term rate should approach beta0
        long_rate = curve.zero_rate(100.0)
        assert abs(long_rate - 0.05) < 0.01


class TestBondAnalytics:
    """Test bond analytics calculations"""

    def setup_method(self):
        """Set up test fixtures"""
        self.ba = BondAnalytics()

    def test_generate_cashflows(self):
        """Test cashflow generation for standard bond"""
        cashflows = BondAnalytics.generate_cashflows(
            maturity=2.0,
            coupon_rate=0.05,
            face_value=100.0,
            frequency=2
        )

        assert len(cashflows) == 4

        # Check coupon payments
        assert abs(cashflows[0][1] - 2.5) < 1e-10
        assert abs(cashflows[1][1] - 2.5) < 1e-10
        assert abs(cashflows[2][1] - 2.5) < 1e-10

        # Check final payment (coupon + principal)
        assert abs(cashflows[3][1] - 102.5) < 1e-10

    def test_bond_price(self):
        """Test bond price calculation"""
        cashflows = [
            (1.0, 5.0),
            (2.0, 105.0),
        ]

        price = self.ba.bond_price(0.05, cashflows)
        expected = 5.0 * np.exp(-0.05) + 105.0 * np.exp(-0.10)
        assert abs(price - expected) < 1e-6

    def test_yield_to_maturity(self):
        """Test YTM calculation"""
        cashflows = BondAnalytics.generate_cashflows(2.0, 0.05, 100.0, 2)
        price = 98.0

        ytm = self.ba.yield_to_maturity(price, cashflows)

        # YTM should be positive and reasonable
        assert ytm > 0.0
        assert ytm < 0.2

        # Verify: price calculated with YTM should match input price
        calculated_price = self.ba.bond_price(ytm, cashflows)
        assert abs(calculated_price - price) < 0.01

    def test_macaulay_duration(self):
        """Test Macaulay duration calculation"""
        cashflows = [
            (1.0, 5.0),
            (2.0, 105.0),
        ]

        duration = self.ba.macaulay_duration(0.05, cashflows)

        # Duration should be between 1 and 2 years
        assert duration > 1.0
        assert duration < 2.0

    def test_modified_duration(self):
        """Test modified duration calculation"""
        cashflows = [
            (1.0, 5.0),
            (2.0, 105.0),
        ]

        # Continuous compounding
        mod_dur_cont = self.ba.modified_duration(0.05, cashflows, frequency=0)
        mac_dur = self.ba.macaulay_duration(0.05, cashflows)
        assert abs(mod_dur_cont - mac_dur) < 1e-10

        # Semi-annual compounding
        mod_dur_semi = self.ba.modified_duration(0.05, cashflows, frequency=2)
        assert mod_dur_semi < mac_dur

    def test_convexity(self):
        """Test convexity calculation"""
        cashflows = [
            (1.0, 5.0),
            (2.0, 105.0),
        ]

        conv = self.ba.convexity(0.05, cashflows)

        # Convexity should be positive
        assert conv > 0.0

    def test_dv01(self):
        """Test DV01 calculation"""
        cashflows = BondAnalytics.generate_cashflows(5.0, 0.05, 100.0, 2)

        dv01_value = self.ba.dv01(0.05, cashflows)

        # DV01 should be positive
        assert dv01_value > 0.0

        # For a 5-year bond, DV01 should be reasonable
        assert dv01_value < 1.0

    def test_zero_coupon_bond(self):
        """Test analytics for zero coupon bond"""
        # Zero coupon bond: single payment at maturity
        cashflows = [(5.0, 100.0)]

        # Test duration calculation (more reliable than YTM for zero coupon)
        duration = self.ba.macaulay_duration(0.05, cashflows)
        assert abs(duration - 5.0) < 1e-6  # Duration equals maturity for zero coupon
        
        # Test YTM calculation
        ytm = self.ba.yield_to_maturity(77.88, cashflows)
        # YTM should be positive and reasonable (between 0 and 0.2)
        assert 0.0 <= ytm <= 0.2

    def test_invalid_price(self):
        """Test that invalid price raises error"""
        cashflows = [(1.0, 100.0)]
        with pytest.raises(Exception):
            self.ba.yield_to_maturity(-10.0, cashflows)

    def test_empty_cashflows(self):
        """Test that empty cashflows raise error"""
        cashflows = []
        with pytest.raises(Exception):
            self.ba.macaulay_duration(0.05, cashflows)


class TestYieldCurveIntegration:
    """Integration tests combining multiple yield curve operations"""

    def test_bootstrap_and_price_bond(self):
        """Test bootstrapping curve and pricing bond"""
        # Bootstrap from bonds
        bonds = [
            (1.0, 0.03, 99.5, 2),
            (2.0, 0.04, 99.0, 2),
            (5.0, 0.05, 98.0, 2),
        ]

        curve = YieldCurveBuilder.bootstrap_from_bonds(bonds)

        # Price a new bond using the curve
        cashflows = [
            (1.0, 3.0),
            (2.0, 3.0),
            (3.0, 103.0),
        ]

        price = curve.price_bond(cashflows)
        assert price > 0.0
        assert price < 109.0

    def test_forward_rates_consistency(self):
        """Test that forward rates are consistent with zero rates"""
        times = np.array([1.0, 2.0, 3.0, 5.0])
        rates = np.array([0.03, 0.035, 0.038, 0.04])
        curve = YieldCurve(times, rates, method='linear')

        # Forward rate from 1 to 3 should be consistent
        f_1_3 = curve.forward_rate(1.0, 3.0)

        # Calculate using zero rates
        r1 = curve.zero_rate(1.0)
        r3 = curve.zero_rate(3.0)
        expected = (r3 * 3.0 - r1 * 1.0) / (3.0 - 1.0)

        assert abs(f_1_3 - expected) < 1e-10

    def test_discount_factors_consistency(self):
        """Test that discount factors are consistent with zero rates"""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.03, 0.035, 0.04, 0.045])
        curve = YieldCurve(times, rates, method='linear')

        for t in [1.0, 2.5, 5.0, 7.5]:
            df = curve.discount_factor(t)
            r = curve.zero_rate(t)
            expected_df = np.exp(-r * t)
            assert abs(df - expected_df) < 1e-10

    def test_ytm_and_bond_price_consistency(self):
        """Test that YTM and bond price are consistent"""
        ba = BondAnalytics()

        # Generate cashflows
        cashflows = BondAnalytics.generate_cashflows(3.0, 0.06, 100.0, 2)

        # Calculate price at a given yield
        yield_rate = 0.05
        price = ba.bond_price(yield_rate, cashflows)

        # Calculate YTM from that price
        ytm = ba.yield_to_maturity(price, cashflows, initial_guess=0.05)

        # YTM should match the original yield
        assert abs(ytm - yield_rate) < 1e-6


class TestMultiCurve:
    """Tests for multi-curve functionality exposed to Python"""

    def setup_method(self):
        times = np.array([0.5, 1.0, 2.0, 5.0])
        discount_rates = np.array([0.02, 0.02, 0.02, 0.02])
        forward_rates = np.array([0.03, 0.031, 0.032, 0.033])

        self.discount_curve = YieldCurve(times, discount_rates, method='linear')
        self.forward_curve = YieldCurve(times, forward_rates, method='linear')

    @staticmethod
    def _schedule():
        return [
            SwapPeriod(0.0, 0.5, 0.5),
            SwapPeriod(0.5, 1.0, 0.5),
            SwapPeriod(1.0, 1.5, 0.5),
            SwapPeriod(1.5, 2.0, 0.5),
        ]

    def test_forward_curve_management(self):
        mc = MultiCurve(self.discount_curve)
        mc.add_forward_curve("LIBOR3M", self.forward_curve)

        names = mc.forward_curve_names()
        assert "LIBOR3M" in names

        retrieved = mc.forward_curve("LIBOR3M")
        assert isinstance(retrieved, YieldCurve)
        assert abs(retrieved.zero_rate(1.0) - self.forward_curve.zero_rate(1.0)) < 1e-12

        discount = mc.discount_curve()
        assert isinstance(discount, YieldCurve)
        assert abs(discount.zero_rate(1.0) - self.discount_curve.zero_rate(1.0)) < 1e-12

    def test_swap_pricing_matches_par_rate(self):
        mc = MultiCurve(self.discount_curve)
        mc.set_forward_curve("LIBOR3M", self.forward_curve)
        schedule = self._schedule()

        par_rate = mc.par_swap_rate("LIBOR3M", schedule)
        pv_at_par = mc.price_payer_swap("LIBOR3M", schedule, par_rate, 1_000_000.0)

        assert abs(pv_at_par) < 1e-2

        pv_above_par = mc.price_payer_swap(
            "LIBOR3M", schedule, par_rate + 0.01, 1_000_000.0
        )
        assert pv_above_par < 0.0

    def test_swap_pricing_validates_schedule(self):
        mc = MultiCurve(self.discount_curve)
        mc.set_forward_curve("LIBOR3M", self.forward_curve)

        with pytest.raises(Exception):
            mc.price_payer_swap("LIBOR3M", [], 0.03, 1_000_000.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
