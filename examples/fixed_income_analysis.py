"""Fixed-income analytics example using the yield-curve toolkit."""

from __future__ import annotations

import numpy as np

from dervflow.yield_curve import BondAnalytics, YieldCurveBuilder


def build_treasury_yield_curve():
    """Bootstrap a spot curve from on-the-run Treasury notes."""
    print("=" * 72)
    print("Bootstrapping Treasury zero curve")
    print("=" * 72)

    # (maturity in years, annual coupon rate, clean price, coupon frequency)
    treasury_bonds = [
        (0.5, 0.0150, 100.12, 2),
        (1.0, 0.0175, 100.35, 2),
        (2.0, 0.0210, 100.80, 2),
        (3.0, 0.0240, 101.10, 2),
        (5.0, 0.0275, 101.70, 2),
        (7.0, 0.0300, 102.05, 2),
        (10.0, 0.0325, 102.60, 2),
    ]

    builder = YieldCurveBuilder()
    curve = builder.bootstrap_from_bonds(treasury_bonds)

    maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    zero_rates = np.array([curve.zero_rate(m) for m in maturities])
    discount_factors = np.array([curve.discount_factor(m) for m in maturities])

    print(f"{'Maturity':>12} {'Zero Rate':>12} {'Discount Factor':>20}")
    print("-" * 48)
    for t, z, df in zip(maturities, zero_rates, discount_factors):
        print(f"{t:>9.1f}y {100 * z:>10.2f}% {df:>18.6f}")

    spot_2y = curve.zero_rate(2.0)
    fwd_2y3y = curve.forward_rate(2.0, 3.0)
    print("\nKey curve levels:")
    print(f"  2-year zero rate : {100 * spot_2y:.2f}%")
    print(f"  2y-3y forward rate: {100 * fwd_2y3y:.2f}%")

    return curve


def analyze_corporate_bond(curve):
    """Price and risk-analyse a 5-year corporate bond using the curve."""
    print("\n" + "=" * 72)
    print("Corporate bond valuation and risk measures")
    print("=" * 72)

    face_value = 100.0
    coupon_rate = 0.032  # 3.2% annual coupon
    maturity = 5.0
    frequency = 2  # Semi-annual coupons

    cashflows = BondAnalytics.generate_cashflows(maturity, coupon_rate, face_value, frequency)
    analytics = BondAnalytics()

    price = sum(amount * curve.discount_factor(time) for time, amount in cashflows)
    yield_to_maturity = analytics.yield_to_maturity(price, cashflows)
    macaulay_duration = analytics.macaulay_duration(yield_to_maturity, cashflows)
    modified_duration = analytics.modified_duration(
        yield_to_maturity, cashflows, frequency=frequency
    )
    convexity = analytics.convexity(yield_to_maturity, cashflows)
    dv01 = analytics.dv01(yield_to_maturity, cashflows)

    print(f"Clean price from curve : {price:8.3f}")
    print(f"Yield to maturity     : {100 * yield_to_maturity:8.3f}%")
    print(f"Macaulay duration     : {macaulay_duration:8.4f} years")
    print(f"Modified duration     : {modified_duration:8.4f}")
    print(f"Convexity             : {convexity:8.4f}")
    print(f"DV01 (per bp)         : {dv01:8.5f}")

    parallel_shift = 0.0025  # +25 bp
    shocked_yield = yield_to_maturity + parallel_shift
    shocked_price = analytics.bond_price(shocked_yield, cashflows)
    actual_change = price - shocked_price
    duration_approx = dv01 * (parallel_shift / 0.0001)

    print("\n25 bp parallel shift analysis:")
    print(f"  New yield           : {100 * shocked_yield:8.3f}%")
    print(f"  Repriced value      : {shocked_price:8.3f}")
    print(f"  Actual price change : {actual_change:8.4f}")
    print(f"  DV01 approximation  : {duration_approx:8.4f}")


def main():
    curve = build_treasury_yield_curve()
    analyze_corporate_bond(curve)
    print("\n" + "=" * 72)
    print("Fixed-income analysis completed successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()
