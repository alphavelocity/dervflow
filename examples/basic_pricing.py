# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic Option Pricing Examples

This script demonstrates basic option pricing using dervflow including:
- Black-Scholes pricing for European options
- Binomial tree pricing for American options
- Monte Carlo simulation pricing
- Implied volatility calculation
- Greeks calculation
"""

import numpy as np

from dervflow import BinomialTreeModel, BlackScholesModel, MonteCarloOptionPricer


def black_scholes_example():
    """Demonstrate Black-Scholes pricing"""
    print("=" * 60)
    print("Black-Scholes European Option Pricing")
    print("=" * 60)

    # Create Black-Scholes model
    bs = BlackScholesModel()

    # Option parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.0
    volatility = 0.2
    time = 1.0

    # Price call option
    call_price = bs.price(spot, strike, rate, dividend, volatility, time, "call")
    print(f"\nCall Option Price: ${call_price:.4f}")

    # Price put option
    put_price = bs.price(spot, strike, rate, dividend, volatility, time, "put")
    print(f"Put Option Price: ${put_price:.4f}")

    # Verify put-call parity: C - P = S - K * exp(-r*T)
    parity_lhs = call_price - put_price
    parity_rhs = spot - strike * np.exp(-rate * time)
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.4f}")
    print(f"  S - K*exp(-rT) = {parity_rhs:.4f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.6f}")

    # Calculate Greeks
    greeks = bs.greeks(spot, strike, rate, dividend, volatility, time, "call")
    print(f"\nCall Option Greeks:")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Vega: {greeks['vega']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f}")
    print(f"  Rho: {greeks['rho']:.4f}")


def implied_volatility_example():
    """Demonstrate implied volatility calculation"""
    print("\n" + "=" * 60)
    print("Implied Volatility Calculation")
    print("=" * 60)

    bs = BlackScholesModel()

    # Option parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.0
    time = 1.0

    # True volatility
    true_vol = 0.25

    # Calculate market price using true volatility
    market_price = bs.price(spot, strike, rate, dividend, true_vol, time, "call")
    print(f"\nMarket Price: ${market_price:.4f}")
    print(f"True Volatility: {true_vol:.4f}")

    # Calculate implied volatility from market price
    implied_vol = bs.implied_vol(market_price, spot, strike, rate, dividend, time, "call")
    print(f"Implied Volatility: {implied_vol:.4f}")
    print(f"Difference: {abs(implied_vol - true_vol):.6f}")


def binomial_tree_example():
    """Demonstrate binomial tree pricing"""
    print("\n" + "=" * 60)
    print("Binomial Tree Pricing")
    print("=" * 60)

    tree = BinomialTreeModel()

    # Option parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.0
    volatility = 0.2
    time = 1.0

    # European put option
    european_price = tree.price(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        time,
        steps=100,
        style="european",
        option_type="put",
    )
    print(f"\nEuropean Put (Binomial Tree): ${european_price:.4f}")

    # Compare with Black-Scholes
    bs = BlackScholesModel()
    bs_price = bs.price(spot, strike, rate, dividend, volatility, time, "put")
    print(f"European Put (Black-Scholes): ${bs_price:.4f}")
    print(f"Difference: ${abs(european_price - bs_price):.4f}")

    # American put option (early exercise premium)
    american_price = tree.price(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        time,
        steps=100,
        style="american",
        option_type="put",
    )
    print(f"\nAmerican Put (Binomial Tree): ${american_price:.4f}")
    print(f"Early Exercise Premium: ${american_price - european_price:.4f}")


def monte_carlo_example():
    """Demonstrate Monte Carlo pricing"""
    print("\n" + "=" * 60)
    print("Monte Carlo Option Pricing")
    print("=" * 60)

    mc = MonteCarloOptionPricer()

    # Option parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.0
    volatility = 0.2
    time = 1.0

    # Price European call with Monte Carlo
    result = mc.price_european(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        time,
        option_type="call",
        num_paths=100000,
        use_antithetic=True,
    )

    print(f"\nMonte Carlo European Call:")
    print(f"  Price: ${result['price']:.4f}")
    print(f"  Standard Error: ${result['std_error']:.4f}")
    print(
        f"  95% Confidence Interval: [${result['price'] - 1.96*result['std_error']:.4f}, "
        f"${result['price'] + 1.96*result['std_error']:.4f}]"
    )

    # Compare with Black-Scholes
    bs = BlackScholesModel()
    bs_price = bs.price(spot, strike, rate, dividend, volatility, time, "call")
    print(f"\nBlack-Scholes Price: ${bs_price:.4f}")
    print(f"Difference: ${abs(result['price'] - bs_price):.4f}")


def batch_pricing_example():
    """Demonstrate batch pricing"""
    print("\n" + "=" * 60)
    print("Batch Option Pricing")
    print("=" * 60)

    bs = BlackScholesModel()

    # Create a portfolio of options
    n_options = 5
    spots = np.full(n_options, 100.0, dtype=np.float64)
    strikes = np.array([95.0, 100.0, 105.0, 110.0, 115.0], dtype=np.float64)
    rates = np.full(n_options, 0.05, dtype=np.float64)
    dividends = np.zeros(n_options, dtype=np.float64)
    volatilities = np.full(n_options, 0.2, dtype=np.float64)
    times = np.full(n_options, 1.0, dtype=np.float64)
    option_types = ["call"] * n_options

    # Batch price all options
    prices = bs.price_batch(spots, strikes, rates, dividends, volatilities, times, option_types)

    print(f"\nOption Portfolio (All Calls, S=${spots[0]}, T={times[0]}y, σ={volatilities[0]}):")
    print(f"{'Strike':<10} {'Price':<10}")
    print("-" * 20)
    for strike, price in zip(strikes, prices):
        print(f"${strike:<9.2f} ${price:<9.4f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DERVFLOW - Basic Option Pricing Examples")
    print("=" * 60)

    black_scholes_example()
    implied_volatility_example()
    binomial_tree_example()
    monte_carlo_example()
    batch_pricing_example()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
