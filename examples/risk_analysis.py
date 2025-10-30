# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Risk Analysis Examples

This script demonstrates risk analytics using dervflow including:
- Greeks calculation for single options and portfolios
- Value at Risk (VaR) using multiple methods
- Conditional Value at Risk (CVaR)
- Risk-adjusted performance metrics
- Drawdown analysis
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    plt = None

from dervflow import BlackScholesModel, GreeksCalculator, RiskMetrics

_HAS_MATPLOTLIB = plt is not None


def greeks_calculation_example():
    """Demonstrate Greeks calculation"""
    print("=" * 60)
    print("Option Greeks Calculation")
    print("=" * 60)

    greeks_calc = GreeksCalculator()

    # Option parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.0
    volatility = 0.2
    time = 1.0

    # Calculate Greeks for call option
    print("\n" + "-" * 60)
    print("Call Option Greeks (ATM)")
    print("-" * 60)
    call_greeks = greeks_calc.calculate(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        time,
        option_type="call",
    )

    print(f"Spot Price: ${spot:.2f}")
    print(f"Strike Price: ${strike:.2f}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Time to Maturity: {time:.2f} years")
    print(f"\nFirst-Order Greeks:")
    print(f"  Delta: {call_greeks['delta']:.4f}")
    print(f"  Vega: {call_greeks['vega']:.4f}")
    print(f"  Theta: {call_greeks['theta']:.4f}")
    print(f"  Rho: {call_greeks['rho']:.4f}")
    print(f"\nSecond-Order Greeks:")
    print(f"  Gamma: {call_greeks['gamma']:.4f}")
    print(f"  Vanna: {call_greeks.get('vanna', 0.0):.4f}")
    print(f"  Volga: {call_greeks.get('volga', 0.0):.4f}")

    # Calculate Greeks for put option
    print("\n" + "-" * 60)
    print("Put Option Greeks (ATM)")
    print("-" * 60)
    put_greeks = greeks_calc.calculate(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        time,
        option_type="put",
    )

    print(f"\nFirst-Order Greeks:")
    print(f"  Delta: {put_greeks['delta']:.4f}")
    print(f"  Vega: {put_greeks['vega']:.4f}")
    print(f"  Theta: {put_greeks['theta']:.4f}")
    print(f"  Rho: {put_greeks['rho']:.4f}")

    # Verify put-call parity for Delta
    print(f"\nPut-Call Parity Check (Delta):")
    print(f"  Call Delta - Put Delta = {call_greeks['delta'] - put_greeks['delta']:.4f}")
    print(f"  Expected (exp(-qT)): {np.exp(-dividend * time):.4f}")


def greeks_sensitivity_example():
    """Demonstrate Greeks sensitivity to parameters"""
    print("\n" + "=" * 60)
    print("Greeks Sensitivity Analysis")
    print("=" * 60)

    greeks_calc = GreeksCalculator()

    # Base parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.0
    volatility = 0.2
    time = 1.0

    # Analyze Delta across different spot prices
    print("\n" + "-" * 60)
    print("Delta vs Spot Price (Call Option)")
    print("-" * 60)

    spot_range = np.linspace(80, 120, 9)
    deltas = []

    print(f"{'Spot':<10} {'Delta':<10} {'Moneyness':<15}")
    print("-" * 35)

    for s in spot_range:
        greeks = greeks_calc.calculate(
            s, strike, rate, dividend, volatility, time, option_type="call"
        )
        deltas.append(greeks["delta"])
        moneyness = "ITM" if s > strike else ("ATM" if s == strike else "OTM")
        print(f"${s:<9.2f} {greeks['delta']:<9.4f} {moneyness:<15}")

    # Analyze Vega across different volatilities
    print("\n" + "-" * 60)
    print("Vega vs Volatility (ATM Call Option)")
    print("-" * 60)

    vol_range = np.linspace(0.1, 0.5, 9)
    vegas = []

    print(f"{'Volatility':<12} {'Vega':<10}")
    print("-" * 22)

    for vol in vol_range:
        greeks = greeks_calc.calculate(spot, strike, rate, dividend, vol, time, option_type="call")
        vegas.append(greeks["vega"])
        print(f"{vol:<11.2%} {greeks['vega']:<9.4f}")


def portfolio_greeks_example():
    """Demonstrate portfolio Greeks calculation"""
    print("\n" + "=" * 60)
    print("Portfolio Greeks Calculation")
    print("=" * 60)

    greeks_calc = GreeksCalculator()

    # Define a portfolio of options
    positions = [
        {
            "spot": 100.0,
            "strike": 95.0,
            "volatility": 0.2,
            "time": 1.0,
            "option_type": "call",
            "quantity": 10,  # Long 10 calls
        },
        {
            "spot": 100.0,
            "strike": 100.0,
            "volatility": 0.2,
            "time": 1.0,
            "option_type": "call",
            "quantity": -20,  # Short 20 calls
        },
        {
            "spot": 100.0,
            "strike": 105.0,
            "volatility": 0.2,
            "time": 1.0,
            "option_type": "call",
            "quantity": 10,  # Long 10 calls
        },
    ]

    rate = 0.05
    dividend = 0.0

    print("\nPortfolio Positions:")
    print(f"{'Position':<15} {'Strike':<10} {'Type':<10} {'Quantity':<10}")
    print("-" * 45)
    for i, pos in enumerate(positions, 1):
        qty_str = f"+{pos['quantity']}" if pos["quantity"] > 0 else str(pos["quantity"])
        print(f"Position {i:<7} ${pos['strike']:<9.2f} {pos['option_type']:<10} {qty_str:<10}")

    # Calculate portfolio Greeks
    spots = np.array([pos["spot"] for pos in positions], dtype=np.float64)
    strikes = np.array([pos["strike"] for pos in positions], dtype=np.float64)
    rates = np.full(len(positions), rate, dtype=np.float64)
    dividends = np.full(len(positions), dividend, dtype=np.float64)
    volatilities = np.array([pos["volatility"] for pos in positions], dtype=np.float64)
    times = np.array([pos["time"] for pos in positions], dtype=np.float64)
    option_types = [pos["option_type"] for pos in positions]
    quantities = np.array([pos["quantity"] for pos in positions], dtype=np.float64)

    portfolio_greeks = greeks_calc.portfolio_greeks(
        spots,
        strikes,
        rates,
        dividends,
        volatilities,
        times,
        option_types,
        quantities,
    )

    print("\n" + "-" * 60)
    print("Portfolio Greeks")
    print("-" * 60)
    print(f"Delta: {portfolio_greeks['delta']:.4f}")
    print(f"Gamma: {portfolio_greeks['gamma']:.4f}")
    print(f"Vega: {portfolio_greeks['vega']:.4f}")
    print(f"Theta: {portfolio_greeks['theta']:.4f}")
    print(f"Rho: {portfolio_greeks['rho']:.4f}")

    print("\nInterpretation:")
    if abs(portfolio_greeks["delta"]) < 0.1:
        print("  - Portfolio is approximately delta-neutral")
    if abs(portfolio_greeks["gamma"]) < 0.01:
        print("  - Portfolio has low gamma exposure")


def var_calculation_example():
    """Demonstrate VaR calculation using different methods"""
    print("\n" + "=" * 60)
    print("Value at Risk (VaR) Calculation")
    print("=" * 60)

    risk_metrics = RiskMetrics()

    # Generate sample returns (simulating daily returns)
    np.random.seed(42)
    n_days = 1000
    mean_return = 0.0005  # 0.05% daily
    volatility = 0.015  # 1.5% daily
    returns = np.random.normal(mean_return, volatility, n_days).astype(np.float64)

    print(f"\nReturn Statistics:")
    print(f"  Number of observations: {n_days}")
    print(f"  Mean return: {np.mean(returns):.4%}")
    print(f"  Volatility: {np.std(returns):.4%}")
    print(f"  Min return: {np.min(returns):.4%}")
    print(f"  Max return: {np.max(returns):.4%}")

    # Calculate VaR at different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]

    print("\n" + "-" * 60)
    print("Historical Simulation VaR")
    print("-" * 60)
    print(f"{'Confidence':<15} {'VaR':<15} {'Interpretation':<30}")
    print("-" * 60)

    for conf in confidence_levels:
        var_result = risk_metrics.var(returns, conf, method="historical")
        var_value = var_result["var"]
        print(
            f"{conf:.0%}{'':>12} {var_value:>13.4%}  "
            f"Loss exceeds {abs(var_value):.2%} in {(1-conf)*100:.0f}% of cases"
        )

    # Calculate CVaR (Expected Shortfall)
    print("\n" + "-" * 60)
    print("Conditional VaR (Expected Shortfall)")
    print("-" * 60)
    print(f"{'Confidence':<15} {'CVaR':<15} {'Interpretation':<30}")
    print("-" * 60)

    for conf in confidence_levels:
        cvar_result = risk_metrics.cvar(returns, conf)
        cvar_value = cvar_result["cvar"]
        print(f"{conf:.0%}{'':>12} {cvar_value:>13.4%}  " f"Expected loss when VaR is exceeded")

    # Compare VaR methods
    print("\n" + "-" * 60)
    print("VaR Method Comparison (95% confidence)")
    print("-" * 60)

    methods = ["historical", "parametric"]
    for method in methods:
        var_result = risk_metrics.var(returns, 0.95, method=method)
        print(f"{method.capitalize():<20} VaR: {var_result['var']:.4%}")


def drawdown_analysis_example():
    """Demonstrate drawdown analysis"""
    print("\n" + "=" * 60)
    print("Drawdown Analysis")
    print("=" * 60)

    risk_metrics = RiskMetrics()

    # Generate sample price series
    np.random.seed(42)
    n_days = 252
    initial_price = 100.0
    daily_returns = np.random.normal(0.0005, 0.015, n_days)
    prices = initial_price * np.exp(np.cumsum(daily_returns))
    returns = (prices[1:] / prices[:-1] - 1).astype(np.float64)

    print(f"\nPrice Series Statistics:")
    print(f"  Initial Price: ${initial_price:.2f}")
    print(f"  Final Price: ${prices[-1]:.2f}")
    print(f"  Total Return: {(prices[-1]/initial_price - 1):.2%}")
    print(f"  Min Price: ${np.min(prices):.2f}")
    print(f"  Max Price: ${np.max(prices):.2f}")

    # Calculate maximum drawdown
    max_dd = risk_metrics.max_drawdown(returns)

    print(f"\nMaximum Drawdown: {max_dd:.2%}")

    # Calculate drawdown series
    cummax = np.maximum.accumulate(prices)
    drawdowns = (prices - cummax) / cummax

    # Find the maximum drawdown period
    max_dd_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(cummax[: max_dd_idx + 1])

    print(f"\nMaximum Drawdown Details:")
    print(f"  Peak Price: ${prices[peak_idx]:.2f} (Day {peak_idx})")
    print(f"  Trough Price: ${prices[max_dd_idx]:.2f} (Day {max_dd_idx})")
    print(f"  Drawdown Duration: {max_dd_idx - peak_idx} days")

    if not _HAS_MATPLOTLIB:
        print("\n(Matplotlib not installed - skipping drawdown analysis plot.)")
        return

    # Plot drawdown series
    plt.figure(figsize=(12, 8))

    # Plot price series
    plt.subplot(2, 1, 1)
    plt.plot(prices, "b-", linewidth=1.5, label="Price")
    plt.plot(cummax, "r--", linewidth=1, label="Running Maximum")
    plt.scatter(
        [peak_idx, max_dd_idx],
        [prices[peak_idx], prices[max_dd_idx]],
        c="red",
        s=100,
        zorder=5,
    )
    plt.ylabel("Price ($)", fontsize=11)
    plt.title("Price Series and Maximum Drawdown", fontsize=13, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    # Plot drawdown series
    plt.subplot(2, 1, 2)
    plt.fill_between(
        range(len(drawdowns)),
        drawdowns,
        0,
        where=(drawdowns < 0),
        color="red",
        alpha=0.3,
        label="Drawdown",
    )
    plt.plot(drawdowns, "r-", linewidth=1)
    plt.axhline(
        y=max_dd,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Max Drawdown: {max_dd:.2%}",
    )
    plt.ylabel("Drawdown (%)", fontsize=11)
    plt.xlabel("Trading Days", fontsize=11)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/drawdown_analysis.png", dpi=150)
    print("\nDrawdown analysis plot saved to 'examples/drawdown_analysis.png'")
    plt.close()


def risk_adjusted_metrics_example():
    """Demonstrate risk-adjusted performance metrics"""
    print("\n" + "=" * 60)
    print("Risk-Adjusted Performance Metrics")
    print("=" * 60)

    risk_metrics = RiskMetrics()

    # Generate sample returns for different strategies
    np.random.seed(42)
    n_days = 252

    strategies = {
        "Conservative": (0.0003, 0.008),  # Low return, low risk
        "Moderate": (0.0005, 0.012),  # Medium return, medium risk
        "Aggressive": (0.0008, 0.020),  # High return, high risk
        "Volatile": (0.0005, 0.025),  # Medium return, high risk
    }

    risk_free_rate = 0.03 / 252  # Daily risk-free rate

    print(f"\n{'Strategy':<15} {'Return':<10} {'Risk':<10} {'Sharpe':<10} {'Sortino':<10}")
    print("-" * 55)

    for name, (mean_ret, vol) in strategies.items():
        returns = np.random.normal(mean_ret, vol, n_days).astype(np.float64)

        # Annualize metrics
        annual_return = (1 + mean_ret) ** 252 - 1
        annual_vol = vol * np.sqrt(252)

        # Calculate Sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        # Calculate Sortino ratio
        sortino = risk_metrics.sortino_ratio(
            returns.astype(np.float64),
            risk_free_rate=risk_free_rate,
            target_return=0.0,
        )

        print(
            f"{name:<15} {annual_return:>8.2%} {annual_vol:>8.2%} "
            f"{sharpe:>9.4f} {sortino:>9.4f}"
        )

    print("\nInterpretation:")
    print("  - Sharpe Ratio: Measures excess return per unit of total risk")
    print("  - Sortino Ratio: Measures excess return per unit of downside risk")
    print("  - Higher values indicate better risk-adjusted performance")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DERVFLOW - Risk Analysis Examples")
    print("=" * 60)

    greeks_calculation_example()
    greeks_sensitivity_example()
    portfolio_greeks_example()
    var_calculation_example()
    drawdown_analysis_example()
    risk_adjusted_metrics_example()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
