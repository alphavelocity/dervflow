# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Portfolio Optimization Examples

This script demonstrates portfolio optimization using dervflow including:
- Mean-variance optimization
- Efficient frontier construction
- Risk parity allocation
- Portfolio constraints handling
- Risk-adjusted performance metrics
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    plt = None

from dervflow import PortfolioOptimizer, RiskParityOptimizer


_HAS_MATPLOTLIB = plt is not None


def generate_sample_data(n_assets=5, n_periods=252):
    """Generate sample return data for demonstration"""
    np.random.seed(42)
    
    # Generate random returns with some correlation structure
    mean_returns = np.random.uniform(0.05, 0.15, n_assets)
    volatilities = np.random.uniform(0.15, 0.35, n_assets)
    
    # Create correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = np.random.uniform(0.2, 0.6)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Convert to covariance matrix
    D = np.diag(volatilities)
    covariance = D @ correlation @ D
    
    return mean_returns, covariance


def mean_variance_optimization_example():
    """Demonstrate mean-variance optimization"""
    print("=" * 60)
    print("Mean-Variance Portfolio Optimization")
    print("=" * 60)
    
    # Generate sample data
    n_assets = 5
    expected_returns, covariance = generate_sample_data(n_assets)
    
    print(f"\nAsset Expected Returns:")
    for i, ret in enumerate(expected_returns):
        print(f"  Asset {i+1}: {ret:.2%}")
    
    print(f"\nAsset Volatilities:")
    for i in range(n_assets):
        vol = np.sqrt(covariance[i, i])
        print(f"  Asset {i+1}: {vol:.2%}")
    
    # Create optimizer
    optimizer = PortfolioOptimizer(expected_returns, covariance)
    
    # Minimum variance portfolio
    print("\n" + "-" * 60)
    print("Minimum Variance Portfolio")
    print("-" * 60)
    result = optimizer.optimize()
    print(f"Weights: {result['weights']}")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    
    # Target return portfolio
    target_return = 0.10
    print("\n" + "-" * 60)
    print(f"Portfolio with Target Return = {target_return:.2%}")
    print("-" * 60)
    result = optimizer.optimize(target_return=target_return)
    print(f"Weights: {result['weights']}")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    
    # Maximum Sharpe ratio portfolio
    risk_free_rate = 0.03
    print("\n" + "-" * 60)
    print(f"Maximum Sharpe Ratio Portfolio (rf = {risk_free_rate:.2%})")
    print("-" * 60)
    result = optimizer.optimize(risk_free_rate=risk_free_rate)
    print(f"Weights: {result['weights']}")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")


def constrained_optimization_example():
    """Demonstrate optimization with constraints"""
    print("\n" + "=" * 60)
    print("Constrained Portfolio Optimization")
    print("=" * 60)
    
    # Generate sample data
    n_assets = 5
    expected_returns, covariance = generate_sample_data(n_assets)
    
    optimizer = PortfolioOptimizer(expected_returns, covariance)
    
    # Unconstrained optimization
    print("\n" + "-" * 60)
    print("Unconstrained Optimization")
    print("-" * 60)
    result = optimizer.optimize(target_return=0.10)
    print(f"Weights: {result['weights']}")
    print(f"Volatility: {result['volatility']:.2%}")
    
    # Box constraints (min/max weights)
    print("\n" + "-" * 60)
    print("With Box Constraints (10% min, 30% max per asset)")
    print("-" * 60)
    min_weights = np.array([0.10] * n_assets)
    max_weights = np.array([0.30] * n_assets)
    result = optimizer.optimize(
        target_return=0.10,
        min_weights=min_weights,
        max_weights=max_weights
    )
    print(f"Weights: {result['weights']}")
    print(f"Volatility: {result['volatility']:.2%}")
    
    # Long-only constraint
    print("\n" + "-" * 60)
    print("Long-Only Constraint (no short selling)")
    print("-" * 60)
    min_weights = np.array([0.0] * n_assets)
    max_weights = np.array([1.0] * n_assets)
    result = optimizer.optimize(
        target_return=0.10,
        min_weights=min_weights,
        max_weights=max_weights
    )
    print(f"Weights: {result['weights']}")
    print(f"Volatility: {result['volatility']:.2%}")


def efficient_frontier_example():
    """Demonstrate efficient frontier construction"""
    print("\n" + "=" * 60)
    print("Efficient Frontier")
    print("=" * 60)
    
    # Generate sample data
    n_assets = 5
    expected_returns, covariance = generate_sample_data(n_assets)
    
    optimizer = PortfolioOptimizer(expected_returns, covariance)
    
    # Generate efficient frontier
    print("\nGenerating efficient frontier with 20 points...")
    frontier = optimizer.efficient_frontier(num_points=20)
    
    # Extract returns and risks
    returns = [p['expected_return'] for p in frontier]
    risks = [p['volatility'] for p in frontier]
    
    print(f"\nFrontier Statistics:")
    print(f"  Minimum Risk: {min(risks):.2%}")
    print(f"  Maximum Return: {max(returns):.2%}")
    print(f"  Return Range: {min(returns):.2%} to {max(returns):.2%}")
    
    if not _HAS_MATPLOTLIB:
        print("\n(Matplotlib not installed - skipping efficient frontier plot.)")
        return

    # Plot efficient frontier
    plt.figure(figsize=(10, 6))
    plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')

    # Plot individual assets
    for i in range(n_assets):
        asset_return = expected_returns[i]
        asset_risk = np.sqrt(covariance[i, i])
        plt.scatter(asset_risk, asset_return, s=100, marker='o', label=f'Asset {i+1}')

    # Highlight special portfolios
    min_var_idx = np.argmin(risks)
    max_return_idx = np.argmax(returns)

    plt.scatter(
        risks[min_var_idx],
        returns[min_var_idx],
        s=200,
        marker='*',
        color='red',
        label='Min Variance',
        zorder=5,
    )
    plt.scatter(
        risks[max_return_idx],
        returns[max_return_idx],
        s=200,
        marker='*',
        color='green',
        label='Max Return',
        zorder=5,
    )

    plt.xlabel('Risk (Volatility)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('examples/efficient_frontier.png', dpi=150)
    print("\nEfficient frontier plot saved to 'examples/efficient_frontier.png'")
    plt.close()


def risk_parity_example():
    """Demonstrate risk parity allocation"""
    print("\n" + "=" * 60)
    print("Risk Parity Portfolio Allocation")
    print("=" * 60)
    
    # Generate sample data
    n_assets = 5
    expected_returns, covariance = generate_sample_data(n_assets)
    
    # Create risk parity optimizer
    rp_optimizer = RiskParityOptimizer(covariance)
    
    # Equal risk contribution
    print("\n" + "-" * 60)
    print("Equal Risk Contribution Portfolio")
    print("-" * 60)
    weights = rp_optimizer.optimize()
    risk_contributions = rp_optimizer.risk_contributions(weights)
    
    print(f"\nWeights:")
    for i, w in enumerate(weights):
        print(f"  Asset {i+1}: {w:.4f} ({w*100:.2f}%)")
    
    print(f"\nRisk Contributions:")
    for i, rc in enumerate(risk_contributions):
        print(f"  Asset {i+1}: {rc:.4f} ({rc*100:.2f}%)")
    
    # Verify equal risk contributions
    print(f"\nRisk Contribution Std Dev: {np.std(risk_contributions):.6f}")
    
    # Custom risk contributions
    print("\n" + "-" * 60)
    print("Custom Risk Contribution Portfolio (50%, 30%, 10%, 5%, 5%)")
    print("-" * 60)
    target_rc = np.array([0.50, 0.30, 0.10, 0.05, 0.05])
    weights = rp_optimizer.optimize(target_risk_contributions=target_rc)
    risk_contributions = rp_optimizer.risk_contributions(weights)
    
    print(f"\nWeights:")
    for i, w in enumerate(weights):
        print(f"  Asset {i+1}: {w:.4f} ({w*100:.2f}%)")
    
    print(f"\nActual Risk Contributions:")
    for i, (target, actual) in enumerate(zip(target_rc, risk_contributions)):
        print(f"  Asset {i+1}: Target={target:.4f}, Actual={actual:.4f}, "
              f"Diff={abs(target-actual):.6f}")


def compare_strategies_example():
    """Compare different portfolio strategies"""
    print("\n" + "=" * 60)
    print("Portfolio Strategy Comparison")
    print("=" * 60)
    
    # Generate sample data
    n_assets = 5
    expected_returns, covariance = generate_sample_data(n_assets)
    
    # Equal weight portfolio
    equal_weights = np.ones(n_assets) / n_assets
    
    # Mean-variance optimizer
    mv_optimizer = PortfolioOptimizer(expected_returns, covariance)
    
    # Minimum variance
    min_var_result = mv_optimizer.optimize()
    
    # Maximum Sharpe ratio
    max_sharpe_result = mv_optimizer.optimize(risk_free_rate=0.03)
    
    # Risk parity
    rp_optimizer = RiskParityOptimizer(covariance)
    rp_weights = rp_optimizer.optimize()
    
    # Calculate metrics for all strategies
    strategies = {
        'Equal Weight': equal_weights,
        'Minimum Variance': min_var_result['weights'],
        'Maximum Sharpe': max_sharpe_result['weights'],
        'Risk Parity': rp_weights
    }
    
    print(f"\n{'Strategy':<20} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
    print("-" * 50)
    
    for name, weights in strategies.items():
        ret = mv_optimizer.portfolio_return(weights)
        risk = mv_optimizer.portfolio_volatility(weights)
        sharpe = mv_optimizer.sharpe_ratio(weights, risk_free_rate=0.03)
        print(f"{name:<20} {ret:>8.2%} {risk:>8.2%} {sharpe:>9.4f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DERVFLOW - Portfolio Optimization Examples")
    print("=" * 60)
    
    mean_variance_optimization_example()
    constrained_optimization_example()
    efficient_frontier_example()
    risk_parity_example()
    compare_strategies_example()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
