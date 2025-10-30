"""Market risk measurement example using the risk analytics toolkit."""

from __future__ import annotations

import numpy as np

from dervflow.risk import RiskMetrics


def simulate_portfolio_returns(seed: int = 11):
    """Simulate correlated daily returns for a three-asset portfolio."""
    rng = np.random.default_rng(seed)

    days = 1000
    mean_daily = np.array([0.0004, 0.0003, 0.0002])
    vol_daily = np.array([0.012, 0.015, 0.010])
    corr = np.array(
        [
            [1.0, 0.55, 0.35],
            [0.55, 1.0, 0.40],
            [0.35, 0.40, 1.0],
        ]
    )
    cov = np.outer(vol_daily, vol_daily) * corr
    asset_returns = rng.multivariate_normal(mean_daily, cov, size=days)

    weights = np.array([0.5, 0.3, 0.2])
    portfolio_returns = asset_returns @ weights

    return asset_returns, portfolio_returns, weights


def compute_risk_metrics(asset_returns, portfolio_returns, weights):
    """Calculate VaR, CVaR, drawdown, and decomposition metrics."""
    risk_metrics = RiskMetrics()

    var_95 = risk_metrics.var(returns=portfolio_returns, confidence_level=0.95, method="historical")
    var_99 = risk_metrics.var(returns=portfolio_returns, confidence_level=0.99, method="parametric")
    var_mc = risk_metrics.var(
        confidence_level=0.99,
        method="monte_carlo",
        mean=np.mean(portfolio_returns),
        std_dev=np.std(portfolio_returns, ddof=1),
        num_simulations=200000,
        seed=42,
    )
    cvar_95 = risk_metrics.cvar(returns=portfolio_returns, confidence_level=0.95)
    max_drawdown = risk_metrics.max_drawdown(portfolio_returns)
    sortino = risk_metrics.sortino_ratio(
        portfolio_returns, risk_free_rate=0.02 / 252, target_return=0.0
    )

    print("=" * 68)
    print("Daily risk measures")
    print("=" * 68)
    print(f"95% historical VaR     : {var_95['var']*100:6.3f}%")
    print(f"99% parametric VaR     : {var_99['var']*100:6.3f}%")
    print(f"99% Monte Carlo VaR    : {var_mc['var']*100:6.3f}%")
    print(f"95% Conditional VaR    : {cvar_95['cvar']*100:6.3f}%")
    print(f"Maximum drawdown       : {max_drawdown*100:6.3f}%")
    print(f"Sortino ratio          : {sortino:6.3f}")

    cov = np.cov(asset_returns, rowvar=False)
    exp_returns = np.mean(asset_returns, axis=0)
    metrics = risk_metrics.portfolio_metrics(
        weights, cov, expected_returns=exp_returns, risk_free_rate=0.02 / 252
    )

    annual_return = np.power(1.0 + metrics["expected_return"], 252) - 1.0
    annual_vol = metrics["volatility"] * np.sqrt(252.0)

    print("\n" + "=" * 68)
    print("Portfolio decomposition")
    print("=" * 68)
    print(f"Annualised return      : {annual_return*100:6.2f}%")
    print(f"Annualised volatility  : {annual_vol*100:6.2f}%")
    print(f"Sharpe ratio           : {metrics['sharpe_ratio']:.3f}")
    print(f"Diversification ratio  : {metrics['diversification_ratio']:.3f}")
    print("Risk contributions (%) :", metrics["risk_contributions"]["percentage"])


def main():
    asset_returns, portfolio_returns, weights = simulate_portfolio_returns()
    compute_risk_metrics(asset_returns, portfolio_returns, weights)
    print("\n" + "=" * 68)
    print("Market risk analysis completed successfully")
    print("=" * 68)


if __name__ == "__main__":
    main()
