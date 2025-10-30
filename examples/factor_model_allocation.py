"""Multi-factor analysis and portfolio construction example."""


from __future__ import annotations

import numpy as np

from dervflow.portfolio import (
    BlackLittermanModel,
    FactorModel,
    InvestorViews,
    PortfolioOptimizer,
)


def simulate_monthly_data(seed: int = 7):
    """Generate synthetic monthly factor and asset returns."""
    rng = np.random.default_rng(seed)

    periods = 120  # 10 years of monthly data
    factor_means = np.array([0.0050, 0.0020, 0.0015])
    factor_vols = np.array([0.035, 0.020, 0.018])
    factor_corr = np.array([
        [1.00, 0.30, 0.20],
        [0.30, 1.00, 0.35],
        [0.20, 0.35, 1.00],
    ])
    factor_cov = np.outer(factor_vols, factor_vols) * factor_corr
    factor_returns = rng.multivariate_normal(factor_means, factor_cov, size=periods)

    betas = np.array([
        [1.05, 0.15, 0.05],
        [0.90, -0.05, 0.25],
        [1.10, 0.30, 0.10],
        [0.80, 0.40, -0.02],
    ])
    alpha = np.array([0.0010, 0.0008, 0.0012, 0.0010])
    idio_vol = np.array([0.020, 0.018, 0.025, 0.020])
    noise = rng.normal(scale=idio_vol, size=(periods, betas.shape[0]))

    asset_returns = factor_returns @ betas.T + alpha + noise
    return asset_returns, factor_returns, factor_means


def run_factor_model(asset_returns, factor_returns, factor_premia):
    """Estimate the factor model and report exposures."""
    model = FactorModel(asset_returns, factor_returns)
    exposures = model.factor_exposures()
    alphas = model.alphas()
    r_squared = model.r_squared()
    residual_vol = model.residual_volatility()
    expected_returns = model.expected_returns(factor_premia)

    print("=" * 76)
    print("Multi-factor regression diagnostics")
    print("=" * 76)
    for idx, (beta, alpha, r2, resid) in enumerate(zip(exposures, alphas, r_squared, residual_vol), start=1):
        print(
            f"Asset {idx}: betas={beta}, alpha={alpha:+.5f}, R^2={r2:.3f}, residual vol={resid:.4f}"
        )

    annual_expected = np.power(1.0 + expected_returns, 12) - 1.0
    print("\nAnnualised expected returns (factor implied):")
    for idx, (monthly, annual) in enumerate(zip(expected_returns, annual_expected), start=1):
        print(f"  Asset {idx}: {monthly*100:6.3f}% per month  |  {annual*100:6.2f}% per year")

    equal_weights = np.full(expected_returns.shape[0], 1.0 / expected_returns.shape[0])
    eq_return = model.portfolio_expected_return(equal_weights, factor_premia)
    eq_factors = model.portfolio_factor_exposure(equal_weights)
    eq_attribution = model.factor_attribution(equal_weights, factor_premia)

    print("\nEqual-weight portfolio diagnostics:")
    print(f"  Expected return (monthly): {eq_return*100:6.3f}%")
    print(f"  Factor exposures         : {eq_factors}")
    print(f"  Factor attribution       : {eq_attribution}")

    return model, expected_returns, annual_expected


def optimize_portfolio(asset_returns, monthly_rf, max_weight=0.5):
    """Solve for the maximum Sharpe portfolio with position limits."""
    optimizer = PortfolioOptimizer(asset_returns)
    bounds = np.full(asset_returns.shape[1], max_weight)
    result = optimizer.optimize(risk_free_rate=monthly_rf, max_weights=bounds)

    annual_return = np.power(1.0 + result["expected_return"], 12) - 1.0
    annual_vol = result["volatility"] * np.sqrt(12.0)

    print("\n" + "=" * 76)
    print("Constrained maximum Sharpe portfolio")
    print("=" * 76)
    print(f"Weights                : {result['weights']}")
    print(f"Expected return (ann.) : {annual_return*100:6.2f}%")
    print(f"Volatility (ann.)      : {annual_vol*100:6.2f}%")
    print(f"Sharpe ratio           : {result['sharpe_ratio']:.3f}")

    return result


def black_litterman_update(asset_returns, prior_weights, posterior_view=0.006):
    """Apply Black-Litterman with simple relative and absolute views."""
    monthly_cov = np.cov(asset_returns, rowvar=False)
    annual_cov = monthly_cov * 12.0

    bl = BlackLittermanModel(prior_weights, annual_cov, tau=0.05, risk_aversion=3.0)

    pick = np.array([
        [1.0, -1.0, 0.0, 0.0],   # Asset 1 expected to outperform Asset 2 by 60 bps annually
        [0.0, 0.0, 0.0, 1.0],    # Absolute view on Asset 4's expected return
    ])
    q = np.array([posterior_view, 0.05])  # annualised view returns
    uncertainty = np.diag([0.0009, 0.0025])
    views = InvestorViews(pick, q).with_uncertainty(uncertainty)

    posterior = bl.posterior(views)

    print("\n" + "=" * 76)
    print("Black-Litterman posterior allocation")
    print("=" * 76)
    print(f"Equilibrium returns (ann.): {posterior['equilibrium_returns']}")
    print(f"Posterior returns (ann.)  : {posterior['posterior_returns']}")
    print(f"Optimal weights           : {posterior['optimal_weights']}")

    return posterior


def main():
    asset_returns, factor_returns, factor_premia = simulate_monthly_data()
    run_factor_model(asset_returns, factor_returns, factor_premia)
    monthly_risk_free = 0.03 / 12.0
    optimize_portfolio(asset_returns, monthly_risk_free)

    market_weights = np.array([0.40, 0.25, 0.20, 0.15])
    black_litterman_update(asset_returns, market_weights)
    print("\n" + "=" * 76)
    print("Factor-based allocation workflow complete")
    print("=" * 76)


if __name__ == "__main__":
    main()
