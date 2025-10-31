<h1 align="center">

DervFlow

</h1>

<h3 align="center">

High-performance Mathematics & Quantitative Finance Toolkit

</h3>

DervFlow delivers production-ready option pricers, risk analytics, portfolio construction utilities, time-series diagnostics, and yield-curve analytics backed by rigorously tested numerical kernels.

<div align="center">

[![Current Release](https://img.shields.io/github/release/neuralsorcerer/dervflow.svg)](https://github.com/neuralsorcerer/dervflow/releases)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![rustc 1.85+](https://img.shields.io/badge/rustc-1.85+-blue.svg?logo=rust&logoColor=white)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Test Linux](https://github.com/neuralsorcerer/dervflow/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/dervflow/actions/workflows/test_ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/dervflow/actions/workflows/test_windows.yml/badge.svg)](https://github.com/neuralsorcerer/dervflow/actions/workflows/test_windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/dervflow/actions/workflows/test_macos.yml/badge.svg)](https://github.com/neuralsorcerer/dervflow/actions/workflows/test_macos.yml?query=branch%3Amain)
[![Lints](https://github.com/neuralsorcerer/dervflow/actions/workflows/lints.yml/badge.svg)](https://github.com/neuralsorcerer/dervflow/actions/workflows/lints.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

> [!CAUTION]
> Active development. APIs may evolve before a 1.0 release.

## Highlights
- **Options pricing** – Black-Scholes-Merton analytics, implied volatility solvers, binomial trees, Longstaff–Schwartz Monte Carlo, and exotic payoffs (Asian, barrier, lookback, digital) exposed through ergonomic Python classes.
- **Risk analytics** – Portfolio Greeks aggregation, historical/parametric/Monte Carlo VaR and CVaR, drawdown analytics, and performance ratios.
- **Portfolio construction** – Mean-variance optimisation with constraints, efficient frontiers, risk parity allocation, and Black–Litterman blending of market and investor views.
- **Yield curves** – Bootstrapping from bonds or swaps, Nelson–Siegel(-Svensson) parametrisations, zero/forward/discount curve queries, and bond analytics (duration, convexity, DV01).
- **Time series** – Return transformations, rolling and EW statistics, autocorrelation diagnostics, hypothesis tests, and GARCH-family volatility models.
- **Monte Carlo** – High-throughput stochastic process simulators (GBM, Ornstein–Uhlenbeck, CIR, Vasicek) with optional parallel path generation.
- **Numerical kernels** – Deterministic integrators, root-finders, optimisers, quasi-random generators, and dense linear-algebra helpers implemented in Rust for safety and speed.


## Python quick start
The package exposes the compiled extension as `dervflow`.

### Option pricing and risk
```python
from dervflow import BlackScholesModel, MonteCarloOptionPricer, RiskMetrics
import numpy as np

# Analytical Black–Scholes pricing with Greeks
bs = BlackScholesModel()
call_price = bs.price(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    dividend=0.02,
    volatility=0.20,
    time=1.0,
    option_type="call",
)
call_greeks = bs.greeks(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "call")
print(f"Call price: {call_price:.2f}")
print(f"Delta: {call_greeks['delta']:.4f}, Vega: {call_greeks['vega']:.4f}")

# Monte Carlo pricing with antithetic variates and parallel execution
mc = MonteCarloOptionPricer()
mc_result = mc.price_european(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    dividend=0.02,
    volatility=0.20,
    time=1.0,
    option_type="call",
    num_paths=100_000,
    use_antithetic=True,
    seed=7,
    parallel=True,
)
print(f"MC price: {mc_result['price']:.2f} ± {mc_result['std_error']:.4f}")

# Historical Value at Risk on simulated P&L
rng = np.random.default_rng(seed=123)
pl = rng.normal(0.0, 0.02, size=10_000)
risk = RiskMetrics()
var = risk.var(pl, confidence_level=0.95, method="historical")
print(f"95% VaR: {var['var']:.4%}")
```

### Monte Carlo simulation
```python
from dervflow import MonteCarloEngine
import matplotlib.pyplot as plt

engine = MonteCarloEngine(seed=42)
paths = engine.simulate_gbm(
    s0=100.0,
    mu=0.05,
    sigma=0.20,
    T=1.0,
    steps=252,
    paths=2_000,
    parallel=True,
)

plt.figure(figsize=(10, 6))
plt.plot(paths[:10].T)
plt.title("Sample GBM paths")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.tight_layout()
plt.show()
```

### Portfolio optimisation
```python
from dervflow import PortfolioOptimizer
import numpy as np

rng = np.random.default_rng(seed=0)
# Daily returns for four assets (252 trading days)
returns = rng.normal(loc=0.0005, scale=0.015, size=(252, 4))

optimizer = PortfolioOptimizer(returns)
min_weights = np.zeros(returns.shape[1])
max_weights = np.full(returns.shape[1], 0.6)

result = optimizer.optimize(
    target_return=float(returns.mean(axis=0).mean()),
    min_weights=min_weights,
    max_weights=max_weights,
)
print("Optimal weights:", result["weights"])
print(f"Expected return: {result['expected_return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")

frontier = optimizer.efficient_frontier(num_points=10, min_weights=min_weights, max_weights=max_weights)
for point in frontier[:3]:
    print(
        f"Frontier portfolio → return: {point['expected_return']:.2%}, volatility: {point['volatility']:.2%}"
    )
```

### Yield curves and fixed income
```python
from dervflow import YieldCurve, YieldCurveBuilder, BondAnalytics
import numpy as np

# Interpolate zero rates
times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
rates = np.array([0.020, 0.024, 0.028, 0.034, 0.038])
curve = YieldCurve(times, rates, method="cubic_spline_natural")
print(f"5Y discount factor: {curve.discount_factor(5.0):.6f}")

# Bootstrap from bond quotes (maturity, coupon, clean price, frequency)
bonds = [
    (0.5, 0.02, 99.50, 2),
    (1.0, 0.025, 99.80, 2),
    (2.0, 0.030, 100.20, 2),
    (5.0, 0.035, 101.10, 2),
]
builder = YieldCurveBuilder()
bootstrapped = builder.bootstrap_from_bonds(bonds)
print(f"2Y zero rate: {bootstrapped.zero_rate(2.0):.4%}")

analytics = BondAnalytics()
duration = analytics.duration(
    yield_rate=0.032,
    coupon_rate=0.03,
    years_to_maturity=5.0,
    frequency=2,
    duration_type="modified",
)
print(f"Modified duration: {duration:.4f}")
```

### Time-series analytics
```python
from dervflow import TimeSeriesAnalyzer
import numpy as np

prices = np.array([100.0, 102.0, 101.5, 103.0, 104.5, 103.8, 105.2])
analyzer = TimeSeriesAnalyzer(prices)
log_returns = analyzer.returns(method="log")
summary = analyzer.stat()
print(f"Mean: {summary['mean']:.5f}, Std Dev: {summary['std_dev']:.5f}")
acf = analyzer.autocorrelation(max_lag=5)
print("Autocorrelation (lags 0–5):", acf)
```

## Rust crate usage
The Rust crate can be used directly without the Python bindings. Enable the domains you need via feature flags:

```rust
use dervflow::options::analytical::black_scholes_price;
use dervflow::common::types::{OptionParams, OptionType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = OptionParams::new(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call);
    let price = black_scholes_price(&params)?;
    println!("Call price: {:.2}", price);
    Ok(())
}
```

When depending on DervFlow from another Rust project, disable the default Python bindings and opt into the domains you require:

```toml
[dependencies]
dervflow = { path = "../dervflow", default-features = false, features = ["options", "numerical"] }
```

## Installation
### Python development workflow
1. Install Rust (1.74 or newer recommended) and Python 3.10–3.14.
2. Create/activate a virtual environment.
3. Install the build backend:
   ```bash
   pip install "maturin[patchelf]"
   ```
4. Build and install the extension in editable mode:
   ```bash
   maturin develop --release
   ```
   The command compiles the Rust sources with the `python` feature enabled and installs the `dervflow` package into the active environment.


### Using the Rust crate only
If you only need the Rust APIs, compile without the Python bindings:
```bash
cargo build --no-default-features --features core
```
You can further narrow the domains (for example `--features options,portfolio`) depending on your needs.

## Feature flags
DervFlow exposes granular Cargo features:

| Feature | Enables |
| --- | --- |
| `python` | PyO3 bindings (`pyo3`, `numpy`) for the Python extension module. Enabled by default. |
| `core` | Umbrella that enables every domain-specific module listed below. Enabled by default. |
| `numerical` | Foundational numerical routines (integration, optimisation, random generation). |
| `options` | Option pricing engines and volatility analytics. Depends on `numerical`. |
| `risk` | Greeks aggregation, VaR/CVaR, and portfolio risk metrics. Depends on `numerical`. |
| `portfolio` | Mean-variance optimisation, efficient frontier, risk parity, Black–Litterman. |
| `yield_curve` | Yield-curve bootstrapping, interpolation, and bond analytics. Depends on `numerical`. |
| `timeseries` | Returns, rolling statistics, correlation diagnostics, GARCH models. |
| `monte_carlo` | Stochastic process simulators and Monte Carlo pricing. Depends on `numerical`. |


## Development and testing
Run the full test suite locally before opening a pull request:
```bash
# Ensure PyO3 uses the intended interpreter
export PYO3_PYTHON="$(which python3)"

# Rust unit and integration tests (all domains + Python bindings)
cargo test --all-features

# Python tests (requires maturin develop)
pytest tests/python -v

# Linting and formatting
cargo fmt
cargo clippy --all-targets --all-features
black python/
mypy python/dervflow
```
If linking against CPython fails when building with the `python` feature, install the appropriate `python3.x-dev` package and export `PYO3_PYTHON` to the desired interpreter before invoking `cargo`.

## Citation
If you use dervflow in your work and wish to refer to it, please use the following BibTeX entry.
```bibtex
@software{dervflow,
  author = {Soumyadip Sarkar},
  title = {DervFlow: High-Performance Quantitative Finance Library},
  year = {2025},
  url = {https://github.com/neuralsorcerer/dervflow}
}
```

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
