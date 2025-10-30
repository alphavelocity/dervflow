Parameter Reference
===================

This page provides detailed information about common parameters used throughout dervflow's API.

.. contents:: Table of Contents
   :local:
   :depth: 2

Option Pricing Parameters
--------------------------

spot
~~~~

:Type: float
:Required: Yes
:Constraints: Must be positive (> 0)
:Description: Current price of the underlying asset

The spot price represents the current market price of the underlying asset (stock, index, commodity, etc.).
It is the reference point from which option payoffs are calculated.

**Examples:**

* Stock: $100.00
* Index: 4500.00
* FX rate: 1.2500

**Common Issues:**

* Negative or zero values will raise ``ValueError``
* Very small values (< 0.01) may cause numerical instability in some calculations

strike
~~~~~~

:Type: float
:Required: Yes
:Constraints: Must be positive (> 0)
:Description: Strike price (exercise price) of the option

The strike price is the predetermined price at which the option holder can buy (call) or sell (put)
the underlying asset upon exercise.

**Moneyness:**

* **In-the-Money (ITM)**: Call when spot > strike, Put when spot < strike
* **At-the-Money (ATM)**: spot ≈ strike
* **Out-of-the-Money (OTM)**: Call when spot < strike, Put when spot > strike

**Examples:**

* ATM option: strike = 100.00, spot = 100.00
* 5% OTM call: strike = 105.00, spot = 100.00
* 10% ITM put: strike = 110.00, spot = 100.00

rate
~~~~

:Type: float
:Required: Yes
:Constraints: Can be any real number (including negative)
:Units: Annualized decimal (e.g., 0.05 for 5%)
:Description: Risk-free interest rate

The risk-free rate represents the theoretical return on an investment with zero risk, typically
approximated by government bond yields.

**Common Values:**

* US Treasury rates: 0.01 to 0.05 (1% to 5%)
* Negative rates (some European bonds): -0.005 to 0.00
* High inflation environments: 0.05 to 0.15

**Important Notes:**

* Must be continuously compounded rate
* Should match the currency of the underlying asset
* For multi-currency options, use the domestic risk-free rate

dividend
~~~~~~~~

:Type: float
:Required: Yes
:Constraints: Can be any real number (typically non-negative)
:Units: Annualized decimal (e.g., 0.02 for 2%)
:Description: Continuous dividend yield

The dividend yield represents the expected dividend payments as a percentage of the stock price.

**Calculation:**

For discrete dividends, convert to continuous yield:

.. math::

   q = -\frac{1}{T} \ln\left(1 - \frac{D}{S_0}\right)

where D is the present value of dividends and T is time to maturity.

**Common Values:**

* Growth stocks: 0.00 to 0.02 (0% to 2%)
* Value stocks: 0.02 to 0.05 (2% to 5%)
* Indices: 0.015 to 0.03 (1.5% to 3%)
* Commodities/FX: 0.00 (no dividends)

**Special Cases:**

* For FX options, use foreign risk-free rate as dividend yield
* For futures options, set dividend = rate (cost of carry)

volatility
~~~~~~~~~~

:Type: float
:Required: Yes
:Constraints: Must be positive (> 0)
:Units: Annualized decimal (e.g., 0.20 for 20%)
:Description: Volatility (standard deviation) of returns

Volatility measures the degree of variation in the underlying asset's returns. It is the most
important parameter for option pricing and the only one not directly observable in the market.

**Typical Ranges:**

* Low volatility (bonds, large-cap stocks): 0.10 to 0.20 (10% to 20%)
* Medium volatility (most stocks): 0.20 to 0.40 (20% to 40%)
* High volatility (small-cap, emerging markets): 0.40 to 0.80 (40% to 80%)
* Very high volatility (cryptocurrencies, biotech): 0.80 to 2.00 (80% to 200%)

**Historical vs Implied:**

* **Historical volatility**: Calculated from past price data
* **Implied volatility**: Derived from market option prices (forward-looking)

**Volatility Smile/Skew:**

In practice, volatility varies by strike and maturity:

* **Smile**: Higher volatility for OTM options (both calls and puts)
* **Skew**: Higher volatility for OTM puts (equity markets)

time
~~~~

:Type: float
:Required: Yes
:Constraints: Must be positive (> 0)
:Units: Years (decimal)
:Description: Time to maturity (expiration)

Time to maturity represents the remaining life of the option.

**Conversion Formulas:**

.. code-block:: python

   # Days to years
   time_years = days / 365.0  # or 365.25 for leap years

   # Business days to years
   time_years = business_days / 252.0  # typical trading days

   # Months to years
   time_years = months / 12.0

   # From datetime
   from datetime import datetime
   expiry = datetime(2024, 12, 31)
   today = datetime.now()
   time_years = (expiry - today).days / 365.0

**Examples:**

* 1 month: 0.0833 (1/12)
* 3 months: 0.25
* 6 months: 0.5
* 1 year: 1.0
* 1 week: 0.0192 (7/365)
* 1 day: 0.00274 (1/365)

**Important Notes:**

* Very short times (< 0.01 years ≈ 3.65 days) may cause numerical issues
* Time decay (theta) accelerates as expiration approaches
* American options become more valuable with longer time to maturity

option_type
~~~~~~~~~~~

:Type: str
:Required: Yes
:Constraints: Must be 'call' or 'put' (case-insensitive)
:Description: Type of option

**Call Option:**

* Right to **buy** the underlying at the strike price
* Payoff: max(S_T - K, 0)
* Profits from price increases
* Unlimited upside potential

**Put Option:**

* Right to **sell** the underlying at the strike price
* Payoff: max(K - S_T, 0)
* Profits from price decreases
* Maximum profit: K (when S_T = 0)

**Examples:**

.. code-block:: python

   # Valid option types
   option_type = 'call'
   option_type = 'Call'
   option_type = 'CALL'
   option_type = 'put'
   option_type = 'Put'
   option_type = 'PUT'

   # Invalid option types (will raise ValueError)
   option_type = 'c'
   option_type = 'p'
   option_type = 'option'

Portfolio Parameters
--------------------

weights
~~~~~~~

:Type: array-like (list, numpy.ndarray)
:Required: Yes
:Constraints: Must sum to 1.0 (within tolerance)
:Description: Portfolio allocation weights

Portfolio weights represent the proportion of capital allocated to each asset.

**Properties:**

* Sum must equal 1.0 (fully invested)
* Can be negative (short positions)
* Typically between -1.0 and 1.0 for each asset

**Examples:**

.. code-block:: python

   import numpy as np

   # Equal weight portfolio (3 assets)
   weights = np.array([1/3, 1/3, 1/3])

   # Long-short portfolio
   weights = np.array([0.6, 0.5, -0.1])  # 60% long, 50% long, 10% short

   # Concentrated portfolio
   weights = np.array([0.5, 0.3, 0.2])  # 50%, 30%, 20%

expected_returns
~~~~~~~~~~~~~~~~

:Type: numpy.ndarray
:Required: Yes
:Constraints: 1D array, length = number of assets
:Units: Annualized decimal
:Description: Expected returns for each asset

Expected returns represent the anticipated return for each asset over the investment horizon.

**Estimation Methods:**

1. **Historical mean**: Average of past returns
2. **CAPM**: Risk-free rate + beta × market risk premium
3. **Factor models**: Multi-factor expected return models
4. **Analyst forecasts**: Forward-looking estimates

**Example:**

.. code-block:: python

   import numpy as np

   # 5 assets with different expected returns
   expected_returns = np.array([0.08, 0.10, 0.12, 0.09, 0.11])
   # 8%, 10%, 12%, 9%, 11% annual returns

covariance
~~~~~~~~~~

:Type: numpy.ndarray
:Required: Yes
:Constraints: 2D array, symmetric, positive semi-definite
:Units: Annualized variance
:Description: Covariance matrix of asset returns

The covariance matrix captures the variance of each asset and the covariances between assets.

**Properties:**

* Diagonal elements: variances (σ²)
* Off-diagonal elements: covariances (σᵢⱼ)
* Symmetric: cov[i,j] = cov[j,i]
* Positive semi-definite: all eigenvalues ≥ 0

**Estimation:**

.. code-block:: python

   import numpy as np

   # From historical returns (n_periods × n_assets)
   returns = np.random.randn(252, 5) * 0.01
   covariance = np.cov(returns.T) * 252  # Annualize

   # Check properties
   assert covariance.shape[0] == covariance.shape[1]  # Square
   assert np.allclose(covariance, covariance.T)  # Symmetric
   assert np.all(np.linalg.eigvals(covariance) >= -1e-10)  # PSD

Monte Carlo Parameters
----------------------

num_paths
~~~~~~~~~

:Type: int
:Required: Yes
:Constraints: Must be positive (> 0)
:Description: Number of simulation paths

The number of paths determines the accuracy of Monte Carlo estimates. More paths provide
better accuracy but require more computation time.

**Accuracy:**

Standard error decreases with √n:

.. math::

   SE = \frac{\sigma}{\sqrt{n}}

where σ is the standard deviation of payoffs and n is num_paths.

**Recommended Values:**

* Quick estimate: 10,000 paths
* Standard accuracy: 100,000 paths
* High accuracy: 1,000,000 paths
* Production/research: 10,000,000+ paths

**Trade-offs:**

* 10x more paths → 3.16x better accuracy
* 100x more paths → 10x better accuracy
* Computation time scales linearly with num_paths

num_steps
~~~~~~~~~

:Type: int
:Required: Yes (for path-dependent options)
:Constraints: Must be positive (> 0)
:Description: Number of time steps in simulation

The number of time steps determines how finely the continuous-time process is discretized.

**Recommended Values:**

* European options: 1 step (only terminal value needed)
* American options: 50-100 steps
* Path-dependent options: 252 steps (daily monitoring)
* Barrier options: 252-1000 steps (frequent monitoring)

**Discretization Error:**

Error decreases with step size (Δt = T/num_steps):

* Euler scheme: O(Δt)
* Milstein scheme: O(Δt²)

antithetic
~~~~~~~~~~

:Type: bool
:Required: No (default: False)
:Description: Use antithetic variates for variance reduction

Antithetic variates reduce variance by using negated random numbers, effectively doubling
the number of paths with minimal additional computation.

**Variance Reduction:**

Can reduce variance by up to 50% for symmetric payoffs.

**Example:**

.. code-block:: python

   import dervflow

   mc = dervflow.MonteCarloEngine()

   # Without variance reduction
   result1 = mc.simulate_gbm(100, 0.1, 0.2, 1.0, 252, 10000, antithetic=False)

   # With variance reduction (same accuracy with fewer paths)
   result2 = mc.simulate_gbm(100, 0.1, 0.2, 1.0, 252, 5000, antithetic=True)

Risk Metrics Parameters
-----------------------

confidence
~~~~~~~~~~

:Type: float
:Required: Yes
:Constraints: Must be between 0 and 1
:Description: Confidence level for VaR/CVaR calculation

The confidence level represents the probability that losses will not exceed the VaR estimate.

**Common Values:**

* 0.90 (90%): Regulatory minimum for some applications
* 0.95 (95%): Standard for risk management
* 0.99 (99%): Conservative, used by banks
* 0.999 (99.9%): Extreme risk assessment

**Interpretation:**

* 95% confidence: 5% chance of exceeding VaR
* 99% confidence: 1% chance of exceeding VaR (1 in 100 days)
* 99.9% confidence: 0.1% chance of exceeding VaR (1 in 1000 days)

method
~~~~~~

:Type: str
:Required: No (default varies by function)
:Description: Calculation method to use

Different methods have different trade-offs in terms of accuracy, assumptions, and computation time.

**VaR Methods:**

* **'historical'**: Non-parametric, no distribution assumptions, requires sufficient data
* **'parametric'**: Assumes normal distribution, fast, may underestimate tail risk
* **'monte_carlo'**: Flexible, can handle complex portfolios, computationally intensive

**Greeks Methods:**

* **'analytical'**: Exact formulas, fast, only available for some models
* **'numerical'**: Finite differences, works for any model, slower, approximation error

**Interpolation Methods:**

* **'linear'**: Simple, fast, not smooth
* **'cubic_spline'**: Smooth, good for most applications
* **'nelson_siegel'**: Parametric, enforces shape constraints

See Also
--------

* :doc:`complete_reference` - Complete API reference
* :doc:`examples` - Usage examples
* :doc:`../user_guide/quickstart` - Getting started guide
