Value at Risk (VaR)
===================

Value at Risk (VaR) is a statistical measure of the potential loss in value of a portfolio over a defined period for a given confidence interval.

Definition
----------

VaR answers the question: "What is the maximum loss over a given time horizon at a given confidence level?"

Formally, VaR at confidence level :math:`\alpha` is defined as:

.. math::

   \text{VaR}_\alpha = \inf\{x : P(L \leq x) \geq \alpha\}

where :math:`L` is the loss distribution.

Equivalently, for a return distribution:

.. math::

   P(R \leq -\text{VaR}_\alpha) = 1 - \alpha

**Example:** A 1-day 95% VaR of $1 million means there is a 5% chance that the portfolio will lose more than $1 million in one day.

VaR Calculation Methods
-----------------------

1. Historical Simulation
~~~~~~~~~~~~~~~~~~~~~~~~

The historical simulation method uses actual historical returns to estimate VaR.

**Algorithm:**

1. Collect historical returns :math:`r_1, r_2, \ldots, r_n`
2. Sort returns in ascending order
3. VaR is the :math:`(1-\alpha)`-th quantile

.. math::

   \text{VaR}_\alpha = -r_{(k)}

where :math:`k = \lfloor n(1-\alpha) \rfloor` and :math:`r_{(k)}` is the :math:`k`-th order statistic.

**Advantages:**

* No distributional assumptions
* Captures fat tails and skewness
* Easy to understand and implement

**Disadvantages:**

* Requires large historical dataset
* Assumes future will resemble past
* Sensitive to outliers
* Cannot model scenarios not in historical data

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   # Historical returns
   returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.00,
                       0.01, 0.02, 0.03, 0.04, 0.05])

   risk_metrics = dervflow.RiskMetrics()
   var_95 = risk_metrics.var(returns, confidence=0.95, method='historical')
   print(f"95% VaR: {var_95:.2%}")

2. Parametric (Variance-Covariance) Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assumes returns follow a normal distribution.

For a portfolio with return :math:`R_p` and standard deviation :math:`\sigma_p`:

.. math::

   \text{VaR}_\alpha = -(\mu_p - z_\alpha \sigma_p)

where :math:`z_\alpha` is the :math:`(1-\alpha)`-th quantile of the standard normal distribution.

For a 95% confidence level, :math:`z_{0.95} = 1.645`.

**For a portfolio of assets:**

.. math::

   \sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}

where :math:`\mathbf{w}` is the vector of portfolio weights and :math:`\Sigma` is the covariance matrix.

**Advantages:**

* Fast computation
* Requires less data than historical simulation
* Analytically tractable

**Disadvantages:**

* Assumes normal distribution (underestimates tail risk)
* Cannot capture skewness and kurtosis
* Poor for non-linear instruments (options)

**Cornish-Fisher Expansion:**

To account for skewness and kurtosis:

.. math::

   z_\alpha^* = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)(K-3) - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2

where :math:`S` is skewness and :math:`K` is kurtosis.

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   returns = np.random.randn(1000) * 0.02  # 2% daily volatility

   risk_metrics = dervflow.RiskMetrics()
   var_95 = risk_metrics.var(returns, confidence=0.95, method='parametric')
   print(f"95% VaR (Parametric): {var_95:.2%}")

3. Monte Carlo Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate future portfolio values using stochastic models.

**Algorithm:**

1. Model asset price dynamics (e.g., GBM)
2. Simulate :math:`N` price paths
3. Calculate portfolio value for each path
4. Compute returns distribution
5. VaR is the :math:`(1-\alpha)`-th quantile

**Advantages:**

* Handles non-linear instruments (options, derivatives)
* Can model complex dependencies
* Flexible for various distributions and scenarios

**Disadvantages:**

* Computationally intensive
* Requires model specification
* Model risk (wrong model assumptions)

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   # Simulate returns using Monte Carlo
   mc_engine = dervflow.MonteCarloEngine()

   # Simulate GBM paths
   paths = mc_engine.simulate_gbm(
       s0=100, mu=0.05, sigma=0.2,
       T=1/252, steps=1, paths=10000
   )

   returns = (paths[-1, :] - 100) / 100

   risk_metrics = dervflow.RiskMetrics()
   var_95 = risk_metrics.var(returns, confidence=0.95, method='monte_carlo')
   print(f"95% VaR (Monte Carlo): {var_95:.2%}")

Conditional Value at Risk (CVaR)
---------------------------------

Also known as Expected Shortfall (ES), CVaR measures the expected loss given that the loss exceeds VaR.

.. math::

   \text{CVaR}_\alpha = E[L | L \geq \text{VaR}_\alpha]

**Properties:**

* CVaR is always greater than or equal to VaR
* CVaR is a coherent risk measure (VaR is not)
* Provides information about tail risk beyond VaR

**Calculation:**

For historical simulation:

.. math::

   \text{CVaR}_\alpha = \frac{1}{n(1-\alpha)} \sum_{i: r_i \leq -\text{VaR}_\alpha} |r_i|

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   returns = np.random.randn(1000) * 0.02

   risk_metrics = dervflow.RiskMetrics()
   var_95 = risk_metrics.var(returns, confidence=0.95)
   cvar_95 = risk_metrics.cvar(returns, confidence=0.95)

   print(f"95% VaR: {var_95:.2%}")
   print(f"95% CVaR: {cvar_95:.2%}")

VaR for Options and Non-Linear Instruments
-------------------------------------------

For portfolios containing options, the parametric method is inadequate due to non-linearity.

Delta-Normal VaR
~~~~~~~~~~~~~~~~

Approximate using first-order Taylor expansion:

.. math::

   \Delta V \approx \Delta \cdot \Delta S

.. math::

   \text{VaR} = |\Delta| \cdot \text{VaR}_{\text{underlying}}

**Limitation:** Ignores gamma and higher-order effects.

Delta-Gamma VaR
~~~~~~~~~~~~~~~

Include second-order term:

.. math::

   \Delta V \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2

The distribution of :math:`\Delta V` is no longer normal due to the quadratic term.

**Cornish-Fisher approximation** or **Monte Carlo simulation** is typically used.

Full Revaluation
~~~~~~~~~~~~~~~~

Most accurate method: reprice the entire portfolio for each scenario.

.. code-block:: python

   import dervflow
   import numpy as np

   # Portfolio with options
   bs_model = dervflow.BlackScholesModel()

   # Current portfolio value
   spot = 100
   option_price = bs_model.price(spot, 100, 0.05, 0, 0.25, 1, 'call')
   portfolio_value = 100 * option_price  # 100 options

   # Simulate spot price scenarios
   scenarios = np.random.lognormal(
       mean=np.log(spot) - 0.5 * 0.25**2 / 252,
       sigma=0.25 / np.sqrt(252),
       size=10000
   )

   # Reprice portfolio for each scenario
   portfolio_values = np.array([
       100 * bs_model.price(s, 100, 0.05, 0, 0.25, 1 - 1/252, 'call')
       for s in scenarios
   ])

   returns = (portfolio_values - portfolio_value) / portfolio_value
   var_95 = -np.percentile(returns, 5)
   print(f"95% VaR: {var_95:.2%}")

Time Scaling
------------

VaR is often scaled across different time horizons using the square root of time rule:

.. math::

   \text{VaR}_T = \text{VaR}_1 \cdot \sqrt{T}

where :math:`\text{VaR}_1` is the 1-day VaR and :math:`T` is the number of days.

**Assumptions:**

* Returns are i.i.d.
* No autocorrelation
* Constant volatility

**Example:**

.. code-block:: python

   var_1day = 0.02  # 2% daily VaR
   var_10day = var_1day * np.sqrt(10)
   print(f"10-day VaR: {var_10day:.2%}")

Backtesting VaR
---------------

Backtesting validates VaR models by comparing predicted VaR with actual losses.

**Violation Ratio:**

.. math::

   \text{Violation Ratio} = \frac{\text{Number of VaR breaches}}{\text{Total observations}}

For a well-calibrated model at confidence level :math:`\alpha`, the violation ratio should be approximately :math:`1 - \alpha`.

**Kupiec Test:**

Tests whether the number of violations is consistent with the confidence level.

Test statistic:

.. math::

   LR = -2 \ln\left[\frac{(1-\alpha)^{T-N}\alpha^N}{(1-N/T)^{T-N}(N/T)^N}\right]

where :math:`N` is the number of violations and :math:`T` is the total number of observations.

Under the null hypothesis, :math:`LR \sim \chi^2(1)`.

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np
   from scipy import stats

   # Simulate returns and VaR predictions
   returns = np.random.randn(250) * 0.02
   var_predictions = np.full(250, 0.033)  # 95% VaR prediction

   # Count violations
   violations = np.sum(returns < -var_predictions)
   violation_ratio = violations / len(returns)

   print(f"Violations: {violations} ({violation_ratio:.2%})")
   print(f"Expected: {0.05 * len(returns):.0f} (5%)")

   # Kupiec test
   T = len(returns)
   N = violations
   alpha = 0.05

   if N > 0 and N < T:
       LR = -2 * (np.log((1-alpha)**(T-N) * alpha**N) -
                  np.log((1-N/T)**(T-N) * (N/T)**N))
       p_value = 1 - stats.chi2.cdf(LR, df=1)
       print(f"Kupiec test p-value: {p_value:.4f}")

Limitations of VaR
------------------

1. **Not a coherent risk measure:** VaR is not subadditive, meaning:

   .. math::

      \text{VaR}(X + Y) \not\leq \text{VaR}(X) + \text{VaR}(Y)

   This can discourage diversification.

2. **Ignores tail risk:** VaR only provides a threshold, not information about losses beyond that threshold.

3. **Model risk:** Results depend heavily on assumptions (distribution, parameters, historical period).

4. **Procyclicality:** VaR can amplify market volatility during crises.

**Alternatives:**

* **CVaR (Expected Shortfall):** Coherent and captures tail risk
* **Stress testing:** Scenario-based analysis
* **Maximum drawdown:** Peak-to-trough decline

See Also
--------

* :doc:`../api/risk` - VaR calculation API
* :doc:`../user_guide/risk_analytics` - Practical risk management
* :doc:`portfolio_theory` - Portfolio risk measures
