Modern Portfolio Theory
=======================

Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952, provides a mathematical framework for constructing portfolios that optimize expected return for a given level of risk.

Expected Return and Risk
------------------------

Portfolio Return
~~~~~~~~~~~~~~~~

For a portfolio of :math:`n` assets with weights :math:`w_i` and expected returns :math:`\mu_i`:

.. math::

   \mu_p = \sum_{i=1}^n w_i \mu_i = \mathbf{w}^T \boldsymbol{\mu}

where :math:`\mathbf{w} = (w_1, \ldots, w_n)^T` and :math:`\boldsymbol{\mu} = (\mu_1, \ldots, \mu_n)^T`.

Portfolio Variance
~~~~~~~~~~~~~~~~~~

The portfolio variance is:

.. math::

   \sigma_p^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij} = \mathbf{w}^T \Sigma \mathbf{w}

where :math:`\Sigma` is the covariance matrix with elements :math:`\sigma_{ij} = \text{Cov}(R_i, R_j)`.

The portfolio standard deviation (volatility) is:

.. math::

   \sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}

Diversification Benefit
~~~~~~~~~~~~~~~~~~~~~~~

For two assets with correlation :math:`\rho`:

.. math::

   \sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\sigma_1\sigma_2

When :math:`\rho < 1`, the portfolio variance is less than the weighted average of individual variances, demonstrating the benefit of diversification.

Mean-Variance Optimization
---------------------------

The Optimization Problem
~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum Variance Portfolio:**

.. math::

   \min_{\mathbf{w}} \quad & \mathbf{w}^T \Sigma \mathbf{w} \\
   \text{subject to} \quad & \mathbf{w}^T \mathbf{1} = 1

**Target Return Portfolio:**

.. math::

   \min_{\mathbf{w}} \quad & \mathbf{w}^T \Sigma \mathbf{w} \\
   \text{subject to} \quad & \mathbf{w}^T \boldsymbol{\mu} = \mu_{\text{target}} \\
   & \mathbf{w}^T \mathbf{1} = 1

**Maximum Sharpe Ratio:**

.. math::

   \max_{\mathbf{w}} \quad & \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \Sigma \mathbf{w}}} \\
   \text{subject to} \quad & \mathbf{w}^T \mathbf{1} = 1

where :math:`r_f` is the risk-free rate.

Analytical Solution
~~~~~~~~~~~~~~~~~~~

For the minimum variance portfolio without constraints:

.. math::

   \mathbf{w}_{\text{min}} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}

For a target return :math:`\mu_{\text{target}}`:

.. math::

   \mathbf{w} = \frac{A\Sigma^{-1}\mathbf{1} - B\Sigma^{-1}\boldsymbol{\mu}}{AC - B^2}\mu_{\text{target}} + \frac{C\Sigma^{-1}\boldsymbol{\mu} - B\Sigma^{-1}\mathbf{1}}{AC - B^2}

where:

.. math::

   A &= \mathbf{1}^T \Sigma^{-1} \mathbf{1} \\
   B &= \mathbf{1}^T \Sigma^{-1} \boldsymbol{\mu} \\
   C &= \boldsymbol{\mu}^T \Sigma^{-1} \boldsymbol{\mu}

Efficient Frontier
------------------

The efficient frontier is the set of portfolios that offer the highest expected return for each level of risk.

Parametric Form
~~~~~~~~~~~~~~~

The efficient frontier can be parameterized as:

.. math::

   \mu_p(\lambda) &= \lambda \mu_A + (1-\lambda) \mu_B \\
   \sigma_p^2(\lambda) &= \lambda^2 \sigma_A^2 + (1-\lambda)^2 \sigma_B^2 + 2\lambda(1-\lambda)\rho_{AB}\sigma_A\sigma_B

where portfolios A and B are any two efficient portfolios.

Hyperbola in Mean-Variance Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The efficient frontier forms a hyperbola in :math:`(\sigma_p, \mu_p)` space:

.. math::

   \sigma_p^2 = \frac{C - 2B\mu_p + A\mu_p^2}{AC - B^2}

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np
   import matplotlib.pyplot as plt

   # Generate sample returns
   np.random.seed(42)
   returns = np.random.randn(252, 5) * 0.01 + 0.0005

   # Create optimizer
   optimizer = dervflow.PortfolioOptimizer(returns)

   # Calculate efficient frontier
   frontier = optimizer.efficient_frontier(num_points=50)

   # Plot
   risks = [p['volatility'] for p in frontier]
   returns_ef = [p['expected_return'] for p in frontier]

   plt.plot(risks, returns_ef, 'b-', linewidth=2)
   plt.xlabel('Volatility (Risk)')
   plt.ylabel('Expected Return')
   plt.title('Efficient Frontier')
   plt.grid(True)
   plt.show()

Capital Market Line (CML)
--------------------------

When a risk-free asset is available, investors can combine it with the market portfolio to achieve any desired risk-return profile.

The Capital Market Line is:

.. math::

   \mu_p = r_f + \frac{\mu_M - r_f}{\sigma_M} \sigma_p

where :math:`(\mu_M, \sigma_M)` is the market portfolio (tangency portfolio).

The tangency portfolio maximizes the Sharpe ratio:

.. math::

   \mathbf{w}_{\text{tan}} = \frac{\Sigma^{-1}(\boldsymbol{\mu} - r_f \mathbf{1})}{\mathbf{1}^T \Sigma^{-1}(\boldsymbol{\mu} - r_f \mathbf{1})}

Constraints
-----------

Real-world portfolios often have constraints:

Box Constraints
~~~~~~~~~~~~~~~

Limit individual asset weights:

.. math::

   l_i \leq w_i \leq u_i \quad \forall i

Common constraints:

* No short selling: :math:`w_i \geq 0`
* Maximum position: :math:`w_i \leq 0.2` (20% max)

Sector Constraints
~~~~~~~~~~~~~~~~~~

Limit exposure to sectors:

.. math::

   \sum_{i \in S_j} w_i \leq c_j

where :math:`S_j` is the set of assets in sector :math:`j`.

Turnover Constraints
~~~~~~~~~~~~~~~~~~~~

Limit portfolio changes:

.. math::

   \sum_{i=1}^n |w_i - w_i^{\text{old}}| \leq \tau

where :math:`\tau` is the maximum turnover.

Cardinality Constraints
~~~~~~~~~~~~~~~~~~~~~~~

Limit the number of assets:

.. math::

   \sum_{i=1}^n \mathbb{1}_{w_i \neq 0} \leq K

This is a mixed-integer programming problem.

**Example with Constraints:**

.. code-block:: python

   import dervflow
   import numpy as np

   returns = np.random.randn(252, 5) * 0.01 + 0.0005
   optimizer = dervflow.PortfolioOptimizer(returns)

   # Optimize with constraints
   result = optimizer.optimize(
       target_return=0.10,
       constraints={
           'min_weight': 0.0,      # No short selling
           'max_weight': 0.4,      # Max 40% per asset
           'sector_limits': {
               'tech': 0.5,        # Max 50% in tech
               'finance': 0.3      # Max 30% in finance
           }
       }
   )

   print(f"Optimal weights: {result['weights']}")
   print(f"Expected return: {result['expected_return']:.2%}")
   print(f"Volatility: {result['volatility']:.2%}")

Risk Parity
-----------

Risk parity allocates capital so that each asset contributes equally to portfolio risk.

Risk Contribution
~~~~~~~~~~~~~~~~~

The risk contribution of asset :math:`i` is:

.. math::

   RC_i = w_i \frac{\partial \sigma_p}{\partial w_i} = w_i \frac{(\Sigma \mathbf{w})_i}{\sigma_p}

Risk parity requires:

.. math::

   RC_i = \frac{\sigma_p}{n} \quad \forall i

or equivalently:

.. math::

   w_i (\Sigma \mathbf{w})_i = w_j (\Sigma \mathbf{w})_j \quad \forall i, j

**Naive Risk Parity (Inverse Volatility):**

.. math::

   w_i = \frac{1/\sigma_i}{\sum_{j=1}^n 1/\sigma_j}

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   returns = np.random.randn(252, 5) * 0.01 + 0.0005
   optimizer = dervflow.PortfolioOptimizer(returns)

   # Risk parity allocation
   result = optimizer.risk_parity()

   print(f"Risk parity weights: {result['weights']}")

   # Verify equal risk contribution
   for i, rc in enumerate(result['risk_contributions']):
       print(f"Asset {i+1} risk contribution: {rc:.4f}")

Black-Litterman Model
---------------------

The Black-Litterman model combines market equilibrium with investor views.

Market Equilibrium Returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implied equilibrium returns from market capitalization weights:

.. math::

   \boldsymbol{\Pi} = \lambda \Sigma \mathbf{w}_{\text{mkt}}

where :math:`\lambda` is the risk aversion coefficient.

Incorporating Views
~~~~~~~~~~~~~~~~~~~

Investor views are expressed as:

.. math::

   P\boldsymbol{\mu} = Q + \boldsymbol{\epsilon}

where:

* :math:`P` is the pick matrix (which assets the views concern)
* :math:`Q` is the vector of view returns
* :math:`\boldsymbol{\epsilon} \sim N(0, \Omega)` is the uncertainty in views

Posterior Returns
~~~~~~~~~~~~~~~~~

The posterior expected returns are:

.. math::

   E[\boldsymbol{\mu}] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\boldsymbol{\Pi} + P^T\Omega^{-1}Q]

Posterior Covariance
~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{Cov}[\boldsymbol{\mu}] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}

Performance Metrics
-------------------

Sharpe Ratio
~~~~~~~~~~~~

Risk-adjusted return:

.. math::

   \text{Sharpe} = \frac{\mu_p - r_f}{\sigma_p}

Sortino Ratio
~~~~~~~~~~~~~

Uses downside deviation instead of total volatility:

.. math::

   \text{Sortino} = \frac{\mu_p - r_f}{\sigma_{\text{downside}}}

where:

.. math::

   \sigma_{\text{downside}} = \sqrt{\frac{1}{n}\sum_{i=1}^n \min(r_i - r_f, 0)^2}

Information Ratio
~~~~~~~~~~~~~~~~~

Measures active return per unit of active risk:

.. math::

   \text{IR} = \frac{\mu_p - \mu_b}{\sigma_{p-b}}

where :math:`\mu_b` is the benchmark return and :math:`\sigma_{p-b}` is the tracking error.

Maximum Drawdown
~~~~~~~~~~~~~~~~

Largest peak-to-trough decline:

.. math::

   \text{MDD} = \max_{t \in [0,T]} \left[\max_{s \in [0,t]} V_s - V_t\right]

where :math:`V_t` is the portfolio value at time :math:`t`.

**Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   returns = np.random.randn(252) * 0.02 + 0.0005

   risk_metrics = dervflow.RiskMetrics()

   sharpe = risk_metrics.sharpe_ratio(returns, rf_rate=0.02)
   sortino = risk_metrics.sortino_ratio(returns, rf_rate=0.02)
   max_dd = risk_metrics.max_drawdown(returns)

   print(f"Sharpe Ratio: {sharpe:.2f}")
   print(f"Sortino Ratio: {sortino:.2f}")
   print(f"Maximum Drawdown: {max_dd:.2%}")

Practical Considerations
------------------------

Estimation Error
~~~~~~~~~~~~~~~~

Covariance matrix estimation is subject to error, especially with:

* Limited historical data
* High-dimensional portfolios
* Non-stationary returns

**Solutions:**

* Shrinkage estimators (Ledoit-Wolf)
* Factor models
* Robust optimization

Transaction Costs
~~~~~~~~~~~~~~~~~

Include transaction costs in optimization:

.. math::

   \min_{\mathbf{w}} \quad \mathbf{w}^T \Sigma \mathbf{w} + \kappa \sum_{i=1}^n |w_i - w_i^{\text{old}}|

where :math:`\kappa` is the transaction cost rate.

Rebalancing
~~~~~~~~~~~

Portfolios drift from optimal weights over time. Rebalancing strategies:

* **Calendar rebalancing:** Fixed intervals (monthly, quarterly)
* **Threshold rebalancing:** When weights deviate by a threshold
* **Volatility-based:** More frequent during high volatility

See Also
--------

* :doc:`../api/portfolio` - Portfolio optimization API
* :doc:`../user_guide/portfolio_optimization` - Practical portfolio construction
* :doc:`var` - Risk measurement
