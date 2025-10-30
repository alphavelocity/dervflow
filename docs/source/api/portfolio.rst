Portfolio Optimisation API
==========================

.. currentmodule:: dervflow

The portfolio toolkit combines mean-variance optimisation, risk-parity solving,
Black-Litterman blending and factor modelling utilities. All classes are exposed
at the top level of :mod:`dervflow`.

PortfolioOptimizer
------------------

.. autoclass:: PortfolioOptimizer
   :members: optimize, efficient_frontier, risk_contributions, portfolio_return, portfolio_volatility, sharpe_ratio, conditional_value_at_risk, value_at_risk, portfolio_summary
   :show-inheritance:

Initialise :class:`PortfolioOptimizer` with either a returns matrix or an
expected returns vector plus covariance matrix. Optimisation supports target
return, risk and leverage constraints. Helper methods evaluate portfolio return,
volatility, risk contributions and risk-adjusted ratios for any weight vector.

.. code-block:: python

   import numpy as np
   import dervflow

   returns = np.random.normal(0.001, 0.02, size=(252, 4))
   optimizer = dervflow.PortfolioOptimizer(returns)

   min_var = optimizer.optimize()
   target = optimizer.optimize(target_return=0.12, min_weights=np.zeros(4), max_weights=np.full(4, 0.5))
   sharpe = optimizer.sharpe_ratio(min_var["weights"], risk_free_rate=0.02)
   frontier = optimizer.efficient_frontier(num_points=20)

RiskParityOptimizer
-------------------

.. autoclass:: RiskParityOptimizer
   :members: optimize, risk_contributions
   :show-inheritance:

Risk parity weights are obtained by solving for equal or custom risk
contributions. :meth:`optimize` accepts target contribution vectors, while
:meth:`risk_contributions` evaluates the contribution of an existing weight set.

BlackLittermanModel
-------------------

.. autoclass:: BlackLittermanModel
   :members: equilibrium_returns, posterior
   :show-inheritance:

Implements the Black-Litterman framework for combining market equilibrium
returns with investor views. :meth:`equilibrium_returns` derives the implied
returns from a covariance matrix and market capitalisation weights.
:meth:`posterior` updates the equilibrium returns with user supplied views (see
:class:`InvestorViews`).

InvestorViews
-------------

.. autoclass:: InvestorViews
   :members: pick_matrix, view_returns, with_uncertainty
   :show-inheritance:

Constructs and manipulates view matrices for the Black-Litterman model. Use
:meth:`pick_matrix` to build pick matrices, :meth:`view_returns` for the view
vector and :meth:`with_uncertainty` to attach view covariance information.

FactorModel
-----------

.. autoclass:: FactorModel
   :members: factor_names, n_assets, n_factors, n_observations, include_intercept, alphas, expected_returns, factor_exposures, factor_attribution, portfolio_expected_return, portfolio_factor_exposure, r_squared, residual_variance, residual_volatility
   :show-inheritance:

The factor model estimates exposures and idiosyncratic risk relative to a set of
factor returns. After fitting, call :meth:`factor_attribution` or
:meth:`portfolio_factor_exposure` to decompose portfolio performance and risk.

See Also
--------

* :doc:`../user_guide/portfolio_optimization` – optimisation walkthroughs
* :doc:`../theory/portfolio_theory` – theoretical foundations for the models
