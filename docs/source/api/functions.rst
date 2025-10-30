Function Reference
==================

This page documents all standalone functions available in dervflow.

.. contents:: Table of Contents
   :local:
   :depth: 2

Validation Functions
--------------------

validate_option_params
~~~~~~~~~~~~~~~~~~~~~~

.. function:: dervflow.utils.validate_option_params(spot, strike, rate, dividend, volatility, time)

   Validate option pricing parameters for correctness.

   :param float spot: Current spot price
   :param float strike: Strike price
   :param float rate: Risk-free interest rate
   :param float dividend: Dividend yield
   :param float volatility: Volatility (annualized)
   :param float time: Time to maturity (in years)
   :return: Tuple of (is_valid, error_message). If valid, error_message is None.
   :rtype: tuple[bool, str | None]

   **Validation Rules:**

   * spot > 0
   * strike > 0
   * volatility > 0
   * time > 0
   * rate and dividend can be any real number (including negative)

   **Example:**

   .. code-block:: python

      import dervflow

      # Valid parameters
      is_valid, error = dervflow.utils.validate_option_params(
          spot=100.0,
          strike=100.0,
          rate=0.05,
          dividend=0.02,
          volatility=0.2,
          time=1.0
      )
      print(f"Valid: {is_valid}")  # True

      # Invalid parameters (negative spot)
      is_valid, error = dervflow.utils.validate_option_params(
          spot=-100.0,
          strike=100.0,
          rate=0.05,
          dividend=0.02,
          volatility=0.2,
          time=1.0
      )
      print(f"Valid: {is_valid}")  # False
      print(f"Error: {error}")     # "Spot price must be positive"

validate_portfolio_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: dervflow.utils.validate_portfolio_weights(weights, tolerance=1e-6)

   Validate that portfolio weights sum to 1.0 within tolerance.

   :param array-like weights: Portfolio weights
   :param float tolerance: Tolerance for sum-to-one check (default: 1e-6)
   :return: Tuple of (is_valid, error_message). If valid, error_message is None.
   :rtype: tuple[bool, str | None]

   **Validation Rules:**

   * Sum of weights must equal 1.0 within tolerance
   * Individual weights can be negative (short positions allowed)

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Valid weights
      weights = np.array([0.3, 0.3, 0.4])
      is_valid, error = dervflow.utils.validate_portfolio_weights(weights)
      print(f"Valid: {is_valid}")  # True

      # Invalid weights (don't sum to 1)
      weights = np.array([0.3, 0.3, 0.3])
      is_valid, error = dervflow.utils.validate_portfolio_weights(weights)
      print(f"Valid: {is_valid}")  # False
      print(f"Error: {error}")     # "Weights sum to 0.9, must sum to 1.0"

      # Valid with short positions
      weights = np.array([0.6, 0.6, -0.2])
      is_valid, error = dervflow.utils.validate_portfolio_weights(weights)
      print(f"Valid: {is_valid}")  # True

Financial Calculation Functions
--------------------------------

annualize_returns
~~~~~~~~~~~~~~~~~

.. function:: dervflow.utils.annualize_returns(returns, periods_per_year=252)

   Annualize a series of period returns.

   :param array-like returns: Period returns (e.g., daily returns)
   :param int periods_per_year: Number of periods per year (default: 252 for daily)
   :return: Annualized return
   :rtype: float

   **Formula:**

   .. math::

      R_{annual} = (1 + R_{period})^{n} - 1

   where n is periods_per_year.

   **Common Values for periods_per_year:**

   * 252: Daily returns (trading days)
   * 52: Weekly returns
   * 12: Monthly returns
   * 4: Quarterly returns

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Daily returns with 1% average
      daily_returns = np.random.normal(0.01/252, 0.02/np.sqrt(252), 252)

      # Annualize
      annual_return = dervflow.utils.annualize_returns(daily_returns, periods_per_year=252)
      print(f"Annualized Return: {annual_return:.2%}")

      # Monthly returns
      monthly_returns = np.array([0.02, 0.01, -0.01, 0.03, 0.02, 0.01,
                                   0.02, 0.01, 0.02, 0.01, 0.02, 0.01])
      annual_return = dervflow.utils.annualize_returns(monthly_returns, periods_per_year=12)
      print(f"Annualized Return: {annual_return:.2%}")

annualize_volatility
~~~~~~~~~~~~~~~~~~~~

.. function:: dervflow.utils.annualize_volatility(volatility, periods_per_year=252)

   Annualize period volatility.

   :param float volatility: Period volatility (standard deviation of returns)
   :param int periods_per_year: Number of periods per year (default: 252)
   :return: Annualized volatility
   :rtype: float

   **Formula:**

   .. math::

      \sigma_{annual} = \sigma_{period} \times \sqrt{n}

   where n is periods_per_year.

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Daily returns
      daily_returns = np.random.normal(0, 0.02, 252)
      daily_vol = np.std(daily_returns)

      # Annualize
      annual_vol = dervflow.utils.annualize_volatility(daily_vol, periods_per_year=252)
      print(f"Daily Volatility: {daily_vol:.4f}")
      print(f"Annualized Volatility: {annual_vol:.2%}")

sharpe_ratio
~~~~~~~~~~~~

.. function:: dervflow.utils.sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

   Calculate the Sharpe ratio (risk-adjusted return).

   :param array-like returns: Period returns
   :param float risk_free_rate: Risk-free rate (same period as returns)
   :param int periods_per_year: Number of periods per year
   :return: Annualized Sharpe ratio
   :rtype: float

   **Formula:**

   .. math::

      Sharpe = \frac{E[R - R_f]}{\sigma_R} \times \sqrt{n}

   where:

   * E[R - R_f] is the mean excess return
   * σ_R is the standard deviation of returns
   * n is periods_per_year

   **Interpretation:**

   * < 1.0: Poor risk-adjusted performance
   * 1.0 - 2.0: Good performance
   * 2.0 - 3.0: Very good performance
   * > 3.0: Excellent performance

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Generate sample returns
      np.random.seed(42)
      daily_returns = np.random.normal(0.0005, 0.01, 252)  # ~12.5% annual return, 16% vol

      # Calculate Sharpe ratio (assuming 2% risk-free rate)
      sharpe = dervflow.utils.sharpe_ratio(
          returns=daily_returns,
          risk_free_rate=0.02/252,  # Daily risk-free rate
          periods_per_year=252
      )
      print(f"Sharpe Ratio: {sharpe:.2f}")

      # Compare two strategies
      strategy_a = np.random.normal(0.0008, 0.015, 252)
      strategy_b = np.random.normal(0.0006, 0.008, 252)

      sharpe_a = dervflow.utils.sharpe_ratio(strategy_a, 0.02/252, 252)
      sharpe_b = dervflow.utils.sharpe_ratio(strategy_b, 0.02/252, 252)

      print(f"Strategy A Sharpe: {sharpe_a:.2f}")
      print(f"Strategy B Sharpe: {sharpe_b:.2f}")

sortino_ratio
~~~~~~~~~~~~~

.. function:: dervflow.utils.sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0, periods_per_year=252)

   Calculate the Sortino ratio (downside risk-adjusted return).

   The Sortino ratio is similar to the Sharpe ratio but only penalizes downside volatility,
   making it more appropriate for strategies with asymmetric return distributions.

   :param array-like returns: Period returns
   :param float risk_free_rate: Risk-free rate (same period as returns)
   :param float target_return: Target return threshold (default: 0.0)
   :param int periods_per_year: Number of periods per year
   :return: Annualized Sortino ratio
   :rtype: float

   **Formula:**

   .. math::

      Sortino = \frac{E[R - R_f]}{\sigma_{downside}} \times \sqrt{n}

   where σ_downside is the standard deviation of returns below the target.

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Strategy with positive skew (more upside than downside)
      np.random.seed(42)
      returns = np.concatenate([
          np.random.normal(0.001, 0.008, 200),  # Normal periods
          np.random.normal(0.003, 0.012, 52)    # High return periods
      ])

      sharpe = dervflow.utils.sharpe_ratio(returns, 0.02/252, 252)
      sortino = dervflow.utils.sortino_ratio(returns, 0.02/252, 0.0, 252)

      print(f"Sharpe Ratio: {sharpe:.2f}")
      print(f"Sortino Ratio: {sortino:.2f}")
      print(f"Sortino/Sharpe: {sortino/sharpe:.2f}")  # > 1 indicates positive skew

max_drawdown
~~~~~~~~~~~~

.. function:: dervflow.utils.max_drawdown(prices)

   Calculate the maximum drawdown from a price series.

   Maximum drawdown is the largest peak-to-trough decline in the price series,
   representing the worst possible loss an investor could have experienced.

   :param array-like prices: Price series
   :return: Maximum drawdown as a negative decimal (e.g., -0.15 for 15% drawdown)
   :rtype: float

   **Formula:**

   .. math::

      MDD = \min_{t} \left( \frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s} \right)

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Simulate price series with drawdown
      prices = np.array([100, 105, 110, 108, 95, 98, 105, 110, 115])

      mdd = dervflow.utils.max_drawdown(prices)
      print(f"Maximum Drawdown: {mdd:.2%}")  # Should show ~-13.6%

      # Find the drawdown period
      cummax = np.maximum.accumulate(prices)
      drawdown = (prices - cummax) / cummax

      worst_idx = np.argmin(drawdown)
      peak_idx = np.argmax(cummax[:worst_idx+1])

      print(f"Peak: ${prices[peak_idx]:.2f} at index {peak_idx}")
      print(f"Trough: ${prices[worst_idx]:.2f} at index {worst_idx}")

calmar_ratio
~~~~~~~~~~~~

.. function:: dervflow.utils.calmar_ratio(annual_return, max_drawdown)

   Calculate the Calmar ratio (return to maximum drawdown).

   The Calmar ratio measures return relative to downside risk, specifically maximum drawdown.
   Higher values indicate better risk-adjusted performance.

   :param float annual_return: Annualized return (as decimal)
   :param float max_drawdown: Maximum drawdown (as positive decimal)
   :return: Calmar ratio
   :rtype: float

   **Formula:**

   .. math::

      Calmar = \frac{R_{annual}}{|MDD|}

   **Interpretation:**

   * < 0.5: Poor performance relative to drawdown
   * 0.5 - 1.0: Acceptable performance
   * 1.0 - 3.0: Good performance
   * > 3.0: Excellent performance

   **Example:**

   .. code-block:: python

      import numpy as np
      import dervflow

      # Calculate from price series
      prices = np.array([100, 105, 110, 108, 95, 98, 105, 110, 115, 120])

      # Calculate annual return (assuming this is 1 year of data)
      annual_return = (prices[-1] - prices[0]) / prices[0]

      # Calculate max drawdown
      mdd = dervflow.utils.max_drawdown(prices)

      # Calculate Calmar ratio
      calmar = dervflow.utils.calmar_ratio(annual_return, abs(mdd))

      print(f"Annual Return: {annual_return:.2%}")
      print(f"Max Drawdown: {mdd:.2%}")
      print(f"Calmar Ratio: {calmar:.2f}")

See Also
--------

* :doc:`complete_reference` - Complete API reference
* :doc:`../user_guide/quickstart` - Getting started guide
* :doc:`risk` - Risk metrics API
