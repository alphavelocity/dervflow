Error Handling Reference
========================

This page documents all exception types and error handling patterns in dervflow.

.. contents:: Table of Contents
   :local:
   :depth: 2

Exception Hierarchy
-------------------

dervflow defines a hierarchy of exception types for different error conditions:

.. code-block:: text

   Exception
   └── DervflowError (base exception)
       ├── InvalidInputError
       ├── ConvergenceError
       ├── OptimizationError
       └── DataError

All dervflow exceptions inherit from ``DervflowError``, allowing you to catch all library-specific
errors with a single except clause.

Exception Types
---------------

DervflowError
~~~~~~~~~~~~~

.. class:: dervflow.DervflowError

   Base exception class for all dervflow errors.

   This is the parent class for all dervflow-specific exceptions. Catching this exception
   will catch all errors raised by the library.

   **Example:**

   .. code-block:: python

      import dervflow

      try:
          result = dervflow.some_function()
      except dervflow.DervflowError as e:
          print(f"dervflow error occurred: {e}")
      except Exception as e:
          print(f"Other error: {e}")

InvalidInputError
~~~~~~~~~~~~~~~~~

.. class:: dervflow.InvalidInputError

   Raised when input parameters are invalid.

   This exception is raised when function arguments violate constraints such as:

   * Negative prices or volatilities
   * Invalid option types
   * Mismatched array dimensions
   * Parameters outside valid ranges

   **Attributes:**

   * **message** (str): Descriptive error message
   * **parameter** (str, optional): Name of the invalid parameter
   * **value** (any, optional): The invalid value that was provided

   **Common Causes:**

   * Negative spot price or strike price
   * Zero or negative volatility
   * Zero or negative time to maturity
   * Invalid option_type (not 'call' or 'put')
   * Portfolio weights that don't sum to 1.0
   * Mismatched array lengths in batch operations

   **Example:**

   .. code-block:: python

      import dervflow

      bs = dervflow.BlackScholesModel()

      try:
          # Invalid: negative spot price
          price = bs.price(
              spot=-100.0,  # Invalid!
              strike=100.0,
              rate=0.05,
              dividend=0.02,
              volatility=0.2,
              time=1.0,
              option_type='call'
          )
      except dervflow.InvalidInputError as e:
          print(f"Invalid input: {e}")
          # Output: "Invalid input: Spot price must be positive, got -100.0"

      try:
          # Invalid: wrong option type
          price = bs.price(100, 100, 0.05, 0.02, 0.2, 1.0, 'option')
      except dervflow.InvalidInputError as e:
          print(f"Invalid input: {e}")
          # Output: "Invalid input: option_type must be 'call' or 'put', got 'option'"

   **Prevention:**

   Use validation functions before calling pricing functions:

   .. code-block:: python

      from dervflow.utils import validate_option_params

      is_valid, error = validate_option_params(
          spot=100.0,
          strike=100.0,
          rate=0.05,
          dividend=0.02,
          volatility=0.2,
          time=1.0
      )

      if not is_valid:
          print(f"Validation failed: {error}")
      else:
          price = bs.price(100, 100, 0.05, 0.02, 0.2, 1.0, 'call')

ConvergenceError
~~~~~~~~~~~~~~~~

.. class:: dervflow.ConvergenceError

   Raised when numerical methods fail to converge.

   This exception is raised when iterative algorithms (Newton-Raphson, optimization, etc.)
   fail to converge within the specified tolerance and maximum iterations.

   **Attributes:**

   * **message** (str): Descriptive error message
   * **iterations** (int): Number of iterations completed
   * **final_error** (float): Final error value at termination
   * **tolerance** (float): Target tolerance that was not achieved

   **Common Causes:**

   * Implied volatility calculation for deep OTM options
   * Market prices outside arbitrage bounds
   * Poor initial guesses for iterative methods
   * Insufficient maximum iterations
   * Ill-conditioned problems

   **Example:**

   .. code-block:: python

      import dervflow

      bs = dervflow.BlackScholesModel()

      try:
          # This might fail for deep OTM options with very short time
          iv = bs.implied_vol(
              market_price=0.01,  # Very low price
              spot=100.0,
              strike=150.0,       # Deep OTM
              rate=0.05,
              dividend=0.02,
              time=0.05,          # Very short time
              option_type='call'
          )
      except dervflow.ConvergenceError as e:
          print(f"Failed to converge: {e}")
          print(f"Iterations: {e.iterations}")
          print(f"Final error: {e.final_error}")
          # Output: "Failed to converge: Implied volatility calculation did not converge after 100 iterations"
          #         "Iterations: 100"
          #         "Final error: 0.0001234"

   **Handling Strategies:**

   1. **Increase max_iterations:**

      .. code-block:: python

         try:
             iv = bs.implied_vol(
                 market_price, spot, strike, rate, dividend, time, option_type,
                 max_iterations=200  # Increase from default 100
             )
         except dervflow.ConvergenceError:
             print("Still failed with 200 iterations")

   2. **Relax tolerance:**

      .. code-block:: python

         try:
             iv = bs.implied_vol(
                 market_price, spot, strike, rate, dividend, time, option_type,
                 tolerance=1e-4  # Relax from default 1e-6
             )
         except dervflow.ConvergenceError:
             print("Still failed with relaxed tolerance")

   3. **Use fallback method:**

      .. code-block:: python

         try:
             iv = bs.implied_vol(market_price, spot, strike, rate, dividend, time, option_type)
         except dervflow.ConvergenceError:
             # Use a different method or return NaN
             import numpy as np
             iv = np.nan
             print("Using NaN for failed convergence")

   4. **Check for arbitrage violations:**

      .. code-block:: python

         import numpy as np

         # Check lower bound: C >= max(S*e^(-qT) - K*e^(-rT), 0)
         lower_bound = max(
             spot * np.exp(-dividend * time) - strike * np.exp(-rate * time),
             0
         )

         if market_price < lower_bound:
             print(f"Market price {market_price} violates lower bound {lower_bound}")
         else:
             try:
                 iv = bs.implied_vol(market_price, spot, strike, rate, dividend, time, option_type)
             except dervflow.ConvergenceError as e:
                 print(f"Convergence failed: {e}")

OptimizationError
~~~~~~~~~~~~~~~~~

.. class:: dervflow.OptimizationError

   Raised when portfolio optimization is infeasible or fails.

   This exception is raised when the optimization problem cannot be solved, typically due to
   conflicting constraints or numerical issues.

   **Attributes:**

   * **message** (str): Descriptive error message
   * **status** (str): Optimization solver status
   * **violated_constraints** (list, optional): List of constraint names that are violated

   **Common Causes:**

   * Conflicting constraints (e.g., min_weight > max_weight)
   * Infeasible target return (too high or too low)
   * Singular covariance matrix
   * Numerical instability in solver

   **Example:**

   .. code-block:: python

      import dervflow
      import numpy as np

      # Create optimizer
      expected_returns = np.array([0.08, 0.10, 0.12])
      covariance = np.eye(3) * 0.04  # Simple diagonal covariance

      optimizer = dervflow.PortfolioOptimizer(expected_returns, covariance)

      try:
          # Infeasible: target return too high
          result = optimizer.optimize(
              target_return=0.20,  # Higher than any individual asset
              min_weights=np.zeros(3),  # No short selling
              max_weights=np.ones(3)
          )
      except dervflow.OptimizationError as e:
          print(f"Optimization failed: {e}")
          print(f"Status: {e.status}")
          # Output: "Optimization failed: Target return 0.20 is infeasible"
          #         "Status: infeasible"

      try:
          # Conflicting constraints
          result = optimizer.optimize(
              min_weights=np.array([0.5, 0.5, 0.5]),  # Sum to 1.5
              max_weights=np.ones(3)
          )
      except dervflow.OptimizationError as e:
          print(f"Optimization failed: {e}")
          # Output: "Optimization failed: Minimum weights sum to 1.5, must sum to 1.0"

   **Handling Strategies:**

   1. **Validate constraints before optimization:**

      .. code-block:: python

         min_weights = np.array([0.1, 0.1, 0.1])
         max_weights = np.array([0.5, 0.5, 0.5])

         # Check that constraints are feasible
         if min_weights.sum() > 1.0:
             print("Minimum weights sum to more than 1.0")
         elif max_weights.sum() < 1.0:
             print("Maximum weights sum to less than 1.0")
         else:
             result = optimizer.optimize(
                 min_weights=min_weights,
                 max_weights=max_weights
             )

   2. **Check target return feasibility:**

      .. code-block:: python

         max_return = expected_returns.max()
         min_return = expected_returns.min()

         target_return = 0.15

         if target_return > max_return:
             print(f"Target return {target_return} exceeds maximum {max_return}")
             target_return = max_return * 0.95  # Use 95% of maximum

         result = optimizer.optimize(target_return=target_return)

   3. **Handle singular covariance:**

      .. code-block:: python

         # Check condition number
         cond = np.linalg.cond(covariance)
         if cond > 1e10:
             print(f"Covariance matrix is ill-conditioned (cond={cond})")
             # Add regularization
             covariance_reg = covariance + np.eye(len(covariance)) * 1e-8

         optimizer = dervflow.PortfolioOptimizer(expected_returns, covariance_reg)

DataError
~~~~~~~~~

.. class:: dervflow.DataError

   Raised when input data is malformed or inconsistent.

   This exception is raised when data structures are invalid, such as mismatched dimensions,
   missing values, or inconsistent data.

   **Attributes:**

   * **message** (str): Descriptive error message
   * **data_type** (str, optional): Type of data that caused the error

   **Common Causes:**

   * Mismatched array dimensions
   * NaN or infinite values in input data
   * Empty arrays
   * Inconsistent time series data

   **Example:**

   .. code-block:: python

      import dervflow
      import numpy as np

      try:
          # Mismatched dimensions
          expected_returns = np.array([0.08, 0.10, 0.12])
          covariance = np.eye(4) * 0.04  # 4x4 instead of 3x3

          optimizer = dervflow.PortfolioOptimizer(expected_returns, covariance)
      except dervflow.DataError as e:
          print(f"Data error: {e}")
          # Output: "Data error: Expected returns length (3) does not match covariance dimension (4)"

      try:
          # NaN values
          returns = np.array([0.01, 0.02, np.nan, 0.01])
          analyzer = dervflow.TimeSeriesAnalyzer(returns)
          stats = analyzer.stat()
      except dervflow.DataError as e:
          print(f"Data error: {e}")
          # Output: "Data error: Input data contains NaN values"

Error Handling Best Practices
------------------------------

Defensive Programming
~~~~~~~~~~~~~~~~~~~~~

Always validate inputs before calling dervflow functions:

.. code-block:: python

   import dervflow
   import numpy as np

   def safe_option_price(spot, strike, rate, dividend, volatility, time, option_type):
       """Safely price an option with input validation."""

       # Validate inputs
       if spot <= 0:
           raise ValueError(f"Spot must be positive, got {spot}")
       if strike <= 0:
           raise ValueError(f"Strike must be positive, got {strike}")
       if volatility <= 0:
           raise ValueError(f"Volatility must be positive, got {volatility}")
       if time <= 0:
           raise ValueError(f"Time must be positive, got {time}")
       if option_type not in ['call', 'put']:
           raise ValueError(f"Option type must be 'call' or 'put', got {option_type}")

       # Price option
       bs = dervflow.BlackScholesModel()
       try:
           return bs.price(spot, strike, rate, dividend, volatility, time, option_type)
       except dervflow.DervflowError as e:
           print(f"Pricing failed: {e}")
           return None

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

Handle errors gracefully and provide fallback values:

.. code-block:: python

   import dervflow
   import numpy as np

   def calculate_implied_vol_with_fallback(market_price, spot, strike, rate, dividend, time, option_type):
       """Calculate implied volatility with fallback to historical volatility."""

       bs = dervflow.BlackScholesModel()

       try:
           # Try to calculate implied volatility
           return bs.implied_vol(market_price, spot, strike, rate, dividend, time, option_type)
       except dervflow.ConvergenceError as e:
           print(f"IV calculation failed: {e}")
           print("Using historical volatility as fallback")
           return 0.25  # Use 25% as default
       except dervflow.InvalidInputError as e:
           print(f"Invalid input: {e}")
           return np.nan

Batch Error Handling
~~~~~~~~~~~~~~~~~~~~

When processing multiple items, handle errors individually:

.. code-block:: python

   import dervflow
   import numpy as np
   import pandas as pd

   def price_option_chain_safe(spot, strikes, rate, dividend, volatility, time, option_type):
       """Price option chain with individual error handling."""

       bs = dervflow.BlackScholesModel()
       results = []

       for strike in strikes:
           try:
               price = bs.price(spot, strike, rate, dividend, volatility, time, option_type)
               results.append({'strike': strike, 'price': price, 'error': None})
           except dervflow.DervflowError as e:
               results.append({'strike': strike, 'price': np.nan, 'error': str(e)})

       return pd.DataFrame(results)

   # Usage
   strikes = [80, 90, 100, 110, 120]
   df = price_option_chain_safe(100, strikes, 0.05, 0.02, 0.25, 1.0, 'call')
   print(df)

Logging Errors
~~~~~~~~~~~~~~

Use Python's logging module for production code:

.. code-block:: python

   import dervflow
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def calculate_portfolio_metrics(returns, confidence=0.95):
       """Calculate portfolio metrics with logging."""

       risk = dervflow.RiskMetrics()

       try:
           var = risk.var(returns, confidence, method='historical')
           logger.info(f"VaR calculated successfully: {var:.4f}")
           return var
       except dervflow.InvalidInputError as e:
           logger.error(f"Invalid input for VaR calculation: {e}")
           raise
       except dervflow.DervflowError as e:
           logger.error(f"dervflow error in VaR calculation: {e}")
           raise
       except Exception as e:
           logger.exception(f"Unexpected error in VaR calculation: {e}")
           raise

Context Managers
~~~~~~~~~~~~~~~~

Use context managers for resource cleanup:

.. code-block:: python

   import dervflow
   import numpy as np
   from contextlib import contextmanager

   @contextmanager
   def monte_carlo_simulation(seed=None):
       """Context manager for Monte Carlo simulations with cleanup."""

       if seed is not None:
           np.random.seed(seed)

       mc = dervflow.MonteCarloEngine()

       try:
           yield mc
       except dervflow.DervflowError as e:
           print(f"Simulation error: {e}")
           raise
       finally:
           # Cleanup code here
           pass

   # Usage
   with monte_carlo_simulation(seed=42) as mc:
       paths = mc.simulate_gbm(100, 0.1, 0.2, 1.0, 252, 10000)

See Also
--------

* :doc:`complete_reference` - Complete API reference
* :doc:`examples` - Usage examples
* :doc:`parameters` - Parameter reference
