Options Pricing Guide
=====================

This guide covers advanced option pricing techniques in dervflow.

.. note::
   For basic option pricing, see the :doc:`quickstart` guide.

Analytical Models
-----------------

Black-Scholes-Merton
~~~~~~~~~~~~~~~~~~~~

The Black-Scholes model is ideal for European options on non-dividend or continuously dividend-paying stocks.

**When to use:**

* European exercise only
* Liquid markets with continuous trading
* Stable volatility environment
* Quick pricing required (microseconds)

**Advantages:**

* Extremely fast (analytical formula)
* Closed-form Greeks available
* Well-understood and widely accepted

**Limitations:**

* Cannot price American options
* Assumes constant volatility
* No jumps or stochastic volatility

Tree Methods
------------

Binomial Trees
~~~~~~~~~~~~~~

Binomial trees discretize time and price movements, allowing for American exercise.

**When to use:**

* American options
* Early exercise features
* Dividend payments at specific dates
* Moderate accuracy requirements

**Advantages:**

* Handles American exercise
* Intuitive and transparent
* Flexible for various payoffs

**Limitations:**

* Slower than analytical methods
* Convergence can be slow
* Requires many steps for accuracy

Monte Carlo Simulation
----------------------

Monte Carlo methods simulate many price paths and average the payoffs.

**When to use:**

* Path-dependent options (Asian, lookback)
* Multiple underlying assets
* Complex payoff structures
* High-dimensional problems

**Advantages:**

* Handles complex payoffs
* Scales well to multiple assets
* Provides confidence intervals

**Limitations:**

* Computationally intensive
* Slow convergence (√N)
* Difficult for American options

Exotic Options
--------------

Asian Options
~~~~~~~~~~~~~

Asian options have payoffs based on the average price over time.

Barrier Options
~~~~~~~~~~~~~~~

Barrier options activate or deactivate when the underlying crosses a barrier level.

Lookback Options
~~~~~~~~~~~~~~~~

Lookback options have payoffs based on the maximum or minimum price over time.

Digital Options
~~~~~~~~~~~~~~~

Digital (binary) options pay a fixed amount if a condition is met.

Volatility Surface
------------------

Implied Volatility
~~~~~~~~~~~~~~~~~~

Implied volatility is the volatility that makes the model price equal to the market price.

Surface Construction
~~~~~~~~~~~~~~~~~~~~

Build a volatility surface from market option prices across strikes and maturities.

SABR Calibration
~~~~~~~~~~~~~~~~

Calibrate the SABR model to fit the volatility smile.

Performance Considerations
--------------------------

Choosing the Right Method
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Speed**: Black-Scholes > Trees > Monte Carlo
* **Accuracy**: Monte Carlo (with many paths) > Trees (with many steps) > Black-Scholes
* **Flexibility**: Monte Carlo > Trees > Black-Scholes

Batch Processing
~~~~~~~~~~~~~~~~

Always use batch methods when pricing multiple options with similar parameters.

Parallel Processing
~~~~~~~~~~~~~~~~~~~

dervflow automatically uses parallel processing for Monte Carlo simulations and batch operations.

Examples
--------

See the example notebooks for detailed examples:

* ``examples/notebooks/01_option_pricing.ipynb``
* ``examples/notebooks/02_volatility_surface.ipynb``
