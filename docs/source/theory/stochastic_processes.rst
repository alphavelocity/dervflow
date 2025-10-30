Stochastic Processes in Finance
================================

Stochastic processes model the random evolution of financial variables over time. They are fundamental to derivatives pricing, risk management, and portfolio optimization.

Brownian Motion
---------------

Standard Brownian Motion
~~~~~~~~~~~~~~~~~~~~~~~~

A standard Brownian motion (Wiener process) :math:`W_t` has the following properties:

1. :math:`W_0 = 0`
2. Independent increments: :math:`W_t - W_s` is independent of :math:`W_u - W_v` for non-overlapping intervals
3. Normal increments: :math:`W_t - W_s \sim N(0, t-s)`
4. Continuous paths

**Simulation:**

.. math::

   W_{t+\Delta t} = W_t + \sqrt{\Delta t} \cdot Z

where :math:`Z \sim N(0, 1)`.

Geometric Brownian Motion (GBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most common model for stock prices:

.. math::

   dS_t = \mu S_t dt + \sigma S_t dW_t

**Solution:**

.. math::

   S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]

**Properties:**

* :math:`S_t` is log-normally distributed
* :math:`E[S_t] = S_0 e^{\mu t}`
* :math:`\text{Var}[S_t] = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)`

**Discrete simulation:**

.. math::

   S_{t+\Delta t} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right]

**Code Example:**

.. code-block:: python

   import dervflow
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulate GBM paths
   mc_engine = dervflow.MonteCarloEngine()
   
   paths = mc_engine.simulate_gbm(
       s0=100.0,      # Initial price
       mu=0.05,       # Drift (5% annual return)
       sigma=0.2,     # Volatility (20%)
       T=1.0,         # Time horizon (1 year)
       steps=252,     # Daily steps
       paths=1000     # Number of paths
   )

   # Plot sample paths
   plt.figure(figsize=(10, 6))
   plt.plot(paths[:, :10])
   plt.xlabel('Time Steps')
   plt.ylabel('Price')
   plt.title('Geometric Brownian Motion Paths')
   plt.grid(True)
   plt.show()

Mean-Reverting Processes
-------------------------

Ornstein-Uhlenbeck Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models mean-reverting behavior:

.. math::

   dX_t = \theta(\mu - X_t)dt + \sigma dW_t

**Parameters:**

* :math:`\theta`: Speed of mean reversion
* :math:`\mu`: Long-term mean
* :math:`\sigma`: Volatility

**Solution:**

.. math::

   X_t = X_0 e^{-\theta t} + \mu(1 - e^{-\theta t}) + \sigma \int_0^t e^{-\theta(t-s)} dW_s

**Properties:**

* :math:`E[X_t] = X_0 e^{-\theta t} + \mu(1 - e^{-\theta t})`
* :math:`\text{Var}[X_t] = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})`
* Long-term variance: :math:`\frac{\sigma^2}{2\theta}`

**Discrete simulation:**

.. math::

   X_{t+\Delta t} = X_t e^{-\theta \Delta t} + \mu(1 - e^{-\theta \Delta t}) + \sigma\sqrt{\frac{1-e^{-2\theta\Delta t}}{2\theta}} \cdot Z

**Applications:**

* Interest rates
* Volatility
* Commodity prices
* Pairs trading spreads

**Code Example:**

.. code-block:: python

   import dervflow

   # Simulate Ornstein-Uhlenbeck process
   mc_engine = dervflow.MonteCarloEngine()
   
   paths = mc_engine.simulate_ou(
       x0=0.05,       # Initial value
       theta=0.5,     # Mean reversion speed
       mu=0.03,       # Long-term mean
       sigma=0.01,    # Volatility
       T=5.0,         # Time horizon
       steps=1000,    # Time steps
       paths=100      # Number of paths
   )

Cox-Ingersoll-Ross (CIR) Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mean-reverting process that stays positive:

.. math::

   dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t} dW_t

**Parameters:**

* :math:`\kappa`: Speed of mean reversion
* :math:`\theta`: Long-term mean
* :math:`\sigma`: Volatility

**Feller condition:**

If :math:`2\kappa\theta > \sigma^2`, the process never reaches zero.

**Properties:**

* :math:`E[r_t] = r_0 e^{-\kappa t} + \theta(1 - e^{-\kappa t})`
* Always non-negative (under Feller condition)
* Chi-squared distribution

**Applications:**

* Interest rate models
* Volatility models (Heston)

**Code Example:**

.. code-block:: python

   import dervflow

   # Simulate CIR process
   mc_engine = dervflow.MonteCarloEngine()
   
   paths = mc_engine.simulate_cir(
       r0=0.03,       # Initial rate
       kappa=0.5,     # Mean reversion speed
       theta=0.04,    # Long-term mean
       sigma=0.1,     # Volatility
       T=10.0,        # Time horizon
       steps=1000,    # Time steps
       paths=100      # Number of paths
   )

Vasicek Model
~~~~~~~~~~~~~

Similar to Ornstein-Uhlenbeck, used for interest rates:

.. math::

   dr_t = \kappa(\theta - r_t)dt + \sigma dW_t

**Difference from OU:** Typically parameterized for interest rates.

**Zero-coupon bond price:**

.. math::

   P(t, T) = A(t, T) e^{-B(t, T)r_t}

where:

.. math::

   B(t, T) &= \frac{1 - e^{-\kappa(T-t)}}{\kappa} \\
   A(t, T) &= \exp\left[\left(\theta - \frac{\sigma^2}{2\kappa^2}\right)(B(t,T) - T + t) - \frac{\sigma^2 B(t,T)^2}{4\kappa}\right]

Jump-Diffusion Processes
-------------------------

Merton Jump-Diffusion
~~~~~~~~~~~~~~~~~~~~~~

Combines continuous diffusion with discrete jumps:

.. math::

   dS_t = \mu S_t dt + \sigma S_t dW_t + S_t dJ_t

where :math:`J_t` is a compound Poisson process with:

* Jump intensity :math:`\lambda` (jumps per unit time)
* Jump size :math:`Y \sim N(\mu_J, \sigma_J^2)`

**Log price:**

.. math::

   d\ln S_t = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma dW_t + \ln(1 + Y)dN_t

where :math:`N_t` is a Poisson process with intensity :math:`\lambda`.

**Properties:**

* Captures sudden price movements (crashes, earnings announcements)
* Fat tails in return distribution
* Implied volatility smile

**Simulation:**

1. Simulate continuous path with GBM
2. Simulate jump times from Poisson process
3. Simulate jump sizes from normal distribution
4. Apply jumps at jump times

**Code Example:**

.. code-block:: python

   import dervflow

   # Simulate Merton jump-diffusion
   mc_engine = dervflow.MonteCarloEngine()
   
   paths = mc_engine.simulate_jump_diffusion(
       s0=100.0,          # Initial price
       mu=0.05,           # Drift
       sigma=0.2,         # Diffusion volatility
       lambda_=10.0,      # Jump intensity (10 jumps/year)
       jump_mean=-0.02,   # Average jump size (-2%)
       jump_std=0.05,     # Jump volatility (5%)
       T=1.0,             # Time horizon
       steps=252,         # Time steps
       paths=1000         # Number of paths
   )

Kou Double Exponential Jump Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jump sizes follow asymmetric double exponential distribution:

.. math::

   f_Y(y) = p \cdot \eta_1 e^{-\eta_1 y} \mathbb{1}_{y \geq 0} + (1-p) \cdot \eta_2 e^{\eta_2 y} \mathbb{1}_{y < 0}

**Parameters:**

* :math:`p`: Probability of upward jump
* :math:`\eta_1`: Decay rate of upward jumps
* :math:`\eta_2`: Decay rate of downward jumps

**Advantages:**

* Captures asymmetry (larger downward jumps)
* Analytical tractability for option pricing

Stochastic Volatility Models
-----------------------------

Heston Model
~~~~~~~~~~~~

Volatility follows a CIR process:

.. math::

   dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\
   dv_t &= \kappa(\theta - v_t)dt + \sigma_v\sqrt{v_t} dW_t^v

with correlation :math:`\text{Corr}(dW_t^S, dW_t^v) = \rho`.

**Parameters:**

* :math:`v_t`: Instantaneous variance
* :math:`\kappa`: Mean reversion speed of variance
* :math:`\theta`: Long-term variance
* :math:`\sigma_v`: Volatility of volatility
* :math:`\rho`: Correlation (typically negative: leverage effect)

**Properties:**

* Captures volatility clustering
* Generates volatility smile
* Semi-analytical option pricing formulas

.. note::

   The current :class:`dervflow.MonteCarloEngine` exposes diffusion (GBM, OU, CIR, Vasicek)
   and jump-diffusion processes. Dedicated stochastic-volatility simulators such as
   Heston and SABR are planned but not yet available in the Python bindings.

SABR Model
~~~~~~~~~~

Stochastic Alpha Beta Rho model for forward rates:

.. math::

   dF_t &= \alpha_t F_t^\beta dW_t^F \\
   d\alpha_t &= \nu \alpha_t dW_t^\alpha

with correlation :math:`\text{Corr}(dW_t^F, dW_t^\alpha) = \rho`.

**Parameters:**

* :math:`\beta`: CEV parameter (0 = normal, 1 = lognormal)
* :math:`\alpha`: Initial volatility
* :math:`\nu`: Volatility of volatility
* :math:`\rho`: Correlation

**Applications:**

* Interest rate derivatives
* FX options
* Volatility surface modeling

.. note::

   Use :meth:`dervflow.MonteCarloEngine.simulate_jump_diffusion` or
   :meth:`~dervflow.MonteCarloEngine.simulate_correlated` to approximate stochastic volatility
   dynamics until a dedicated SABR simulator is provided.

Correlated Multi-Asset Processes
---------------------------------

Cholesky Decomposition
~~~~~~~~~~~~~~~~~~~~~~~

To simulate :math:`n` correlated Brownian motions with correlation matrix :math:`\rho`:

1. Compute Cholesky decomposition: :math:`\rho = LL^T`
2. Generate independent standard normals :math:`Z_1, \ldots, Z_n`
3. Correlated normals: :math:`\mathbf{X} = L\mathbf{Z}`

**Code Example:**

.. code-block:: python

   import dervflow
   import numpy as np

   # Define correlation matrix
   correlation = np.array([
       [1.0, 0.5, 0.3],
       [0.5, 1.0, 0.4],
       [0.3, 0.4, 1.0]
   ])

   # Define processes for each asset
   processes = [
       {'type': 'gbm', 's0': 100, 'mu': 0.05, 'sigma': 0.2},
       {'type': 'gbm', 's0': 50, 'mu': 0.07, 'sigma': 0.25},
       {'type': 'gbm', 's0': 150, 'mu': 0.04, 'sigma': 0.15},
   ]

   # Simulate correlated paths
   mc_engine = dervflow.MonteCarloEngine()
   
   paths = mc_engine.simulate_correlated(
       processes=processes,
       correlation=correlation,
       T=1.0,
       steps=252,
       paths=1000
   )

   # paths is a list of arrays, one for each asset
   for i, asset_paths in enumerate(paths):
       print(f"Asset {i+1} final price: ${asset_paths[-1, :].mean():.2f}")

Copulas
~~~~~~~

Alternative approach for modeling dependence:

1. Model marginal distributions separately
2. Use copula to model dependence structure

**Gaussian Copula:**

.. math::

   C(u_1, \ldots, u_n) = \Phi_\rho(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_n))

where :math:`\Phi_\rho` is the multivariate normal CDF with correlation :math:`\rho`.

Variance Reduction Techniques
------------------------------

Antithetic Variates
~~~~~~~~~~~~~~~~~~~

For each path with random numbers :math:`Z`, simulate another with :math:`-Z`:

.. math::

   \hat{\mu} = \frac{1}{2}(f(Z) + f(-Z))

**Variance reduction:**

.. math::

   \text{Var}[\hat{\mu}] = \frac{1}{2}\text{Var}[f(Z)](1 + \text{Corr}[f(Z), f(-Z)])

Effective when :math:`\text{Corr}[f(Z), f(-Z)] < 0`.

**Code Example:**

.. code-block:: python

   import dervflow

   # Price European option with antithetic variates
   mc_pricer = dervflow.MonteCarloOptionPricer()
   
   result = mc_pricer.price_european(
       spot=100.0,
       strike=100.0,
       rate=0.05,
       dividend=0.0,
       volatility=0.2,
       time=1.0,
       option_type='call',
       num_paths=10000,
       antithetic=True  # Enable variance reduction
   )

   print(f"Price: ${result['price']:.2f}")
   print(f"Std Error: ${result['std_error']:.4f}")

Control Variates
~~~~~~~~~~~~~~~~

Use a correlated variable with known expectation:

.. math::

   \hat{\mu} = \bar{X} - \beta(\bar{Y} - E[Y])

where :math:`\beta` is chosen to minimize variance.

Optimal :math:`\beta`:

.. math::

   \beta^* = \frac{\text{Cov}[X, Y]}{\text{Var}[Y]}

Importance Sampling
~~~~~~~~~~~~~~~~~~~

Sample from a different distribution to reduce variance in tail regions:

.. math::

   E[f(X)] = E\left[f(X)\frac{p(X)}{q(X)}\right]

where :math:`p` is the original density and :math:`q` is the importance sampling density.

Quasi-Random Sequences
~~~~~~~~~~~~~~~~~~~~~~~

Use low-discrepancy sequences instead of pseudo-random numbers:

* **Sobol sequences**
* **Halton sequences**
* **Faure sequences**

**Advantages:**

* Better coverage of sample space
* Faster convergence (:math:`O(1/N)` vs :math:`O(1/\sqrt{N})`)

**Code Example:**

.. code-block:: python

   import dervflow

   # Generate Sobol sequence
   sobol = dervflow.SobolSequence(dimension=2)
   quasi_points = sobol.generate(1000)

   # Use quasi-random draws as custom shocks for post-processing or custom path logic
   mc_engine = dervflow.MonteCarloEngine()
   paths = mc_engine.simulate_gbm(
       s0=100,
       mu=0.05,
       sigma=0.2,
       T=1.0,
       steps=252,
       paths=1000,
   )

Applications
------------

Option Pricing
~~~~~~~~~~~~~~

Monte Carlo is essential for:

* Path-dependent options (Asian, lookback)
* American options (Longstaff-Schwartz)
* Multi-asset options (basket, rainbow)
* Exotic payoffs

Risk Management
~~~~~~~~~~~~~~~

* VaR and CVaR calculation
* Stress testing
* Scenario analysis

Portfolio Optimization
~~~~~~~~~~~~~~~~~~~~~~

* Simulate future portfolio values
* Estimate return distributions
* Optimize under uncertainty

See Also
--------

* :doc:`../api/monte_carlo` - Monte Carlo simulation API
* :doc:`../user_guide/monte_carlo` - Practical simulation guide
* :doc:`black_scholes` - Option pricing theory
