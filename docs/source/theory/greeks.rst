Option Greeks
=============

Greeks measure the sensitivity of option prices to changes in underlying parameters. They are essential tools for risk management and hedging strategies.

First-Order Greeks
------------------

Delta (Δ)
~~~~~~~~~

Delta measures the rate of change of the option price with respect to changes in the underlying asset price.

.. math::

   \Delta = \frac{\partial V}{\partial S}

For a European call option under Black-Scholes:

.. math::

   \Delta_{\text{call}} = N(d_1)

For a European put option:

.. math::

   \Delta_{\text{put}} = N(d_1) - 1 = -N(-d_1)

where :math:`N(\cdot)` is the cumulative standard normal distribution and:

.. math::

   d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}

**Interpretation:**

* Delta ranges from 0 to 1 for calls, -1 to 0 for puts
* A delta of 0.5 means the option price changes by $0.50 for every $1 change in the underlying
* Delta is also the hedge ratio: to delta-hedge, hold Δ units of the underlying per option sold

Vega (ν)
~~~~~~~~

Vega measures the sensitivity of the option price to changes in volatility.

.. math::

   \nu = \frac{\partial V}{\partial \sigma}

For European options under Black-Scholes:

.. math::

   \nu = S e^{-qT} N'(d_1) \sqrt{T}

where :math:`N'(\cdot)` is the standard normal probability density function:

.. math::

   N'(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}

**Interpretation:**

* Vega is always positive for long options (both calls and puts)
* Higher vega means the option is more sensitive to volatility changes
* At-the-money options have the highest vega
* Vega decreases as expiration approaches

Theta (Θ)
~~~~~~~~~

Theta measures the rate of change of the option price with respect to the passage of time (time decay).

.. math::

   \Theta = \frac{\partial V}{\partial t} = -\frac{\partial V}{\partial T}

For a European call option:

.. math::

   \Theta_{\text{call}} = -\frac{S e^{-qT} N'(d_1) \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)

For a European put option:

.. math::

   \Theta_{\text{put}} = -\frac{S e^{-qT} N'(d_1) \sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2) - qSe^{-qT}N(-d_1)

**Interpretation:**

* Theta is typically negative for long options (time decay works against you)
* Measures the dollar amount an option loses per day
* Accelerates as expiration approaches
* At-the-money options have the highest theta

Rho (ρ)
~~~~~~~

Rho measures the sensitivity of the option price to changes in the risk-free interest rate.

.. math::

   \rho = \frac{\partial V}{\partial r}

For a European call option:

.. math::

   \rho_{\text{call}} = KTe^{-rT}N(d_2)

For a European put option:

.. math::

   \rho_{\text{put}} = -KTe^{-rT}N(-d_2)

**Interpretation:**

* Rho is positive for calls, negative for puts
* Less important for short-dated options
* More significant for long-dated options and interest rate derivatives

Second-Order Greeks
-------------------

Gamma (Γ)
~~~~~~~~~

Gamma measures the rate of change of delta with respect to changes in the underlying asset price.

.. math::

   \Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}

For European options under Black-Scholes:

.. math::

   \Gamma = \frac{e^{-qT} N'(d_1)}{S\sigma\sqrt{T}}

**Interpretation:**

* Gamma is always positive for long options
* Measures the curvature of the option price curve
* High gamma means delta changes rapidly
* At-the-money options have the highest gamma
* Gamma increases as expiration approaches
* Important for delta hedging: high gamma requires frequent rebalancing

Vanna
~~~~~

Vanna measures the sensitivity of delta to changes in volatility, or equivalently, the sensitivity of vega to changes in the underlying price.

.. math::

   \text{Vanna} = \frac{\partial^2 V}{\partial S \partial \sigma} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \nu}{\partial S}

For European options:

.. math::

   \text{Vanna} = -e^{-qT} N'(d_1) \frac{d_2}{\sigma}

**Interpretation:**

* Measures how delta changes with volatility
* Important for managing volatility risk in delta-hedged portfolios
* Can be positive or negative depending on moneyness

Volga (Vomma)
~~~~~~~~~~~~~

Volga measures the sensitivity of vega to changes in volatility.

.. math::

   \text{Volga} = \frac{\partial^2 V}{\partial \sigma^2} = \frac{\partial \nu}{\partial \sigma}

For European options:

.. math::

   \text{Volga} = S e^{-qT} N'(d_1) \sqrt{T} \frac{d_1 d_2}{\sigma}

**Interpretation:**

* Measures the convexity of the option price with respect to volatility
* Important for volatility trading and exotic options
* Always positive for vanilla options

Third-Order Greeks
------------------

Speed
~~~~~

Speed measures the rate of change of gamma with respect to the underlying price.

.. math::

   \text{Speed} = \frac{\partial^3 V}{\partial S^3} = \frac{\partial \Gamma}{\partial S}

**Interpretation:**

* Measures how quickly gamma changes
* Important for understanding gamma hedging costs

Zomma
~~~~~

Zomma measures the sensitivity of gamma to changes in volatility.

.. math::

   \text{Zomma} = \frac{\partial^3 V}{\partial S^2 \partial \sigma} = \frac{\partial \Gamma}{\partial \sigma}

Color (Gamma Decay)
~~~~~~~~~~~~~~~~~~~

Color measures the rate of change of gamma with respect to time.

.. math::

   \text{Color} = \frac{\partial^3 V}{\partial S^2 \partial t} = \frac{\partial \Gamma}{\partial t}

Greeks Relationships
--------------------

Several important relationships exist between Greeks:

1. **Put-Call Parity for Delta:**

   .. math::

      \Delta_{\text{call}} - \Delta_{\text{put}} = e^{-qT}

2. **Gamma Equality:**

   .. math::

      \Gamma_{\text{call}} = \Gamma_{\text{put}}

3. **Vega Equality:**

   .. math::

      \nu_{\text{call}} = \nu_{\text{put}}

4. **Portfolio Greeks:**

   For a portfolio of options, Greeks are additive:

   .. math::

      \Delta_{\text{portfolio}} = \sum_{i} n_i \Delta_i

   where :math:`n_i` is the quantity of option :math:`i`.

Numerical Calculation
---------------------

When analytical formulas are not available, Greeks can be calculated using finite differences:

**Central Difference for Delta:**

.. math::

   \Delta \approx \frac{V(S + \epsilon) - V(S - \epsilon)}{2\epsilon}

**Central Difference for Gamma:**

.. math::

   \Gamma \approx \frac{V(S + \epsilon) - 2V(S) + V(S - \epsilon)}{\epsilon^2}

**Forward Difference for Theta:**

.. math::

   \Theta \approx \frac{V(t + \Delta t) - V(t)}{\Delta t}

The choice of :math:`\epsilon` involves a trade-off between truncation error (too large) and round-off error (too small). Typical values are :math:`\epsilon = 0.01S` for delta and :math:`\epsilon = 0.01\sigma` for vega.

Practical Applications
----------------------

Hedging Strategies
~~~~~~~~~~~~~~~~~~

**Delta Hedging:**

To create a delta-neutral portfolio, hold :math:`-\Delta` units of the underlying for each option:

.. code-block:: python

   import dervflow

   bs_model = dervflow.BlackScholesModel()
   greeks = bs_model.greeks(100, 100, 0.05, 0, 0.25, 1, 'call')

   options_held = 100  # Long 100 call options
   hedge_ratio = -greeks['delta'] * options_held
   print(f"Hedge with {hedge_ratio:.2f} shares")

**Gamma Scalping:**

Profit from gamma by rebalancing delta hedge as the underlying moves:

.. code-block:: python

   # High gamma means more frequent rebalancing needed
   if greeks['gamma'] > 0.05:
       print("High gamma - rebalance frequently")

Risk Management
~~~~~~~~~~~~~~~

**Vega Risk:**

Monitor vega exposure to manage volatility risk:

.. code-block:: python

   portfolio_vega = sum(position.quantity * position.vega
                        for position in portfolio)
   print(f"Portfolio vega: {portfolio_vega:.2f}")

**Theta Decay:**

Understand time decay impact on portfolio value:

.. code-block:: python

   daily_theta = sum(position.quantity * position.theta
                     for position in portfolio)
   print(f"Expected daily P&L from time decay: ${daily_theta:.2f}")

See Also
--------

* :doc:`black_scholes` - Black-Scholes model derivation
* :doc:`../api/risk` - Greeks calculation API
* :doc:`../user_guide/risk_analytics` - Practical risk management guide
