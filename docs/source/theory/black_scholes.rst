Black-Scholes Model
===================

The Black-Scholes-Merton model is the foundational framework for pricing European options.

Model Assumptions
-----------------

The Black-Scholes model makes the following assumptions:

1. **Efficient Markets**: No arbitrage opportunities exist
2. **Log-Normal Returns**: Asset prices follow geometric Brownian motion
3. **Constant Volatility**: Volatility :math:`\sigma` is constant over time
4. **Constant Interest Rate**: Risk-free rate :math:`r` is constant
5. **No Transaction Costs**: Trading is frictionless
6. **Continuous Trading**: Assets can be traded continuously
7. **No Dividends**: Or dividends are paid continuously at rate :math:`q`
8. **European Exercise**: Options can only be exercised at maturity

Stochastic Differential Equation
---------------------------------

Under the Black-Scholes model, the asset price :math:`S_t` follows:

.. math::

   dS_t = (r - q) S_t dt + \sigma S_t dW_t

where:

* :math:`S_t` is the asset price at time :math:`t`
* :math:`r` is the risk-free interest rate
* :math:`q` is the continuous dividend yield
* :math:`\sigma` is the volatility
* :math:`W_t` is a Wiener process (Brownian motion)

Pricing Formula
---------------

Call Option
~~~~~~~~~~~

The price of a European call option is:

.. math::

   C(S, K, r, q, \sigma, T) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)

where:

.. math::

   d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}

.. math::

   d_2 = d_1 - \sigma\sqrt{T}

and :math:`N(\cdot)` is the cumulative standard normal distribution function.

Put Option
~~~~~~~~~~

The price of a European put option is:

.. math::

   P(S, K, r, q, \sigma, T) = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)

Put-Call Parity
~~~~~~~~~~~~~~~

European call and put options satisfy the put-call parity relationship:

.. math::

   C - P = S e^{-qT} - K e^{-rT}

This relationship must hold to prevent arbitrage opportunities.

Derivation
----------

The Black-Scholes formula is derived using the following approach:

1. **Construct a Hedged Portfolio**: Create a portfolio consisting of:

   * Long one option
   * Short :math:`\Delta` units of the underlying asset

2. **Apply Itô's Lemma**: The option value :math:`V(S, t)` satisfies:

   .. math::

      dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (dS)^2

3. **Eliminate Risk**: Choose :math:`\Delta = \frac{\partial V}{\partial S}` to eliminate the stochastic term

4. **No-Arbitrage Condition**: The risk-free portfolio must earn the risk-free rate:

   .. math::

      \frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV

5. **Boundary Conditions**: Apply terminal conditions:

   * Call: :math:`V(S, T) = \max(S - K, 0)`
   * Put: :math:`V(S, T) = \max(K - S, 0)`

6. **Solve PDE**: The solution to this partial differential equation yields the Black-Scholes formula

Risk-Neutral Valuation
----------------------

An alternative derivation uses risk-neutral valuation:

1. **Risk-Neutral Measure**: Under the risk-neutral measure :math:`\mathbb{Q}`:

   .. math::

      dS_t = (r - q) S_t dt + \sigma S_t dW_t^{\mathbb{Q}}

2. **Discounted Expectation**: The option value is:

   .. math::

      V(S, t) = e^{-r(T-t)} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T) | S_t = S]

3. **Log-Normal Distribution**: Under :math:`\mathbb{Q}`, :math:`\ln(S_T)` is normally distributed:

   .. math::

      \ln(S_T) \sim \mathcal{N}\left(\ln(S) + (r - q - \sigma^2/2)(T-t), \sigma^2(T-t)\right)

4. **Evaluate Expectation**: Computing the expectation yields the Black-Scholes formula

Limitations
-----------

The Black-Scholes model has several limitations:

1. **Constant Volatility**: Real markets exhibit volatility smiles and term structures
2. **Continuous Trading**: Transaction costs and discrete trading affect hedging
3. **Log-Normal Returns**: Empirical returns have fatter tails (excess kurtosis)
4. **Jumps**: Asset prices can jump discontinuously
5. **Stochastic Volatility**: Volatility itself is random
6. **Interest Rate Risk**: Interest rates are not constant

Extensions
----------

Several extensions address these limitations:

* **Stochastic Volatility**: Heston model, SABR model
* **Jump Processes**: Merton jump-diffusion, Kou model
* **Local Volatility**: Dupire's local volatility model
* **Stochastic Interest Rates**: Hull-White, CIR models

Implementation Notes
--------------------

Numerical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

When implementing the Black-Scholes formula:

1. **Near Expiry**: For :math:`T \to 0`, use intrinsic value
2. **Deep ITM/OTM**: Use asymptotic approximations for :math:`N(d_1)` and :math:`N(d_2)`
3. **Numerical Stability**: Compute :math:`\ln(S/K)` carefully to avoid cancellation errors
4. **Vectorization**: Use SIMD operations for batch pricing

Example Calculation
~~~~~~~~~~~~~~~~~~~

Consider a call option with:

* :math:`S = 100` (spot price)
* :math:`K = 105` (strike price)
* :math:`r = 0.05` (5% risk-free rate)
* :math:`q = 0.02` (2% dividend yield)
* :math:`\sigma = 0.25` (25% volatility)
* :math:`T = 1.0` (1 year to maturity)

Step 1: Calculate :math:`d_1` and :math:`d_2`:

.. math::

   d_1 = \frac{\ln(100/105) + (0.05 - 0.02 + 0.25^2/2) \times 1}{0.25\sqrt{1}} = 0.0553

.. math::

   d_2 = 0.0553 - 0.25\sqrt{1} = -0.1947

Step 2: Evaluate cumulative normal:

.. math::

   N(d_1) = N(0.0553) = 0.5221

.. math::

   N(d_2) = N(-0.1947) = 0.4228

Step 3: Calculate option price:

.. math::

   C = 100 \times e^{-0.02 \times 1} \times 0.5221 - 105 \times e^{-0.05 \times 1} \times 0.4228 = 8.92

The call option is worth approximately $8.92.

References
----------

* Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities". *Journal of Political Economy*, 81(3), 637-654.
* Merton, R. C. (1973). "Theory of Rational Option Pricing". *Bell Journal of Economics and Management Science*, 4(1), 141-183.
* Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
