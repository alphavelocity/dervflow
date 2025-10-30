Yield Curves and Interest Rate Models
======================================

Yield curves represent the relationship between interest rates (or yields) and time to maturity. They are fundamental to fixed income pricing, risk management, and monetary policy analysis.

Term Structure of Interest Rates
---------------------------------

Zero-Coupon Rates
~~~~~~~~~~~~~~~~~

The zero-coupon rate (spot rate) :math:`r(t)` is the yield on a zero-coupon bond maturing at time :math:`t`.

The price of a zero-coupon bond is:

.. math::

   P(0, t) = e^{-r(t) \cdot t}

or with discrete compounding:

.. math::

   P(0, t) = \frac{1}{(1 + r(t))^t}

Discount Factors
~~~~~~~~~~~~~~~~

The discount factor :math:`D(t)` is the present value of $1 received at time :math:`t`:

.. math::

   D(t) = e^{-r(t) \cdot t}

Forward Rates
~~~~~~~~~~~~~

The forward rate :math:`f(t_1, t_2)` is the interest rate agreed today for borrowing/lending between times :math:`t_1` and :math:`t_2`.

**Relationship with spot rates:**

.. math::

   f(t_1, t_2) = \frac{r(t_2) \cdot t_2 - r(t_1) \cdot t_1}{t_2 - t_1}

**Instantaneous forward rate:**

.. math::

   f(t) = r(t) + t \frac{dr(t)}{dt} = -\frac{d \ln D(t)}{dt}

Par Rates
~~~~~~~~~

The par rate is the coupon rate that makes a bond trade at par (price = face value).

For a bond with annual coupons:

.. math::

   c = \frac{1 - D(T)}{\sum_{i=1}^n D(t_i)}

Yield Curve Construction
------------------------

Bootstrapping
~~~~~~~~~~~~~

Bootstrapping builds the zero curve iteratively from market instruments.

**Algorithm:**

1. Start with shortest maturity instrument
2. Solve for the zero rate that prices the instrument correctly
3. Use solved rates to price next instrument
4. Repeat until all maturities are covered

**Example with bonds:**

For a bond with price :math:`P`, coupon :math:`c`, and maturity :math:`T`:

.. math::

   P = \sum_{i=1}^n c \cdot D(t_i) + D(T)

Given :math:`D(t_1), \ldots, D(t_{n-1})`, solve for :math:`D(T)`:

.. math::

   D(T) = P - \sum_{i=1}^{n-1} c \cdot D(t_i)

**Code Example:**

.. code-block:: python

   import dervflow

   # Market bond data
   bonds = [
       {'maturity': 0.5, 'coupon': 0.02, 'price': 100.5},
       {'maturity': 1.0, 'coupon': 0.025, 'price': 101.2},
       {'maturity': 2.0, 'coupon': 0.03, 'price': 102.5},
   ]

   # Bootstrap yield curve
   builder = dervflow.YieldCurveBuilder()
   curve = builder.bootstrap(bonds)

   # Query rates
   print(f"1-year zero rate: {curve.zero_rate(1.0):.4f}")
   print(f"2-year zero rate: {curve.zero_rate(2.0):.4f}")

Interpolation Methods
---------------------

Linear Interpolation
~~~~~~~~~~~~~~~~~~~~

Simple linear interpolation between known points:

.. math::

   r(t) = r(t_1) + \frac{r(t_2) - r(t_1)}{t_2 - t_1}(t - t_1)

**Advantages:** Simple, fast
**Disadvantages:** Not smooth, unrealistic forward rates

Cubic Spline
~~~~~~~~~~~~

Piecewise cubic polynomials with continuous first and second derivatives.

For interval :math:`[t_i, t_{i+1}]`:

.. math::

   r(t) = a_i + b_i(t - t_i) + c_i(t - t_i)^2 + d_i(t - t_i)^3

**Advantages:** Smooth, continuous derivatives
**Disadvantages:** Can produce unrealistic forward rates

**Code Example:**

.. code-block:: python

   import dervflow

   # Create curve with cubic spline interpolation
   dates = [0.5, 1.0, 2.0, 5.0, 10.0]
   rates = [0.02, 0.025, 0.03, 0.035, 0.04]

   curve = dervflow.YieldCurve(dates, rates, method='cubic_spline')

   # Interpolate at any maturity
   rate_3y = curve.zero_rate(3.0)
   print(f"3-year rate: {rate_3y:.4f}")

Nelson-Siegel Model
~~~~~~~~~~~~~~~~~~~

Parametric model with four parameters:

.. math::

   r(t) = \beta_0 + \beta_1 \frac{1 - e^{-t/\tau}}{t/\tau} + \beta_2 \left(\frac{1 - e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\right)

**Parameters:**

* :math:`\beta_0`: Long-term level
* :math:`\beta_1`: Short-term component
* :math:`\beta_2`: Medium-term component
* :math:`\tau`: Decay parameter

**Advantages:** Smooth, economically interpretable, few parameters
**Disadvantages:** May not fit complex shapes

Nelson-Siegel-Svensson Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds a fourth term for better flexibility:

.. math::

   r(t) = \beta_0 + \beta_1 \frac{1 - e^{-t/\tau_1}}{t/\tau_1} + \beta_2 \left(\frac{1 - e^{-t/\tau_1}}{t/\tau_1} - e^{-t/\tau_1}\right) + \beta_3 \left(\frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2}\right)

**Code Example:**

.. code-block:: python

   import dervflow

   # Fit Nelson-Siegel model to market data
   dates = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
   rates = [0.015, 0.018, 0.02, 0.025, 0.028, 0.032, 0.035, 0.038, 0.04, 0.041]

   curve = dervflow.YieldCurve(dates, rates, method='nelson_siegel')

   # Get fitted parameters
   params = curve.get_parameters()
   print(f"β0 (level): {params['beta0']:.4f}")
   print(f"β1 (slope): {params['beta1']:.4f}")
   print(f"β2 (curvature): {params['beta2']:.4f}")

Bond Pricing and Analytics
---------------------------

Bond Price
~~~~~~~~~~

The price of a bond with cash flows :math:`C_i` at times :math:`t_i`:

.. math::

   P = \sum_{i=1}^n C_i \cdot D(t_i)

For a coupon bond:

.. math::

   P = \sum_{i=1}^n c \cdot D(t_i) + F \cdot D(T)

where :math:`c` is the coupon payment and :math:`F` is the face value.

Yield to Maturity (YTM)
~~~~~~~~~~~~~~~~~~~~~~~~

The YTM :math:`y` is the single discount rate that equates the bond price to the present value of cash flows:

.. math::

   P = \sum_{i=1}^n \frac{C_i}{(1 + y)^{t_i}}

YTM is found by solving this equation numerically (e.g., Newton-Raphson).

**Code Example:**

.. code-block:: python

   import dervflow

   # Calculate YTM
   bond_price = 102.5
   coupon = 0.03
   maturity = 5.0
   face_value = 100.0

   ytm = dervflow.bond_ytm(bond_price, coupon, maturity, face_value)
   print(f"Yield to maturity: {ytm:.4f}")

Duration
~~~~~~~~

**Macaulay Duration:**

Weighted average time to receive cash flows:

.. math::

   D_{\text{Mac}} = \frac{1}{P} \sum_{i=1}^n t_i \cdot C_i \cdot D(t_i)

**Modified Duration:**

Measures price sensitivity to yield changes:

.. math::

   D_{\text{Mod}} = \frac{D_{\text{Mac}}}{1 + y}

**Price change approximation:**

.. math::

   \frac{\Delta P}{P} \approx -D_{\text{Mod}} \cdot \Delta y

**Code Example:**

.. code-block:: python

   import dervflow

   # Calculate duration
   curve = dervflow.YieldCurve(dates, rates)

   duration = dervflow.bond_duration(
       coupon=0.03,
       maturity=5.0,
       yield_curve=curve
   )

   print(f"Macaulay duration: {duration['macaulay']:.2f} years")
   print(f"Modified duration: {duration['modified']:.2f}")

Convexity
~~~~~~~~~

Measures the curvature of the price-yield relationship:

.. math::

   C = \frac{1}{P} \sum_{i=1}^n t_i^2 \cdot C_i \cdot D(t_i)

**Price change with convexity:**

.. math::

   \frac{\Delta P}{P} \approx -D_{\text{Mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2

DV01 (Dollar Value of 01)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Change in bond price for a 1 basis point (0.01%) change in yield:

.. math::

   \text{DV01} = -D_{\text{Mod}} \cdot P \cdot 0.0001

Multi-Curve Framework
---------------------

Post-2008 financial crisis, different curves are used for discounting and forecasting.

OIS Discounting
~~~~~~~~~~~~~~~

Overnight Index Swap (OIS) rates are used for discounting collateralized cash flows.

LIBOR/SOFR Projection
~~~~~~~~~~~~~~~~~~~~~~

LIBOR (or SOFR) curves are used for projecting floating rate cash flows.

**Basis Spread:**

.. math::

   \text{Spread} = r_{\text{LIBOR}}(t) - r_{\text{OIS}}(t)

**Code Example:**

.. code-block:: python

   import dervflow

   # Build multi-curve framework
   ois_curve = dervflow.YieldCurve(ois_dates, ois_rates)
   libor_curve = dervflow.YieldCurve(libor_dates, libor_rates)

   # Price swap with different curves
   swap_value = dervflow.price_swap(
       notional=1000000,
       fixed_rate=0.03,
       maturity=5.0,
       discount_curve=ois_curve,
       projection_curve=libor_curve
   )

Interest Rate Models
--------------------

Short Rate Models
~~~~~~~~~~~~~~~~~

Model the instantaneous interest rate :math:`r_t`.

**Vasicek Model:**

.. math::

   dr_t = \kappa(\theta - r_t)dt + \sigma dW_t

* Mean-reverting
* Normally distributed (can be negative)

**Cox-Ingersoll-Ross (CIR) Model:**

.. math::

   dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t} dW_t

* Mean-reverting
* Always positive (if :math:`2\kappa\theta > \sigma^2`)

**Hull-White Model:**

.. math::

   dr_t = [\theta(t) - \kappa r_t]dt + \sigma dW_t

* Time-dependent mean reversion
* Calibrated to match initial term structure

Heath-Jarrow-Morton (HJM) Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models the entire forward rate curve:

.. math::

   df(t, T) = \alpha(t, T)dt + \sigma(t, T)dW_t

**No-arbitrage condition:**

.. math::

   \alpha(t, T) = \sigma(t, T) \int_t^T \sigma(t, s)ds

LIBOR Market Model (LMM)
~~~~~~~~~~~~~~~~~~~~~~~~~

Models forward LIBOR rates directly:

.. math::

   dL_i(t) = \mu_i(t)L_i(t)dt + \sigma_i(t)L_i(t)dW_i(t)

* Market-consistent
* Used for pricing caps, floors, swaptions

Practical Applications
----------------------

Pricing Fixed Income Securities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import dervflow

   # Build yield curve
   curve = dervflow.YieldCurve(dates, rates, method='cubic_spline')

   # Price a bond
   bond_price = dervflow.price_bond(
       coupon=0.04,
       maturity=10.0,
       face_value=100.0,
       yield_curve=curve
   )
   print(f"Bond price: ${bond_price:.2f}")

   # Calculate forward rate
   forward_rate = curve.forward_rate(2.0, 5.0)
   print(f"2y5y forward rate: {forward_rate:.4f}")

Interest Rate Risk Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import dervflow

   # Calculate bond portfolio risk
   portfolio = [
       {'coupon': 0.03, 'maturity': 5.0, 'notional': 1000000},
       {'coupon': 0.04, 'maturity': 10.0, 'notional': 2000000},
   ]

   curve = dervflow.YieldCurve(dates, rates)

   # Portfolio duration and DV01
   portfolio_duration = 0
   portfolio_dv01 = 0

   for bond in portfolio:
       duration = dervflow.bond_duration(
           bond['coupon'], bond['maturity'], curve
       )
       price = dervflow.price_bond(
           bond['coupon'], bond['maturity'], 100, curve
       )

       portfolio_duration += duration['modified'] * bond['notional'] * price / 100
       portfolio_dv01 += duration['modified'] * bond['notional'] * price / 100 * 0.0001

   print(f"Portfolio duration: {portfolio_duration:.2f}")
   print(f"Portfolio DV01: ${portfolio_dv01:.2f}")

See Also
--------

* :doc:`../api/yield_curve` - Yield curve API
* :doc:`../user_guide/yield_curves` - Practical yield curve construction
* :doc:`stochastic_processes` - Interest rate process simulation
