Yield Curve Guide
=================

DervFlow provides high-performance tools for constructing and interrogating yield
curves. The key classes are :class:`dervflow.YieldCurve` for interpolation and
analytics, :class:`dervflow.YieldCurveBuilder` for bootstrapping from market
instruments, and :class:`dervflow.BondAnalytics` for bond-specific measures.

Bootstrapping Curves
--------------------

``YieldCurveBuilder`` exposes convenience constructors for common datasets. The
inputs are simple tuples, so no Pandas dependency is required.

From Bond Quotes
~~~~~~~~~~~~~~~~

.. code-block:: python

   import dervflow

   bonds = [
       (0.5, 0.00, 99.5, 2),   # 6M zero-coupon
       (1.0, 0.03, 100.2, 2),  # 1Y coupon bond
       (2.0, 0.04, 101.5, 2),  # 2Y coupon bond
       (5.0, 0.045, 103.0, 2), # 5Y coupon bond
   ]

   builder = dervflow.YieldCurveBuilder()
   curve = builder.bootstrap_from_bonds(bonds)

   print(f"Bootstrapped tenors: {curve.times().shape[0]}")
   print(f"1Y zero rate: {curve.zero_rate(1.0):.4%}")

From Swap Quotes
~~~~~~~~~~~~~~~~

.. code-block:: python

   swaps = [
       (1.0, 0.025, 2),
       (2.0, 0.030, 2),
       (5.0, 0.035, 2),
       (10.0, 0.040, 2),
   ]

   swap_curve = builder.bootstrap_from_swaps(swaps)
   print(f"10Y zero rate: {swap_curve.zero_rate(10.0):.4%}")

Model-based Curves
~~~~~~~~~~~~~~~~~~

Nelson-Siegel and Nelson-Siegel-Svensson helper constructors are also available.
Pass the factor parameters and evaluation times to obtain a calibrated curve.

.. code-block:: python

   import numpy as np

   times = np.linspace(0.25, 30.0, 20)
   ns_curve = builder.from_nelson_siegel(
       beta0=0.03,
       beta1=-0.02,
       beta2=-0.01,
       lambda_=1.2,
       times=times,
   )
   print(f"5Y NS zero rate: {ns_curve.zero_rate(5.0):.4%}")

Direct Construction
-------------------

If zero rates are already known, instantiate :class:`dervflow.YieldCurve`
directly. Interpolation methods include ``'linear'``, ``'cubic_spline_natural'``,
``'cubic_spline_clamped'``, ``'nelson_siegel'``, and ``'nelson_siegel_svensson'``.

.. code-block:: python

   times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
   rates = np.array([0.02, 0.024, 0.028, 0.033, 0.037])

   curve = dervflow.YieldCurve(times, rates, method='cubic_spline_natural')
   print(f"Discount factor (5Y): {curve.discount_factor(5.0):.6f}")
   print(f"Forward rate 1Y→2Y: {curve.forward_rate(1.0, 2.0):.4%}")

The underlying time points and rates are retrievable as NumPy arrays:

.. code-block:: python

   print(curve.times())
   print(curve.rates())

Bond Analytics
--------------

The :class:`dervflow.BondAnalytics` helper works with cash-flow schedules
expressed as ``(time, amount)`` tuples.

.. code-block:: python

   analytics = dervflow.BondAnalytics()

   cashflows = analytics.generate_cashflows(
       maturity=5.0,
       coupon_rate=0.04,
       face_value=100.0,
       frequency=2,
   )

   # Price from a yield and derive analytics
   price = analytics.bond_price(yield_rate=0.032, cashflows=cashflows)
   ytm = analytics.yield_to_maturity(price=price, cashflows=cashflows)
   macaulay = analytics.macaulay_duration(ytm, cashflows)
   modified = analytics.modified_duration(ytm, cashflows, frequency=2)
   convexity = analytics.convexity(ytm, cashflows)
   dv01 = analytics.dv01(ytm, cashflows)

   print(f"Bond price: {price:.2f}")
   print(f"Yield to maturity: {ytm:.4%}")
   print(f"Macaulay duration: {macaulay:.2f} years")
   print(f"Modified duration: {modified:.2f}")
   print(f"Convexity: {convexity:.2f}")
   print(f"DV01: {dv01:.4f}")

Next Steps
----------

* :doc:`../api/yield_curve` – API reference for curve and analytics types.
* :doc:`../theory/yield_curves` – Mathematical background for interpolation and
  bootstrapping routines.
