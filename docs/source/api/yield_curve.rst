Yield Curve and Fixed Income API
================================

.. currentmodule:: dervflow

The fixed income toolkit covers single- and multi-curve term structures, bond
analytics and swap scheduling helpers. All classes are available directly from
the :mod:`dervflow` namespace.

YieldCurve
----------

.. autoclass:: YieldCurve
   :members: zero_rate, forward_rate, discount_factor, price_bond, rates, times
   :show-inheritance:

Create :class:`YieldCurve` with a grid of maturities and zero rates. Methods
provide discount factors, forward rates, bond pricing and access to the
underlying term structure arrays.

.. code-block:: python

   import numpy as np
   import dervflow

   maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
   rates = np.array([0.02, 0.022, 0.025, 0.03, 0.035, 0.04])
   curve = dervflow.YieldCurve(maturities, rates, method="cubic_spline")

   df_3y = curve.discount_factor(3.0)
   fwd_1y2y = curve.forward_rate(1.0, 2.0)
   analytics = dervflow.BondAnalytics()
   cashflows = analytics.generate_cashflows(
       maturity=5.0,
       coupon_rate=0.03,
       face_value=100.0,
       frequency=2,
   )
   bond = curve.price_bond(cashflows)

YieldCurveBuilder
-----------------

.. autoclass:: YieldCurveBuilder
   :members: bootstrap_from_bonds, bootstrap_from_swaps, from_nelson_siegel, from_nelson_siegel_svensson
   :show-inheritance:

Factory methods for constructing :class:`YieldCurve` instances from market
instruments or parametric curve families.

BondAnalytics
-------------

.. autoclass:: BondAnalytics
   :members: bond_price, yield_to_maturity, macaulay_duration, modified_duration, convexity, dv01, generate_cashflows
   :show-inheritance:

Analytical helpers for bond pricing and risk metrics (duration, convexity, DV01)
with utilities to generate coupon cashflows.

MultiCurve
----------

.. autoclass:: MultiCurve
   :members: discount_curve, discount_factor, forward_curve, forward_curve_names, set_forward_curve, add_forward_curve, forward_rate, par_swap_rate, present_value, price_payer_swap
   :show-inheritance:

The multi-curve container tracks discounting and forwarding curves separately.
Use :meth:`add_forward_curve` or :meth:`set_forward_curve` to register new
forward indices and evaluate swap pricing via :meth:`price_payer_swap` or
:meth:`par_swap_rate`.

SwapPeriod
----------

.. autoclass:: SwapPeriod
   :members: start, end, year_fraction, as_tuple
   :show-inheritance:

Represents accrual periods used for swap scheduling and cashflow generation.

See Also
--------

* :doc:`../user_guide/yield_curves` – building curves from market data
* :doc:`../theory/yield_curves` – theoretical background on interest rate term
  structures
