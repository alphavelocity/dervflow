Numerical Methods API
=====================

The numerical methods module provides foundational algorithms for integration, root finding, optimization,
linear algebra, and random number generation. All functionality is exposed through classes on the
:mod:`dervflow` top-level namespace.

.. currentmodule:: dervflow

Integration
-----------

Adaptive Quadrature Engines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaptiveSimpsonsIntegrator
   :members: integrate
   :show-inheritance:

.. autoclass:: AdaptiveGaussLegendreIntegrator
   :members: integrate
   :show-inheritance:

Fixed-Order Quadrature
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussLegendreIntegrator
   :members: integrate
   :show-inheritance:

.. code-block:: python

   import numpy as np
   import dervflow

   integrator = dervflow.AdaptiveSimpsonsIntegrator()

   def f(x):
       return np.exp(-x**2)

   result = integrator.integrate(f, 0.0, 1.0, tolerance=1e-6)
   print(f"Integral: {result.value:.6f}, error ≤ {result.error_estimate:.1e}")

Root Finding
------------

.. autoclass:: NewtonRaphsonSolver
   :members: solve
   :show-inheritance:

.. autoclass:: BrentSolver
   :members: solve
   :show-inheritance:

.. autoclass:: BisectionSolver
   :members: solve
   :show-inheritance:

.. autoclass:: SecantSolver
   :members: solve
   :show-inheritance:

.. code-block:: python

   import dervflow

   solver = dervflow.BrentSolver()

   def polynomial(x: float) -> float:
       return x**2 - 2

   result = solver.solve(polynomial, a=0.0, b=2.0)
   print(f"Root: {result.root:.6f}, converged={result.converged}")

Optimization
------------

.. autoclass:: GradientDescentOptimizer
   :members: optimize
   :show-inheritance:

.. autoclass:: BFGSOptimizer
   :members: optimize
   :show-inheritance:

.. autoclass:: NelderMeadOptimizer
   :members: optimize
   :show-inheritance:

.. code-block:: python

   import numpy as np
   import dervflow

   def rosenbrock(x: np.ndarray) -> float:
       return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

   def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
       return np.array([
           -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
           200 * (x[1] - x[0]**2),
       ])

   optimizer = dervflow.BFGSOptimizer()
   x0 = np.array([0.0, 0.0])
   result = optimizer.optimize(rosenbrock, rosenbrock_grad, x0)
   print(f"Minimum located at: {result.x}, converged={result.converged}")

Linear Algebra Utilities
------------------------

.. autoclass:: LinearAlgebra
   :members:
   :show-inheritance:

.. code-block:: python

   import numpy as np
   import dervflow

   la = dervflow.LinearAlgebra()
   matrix = np.array([[4.0, 2.0], [2.0, 3.0]])

   lower = la.cholesky(matrix)
   inverse = la.matrix_inverse(matrix)
   eigenvalues, eigenvectors = la.eigen_decomposition(matrix)

Random Number Generation
------------------------

.. autoclass:: RandomGenerator
   :members:
   :show-inheritance:

.. autoclass:: ThreadLocalRandom
   :members:
   :show-inheritance:

.. code-block:: python

   import dervflow

   rng = dervflow.RandomGenerator(seed=42)
   sample = rng.standard_normal_vec(1_000)
   uniform_draw = rng.uniform_range(-1.0, 1.0)

   tls_rng = dervflow.ThreadLocalRandom()
   tls_rng.seed(1234)
   scalar = tls_rng.normal(mean=0.0, std_dev=1.0)

Quasi-Random Sequences
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SobolSequence
   :members:
   :show-inheritance:

.. autoclass:: HaltonSequence
   :members:
   :show-inheritance:

.. code-block:: python

   import dervflow

   sobol = dervflow.SobolSequence(dimension=2)
   first_point = sobol.next_point()
   sample = sobol.generate(1024)

   halton = dervflow.HaltonSequence(dimension=3)
   grid = halton.generate(512)

Result Objects
--------------

.. autoclass:: IntegrationResult
   :members:
   :show-inheritance:

.. autoclass:: RootFindingResult
   :members:
   :show-inheritance:

.. autoclass:: OptimizationResult
   :members:
   :show-inheritance:

See Also
--------

* :doc:`../user_guide/options_pricing` - Using numerical methods for option pricing
* :doc:`../theory/black_scholes` - Mathematical background
