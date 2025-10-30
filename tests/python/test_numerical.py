# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for numerical module bindings."""

import math

import numpy as np
import pytest

from dervflow.numerical import (
    AdaptiveGaussLegendreIntegrator,
    AdaptiveSimpsonsIntegrator,
    BFGSOptimizer,
    BisectionSolver,
    BrentSolver,
    GaussLegendreIntegrator,
    GradientDescentOptimizer,
    HaltonSequence,
    LinearAlgebra,
    NelderMeadOptimizer,
    NewtonRaphsonSolver,
    RandomGenerator,
    SecantSolver,
    SobolSequence,
    ThreadLocalRandom,
)


def test_integration_algorithms_agree():
    """Adaptive Simpson and Gauss-Legendre should integrate x^2 accurately."""

    def func(x: float) -> float:
        return x * x

    adaptive = AdaptiveSimpsonsIntegrator()
    gauss = AdaptiveGaussLegendreIntegrator()
    fixed = GaussLegendreIntegrator()

    adaptive_result = adaptive.integrate(func, 0.0, 1.0)
    ga_result = gauss.integrate(func, 0.0, 1.0)
    fixed_result = fixed.integrate(func, 0.0, 1.0, n_points=5)

    exact = 1.0 / 3.0
    for result in (adaptive_result, ga_result, fixed_result):
        assert result.converged
        assert abs(result.value - exact) < 1e-6


def test_root_finding_methods():
    """All root solvers should recover sqrt(2) for x^2 - 2."""

    def func(x: float) -> float:
        return x * x - 2.0

    def dfunc(x: float) -> float:
        return 2.0 * x

    newton = NewtonRaphsonSolver()
    brent = BrentSolver()
    bisection = BisectionSolver()
    secant = SecantSolver()

    nr_res = newton.solve(func, dfunc, initial_guess=1.0)
    br_res = brent.solve(func, a=0.0, b=2.0)
    bi_res = bisection.solve(func, a=0.0, b=2.0)
    se_res = secant.solve(func, x0=0.5, x1=2.0)

    root = math.sqrt(2.0)
    for result in (nr_res, br_res, bi_res, se_res):
        assert result.converged
        assert abs(result.root - root) < 1e-6


def test_gradient_based_optimization():
    """Gradient descent and BFGS should minimize quadratic bowl."""

    def objective(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    def gradient(x: np.ndarray) -> list[float]:
        return (2.0 * x).tolist()

    x0 = np.array([3.0, -4.0])

    gd = GradientDescentOptimizer()
    gd_res = gd.optimize(objective, gradient, x0)
    assert gd_res.converged
    assert np.allclose(gd_res.x, np.zeros_like(x0), atol=1e-6)

    bfgs = BFGSOptimizer()
    bfgs_res = bfgs.optimize(objective, gradient, x0)
    assert bfgs_res.converged
    assert np.allclose(bfgs_res.x, np.zeros_like(x0), atol=1e-6)

    nm = NelderMeadOptimizer()
    nm_res = nm.optimize(objective, x0)
    assert nm_res.converged
    assert np.allclose(nm_res.x, np.zeros_like(x0), atol=1e-4)


def test_linear_algebra_operations():
    """Cholesky and linear system solvers should match NumPy."""

    la = LinearAlgebra()

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])

    chol = la.cholesky(matrix)
    assert np.allclose(chol @ chol.T, matrix, atol=1e-10)

    solution = la.solve_linear_system(matrix, rhs)
    assert np.allclose(matrix @ solution, rhs, atol=1e-10)

    assert la.is_positive_definite(matrix)

    det = la.determinant(matrix)
    tr = la.trace(matrix)
    assert det == pytest.approx(11.0, abs=1e-12)
    assert tr == pytest.approx(7.0, abs=1e-12)

    matrix_sq = la.matrix_power(matrix, 2)
    assert np.allclose(matrix_sq, matrix @ matrix, atol=1e-10)

    product = la.matrix_multiply(matrix, np.eye(2))
    assert np.allclose(product, matrix, atol=1e-12)

    inverse = la.matrix_inverse(matrix)
    assert np.allclose(inverse, np.linalg.inv(matrix), atol=1e-10)

    # Least squares on over-determined system
    ls_matrix = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    ls_rhs = np.array([1.0, 2.0, 2.0])
    ls_solution = la.solve_least_squares(ls_matrix, ls_rhs)
    normal_solution = np.linalg.lstsq(ls_matrix, ls_rhs, rcond=None)[0]
    assert np.allclose(ls_solution, normal_solution, atol=1e-8)

    q, r = la.qr_decomposition(ls_matrix)
    assert q.shape == (3, 2)
    assert r.shape == (2, 2)
    assert np.allclose(q @ r, ls_matrix, atol=1e-10)
    assert np.allclose(q.T @ q, np.eye(2), atol=1e-10)

    u, s, v_t = la.svd(ls_matrix)
    sigma = np.diag(s)
    reconstructed = u @ sigma @ v_t
    assert np.allclose(reconstructed, ls_matrix, atol=1e-10)

    pseudo_inv = la.pseudo_inverse(ls_matrix)
    assert np.allclose(ls_matrix @ pseudo_inv @ ls_matrix, ls_matrix, atol=1e-10)

    rank = la.matrix_rank(ls_matrix)
    assert rank == 2

    l, u, p = la.lu_decomposition(matrix)
    assert np.allclose(l @ u, p @ matrix, atol=1e-10)

    fro_norm = la.matrix_norm(matrix, ord="fro")
    assert fro_norm == pytest.approx(np.linalg.norm(matrix, ord="fro"), rel=1e-12)

    one_norm = la.matrix_norm(matrix, ord="1")
    assert one_norm == pytest.approx(np.linalg.norm(matrix, ord=1), rel=1e-12)

    spectral_cond = la.condition_number(matrix, ord="spectral")
    expected_cond = np.linalg.cond(matrix, p=2)
    assert spectral_cond == pytest.approx(expected_cond, rel=1e-12)

    eigenvalues, eigenvectors = la.eigen_decomposition(matrix)
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    assert np.allclose(reconstructed, matrix, atol=1e-10)

    exp_matrix = la.matrix_exponential(np.diag([1.0, 2.0]))
    assert np.allclose(exp_matrix, np.diag(np.exp([1.0, 2.0])), atol=1e-12)

    correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = np.array([[0.1, -0.2], [0.3, 0.4]])
    correlated = la.correlate_samples(correlation, samples)
    expected_correlated = la.cholesky(correlation) @ samples
    assert np.allclose(correlated, expected_correlated, atol=1e-12)

    perturbed = np.array([[1.0, 1.2], [0.8, 1.0]])
    nearest = la.nearest_positive_definite(perturbed)
    assert np.allclose(np.diag(nearest), np.ones(2), atol=1e-10)
    assert np.all(np.linalg.eigvalsh(nearest) > 0.0)


def test_random_generators_reproducible():
    """RandomGenerator with seed should be reproducible and match thread local."""

    rng1 = RandomGenerator(seed=42)
    rng2 = RandomGenerator(seed=42)

    seq1 = rng1.standard_normal_vec(5)
    seq2 = rng2.standard_normal_vec(5)
    assert np.allclose(seq1, seq2)

    tl = ThreadLocalRandom()
    tl.seed(1234)
    values = [tl.standard_normal() for _ in range(3)]
    tl.seed(1234)
    values_again = [tl.standard_normal() for _ in range(3)]
    assert np.allclose(values, values_again)


def test_random_generators_validate_std_dev():
    """RNG wrappers should reject non-positive standard deviations."""

    rng = RandomGenerator(seed=1)
    with pytest.raises(ValueError):
        rng.normal(0.0, 0.0)

    tl = ThreadLocalRandom()
    with pytest.raises(ValueError):
        tl.normal(0.0, -1.0)


def test_quasi_random_sequences_shape():
    """Sobol and Halton sequences should produce arrays of expected shape."""

    sobol = SobolSequence(dimension=3)
    points = sobol.generate(4)
    assert points.shape == (4, 3)

    halton = HaltonSequence(dimension=2)
    point = halton.next_point()
    assert point.shape == (2,)
