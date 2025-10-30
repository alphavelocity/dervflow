# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import numpy as np
import pytest

import dervflow.core as core


def test_stat_enhancements() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0])

    assert math.isclose(core.root_mean_square(data), math.sqrt(7.5), rel_tol=1e-12)
    assert math.isclose(core.mean_absolute_deviation(data), 1.0, rel_tol=1e-12)

    # Default scaling corresponds to the robust normal-consistent factor.
    assert math.isclose(core.median_absolute_deviation(data), 1.482602218505602, rel_tol=1e-12)
    assert math.isclose(core.median_absolute_deviation(data, scale=None), 1.0, rel_tol=1e-12)

    cv = core.coefficient_of_variation(data)
    assert math.isclose(cv, 0.5163977794943222, rel_tol=1e-12)

    assert math.isclose(core.central_moment(data, 0), 1.0, rel_tol=1e-12)
    assert math.isclose(core.central_moment(data, 4), 2.5625, rel_tol=1e-12)
    assert math.isclose(core.central_moment(data, 3), 0.0, abs_tol=1e-12)

    with pytest.raises(ValueError):
        core.coefficient_of_variation([-1.0, 1.0])


def test_vector_norms_and_distances() -> None:
    vec = np.array([1.0, -2.0, 3.0])
    other = np.array([4.0, 2.0, 9.0])

    assert math.isclose(core.lp_norm(vec, 1.0), 6.0, rel_tol=1e-12)
    assert math.isclose(core.lp_norm(vec, 2.0), math.sqrt(14.0), rel_tol=1e-12)
    assert math.isclose(core.lp_norm(vec, 3.0), 36.0 ** (1.0 / 3.0), rel_tol=1e-12)
    assert math.isclose(core.lp_norm(vec, math.inf), 3.0, rel_tol=1e-12)

    with pytest.raises(ValueError):
        core.lp_norm(vec, 0.5)

    with pytest.raises(ValueError):
        core.lp_norm(vec, float("nan"))

    assert math.isclose(core.euclidean_distance(vec, other), math.sqrt(61.0), rel_tol=1e-12)
    assert math.isclose(core.manhattan_distance(vec, other), 13.0, rel_tol=1e-12)
    assert math.isclose(core.chebyshev_distance(vec, other), 6.0, rel_tol=1e-12)

    with pytest.raises(ValueError):
        core.euclidean_distance([1.0], [1.0, 2.0])


def test_combinatorics_extensions() -> None:
    assert core.catalan_number(0) == 1
    assert core.catalan_number(6) == 132

    assert core.stirling_number_second(0, 0) == 1
    assert core.stirling_number_second(5, 2) == 15
    assert core.stirling_number_second(6, 3) == 90
    assert core.stirling_number_second(5, 0) == 0

    assert core.stirling_number_first(5, 2) == 50
    assert core.stirling_number_first(7, 3) == 1624
    assert core.stirling_number_first(5, 0) == 0

    assert core.bell_number(0) == 1
    assert core.bell_number(6) == 203

    assert core.lah_number(0, 0) == 1
    assert core.lah_number(5, 2) == 240
    assert core.lah_number(6, 3) == 1200

    assert core.multinomial([2, 1, 1]) == 12
    assert core.multinomial([3, 0, 2]) == 10

    with pytest.raises(ValueError):
        core.stirling_number_second(3, 5)

    with pytest.raises(ValueError):
        core.stirling_number_first(2, 5)

    with pytest.raises(ValueError):
        core.multinomial([])

    with pytest.raises(ValueError):
        core.lah_number(3, 5)


def test_calculus_operations() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 200)
    y = np.sin(x)
    dy = core.derivative(y, spacing=x[1] - x[0])
    d2y = core.second_derivative(y, spacing=x[1] - x[0])

    assert np.allclose(dy, np.cos(x), atol=5e-2)
    assert np.allclose(d2y, -np.sin(x), atol=5e-2)

    integral = core.definite_integral(np.cos(x), spacing=x[1] - x[0])
    assert pytest.approx(integral, rel=1e-6) == np.sin(x[-1]) - np.sin(x[0])

    cumulative = core.cumulative_integral(np.ones_like(x), spacing=0.5)
    assert np.allclose(cumulative, np.linspace(0.0, 0.5 * (len(x) - 1), len(x)))

    grid_x = np.linspace(-1.0, 1.0, 6)
    grid_y = np.linspace(-1.0, 1.0, 5)
    dx = grid_x[1] - grid_x[0]
    dy_spacing = grid_y[1] - grid_y[0]

    scalar_values = []
    vec_x = []
    vec_y = []
    for x_val in grid_x:
        for y_val in grid_y:
            scalar_values.append(x_val * x_val + 3.0 * y_val)
            vec_x.append(2.0 * x_val)
            vec_y.append(3.0 * y_val)

    grad = core.gradient(scalar_values, shape=[len(grid_x), len(grid_y)], spacings=[dx, dy_spacing])
    grad = grad.reshape(len(grid_x), len(grid_y), 2)
    mid_grad = grad[len(grid_x) // 2, len(grid_y) // 2]
    assert pytest.approx(mid_grad[0], rel=5e-2, abs=5e-2) == 2.0 * grid_x[len(grid_x) // 2]
    assert pytest.approx(mid_grad[1], rel=5e-2, abs=5e-2) == 3.0

    field = np.concatenate([vec_x, vec_y])
    divergence = core.divergence(field, shape=[len(grid_x), len(grid_y)], spacings=[dx, dy_spacing])
    assert np.allclose(divergence, 5.0, atol=5e-2)

    grid_z = np.linspace(-1.0, 1.0, 4)
    dz = grid_z[1] - grid_z[0]
    vec_x3 = []
    vec_y3 = []
    vec_z3 = []
    for x_val in grid_x:
        for y_val in grid_y:
            for z_val in grid_z:
                vec_x3.append(-y_val)
                vec_y3.append(x_val)
                vec_z3.append(0.0)

    field3 = np.concatenate([vec_x3, vec_y3, vec_z3])
    curl_values = core.curl(
        field3,
        shape=[len(grid_x), len(grid_y), len(grid_z)],
        spacings=[dx, dy_spacing, dz],
    )
    curl_values = curl_values.reshape(len(grid_x), len(grid_y), len(grid_z), 3)
    mid_curl = curl_values[len(grid_x) // 2, len(grid_y) // 2, len(grid_z) // 2]
    assert pytest.approx(mid_curl[2], rel=5e-2, abs=5e-2) == 2.0
