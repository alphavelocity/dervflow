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
    
    running_mean = core.cumulative_mean(data)
    assert np.allclose(running_mean, np.array([1.0, 1.5, 2.0, 2.5]))

    running_variance = core.cumulative_variance(data)
    expected_variance = np.array([0.0, 0.5, 1.0, 5.0 / 3.0])
    assert np.allclose(running_variance, expected_variance)

    population_variance = core.cumulative_variance(data, unbiased=False)
    assert math.isclose(population_variance[-1], 5.0 / 4.0, rel_tol=1e-12)

    running_std = core.cumulative_std(data)
    expected_std = np.array([0.0, math.sqrt(0.5), 1.0, math.sqrt(5.0 / 3.0)])
    assert np.allclose(running_std, expected_std)

    population_std = core.cumulative_std(data, unbiased=False)
    assert math.isclose(population_std[-1], math.sqrt(5.0 / 4.0), rel_tol=1e-12)

    skew_source = np.array([1.0, 2.0, 3.0, 6.0])
    running_skew = core.cumulative_skewness(skew_source)
    expected_skew = np.array([0.0, 0.0, 0.0, 1.1903401282789947])
    assert np.allclose(running_skew, expected_skew)

    population_skew = core.cumulative_skewness(skew_source, unbiased=False)
    assert math.isclose(population_skew[-1], 0.6872431934890912, rel_tol=1e-12)

    running_kurtosis = core.cumulative_kurtosis(skew_source)
    expected_kurtosis = np.array([0.0, 0.0, 0.0, 1.5])
    assert np.allclose(running_kurtosis, expected_kurtosis)

    population_kurtosis = core.cumulative_kurtosis(skew_source, unbiased=False)
    assert math.isclose(population_kurtosis[-1], -1.0, rel_tol=1e-12)


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

    unit_grad = core.normalized_gradient(
        scalar_values,
        shape=[len(grid_x), len(grid_y)],
        spacings=[dx, dy_spacing],
    )
    unit_grad = unit_grad.reshape(len(grid_x), len(grid_y), 2)
    centre_unit = unit_grad[len(grid_x) // 2, len(grid_y) // 2]
    expected_vec = np.array([2.0 * grid_x[len(grid_x) // 2], 3.0])
    expected_unit = expected_vec / np.linalg.norm(expected_vec)
    assert np.allclose(centre_unit, expected_unit, atol=5e-2)

    grad_mag = core.gradient_magnitude(
        scalar_values,
        shape=[len(grid_x), len(grid_y)],
        spacings=[dx, dy_spacing],
    )
    grad_mag = grad_mag.reshape(len(grid_x), len(grid_y))
    edge_grad_mag = grad_mag[1, len(grid_y) - 2]
    expected_edge = math.sqrt((2.0 * grid_x[1]) ** 2 + 3.0**2)
    assert pytest.approx(edge_grad_mag, rel=5e-2, abs=5e-2) == expected_edge

    direction_result = core.directional_derivative(
        scalar_values,
        shape=[len(grid_x), len(grid_y)],
        spacings=[dx, dy_spacing],
        direction=[1.0, 1.0],
    )
    direction_result = direction_result.reshape(len(grid_x), len(grid_y))
    centre_direction = direction_result[len(grid_x) // 2, len(grid_y) // 2]
    expected_direction = (2.0 * grid_x[len(grid_x) // 2] + 3.0) / math.sqrt(2.0)
    assert pytest.approx(centre_direction, rel=5e-2, abs=5e-2) == expected_direction

    field = np.concatenate([vec_x, vec_y])
    divergence = core.divergence(field, shape=[len(grid_x), len(grid_y)], spacings=[dx, dy_spacing])
    assert np.allclose(divergence, 5.0, atol=5e-2)

    lap = core.laplacian(scalar_values, shape=[len(grid_x), len(grid_y)], spacings=[dx, dy_spacing])
    lap = lap.reshape(len(grid_x), len(grid_y))
    interior = lap[1:-1, 1:-1]
    assert np.allclose(interior, 2.0, atol=1e-1)

    component_one = []
    component_two = []
    for x_val in grid_x:
        for y_val in grid_y:
            component_one.append(2.0 * x_val + y_val)
            component_two.append(x_val - 3.0 * y_val)

    vector_field = np.concatenate([component_one, component_two])
    jac = core.jacobian(
        vector_field,
        shape=[len(grid_x), len(grid_y)],
        spacings=[dx, dy_spacing],
    )
    jac = jac.reshape(len(grid_x), len(grid_y), 2, 2)
    centre = jac[len(grid_x) // 2, len(grid_y) // 2]
    assert np.allclose(centre[0], [2.0, 1.0], atol=5e-2)
    assert np.allclose(centre[1], [1.0, -3.0], atol=5e-2)

    hessian_values = []
    for x_val in grid_x:
        for y_val in grid_y:
            hessian_values.append(3.0 * x_val * x_val + x_val * y_val + 2.0 * y_val * y_val)

    hess = core.hessian(
        hessian_values,
        shape=[len(grid_x), len(grid_y)],
        spacings=[dx, dy_spacing],
    )
    hess = hess.reshape(len(grid_x), len(grid_y), 2, 2)
    centre_hess = hess[len(grid_x) // 2, len(grid_y) // 2]
    assert np.allclose(centre_hess, [[6.0, 1.0], [1.0, 4.0]], atol=1e-1)

    vector_lap = core.vector_laplacian(
        vector_field,
        shape=[len(grid_x), len(grid_y)],
        spacings=[dx, dy_spacing],
    )
    vector_lap = vector_lap.reshape(len(grid_x), len(grid_y), 2)
    assert np.allclose(vector_lap, 0.0, atol=5e-2)

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

    with pytest.raises(ValueError):
        core.directional_derivative(
            scalar_values,
            shape=[len(grid_x), len(grid_y)],
            spacings=[dx, dy_spacing],
            direction=[0.0, 0.0],
        )
