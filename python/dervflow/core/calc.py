# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Calculus utilities backed by the Rust core implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np

from ._backend import ArrayLike, _as_array, _core

__all__ = [
    "derivative",
    "second_derivative",
    "definite_integral",
    "cumulative_integral",
    "gradient",
    "normalized_gradient",
    "gradient_magnitude",
    "directional_derivative",
    "divergence",
    "curl",
    "laplacian",
    "vector_laplacian",
    "jacobian",
    "hessian",
]


def _validate_shape(shape: Sequence[int]) -> list[int]:
    if not isinstance(shape, Sequence):
        raise TypeError("shape must be a sequence of positive integers")

    result: list[int] = []
    for dim in shape:
        dim_int = int(dim)
        if dim_int <= 0:
            raise ValueError("shape dimensions must be positive integers")
        result.append(dim_int)

    if not result:
        raise ValueError("shape must contain at least one dimension")

    return result


def _validate_spacings(spacings: Iterable[float], expected_len: int) -> np.ndarray:
    spacing_arr = np.asarray(list(spacings) if not isinstance(spacings, np.ndarray) else spacings)
    spacing_arr = np.asarray(spacing_arr, dtype=np.float64)

    if spacing_arr.ndim != 1:
        raise ValueError("spacings must be a one-dimensional sequence")
    if spacing_arr.size != expected_len:
        raise ValueError("spacings length must match number of dimensions")
    if not np.all(np.isfinite(spacing_arr)):
        raise ValueError("spacings must contain only finite values")
    if np.any(spacing_arr <= 0.0):
        raise ValueError("spacings must be strictly positive")

    return np.ascontiguousarray(spacing_arr, dtype=np.float64)


def derivative(data: ArrayLike, spacing: float = 1.0) -> np.ndarray:
    """Return the first derivative of equally spaced samples."""

    arr = _as_array("data", data)
    spacing_f = float(spacing)
    result = _core().derivative(arr, spacing_f)
    return np.asarray(result, dtype=np.float64)


def second_derivative(data: ArrayLike, spacing: float = 1.0) -> np.ndarray:
    """Return the second derivative of equally spaced samples."""

    arr = _as_array("data", data)
    spacing_f = float(spacing)
    result = _core().second_derivative(arr, spacing_f)
    return np.asarray(result, dtype=np.float64)


def definite_integral(data: ArrayLike, spacing: float = 1.0) -> float:
    """Return the trapezoidal integral of *data* with uniform spacing."""

    arr = _as_array("data", data)
    spacing_f = float(spacing)
    return float(_core().definite_integral(arr, spacing_f))


def cumulative_integral(data: ArrayLike, spacing: float = 1.0) -> np.ndarray:
    """Return the cumulative trapezoidal integral of *data*."""

    arr = _as_array("data", data)
    spacing_f = float(spacing)
    result = _core().cumulative_integral(arr, spacing_f)
    return np.asarray(result, dtype=np.float64)


def gradient(values: ArrayLike, shape: Sequence[int], spacings: Sequence[float]) -> np.ndarray:
    """Return the gradient of a scalar field defined on a regular grid."""

    arr = _as_array("values", values)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    result = _core().gradient(arr, dims, spacing_arr)
    gradient_arr = np.asarray(result, dtype=np.float64)
    return gradient_arr.reshape(-1, len(dims))


def normalized_gradient(
    values: ArrayLike, shape: Sequence[int], spacings: Sequence[float]
) -> np.ndarray:
    """Return unit vectors pointing in the direction of steepest ascent."""

    arr = _as_array("values", values)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    result = _core().normalized_gradient(arr, dims, spacing_arr)
    normalized_arr = np.asarray(result, dtype=np.float64)
    return normalized_arr.reshape(-1, len(dims))


def gradient_magnitude(values: ArrayLike, shape: Sequence[int], spacings: Sequence[float]) -> np.ndarray:
    """Return the magnitude of the gradient of a scalar field."""

    arr = _as_array("values", values)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    result = _core().gradient_magnitude(arr, dims, spacing_arr)
    magnitude_arr = np.asarray(result, dtype=np.float64)
    return magnitude_arr.reshape(tuple(dims))


def directional_derivative(
    values: ArrayLike,
    shape: Sequence[int],
    spacings: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    """Return the directional derivative of a scalar field along *direction*."""

    arr = _as_array("values", values)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    direction_arr = np.asarray(direction, dtype=np.float64)
    if direction_arr.ndim != 1:
        raise ValueError("direction must be a one-dimensional sequence")
    if direction_arr.size != len(dims):
        raise ValueError("direction length must match the number of dimensions")
    if not np.all(np.isfinite(direction_arr)):
        raise ValueError("direction must contain only finite values")
    if float(np.linalg.norm(direction_arr)) == 0.0:
        raise ValueError("direction must have non-zero magnitude")

    direction_arr = np.ascontiguousarray(direction_arr, dtype=np.float64)
    result = _core().directional_derivative(arr, dims, spacing_arr, direction_arr)
    derivative_arr = np.asarray(result, dtype=np.float64)
    return derivative_arr.reshape(tuple(dims))


def divergence(
    field: ArrayLike,
    shape: Sequence[int],
    spacings: Sequence[float],
) -> np.ndarray:
    """Return the divergence of a vector field sampled on a regular grid."""

    arr = _as_array("field", field)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    result = _core().divergence(arr, dims, spacing_arr)
    return np.asarray(result, dtype=np.float64)


def curl(field: ArrayLike, shape: Sequence[int], spacings: Sequence[float]) -> np.ndarray:
    """Return the curl of a 3D vector field sampled on a regular grid."""

    arr = _as_array("field", field)
    dims = _validate_shape(shape)
    if len(dims) != 3:
        raise ValueError("curl is only defined for three-dimensional grids")

    spacing_arr = _validate_spacings(spacings, len(dims))
    result = _core().curl(arr, dims, spacing_arr)
    curl_arr = np.asarray(result, dtype=np.float64)
    return curl_arr.reshape(-1, 3)


def laplacian(values: ArrayLike, shape: Sequence[int], spacings: Sequence[float]) -> np.ndarray:
    """Return the discrete Laplacian of a scalar field on a regular grid."""

    arr = _as_array("values", values)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    result = _core().laplacian(arr, dims, spacing_arr)
    return np.asarray(result, dtype=np.float64)


def vector_laplacian(
    field: ArrayLike, shape: Sequence[int], spacings: Sequence[float]
) -> np.ndarray:
    """Return the Laplacian of each component of a vector field."""

    arr = _as_array("field", field)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    total_points = int(np.prod(dims, dtype=int))
    if arr.size != total_points * len(dims):
        raise ValueError("field length must equal number of grid points times dimensions")

    result = _core().vector_laplacian(arr, dims, spacing_arr)
    return np.asarray(result, dtype=np.float64)


def jacobian(field: ArrayLike, shape: Sequence[int], spacings: Sequence[float]) -> np.ndarray:
    """Return the Jacobian matrices for a vector field sampled on a grid."""

    arr = _as_array("field", field)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    total_points = int(np.prod(dims, dtype=int))
    if total_points == 0:
        raise ValueError("shape must contain at least one grid point")
    if arr.size % total_points != 0:
        raise ValueError("field length must be a multiple of the number of grid points")

    result = _core().jacobian(arr, dims, spacing_arr)
    components = arr.size // total_points
    jacobian_arr = np.asarray(result, dtype=np.float64)
    return jacobian_arr.reshape(total_points, components, len(dims))


def hessian(values: ArrayLike, shape: Sequence[int], spacings: Sequence[float]) -> np.ndarray:
    """Return the Hessian matrices for a scalar field sampled on a grid."""

    arr = _as_array("values", values)
    dims = _validate_shape(shape)
    spacing_arr = _validate_spacings(spacings, len(dims))

    result = _core().hessian(arr, dims, spacing_arr)
    total_points = int(np.prod(dims, dtype=int))
    hessian_arr = np.asarray(result, dtype=np.float64)
    return hessian_arr.reshape(total_points, len(dims), len(dims))
