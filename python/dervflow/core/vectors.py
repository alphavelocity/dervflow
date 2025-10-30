# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vector algebra helpers wrapping the Rust ``Core`` implementation."""

from __future__ import annotations

import numpy as np

from ._backend import ArrayLike, _as_array, _as_pair, _core

__all__ = [
    "dot",
    "hadamard_product",
    "norm",
    "lp_norm",
    "normalize",
    "cosine_similarity",
    "angle_between",
    "euclidean_distance",
    "manhattan_distance",
    "chebyshev_distance",
    "vector_add",
    "vector_subtract",
    "scalar_multiply",
    "cross_product",
    "projection",
]


def dot(a: ArrayLike, b: ArrayLike) -> float:
    """Return the dot product of vectors *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    return _core().dot(arr_a, arr_b)


def hadamard_product(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Return the element-wise product of *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    result = _core().hadamard_product(arr_a, arr_b)
    return np.asarray(result, dtype=np.float64)


def norm(data: ArrayLike) -> float:
    """Return the Euclidean norm of *data*."""

    return _core().norm(_as_array("data", data))


def lp_norm(data: ArrayLike, p: float = 2.0) -> float:
    """Return the generalised ``L^p`` norm of *data*."""

    return _core().lp_norm(_as_array("data", data), float(p))


def normalize(data: ArrayLike) -> np.ndarray:
    """Return *data* scaled to unit norm."""

    result = _core().normalize(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Return the cosine similarity between *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    return _core().cosine_similarity(arr_a, arr_b)


def angle_between(a: ArrayLike, b: ArrayLike) -> float:
    """Return the angle between *a* and *b* in radians."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    return _core().angle_between(arr_a, arr_b)


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Return the Euclidean distance between *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    return _core().euclidean_distance(arr_a, arr_b)


def manhattan_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Return the Manhattan (``L^1``) distance between *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    return _core().manhattan_distance(arr_a, arr_b)


def chebyshev_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Return the Chebyshev (``L^∞``) distance between *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    return _core().chebyshev_distance(arr_a, arr_b)


def vector_add(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Return the vector sum of *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    result = _core().vector_add(arr_a, arr_b)
    return np.asarray(result, dtype=np.float64)


def vector_subtract(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Return the vector difference ``a - b``."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    result = _core().vector_subtract(arr_a, arr_b)
    return np.asarray(result, dtype=np.float64)


def scalar_multiply(data: ArrayLike, scalar: float) -> np.ndarray:
    """Return *data* scaled by ``scalar``."""

    result = _core().scalar_multiply(_as_array("data", data), float(scalar))
    return np.asarray(result, dtype=np.float64)


def cross_product(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Return the 3D cross product of *a* and *b*."""

    arr_a, arr_b = _as_pair("a", a, "b", b)
    result = _core().cross_product(arr_a, arr_b)
    return np.asarray(result, dtype=np.float64)


def projection(a: ArrayLike, onto: ArrayLike) -> np.ndarray:
    """Project vector *a* onto *onto*."""

    arr_a, arr_onto = _as_pair("a", a, "onto", onto)
    result = _core().projection(arr_a, arr_onto)
    return np.asarray(result, dtype=np.float64)
