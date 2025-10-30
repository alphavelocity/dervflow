# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Statistical measures powered by the Rust ``Core`` bindings."""

from __future__ import annotations

import math

import numpy as np

from ._backend import ArrayLike, _as_array, _as_pair, _core

__all__ = [
    "mean",
    "geometric_mean",
    "harmonic_mean",
    "weighted_mean",
    "root_mean_square",
    "mean_absolute_deviation",
    "median_absolute_deviation",
    "coefficient_of_variation",
    "central_moment",
    "variance",
    "standard_deviation",
    "median",
    "percentile",
    "interquartile_range",
    "skewness",
    "kurtosis",
    "z_scores",
    "covariance",
    "correlation",
]


def mean(data: ArrayLike) -> float:
    """Return the arithmetic mean of *data*."""

    return _core().mean(_as_array("data", data))


def geometric_mean(data: ArrayLike) -> float:
    """Return the geometric mean of *data*."""

    return _core().geometric_mean(_as_array("data", data))


def harmonic_mean(data: ArrayLike) -> float:
    """Return the harmonic mean of *data*."""

    return _core().harmonic_mean(_as_array("data", data))


def weighted_mean(data: ArrayLike, weights: ArrayLike) -> float:
    """Return the weighted mean of *data* with *weights*."""

    arr_data, arr_weights = _as_pair("data", data, "weights", weights)
    return _core().weighted_mean(arr_data, arr_weights)


def root_mean_square(data: ArrayLike) -> float:
    """Return the root-mean-square of *data*."""

    return _core().root_mean_square(_as_array("data", data))


def mean_absolute_deviation(data: ArrayLike) -> float:
    """Return the mean absolute deviation of *data* around its mean."""

    return _core().mean_absolute_deviation(_as_array("data", data))


def median_absolute_deviation(
    data: ArrayLike,
    scale: float | None = 1.482_602_218_505_602,
) -> float:
    """Return the (optionally scaled) median absolute deviation of *data*."""

    arr = _as_array("data", data)
    if scale is None:
        scale_arg = 1.0
    else:
        scale_arg = float(scale)
        if not math.isfinite(scale_arg) or scale_arg <= 0.0:
            raise ValueError("MAD scale factor must be finite and positive")

    return _core().median_absolute_deviation(arr, scale=scale_arg)


def coefficient_of_variation(data: ArrayLike, unbiased: bool = True) -> float:
    """Return the coefficient of variation of *data*."""

    return _core().coefficient_of_variation(_as_array("data", data), unbiased=unbiased)


def central_moment(data: ArrayLike, order: int) -> float:
    """Return the central moment of *data* of integer ``order``."""

    arr = _as_array("data", data)
    order_int = int(order)
    if order_int < 0:
        raise ValueError("moment order must be non-negative")

    return _core().central_moment(arr, order_int)


def variance(data: ArrayLike, unbiased: bool = True) -> float:
    """Return the variance of *data*."""

    return _core().variance(_as_array("data", data), unbiased=unbiased)


def standard_deviation(data: ArrayLike, unbiased: bool = True) -> float:
    """Return the standard deviation of *data*."""

    return _core().standard_deviation(_as_array("data", data), unbiased=unbiased)


def median(data: ArrayLike) -> float:
    """Return the median of *data*."""

    return _core().median(_as_array("data", data))


def percentile(data: ArrayLike, value: float) -> float:
    """Return the ``value`` percentile of *data* (``value`` in ``[0, 1]``)."""

    return _core().percentile(_as_array("data", data), float(value))


def interquartile_range(data: ArrayLike) -> float:
    """Return the interquartile range of *data*."""

    return _core().interquartile_range(_as_array("data", data))


def skewness(data: ArrayLike) -> float:
    """Return the skewness of *data*."""

    return _core().skewness(_as_array("data", data))


def kurtosis(data: ArrayLike) -> float:
    """Return the kurtosis of *data*."""

    return _core().kurtosis(_as_array("data", data))


def z_scores(data: ArrayLike) -> np.ndarray:
    """Return the z-scores of *data*."""

    result = _core().z_scores(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def covariance(x: ArrayLike, y: ArrayLike, unbiased: bool = False) -> float:
    """Return the covariance between *x* and *y*."""

    arr_x, arr_y = _as_pair("x", x, "y", y)
    return _core().covariance(arr_x, arr_y, unbiased=unbiased)


def correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Return the Pearson correlation between *x* and *y*."""

    arr_x, arr_y = _as_pair("x", x, "y", y)
    return _core().correlation(arr_x, arr_y)
