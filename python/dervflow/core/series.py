# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sequential series transforms implemented in Rust."""

from __future__ import annotations

import numpy as np

from ._backend import ArrayLike, _as_array, _core

__all__ = [
    "cumulative_sum",
    "cumulative_product",
    "cumulative_max",
    "cumulative_min",
    "first_difference",
    "moving_average",
]


def cumulative_sum(data: ArrayLike) -> np.ndarray:
    """Return the cumulative sum of *data*."""

    result = _core().cumulative_sum(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def cumulative_product(data: ArrayLike) -> np.ndarray:
    """Return the cumulative product of *data*."""

    result = _core().cumulative_product(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def cumulative_max(data: ArrayLike) -> np.ndarray:
    """Return the cumulative maximum of *data*."""

    result = _core().cumulative_max(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def cumulative_min(data: ArrayLike) -> np.ndarray:
    """Return the cumulative minimum of *data*."""

    result = _core().cumulative_min(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def first_difference(data: ArrayLike) -> np.ndarray:
    """Return the first difference of *data*."""

    result = _core().first_difference(_as_array("data", data))
    return np.asarray(result, dtype=np.float64)


def moving_average(data: ArrayLike, window_size: int) -> np.ndarray:
    """Return the simple moving average of *data*."""

    result = _core().moving_average(_as_array("data", data), int(window_size))
    return np.asarray(result, dtype=np.float64)
