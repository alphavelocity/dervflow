# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for wrapping the Rust core math bindings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Union

import numpy as np

from dervflow._dervflow import Core as _Core

ArrayLike = Union[Sequence[float], Iterable[float], np.ndarray]


@lru_cache(maxsize=1)
def _core() -> _Core:
    """Return a singleton instance of the Rust ``Core`` binding."""

    return _Core()


def _as_array(name: str, values: ArrayLike) -> np.ndarray:
    """Convert *values* to a contiguous ``float64`` NumPy vector."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64)
    elif isinstance(values, Iterable):
        array = np.asarray(list(values), dtype=np.float64)
    else:
        raise TypeError(f"{name} must be convertible to a one-dimensional array")

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array)


def _as_pair(
    name_a: str,
    a: ArrayLike,
    name_b: str,
    b: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return *a* and *b* coerced to ``float64`` vectors."""

    return _as_array(name_a, a), _as_array(name_b, b)
