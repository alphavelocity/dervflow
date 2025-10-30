# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Combinatoric helpers executed by the Rust ``Core`` backend."""

from __future__ import annotations

from collections.abc import Sequence

from ._backend import _core

__all__ = [
    "factorial",
    "permutation",
    "combination",
    "falling_factorial",
    "rising_factorial",
    "binomial_probability",
    "catalan_number",
    "stirling_number_second",
    "multinomial",
    "stirling_number_first",
    "bell_number",
    "lah_number",
]


def factorial(n: int) -> int:
    """Return ``n!`` computed in Rust."""

    return int(_core().factorial(int(n)))


def permutation(n: int, k: int) -> int:
    """Return the number of permutations of ``n`` items taken ``k`` at a time."""

    return int(_core().permutation(int(n), int(k)))


def combination(n: int, k: int) -> int:
    """Return the number of combinations of ``n`` items taken ``k`` at a time."""

    return int(_core().combination(int(n), int(k)))


def falling_factorial(n: int, k: int) -> int:
    """Return the falling factorial of ``n`` over ``k``."""

    return int(_core().falling_factorial(int(n), int(k)))


def rising_factorial(n: int, k: int) -> int:
    """Return the rising factorial of ``n`` over ``k``."""

    return int(_core().rising_factorial(int(n), int(k)))


def binomial_probability(n: int, k: int, p: float) -> float:
    """Return the binomial probability ``P(X = k)`` for parameters ``n`` and ``p``."""

    return float(_core().binomial_probability(int(n), int(k), float(p)))


def catalan_number(n: int) -> int:
    """Return the ``n``th Catalan number."""

    return int(_core().catalan_number(int(n)))


def stirling_number_second(n: int, k: int) -> int:
    """Return the Stirling number of the second kind ``S(n, k)``."""

    return int(_core().stirling_number_second(int(n), int(k)))


def multinomial(counts: Sequence[int]) -> int:
    """Return the multinomial coefficient for the provided counts."""

    return int(_core().multinomial([int(value) for value in counts]))


def stirling_number_first(n: int, k: int) -> int:
    """Return the (unsigned) Stirling number of the first kind ``c(n, k)``."""

    return int(_core().stirling_number_first(int(n), int(k)))


def bell_number(n: int) -> int:
    """Return the ``n``th Bell number."""

    return int(_core().bell_number(int(n)))


def lah_number(n: int, k: int) -> int:
    """Return the Lah number ``L(n, k)``."""

    return int(_core().lah_number(int(n), int(k)))
