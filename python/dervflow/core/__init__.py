# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public interface for the Rust-backed core mathematics toolkit."""

from __future__ import annotations

from . import calc, combinatorics, series, stat, vectors
from ._backend import ArrayLike
from .calc import (cumulative_integral, curl, definite_integral, derivative,
                   divergence, gradient, second_derivative)
from .combinatorics import (bell_number, binomial_probability, catalan_number,
                            combination, factorial, falling_factorial,
                            lah_number, multinomial, permutation,
                            rising_factorial, stirling_number_first,
                            stirling_number_second)
from .series import (cumulative_max, cumulative_min, cumulative_product,
                     cumulative_sum, first_difference, moving_average)
from .stat import (central_moment, coefficient_of_variation, correlation,
                   covariance, geometric_mean, harmonic_mean,
                   interquartile_range, kurtosis, mean,
                   mean_absolute_deviation, median, median_absolute_deviation,
                   percentile, root_mean_square, skewness, standard_deviation,
                   variance, weighted_mean, z_scores)
from .vectors import (angle_between, chebyshev_distance, cosine_similarity,
                      cross_product, dot, euclidean_distance, hadamard_product,
                      lp_norm, manhattan_distance, norm, normalize, projection,
                      scalar_multiply, vector_add, vector_subtract)

__all__ = [
    "ArrayLike",
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
    "correlation",
    "covariance",
    "cumulative_sum",
    "cumulative_product",
    "cumulative_max",
    "cumulative_min",
    "first_difference",
    "moving_average",
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
    "derivative",
    "second_derivative",
    "definite_integral",
    "cumulative_integral",
    "gradient",
    "divergence",
    "curl",
    "stat",
    "vectors",
    "series",
    "combinatorics",
    "calc",
]

# ``skewness`` is implemented in ``stat`` but exposed here for a
# convenient flat namespace.
