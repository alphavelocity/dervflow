# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pytest configuration and fixtures for dervflow tests

This module provides shared fixtures and configuration for all test modules.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Ensure the local Python package is importable without installation
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_SRC = _REPO_ROOT / "python"
if _PYTHON_SRC.is_dir():
    sys.path.insert(0, str(_PYTHON_SRC))
else:
    raise RuntimeError(f"Expected Python package directory at {_PYTHON_SRC}, but it was not found")


# Test data fixtures
@pytest.fixture
def standard_option_params():
    """Standard option parameters for testing"""
    return {
        'spot': 100.0,
        'strike': 100.0,
        'rate': 0.05,
        'dividend': 0.0,
        'volatility': 0.2,
        'time': 1.0
    }


@pytest.fixture
def itm_call_params():
    """In-the-money call option parameters"""
    return {
        'spot': 110.0,
        'strike': 100.0,
        'rate': 0.05,
        'dividend': 0.0,
        'volatility': 0.2,
        'time': 1.0,
        'option_type': 'call'
    }


@pytest.fixture
def otm_put_params():
    """Out-of-the-money put option parameters"""
    return {
        'spot': 110.0,
        'strike': 100.0,
        'rate': 0.05,
        'dividend': 0.0,
        'volatility': 0.2,
        'time': 1.0,
        'option_type': 'put'
    }


@pytest.fixture
def sample_returns():
    """Sample returns data for risk calculations"""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)


@pytest.fixture
def sample_prices():
    """Sample price series for time series analysis"""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def correlation_matrix_2d():
    """2x2 correlation matrix"""
    return np.array([[1.0, 0.5], [0.5, 1.0]])


@pytest.fixture
def correlation_matrix_3d():
    """3x3 correlation matrix"""
    return np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])


@pytest.fixture
def sample_bond_data():
    """Sample bond data for yield curve construction"""
    return [
        {'maturity': 0.25, 'rate': 0.02},
        {'maturity': 0.5, 'rate': 0.025},
        {'maturity': 1.0, 'rate': 0.03},
        {'maturity': 2.0, 'rate': 0.035},
        {'maturity': 5.0, 'rate': 0.04},
        {'maturity': 10.0, 'rate': 0.045}
    ]


# Tolerance fixtures for numerical comparisons
@pytest.fixture
def tight_tolerance():
    """Tight tolerance for numerical comparisons"""
    return 1e-10


@pytest.fixture
def standard_tolerance():
    """Standard tolerance for numerical comparisons"""
    return 1e-6


@pytest.fixture
def loose_tolerance():
    """Loose tolerance for convergence tests"""
    return 0.01


# Random seed fixture
@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility"""
    np.random.seed(42)
    yield
    # Cleanup if needed
