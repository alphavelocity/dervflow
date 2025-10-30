# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Portfolio optimization module

Provides portfolio construction and optimization tools:
- Mean-variance optimization
- Risk parity allocation
- Efficient frontier calculation
- Portfolio constraints handling
- Portfolio risk analytics (risk contributions, VaR, CVaR, summaries)
"""

from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dervflow._dervflow import PortfolioOptimizer as _PortfolioOptimizer
    from dervflow._dervflow import RiskParityOptimizer as _RiskParityOptimizer
    from dervflow._dervflow import BlackLittermanModel as _BlackLittermanModel
    from dervflow._dervflow import InvestorViews as _InvestorViews
    from dervflow._dervflow import FactorModel as _FactorModel
else:
    from dervflow._dervflow import PortfolioOptimizer as _PortfolioOptimizer
    from dervflow._dervflow import RiskParityOptimizer as _RiskParityOptimizer
    from dervflow._dervflow import BlackLittermanModel as _BlackLittermanModel
    from dervflow._dervflow import InvestorViews as _InvestorViews
    from dervflow._dervflow import FactorModel as _FactorModel

__all__ = [
    'PortfolioOptimizer',
    'RiskParityOptimizer',
    'BlackLittermanModel',
    'InvestorViews',
    'FactorModel',
]


# Re-export PyO3 classes with Pythonic docstrings for convenience
BlackLittermanModel = _BlackLittermanModel
InvestorViews = _InvestorViews
FactorModel = _FactorModel

BlackLittermanModel.__doc__ = """Black-Litterman portfolio construction model.

The model combines market equilibrium returns with optional investor views to
produce posterior expected returns and an updated covariance matrix. Use
``posterior`` to retrieve the adjusted distribution and optimal weights.
"""

InvestorViews.__doc__ = """Container describing investor views for Black-Litterman.

Specify a pick matrix selecting asset exposures and a vector of view returns.
Optionally provide a view uncertainty matrix via :meth:`with_uncertainty`.
"""

FactorModel.__doc__ = """Ordinary least squares multi-factor regression model.

The model estimates factor exposures (betas) and intercepts for one or more
assets from historical return data. It supports computing expected returns
from factor premia, portfolio factor exposures, and factor-based performance
attribution directly from Python.

Parameters
----------
asset_returns : np.ndarray
    Matrix of asset returns with shape ``(observations, assets)``
factor_returns : np.ndarray
    Matrix of factor returns with shape ``(observations, factors)``
include_intercept : bool, optional
    Whether to include a regression intercept (default True)
factor_names : Sequence[str], optional
    Optional names for the factors; defaults to ``["factor_0", ...]``
"""


class PortfolioOptimizer:
    """
    Portfolio optimizer using mean-variance optimization.
    
    This class provides methods for portfolio optimization using various
    objectives including minimum variance, target return, and maximum Sharpe ratio.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset (1D array)
    covariance : np.ndarray
        Covariance matrix (2D array, n_assets x n_assets)
    
    Examples
    --------
    >>> import numpy as np
    >>> from dervflow.portfolio import PortfolioOptimizer
    >>> 
    >>> # Define expected returns and covariance
    >>> returns = np.array([0.10, 0.12, 0.08])
    >>> cov = np.array([[0.04, 0.01, 0.005],
    ...                 [0.01, 0.09, 0.01],
    ...                 [0.005, 0.01, 0.0225]])
    >>> 
    >>> # Create optimizer
    >>> optimizer = PortfolioOptimizer(returns, cov)
    >>> 
    >>> # Minimize variance for target return
    >>> result = optimizer.optimize(target_return=0.10)
    >>> print(f"Optimal weights: {result['weights']}")
    >>> print(f"Portfolio volatility: {result['volatility']:.4f}")
    >>> 
    >>> # Maximize Sharpe ratio
    >>> result = optimizer.optimize(risk_free_rate=0.03)
    >>> print(f"Sharpe ratio: {result['sharpe_ratio']:.4f}")
    >>> 
    >>> # Generate efficient frontier
    >>> frontier = optimizer.efficient_frontier(num_points=20)
    >>> returns_list = [p['expected_return'] for p in frontier]
    >>> risks_list = [p['volatility'] for p in frontier]
    """
    
    def __init__(
        self, 
        expected_returns: NDArray[np.float64], 
        covariance: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Initialize portfolio optimizer.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns for each asset (1D) or historical returns (2D)
        covariance : np.ndarray, optional
            Covariance matrix (required if expected_returns is 1D)
        """
        returns_array = np.asarray(expected_returns, dtype=np.float64)
        
        if covariance is not None:
            cov_array = np.asarray(covariance, dtype=np.float64)
            self._optimizer = _PortfolioOptimizer(returns_array, cov_array)
            self.expected_returns = returns_array
            self.covariance = cov_array
            self.n_assets = len(returns_array)
        else:
            # Assume returns_array is 2D historical returns
            self._optimizer = _PortfolioOptimizer(returns_array)
            # Calculate expected returns and covariance from historical data
            self.expected_returns = np.mean(returns_array, axis=0)
            self.covariance = np.cov(returns_array, rowvar=False)
            self.n_assets = returns_array.shape[1]
    
    def optimize(
        self,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
        min_weights: Optional[NDArray[np.float64]] = None,
        max_weights: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """
        Optimize portfolio for a given objective.
        
        Parameters
        ----------
        target_return : float, optional
            Target return for minimum variance optimization
        target_risk : float, optional
            Target risk (volatility) for maximum return optimization
        risk_free_rate : float, optional
            Risk-free rate for Sharpe ratio maximization
        min_weights : np.ndarray, optional
            Minimum weight for each asset (default: 0)
        max_weights : np.ndarray, optional
            Maximum weight for each asset (default: 1)
        
        Returns
        -------
        dict
            Dictionary containing:
            - weights: Optimal portfolio weights (np.ndarray)
            - expected_return: Expected portfolio return (float)
            - volatility: Portfolio volatility (float)
            - sharpe_ratio: Sharpe ratio if risk_free_rate provided (float or None)
            - status: Optimization status (str)
        
        Notes
        -----
        If no objective is specified, minimizes portfolio variance.
        Only one of target_return, target_risk, or risk_free_rate should be specified.
        
        Examples
        --------
        >>> # Minimum variance portfolio
        >>> result = optimizer.optimize()
        >>> 
        >>> # Target return of 10%
        >>> result = optimizer.optimize(target_return=0.10)
        >>> 
        >>> # Maximum Sharpe ratio with 3% risk-free rate
        >>> result = optimizer.optimize(risk_free_rate=0.03)
        >>> 
        >>> # With box constraints
        >>> result = optimizer.optimize(
        ...     target_return=0.10,
        ...     min_weights=np.array([0.1, 0.1, 0.1]),
        ...     max_weights=np.array([0.5, 0.5, 0.5])
        ... )
        """
        # Convert weight bounds to numpy arrays if provided
        if min_weights is not None:
            min_weights = np.asarray(min_weights, dtype=np.float64)
        if max_weights is not None:
            max_weights = np.asarray(max_weights, dtype=np.float64)
        
        return self._optimizer.optimize(
            target_return=target_return,
            target_risk=target_risk,
            risk_free_rate=risk_free_rate,
            min_weights=min_weights,
            max_weights=max_weights,
        )
    
    def efficient_frontier(
        self,
        num_points: int,
        min_weights: Optional[NDArray[np.float64]] = None,
        max_weights: Optional[NDArray[np.float64]] = None,
    ) -> List[Dict[str, float]]:
        """
        Generate efficient frontier points.
        
        Parameters
        ----------
        num_points : int
            Number of points to generate along the frontier
        min_weights : np.ndarray, optional
            Minimum weight for each asset
        max_weights : np.ndarray, optional
            Maximum weight for each asset
        
        Returns
        -------
        list of dict
            List of optimization results for each frontier point
        
        Examples
        --------
        >>> frontier = optimizer.efficient_frontier(num_points=20)
        >>> 
        >>> # Extract returns and risks for plotting
        >>> returns = [p['expected_return'] for p in frontier]
        >>> risks = [p['volatility'] for p in frontier]
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(risks, returns)
        >>> plt.xlabel('Risk (Volatility)')
        >>> plt.ylabel('Expected Return')
        >>> plt.title('Efficient Frontier')
        >>> plt.show()
        """
        if min_weights is not None:
            min_weights = np.asarray(min_weights, dtype=np.float64)
        if max_weights is not None:
            max_weights = np.asarray(max_weights, dtype=np.float64)
        
        return self._optimizer.efficient_frontier(
            num_points=num_points,
            min_weights=min_weights,
            max_weights=max_weights,
        )
    
    def portfolio_return(self, weights: NDArray[np.float64]) -> float:
        """
        Calculate portfolio return for given weights.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        float
            Expected portfolio return
        
        Examples
        --------
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> ret = optimizer.portfolio_return(weights)
        >>> print(f"Portfolio return: {ret:.4f}")
        """
        weights = np.asarray(weights, dtype=np.float64)
        return self._optimizer.portfolio_return(weights)
    
    def portfolio_volatility(self, weights: NDArray[np.float64]) -> float:
        """
        Calculate portfolio volatility for given weights.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        float
            Portfolio volatility (standard deviation)
        
        Examples
        --------
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> vol = optimizer.portfolio_volatility(weights)
        >>> print(f"Portfolio volatility: {vol:.4f}")
        """
        weights = np.asarray(weights, dtype=np.float64)
        return self._optimizer.portfolio_volatility(weights)
    
    def sharpe_ratio(self, weights: NDArray[np.float64], risk_free_rate: float) -> float:
        """
        Calculate Sharpe ratio for given weights.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        risk_free_rate : float
            Risk-free rate
        
        Returns
        -------
        float
            Sharpe ratio
        
        Examples
        --------
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> sharpe = optimizer.sharpe_ratio(weights, risk_free_rate=0.03)
        >>> print(f"Sharpe ratio: {sharpe:.4f}")
        """
        weights = np.asarray(weights, dtype=np.float64)
        return float(self._optimizer.sharpe_ratio(weights, risk_free_rate))

    def risk_contributions(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return percentage risk contributions for each asset.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (must sum to 1)

        Returns
        -------
        np.ndarray
            Percentage risk contribution of each asset (sums to 1)

        Examples
        --------
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> rc = optimizer.risk_contributions(weights)
        >>> print(f"Risk contributions: {rc}")
        """

        weights = np.asarray(weights, dtype=np.float64)
        return np.asarray(self._optimizer.risk_contributions(weights), dtype=np.float64)

    def value_at_risk(
        self,
        weights: NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> float:
        """Compute parametric Value at Risk (VaR) for the portfolio.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, optional
            Confidence level for the loss threshold (default 0.95)

        Returns
        -------
        float
            Value at Risk expressed as a positive loss amount
        """

        weights = np.asarray(weights, dtype=np.float64)
        return float(self._optimizer.value_at_risk(weights, confidence_level))

    def conditional_value_at_risk(
        self,
        weights: NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> float:
        """Compute parametric Conditional Value at Risk (Expected Shortfall).

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, optional
            Confidence level for the tail expectation (default 0.95)

        Returns
        -------
        float
            Expected loss beyond the VaR threshold
        """

        weights = np.asarray(weights, dtype=np.float64)
        return float(
            self._optimizer.conditional_value_at_risk(weights, confidence_level)
        )

    def portfolio_summary(
        self,
        weights: NDArray[np.float64],
        risk_free_rate: Optional[float] = None,
    ) -> Dict[str, np.ndarray | float | None]:
        """Return a comprehensive portfolio risk summary.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        risk_free_rate : float, optional
            Risk-free rate for Sharpe ratio calculation

        Returns
        -------
        dict
            Summary containing expected return, variance, volatility, Sharpe
            ratio, diversification metrics, and risk contributions
        """

        weights = np.asarray(weights, dtype=np.float64)
        return self._optimizer.portfolio_summary(weights, risk_free_rate)


class RiskParityOptimizer:
    """
    Risk parity portfolio optimizer.
    
    Implements risk parity allocation where each asset contributes equally
    to the total portfolio risk.
    
    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix (2D array, n_assets x n_assets)
    
    Examples
    --------
    >>> import numpy as np
    >>> from dervflow.portfolio import RiskParityOptimizer
    >>> 
    >>> # Define covariance matrix
    >>> cov = np.array([[0.04, 0.01, 0.005],
    ...                 [0.01, 0.09, 0.01],
    ...                 [0.005, 0.01, 0.0225]])
    >>> 
    >>> # Create optimizer
    >>> optimizer = RiskParityOptimizer(cov)
    >>> 
    >>> # Equal risk contribution
    >>> weights = optimizer.optimize()
    >>> print(f"Risk parity weights: {weights}")
    >>> 
    >>> # Check risk contributions
    >>> rc = optimizer.risk_contributions(weights)
    >>> print(f"Risk contributions: {rc}")
    >>> 
    >>> # Custom risk contributions
    >>> target_rc = np.array([0.5, 0.3, 0.2])
    >>> weights = optimizer.optimize(target_risk_contributions=target_rc)
    """
    
    def __init__(self, covariance: NDArray[np.float64]) -> None:
        """
        Initialize risk parity optimizer.
        
        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix
        """
        self._optimizer = _RiskParityOptimizer(
            np.asarray(covariance, dtype=np.float64)
        )
        self.covariance: NDArray[np.float64] = np.asarray(covariance, dtype=np.float64)
        self.n_assets: int = covariance.shape[0]
    
    def optimize(
        self,
        target_risk_contributions: Optional[NDArray[np.float64]] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """
        Optimize portfolio using risk parity approach.
        
        Parameters
        ----------
        target_risk_contributions : np.ndarray, optional
            Target risk contributions for each asset (must sum to 1)
            Default: equal risk contributions (1/n for each asset)
        max_iterations : int, optional
            Maximum number of iterations (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-8)
        
        Returns
        -------
        np.ndarray
            Optimal portfolio weights
        
        Examples
        --------
        >>> # Equal risk contribution
        >>> weights = optimizer.optimize()
        >>> 
        >>> # Custom risk contributions (50% from asset 1, 30% from asset 2, 20% from asset 3)
        >>> target_rc = np.array([0.5, 0.3, 0.2])
        >>> weights = optimizer.optimize(target_risk_contributions=target_rc)
        """
        if target_risk_contributions is not None:
            target_risk_contributions = np.asarray(target_risk_contributions, dtype=np.float64)
        
        return self._optimizer.optimize(
            target_risk_contributions=target_risk_contributions,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    
    def risk_contributions(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate risk contributions for given weights.
        
        Risk contribution of asset i measures how much asset i contributes
        to the total portfolio risk.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        np.ndarray
            Risk contributions for each asset (sum to 1)
        
        Examples
        --------
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> rc = optimizer.risk_contributions(weights)
        >>> print(f"Risk contributions: {rc}")
        >>> print(f"Sum of risk contributions: {rc.sum():.6f}")
        """
        weights = np.asarray(weights, dtype=np.float64)
        return self._optimizer.risk_contributions(weights)
