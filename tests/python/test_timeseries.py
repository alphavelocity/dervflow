# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for time series analysis module

Tests cover:
- Return calculations (simple, log, continuous)
- Statistical measures (mean, variance, skewness, kurtosis)
- Correlation analysis (ACF, PACF, Pearson, Spearman, Kendall)
- GARCH model estimation
- Statistical tests (ADF, KPSS, Ljung-Box, Jarque-Bera)
"""

import pytest
import numpy as np
from dervflow import TimeSeriesAnalyzer


class TestReturnCalculations:
    """Test return calculation methods"""

    def test_simple_returns(self):
        """Test simple return calculation"""
        prices = np.array([100.0, 105.0, 103.0, 108.0])
        analyzer = TimeSeriesAnalyzer(prices)
        
        returns = analyzer.returns(method='simple')
        
        # Simple returns: (P_t - P_{t-1}) / P_{t-1}
        expected = np.array([0.05, -0.019047619, 0.048543689])
        
        assert len(returns) == len(prices) - 1
        np.testing.assert_array_almost_equal(returns, expected, decimal=6)

    def test_log_returns(self):
        """Test logarithmic return calculation"""
        prices = np.array([100.0, 105.0, 103.0, 108.0])
        analyzer = TimeSeriesAnalyzer(prices)
        
        returns = analyzer.returns(method='log')
        
        # Log returns: ln(P_t / P_{t-1})
        expected = np.log(prices[1:] / prices[:-1])
        
        assert len(returns) == len(prices) - 1
        np.testing.assert_array_almost_equal(returns, expected, decimal=10)

    def test_continuous_returns(self):
        """Test continuous compounded return calculation"""
        prices = np.array([100.0, 105.0, 103.0, 108.0])
        analyzer = TimeSeriesAnalyzer(prices)
        
        returns = analyzer.returns(method='continuous')
        
        # Continuous returns should equal log returns
        log_returns = analyzer.returns(method='log')
        
        assert len(returns) == len(prices) - 1
        np.testing.assert_array_almost_equal(returns, log_returns, decimal=10)

    def test_rolling_returns(self):
        """Test rolling return calculation"""
        prices = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 107.0])
        analyzer = TimeSeriesAnalyzer(prices)
        
        rolling_returns = analyzer.returns(method='simple', window=2)
        
        # Rolling returns over 2-period windows
        # With window=2, we get returns for each consecutive pair
        assert len(rolling_returns) == len(prices) - 1
        assert all(np.isfinite(rolling_returns))

    def test_returns_invalid_method(self):
        """Test error handling for invalid return method"""
        prices = np.array([100.0, 105.0, 103.0])
        analyzer = TimeSeriesAnalyzer(prices)
        
        with pytest.raises(ValueError, match="Invalid return type"):
            analyzer.returns(method='invalid')


class TestStatisticalMeasures:
    """Test statistical measure calculations"""

    def test_stat_basic(self):
        """Test basic statistical measures"""
        np.random.seed(42)
        data = np.random.normal(0.001, 0.02, 100)
        analyzer = TimeSeriesAnalyzer(data)
        
        stats = analyzer.stat()

        assert 'mean' in stats
        assert 'variance' in stats
        assert 'std_dev' in stats
        assert 'std' in stats
        assert 'std_error' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        assert 'count' in stats
        assert 'sum' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'range' in stats
        assert 'median' in stats
        assert 'q1' in stats
        assert 'q3' in stats
        assert 'iqr' in stats
        assert 'mean_abs_dev' in stats
        assert 'median_abs_dev' in stats
        assert 'root_mean_square' in stats

        assert stats['count'] == 100
        assert stats['std_dev'] > 0
        assert stats['std'] == stats['std_dev']
        assert stats['min'] <= stats['median'] <= stats['max']
        assert stats['range'] >= 0
        assert stats['iqr'] >= 0

    def test_stat_mean(self):
        """Test mean calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        stats = analyzer.stat()
        
        assert abs(stats['mean'] - 3.0) < 1e-10

    def test_stat_variance(self):
        """Test variance calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        stats = analyzer.stat()
        
        # Sample variance
        expected_var = np.var(data, ddof=1)
        assert abs(stats['variance'] - expected_var) < 1e-10

    def test_stat_std_dev(self):
        """Test standard deviation calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        stats = analyzer.stat()
        
        expected_std = np.std(data, ddof=1)
        assert abs(stats['std_dev'] - expected_std) < 1e-10

    def test_stat_skewness(self):
        """Test skewness calculation"""
        # Create positively skewed data
        data = np.array([1.0, 1.0, 1.0, 2.0, 10.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        stats = analyzer.stat()
        
        # Should be positive for right-skewed data
        assert stats['skewness'] > 0

    def test_stat_kurtosis(self):
        """Test kurtosis calculation"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        analyzer = TimeSeriesAnalyzer(data)

        stats = analyzer.stat()

        # Normal distribution should have excess kurtosis near 0
        assert abs(stats['kurtosis']) < 1.0

    def test_stat_additional_metrics(self):
        """Test extended descriptive stat metrics"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)

        stats = analyzer.stat()

        assert abs(stats['sum'] - 15.0) < 1e-10
        assert abs(stats['median'] - 3.0) < 1e-10
        assert abs(stats['q1'] - 2.0) < 1e-10
        assert abs(stats['q3'] - 4.0) < 1e-10
        assert abs(stats['iqr'] - 2.0) < 1e-10
        assert abs(stats['mean_abs_dev'] - 1.2) < 1e-10
        assert abs(stats['median_abs_dev'] - 1.0) < 1e-10
        assert abs(stats['root_mean_square'] - np.sqrt(11.0)) < 1e-10
        assert abs(stats['std_error'] - np.sqrt(2.5) / np.sqrt(5.0)) < 1e-10


class TestCorrelationAnalysis:
    """Test correlation analysis methods"""

    def test_autocorrelation_basic(self):
        """Test basic autocorrelation calculation"""
        np.random.seed(42)
        data = np.random.randn(100)
        analyzer = TimeSeriesAnalyzer(data)
        
        acf = analyzer.autocorrelation(max_lag=10)
        
        assert len(acf) == 11  # lag 0 to max_lag
        assert abs(acf[0] - 1.0) < 1e-10  # ACF at lag 0 should be 1

    def test_autocorrelation_white_noise(self):
        """Test autocorrelation for white noise"""
        np.random.seed(42)
        data = np.random.randn(500)
        analyzer = TimeSeriesAnalyzer(data)
        
        acf = analyzer.autocorrelation(max_lag=20)
        
        # For white noise, ACF should be near zero for lags > 0
        assert all(abs(acf[1:]) < 0.2)

    def test_partial_autocorrelation_basic(self):
        """Test basic partial autocorrelation calculation"""
        np.random.seed(42)
        data = np.random.randn(100)
        analyzer = TimeSeriesAnalyzer(data)
        
        pacf = analyzer.partial_autocorrelation(max_lag=10)
        
        assert len(pacf) == 11  # lag 0 to max_lag
        assert abs(pacf[0] - 1.0) < 1e-10  # PACF at lag 0 should be 1

    def test_pearson_correlation(self):
        """Test Pearson correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        analyzer = TimeSeriesAnalyzer(x)
        
        corr = analyzer.correlation(y, method='pearson')
        
        # Perfect positive correlation
        assert abs(corr - 1.0) < 1e-10

    def test_pearson_correlation_negative(self):
        """Test Pearson correlation with negative relationship"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        analyzer = TimeSeriesAnalyzer(x)
        
        corr = analyzer.correlation(y, method='pearson')
        
        # Perfect negative correlation
        assert abs(corr - (-1.0)) < 1e-10

    def test_spearman_correlation(self):
        """Test Spearman rank correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # Non-linear but monotonic
        analyzer = TimeSeriesAnalyzer(x)
        
        corr = analyzer.correlation(y, method='spearman')
        
        # Perfect monotonic relationship
        assert abs(corr - 1.0) < 1e-10

    def test_kendall_correlation(self):
        """Test Kendall tau correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        analyzer = TimeSeriesAnalyzer(x)
        
        corr = analyzer.correlation(y, method='kendall')
        
        # Perfect concordance
        assert abs(corr - 1.0) < 1e-10

    def test_correlation_invalid_method(self):
        """Test error handling for invalid correlation method"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        analyzer = TimeSeriesAnalyzer(x)
        
        with pytest.raises(ValueError, match="Invalid correlation method"):
            analyzer.correlation(y, method='invalid')


class TestGARCHModel:
    """Test GARCH model estimation"""

    def test_garch_standard_basic(self):
        """Test basic GARCH(1,1) estimation"""
        np.random.seed(42)
        # Generate returns with volatility clustering
        returns = np.random.randn(500) * 0.01
        analyzer = TimeSeriesAnalyzer(returns)
        
        result = analyzer.fit_garch(variant='standard')
        
        assert 'omega' in result
        assert 'alpha' in result
        assert 'beta' in result
        assert 'log_likelihood' in result
        assert 'conditional_variances' in result
        
        # Parameters should be positive
        assert result['omega'] > 0
        assert result['alpha'] >= 0
        assert result['beta'] >= 0

    def test_garch_parameter_constraints(self):
        """Test GARCH parameter constraints"""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        analyzer = TimeSeriesAnalyzer(returns)
        
        result = analyzer.fit_garch(variant='standard')
        
        # Alpha + Beta should be less than 1 for stationarity
        assert result['alpha'] + result['beta'] < 1.0

    def test_garch_conditional_variances(self):
        """Test GARCH conditional variance output"""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        analyzer = TimeSeriesAnalyzer(returns)
        
        result = analyzer.fit_garch(variant='standard')
        
        cond_var = result['conditional_variances']
        assert len(cond_var) == len(returns)
        assert all(cond_var > 0)  # Variances must be positive

    def test_garch_egarch_variant(self):
        """Test EGARCH variant"""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        analyzer = TimeSeriesAnalyzer(returns)
        
        result = analyzer.fit_garch(variant='egarch')
        
        assert 'omega' in result
        assert 'alpha' in result
        assert 'beta' in result
        assert 'gamma' in result  # Asymmetry parameter

    def test_garch_gjr_variant(self):
        """Test GJR-GARCH variant"""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        analyzer = TimeSeriesAnalyzer(returns)
        
        result = analyzer.fit_garch(variant='gjr')
        
        assert 'omega' in result
        assert 'alpha' in result
        assert 'beta' in result
        assert 'gamma' in result  # Leverage parameter


class TestStatisticalTests:
    """Test statistical hypothesis tests"""

    def test_adf_test_stationary(self):
        """Test ADF test on stationary series"""
        np.random.seed(42)
        # White noise is stationary
        data = np.random.randn(200)
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.stationarity_test(test='adf')
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'reject_null' in result
        
        # Should reject null hypothesis (non-stationary), meaning data is stationary
        assert result['reject_null'] == True

    def test_adf_test_non_stationary(self):
        """Test ADF test on non-stationary series"""
        np.random.seed(123)
        # Random walk with drift is non-stationary
        data = np.cumsum(np.random.randn(300) + 0.1)
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.stationarity_test(test='adf')
        
        # Should fail to reject null hypothesis (non-stationary)
        # Note: With random data, this may occasionally fail, but with drift it's more reliable
        assert result['reject_null'] == False or result['p_value'] > 0.01

    def test_kpss_test_stationary(self):
        """Test KPSS test on stationary series"""
        np.random.seed(42)
        data = np.random.randn(200)
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.stationarity_test(test='kpss')
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'reject_null' in result
        
        # KPSS null is stationarity, so should not reject (reject_null=False means stationary)
        assert result['reject_null'] == False

    def test_ljung_box_test(self):
        """Test Ljung-Box test for autocorrelation"""
        np.random.seed(42)
        data = np.random.randn(200)
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.ljung_box_test(lags=10)
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'reject_null' in result
        
        # White noise should have no significant autocorrelation
        assert result['p_value'] > 0.05

    def test_jarque_bera_test_normal(self):
        """Test Jarque-Bera test on normal data"""
        np.random.seed(42)
        data = np.random.randn(1000)
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.jarque_bera_test()
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'reject_null' in result
        
        # Should not reject normality (reject_null=False means normal)
        assert result['reject_null'] == False

    def test_jarque_bera_test_non_normal(self):
        """Test Jarque-Bera test on non-normal data"""
        np.random.seed(42)
        # Exponential distribution is not normal
        data = np.random.exponential(1.0, 1000)
        analyzer = TimeSeriesAnalyzer(data)
        
        result = analyzer.jarque_bera_test()
        
        # Should reject normality (reject_null=True means not normal)
        assert result['reject_null'] == True


class TestRollingStatistics:
    """Test rolling window statistics"""

    def test_rolling_mean(self):
        """Test rolling mean calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        rolling_mean = analyzer.rolling(window=3, statistic='mean')
        
        # Rolling mean over 3-period windows
        expected = np.array([2.0, 3.0, 4.0, 5.0])
        
        assert len(rolling_mean) == len(data) - 3 + 1
        np.testing.assert_array_almost_equal(rolling_mean, expected, decimal=10)

    def test_rolling_std(self):
        """Test rolling standard deviation calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        rolling_std = analyzer.rolling(window=3, statistic='std')
        
        assert len(rolling_std) == len(data) - 3 + 1
        assert all(rolling_std > 0)

    def test_ewma(self):
        """Test exponentially weighted moving average"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        ewma = analyzer.ewma(alpha=0.3)
        
        assert len(ewma) == len(data)
        # EWMA should be between min and max of data
        assert all(ewma >= data.min())
        assert all(ewma <= data.max())


class TestQuantiles:
    """Test quantile calculations"""

    def test_quantile_single(self):
        """Test single quantile calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        median = analyzer.quantile(0.5)
        
        assert abs(median - 3.0) < 1e-10

    def test_quantile_multiple(self):
        """Test multiple quantile calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        quartiles = analyzer.quantile([0.25, 0.5, 0.75])
        
        assert len(quartiles) == 3
        assert quartiles[0] < quartiles[1] < quartiles[2]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_insufficient_data_stat(self):
        """Test error handling for insufficient data"""
        data = np.array([1.0, 2.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        with pytest.raises(Exception):
            analyzer.stat()

    def test_autocorrelation_large_lag(self):
        """Test error handling for lag larger than data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = TimeSeriesAnalyzer(data)
        
        with pytest.raises(Exception):
            analyzer.autocorrelation(max_lag=10)

    def test_correlation_length_mismatch(self):
        """Test error handling for mismatched series lengths"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        analyzer = TimeSeriesAnalyzer(x)
        
        with pytest.raises(Exception):
            analyzer.correlation(y, method='pearson')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
