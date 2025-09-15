import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MathUtils:
    """수학 유틸리티 클래스"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, 
                              risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        volatility = np.std(returns)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, 
                               risk_free_rate: float = 0.02) -> float:
        """소르티노 비율 계산"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns > 0 else 0.0
        
        downside_volatility = np.std(downside_returns)
        return excess_returns / downside_volatility if downside_volatility > 0 else 0.0
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """칼마 비율 계산"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_drawdown = MathUtils.calculate_max_drawdown(returns)
        
        return annual_return / max_drawdown if max_drawdown > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Value at Risk 계산"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Conditional Value at Risk 계산"""
        if len(returns) == 0:
            return 0.0
        
        var = MathUtils.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_beta(portfolio_returns: np.ndarray, 
                      market_returns: np.ndarray) -> float:
        """베타 계산"""
        if len(portfolio_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        min_len = min(len(portfolio_returns), len(market_returns))
        portfolio_aligned = portfolio_returns[:min_len]
        market_aligned = market_returns[:min_len]
        
        covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    @staticmethod
    def calculate_alpha(portfolio_returns: np.ndarray,
                       market_returns: np.ndarray,
                       risk_free_rate: float = 0.02) -> float:
        """알파 계산"""
        if len(portfolio_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        beta = MathUtils.calculate_beta(portfolio_returns, market_returns)
        portfolio_return = np.mean(portfolio_returns) * 252
        market_return = np.mean(market_returns) * 252
        
        return portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> float:
        """정보 비율 계산"""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_aligned = portfolio_returns[:min_len]
        benchmark_aligned = benchmark_returns[:min_len]
        
        excess_returns = portfolio_aligned - benchmark_aligned
        tracking_error = np.std(excess_returns)
        
        return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0
    
    @staticmethod
    def calculate_treynor_ratio(portfolio_returns: np.ndarray,
                               market_returns: np.ndarray,
                               risk_free_rate: float = 0.02) -> float:
        """트레이너 비율 계산"""
        if len(portfolio_returns) == 0:
            return 0.0
        
        beta = MathUtils.calculate_beta(portfolio_returns, market_returns)
        excess_return = np.mean(portfolio_returns) - risk_free_rate / 252
        
        return excess_return / beta if beta > 0 else 0.0
    
    @staticmethod
    def calculate_jensen_alpha(portfolio_returns: np.ndarray,
                              market_returns: np.ndarray,
                              risk_free_rate: float = 0.02) -> float:
        """젠센 알파 계산"""
        return MathUtils.calculate_alpha(portfolio_returns, market_returns, risk_free_rate)
    
    @staticmethod
    def calculate_tracking_error(portfolio_returns: np.ndarray,
                               benchmark_returns: np.ndarray) -> float:
        """추적 오차 계산"""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_aligned = portfolio_returns[:min_len]
        benchmark_aligned = benchmark_returns[:min_len]
        
        excess_returns = portfolio_aligned - benchmark_aligned
        return np.std(excess_returns)
    
    @staticmethod
    def calculate_correlation(returns1: np.ndarray, returns2: np.ndarray) -> float:
        """상관관계 계산"""
        if len(returns1) == 0 or len(returns2) == 0:
            return 0.0
        
        min_len = min(len(returns1), len(returns2))
        returns1_aligned = returns1[:min_len]
        returns2_aligned = returns2[:min_len]
        
        return np.corrcoef(returns1_aligned, returns2_aligned)[0, 1]
    
    @staticmethod
    def calculate_covariance_matrix(returns_data: Dict[str, np.ndarray]) -> np.ndarray:
        """공분산 행렬 계산"""
        if not returns_data:
            return np.array([])
        
        # 모든 수익률 데이터의 길이를 맞춤
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_data = []
        
        for returns in returns_data.values():
            aligned_data.append(returns[:min_length])
        
        aligned_data = np.array(aligned_data)
        return np.cov(aligned_data)
    
    @staticmethod
    def calculate_portfolio_variance(weights: np.ndarray, 
                                   cov_matrix: np.ndarray) -> float:
        """포트폴리오 분산 계산"""
        if len(weights) != cov_matrix.shape[0]:
            raise ValueError("Weights and covariance matrix dimensions don't match")
        
        return np.dot(weights, np.dot(cov_matrix, weights))
    
    @staticmethod
    def calculate_portfolio_return(weights: np.ndarray, 
                                 expected_returns: np.ndarray) -> float:
        """포트폴리오 수익률 계산"""
        if len(weights) != len(expected_returns):
            raise ValueError("Weights and expected returns dimensions don't match")
        
        return np.dot(weights, expected_returns)
    
    @staticmethod
    def calculate_risk_contribution(weights: np.ndarray, 
                                  cov_matrix: np.ndarray) -> np.ndarray:
        """리스크 기여도 계산"""
        portfolio_variance = MathUtils.calculate_portfolio_variance(weights, cov_matrix)
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return risk_contrib
    
    @staticmethod
    def calculate_diversification_ratio(weights: np.ndarray, 
                                      cov_matrix: np.ndarray) -> float:
        """다각화 비율 계산"""
        portfolio_variance = MathUtils.calculate_portfolio_variance(weights, cov_matrix)
        weighted_avg_variance = np.dot(weights**2, np.diag(cov_matrix))
        
        return weighted_avg_variance / portfolio_variance if portfolio_variance > 0 else 0.0
    
    @staticmethod
    def calculate_concentration_risk(weights: np.ndarray) -> Dict[str, float]:
        """집중도 리스크 계산"""
        # HHI (Herfindahl-Hirschman Index)
        hhi = np.sum(weights**2)
        
        # 최대 가중치
        max_weight = np.max(weights)
        
        # 상위 5개 종목 비중
        top5_weight = np.sum(np.sort(weights)[-5:])
        
        # 엔트로피 (다각화 지표)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        
        return {
            'hhi': hhi,
            'max_weight': max_weight,
            'top5_weight': top5_weight,
            'entropy': entropy
        }
    
    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:
        """왜도 계산"""
        if len(returns) == 0:
            return 0.0
        
        return stats.skew(returns)
    
    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:
        """첨도 계산"""
        if len(returns) == 0:
            return 0.0
        
        return stats.kurtosis(returns)
    
    @staticmethod
    def calculate_jarque_bera_statistic(returns: np.ndarray) -> Tuple[float, float]:
        """Jarque-Bera 정규성 검정"""
        if len(returns) == 0:
            return 0.0, 1.0
        
        return stats.jarque_bera(returns)
    
    @staticmethod
    def calculate_shapiro_wilk_statistic(returns: np.ndarray) -> Tuple[float, float]:
        """Shapiro-Wilk 정규성 검정"""
        if len(returns) == 0 or len(returns) > 5000:
            return 0.0, 1.0
        
        return stats.shapiro(returns)
    
    @staticmethod
    def calculate_autocorrelation(returns: np.ndarray, lag: int = 1) -> float:
        """자기상관 계산"""
        if len(returns) <= lag:
            return 0.0
        
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
    
    @staticmethod
    def calculate_ljung_box_statistic(returns: np.ndarray, 
                                    lags: int = 10) -> Tuple[float, float]:
        """Ljung-Box 자기상관 검정"""
        if len(returns) == 0:
            return 0.0, 1.0
        
        from statsmodels.stats.diagnostic import acorr_ljungbox
        result = acorr_ljungbox(returns, lags=lags, return_df=False)
        return result[0][-1], result[1][-1]
    
    @staticmethod
    def calculate_arch_effect(returns: np.ndarray, lags: int = 5) -> Tuple[float, float]:
        """ARCH 효과 검정"""
        if len(returns) == 0:
            return 0.0, 1.0
        
        squared_returns = returns**2
        from statsmodels.stats.diagnostic import het_arch
        result = het_arch(squared_returns, maxlag=lags)
        return result[0], result[1]
    
    @staticmethod
    def calculate_cointegration(series1: np.ndarray, series2: np.ndarray) -> Tuple[float, float]:
        """공적분 검정"""
        if len(series1) == 0 or len(series2) == 0:
            return 0.0, 1.0
        
        from statsmodels.tsa.stattools import coint
        result = coint(series1, series2)
        return result[0], result[1]
    
    @staticmethod
    def calculate_granger_causality(series1: np.ndarray, series2: np.ndarray, 
                                  maxlag: int = 4) -> Tuple[float, float]:
        """그랜저 인과관계 검정"""
        if len(series1) == 0 or len(series2) == 0:
            return 0.0, 1.0
        
        from statsmodels.tsa.stattools import grangercausalitytests
        try:
            data = np.column_stack([series1, series2])
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            # 가장 최근 lag의 결과 반환
            f_stat = result[maxlag][0]['ssr_ftest'][0]
            p_value = result[maxlag][0]['ssr_ftest'][1]
            return f_stat, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def calculate_rolling_statistics(data: np.ndarray, 
                                   window: int,
                                   statistic: str = 'mean') -> np.ndarray:
        """롤링 통계 계산"""
        if len(data) == 0:
            return np.array([])
        
        if statistic == 'mean':
            return pd.Series(data).rolling(window).mean().values
        elif statistic == 'std':
            return pd.Series(data).rolling(window).std().values
        elif statistic == 'min':
            return pd.Series(data).rolling(window).min().values
        elif statistic == 'max':
            return pd.Series(data).rolling(window).max().values
        elif statistic == 'median':
            return pd.Series(data).rolling(window).median().values
        elif statistic == 'skew':
            return pd.Series(data).rolling(window).skew().values
        elif statistic == 'kurt':
            return pd.Series(data).rolling(window).kurt().values
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    @staticmethod
    def calculate_ewm_statistics(data: np.ndarray, 
                                span: int,
                                statistic: str = 'mean') -> np.ndarray:
        """지수 가중 이동 통계 계산"""
        if len(data) == 0:
            return np.array([])
        
        if statistic == 'mean':
            return pd.Series(data).ewm(span=span).mean().values
        elif statistic == 'std':
            return pd.Series(data).ewm(span=span).std().values
        elif statistic == 'var':
            return pd.Series(data).ewm(span=span).var().values
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    @staticmethod
    def calculate_percentile_rank(data: np.ndarray, value: float) -> float:
        """백분위 순위 계산"""
        if len(data) == 0:
            return 0.0
        
        return (np.sum(data <= value) / len(data)) * 100
    
    @staticmethod
    def calculate_z_score(data: np.ndarray, value: float) -> float:
        """Z-점수 계산"""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        return (value - mean) / std if std > 0 else 0.0
    
    @staticmethod
    def calculate_modified_z_score(data: np.ndarray, value: float) -> float:
        """수정된 Z-점수 계산 (중앙값 기반)"""
        if len(data) == 0:
            return 0.0
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        return 0.6745 * (value - median) / mad if mad > 0 else 0.0
    
    @staticmethod
    def calculate_rank_correlation(data1: np.ndarray, data2: np.ndarray) -> float:
        """순위 상관관계 계산 (Spearman)"""
        if len(data1) == 0 or len(data2) == 0:
            return 0.0
        
        min_len = min(len(data1), len(data2))
        data1_aligned = data1[:min_len]
        data2_aligned = data2[:min_len]
        
        return stats.spearmanr(data1_aligned, data2_aligned)[0]
    
    @staticmethod
    def calculate_kendall_tau(data1: np.ndarray, data2: np.ndarray) -> float:
        """Kendall tau 계산"""
        if len(data1) == 0 or len(data2) == 0:
            return 0.0
        
        min_len = min(len(data1), len(data2))
        data1_aligned = data1[:min_len]
        data2_aligned = data2[:min_len]
        
        return stats.kendalltau(data1_aligned, data2_aligned)[0]
    
    @staticmethod
    def calculate_mutual_information(data1: np.ndarray, data2: np.ndarray, 
                                   bins: int = 10) -> float:
        """상호 정보량 계산"""
        if len(data1) == 0 or len(data2) == 0:
            return 0.0
        
        min_len = min(len(data1), len(data2))
        data1_aligned = data1[:min_len]
        data2_aligned = data2[:min_len]
        
        # 히스토그램 계산
        hist_2d, _, _ = np.histogram2d(data1_aligned, data2_aligned, bins=bins)
        
        # 확률 계산
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # 상호 정보량 계산
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        
        return mi
