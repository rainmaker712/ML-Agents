import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """리스크 분석 클래스"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% 무위험 수익률
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Value at Risk (VaR) 계산"""
        if returns.empty:
            return 0.0
        
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Conditional Value at Risk (CVaR) 계산"""
        if returns.empty:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict:
        """최대 낙폭 계산"""
        if returns.empty:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # 최대 낙폭 지속 기간 계산
        dd_duration = 0
        current_dd_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                current_dd_duration += 1
                dd_duration = max(dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_idx,
            'max_drawdown_duration': dd_duration
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """샤프 비율 계산"""
        if returns.empty:
            return 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns.mean() - risk_free_rate
        volatility = returns.std()
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """소르티노 비율 계산"""
        if returns.empty:
            return 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns.mean() - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        
        return excess_returns / downside_volatility if downside_volatility > 0 else 0.0
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """칼마 비율 계산"""
        if returns.empty:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(returns)['max_drawdown'])
        
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    def calculate_treynor_ratio(self, returns: pd.Series, beta: float) -> float:
        """트레이너 비율 계산"""
        if returns.empty or beta <= 0:
            return 0.0
        
        excess_returns = returns.mean() - self.risk_free_rate
        return excess_returns / beta
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> float:
        """정보 비율 계산"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0
        
        # 공통 기간으로 정렬
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return 0.0
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        excess_returns = portfolio_aligned - benchmark_aligned
        tracking_error = excess_returns.std()
        
        return excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
    
    def calculate_beta(self, portfolio_returns: pd.Series, 
                      market_returns: pd.Series) -> float:
        """베타 계산"""
        if portfolio_returns.empty or market_returns.empty:
            return 1.0
        
        # 공통 기간으로 정렬
        common_dates = portfolio_returns.index.intersection(market_returns.index)
        if len(common_dates) < 2:
            return 1.0
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        market_aligned = market_returns.loc[common_dates]
        
        covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """상관관계 행렬 계산"""
        if returns_data.empty:
            return pd.DataFrame()
        
        return returns_data.corr()
    
    def calculate_covariance_matrix(self, returns_data: pd.DataFrame, 
                                   shrinkage: bool = True) -> pd.DataFrame:
        """공분산 행렬 계산"""
        if returns_data.empty:
            return pd.DataFrame()
        
        if shrinkage:
            # Ledoit-Wolf shrinkage estimator 사용
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_data.fillna(0)).covariance_
            return pd.DataFrame(cov_matrix, index=returns_data.columns, columns=returns_data.columns)
        else:
            return returns_data.cov()
    
    def calculate_portfolio_risk_metrics(self, weights: np.ndarray,
                                       returns_data: pd.DataFrame) -> Dict:
        """포트폴리오 리스크 지표 계산"""
        if returns_data.empty or len(weights) != len(returns_data.columns):
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var_95': 0.0,
                'portfolio_cvar_95': 0.0,
                'portfolio_sharpe': 0.0,
                'portfolio_sortino': 0.0,
                'portfolio_calmar': 0.0
            }
        
        # 포트폴리오 수익률 계산
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # 기본 지표
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        portfolio_var_95 = self.calculate_var(portfolio_returns, 0.05)
        portfolio_cvar_95 = self.calculate_cvar(portfolio_returns, 0.05)
        portfolio_sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        portfolio_sortino = self.calculate_sortino_ratio(portfolio_returns)
        portfolio_calmar = self.calculate_calmar_ratio(portfolio_returns)
        
        # 최대 낙폭
        max_dd_info = self.calculate_max_drawdown(portfolio_returns)
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var_95': portfolio_var_95,
            'portfolio_cvar_95': portfolio_cvar_95,
            'portfolio_sharpe': portfolio_sharpe,
            'portfolio_sortino': portfolio_sortino,
            'portfolio_calmar': portfolio_calmar,
            'max_drawdown': max_dd_info['max_drawdown'],
            'max_drawdown_duration': max_dd_info['max_drawdown_duration']
        }
    
    def calculate_risk_contribution(self, weights: np.ndarray,
                                   cov_matrix: np.ndarray) -> np.ndarray:
        """리스크 기여도 계산"""
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return risk_contrib
    
    def calculate_diversification_ratio(self, weights: np.ndarray,
                                      cov_matrix: np.ndarray) -> float:
        """다각화 비율 계산"""
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        weighted_avg_variance = np.dot(weights**2, np.diag(cov_matrix))
        
        return weighted_avg_variance / portfolio_variance if portfolio_variance > 0 else 0.0
    
    def calculate_concentration_risk(self, weights: np.ndarray) -> Dict:
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
            'entropy': entropy,
            'concentration_risk': 'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low'
        }
    
    def stress_test(self, returns_data: pd.DataFrame, 
                   stress_scenarios: Dict[str, float] = None) -> Dict:
        """스트레스 테스트"""
        if returns_data.empty:
            return {}
        
        if stress_scenarios is None:
            stress_scenarios = {
                'market_crash': -0.2,  # 20% 하락
                'recession': -0.1,     # 10% 하락
                'volatility_spike': 0.3,  # 변동성 급증
                'sector_rotation': 0.15   # 섹터 로테이션
            }
        
        stress_results = {}
        
        for scenario_name, shock in stress_scenarios.items():
            if scenario_name == 'volatility_spike':
                # 변동성 급증 시나리오
                stressed_returns = returns_data * (1 + shock)
            else:
                # 가격 충격 시나리오
                stressed_returns = returns_data + shock
            
            # 포트폴리오 손실 계산 (균등 가중 가정)
            equal_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
            portfolio_loss = (stressed_returns * equal_weights).sum(axis=1).mean()
            
            stress_results[scenario_name] = {
                'shock': shock,
                'portfolio_loss': portfolio_loss,
                'worst_case': stressed_returns.min().min()
            }
        
        return stress_results
    
    def calculate_tail_risk(self, returns: pd.Series) -> Dict:
        """테일 리스크 계산"""
        if returns.empty:
            return {'tail_ratio': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
        
        # 왜도 (Skewness)
        skewness = stats.skew(returns.dropna())
        
        # 첨도 (Kurtosis)
        kurtosis = stats.kurtosis(returns.dropna())
        
        # 테일 비율 (5% VaR / 1% VaR)
        var_5 = self.calculate_var(returns, 0.05)
        var_1 = self.calculate_var(returns, 0.01)
        tail_ratio = abs(var_5 / var_1) if var_1 != 0 else 0
        
        return {
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_risk': 'High' if tail_ratio > 2.0 or kurtosis > 3.0 else 'Medium' if tail_ratio > 1.5 else 'Low'
        }
    
    def calculate_liquidity_risk(self, volume_data: pd.DataFrame) -> Dict:
        """유동성 리스크 계산"""
        if volume_data.empty:
            return {'liquidity_risk': 'Unknown', 'avg_volume': 0}
        
        # 평균 거래량
        avg_volume = volume_data.mean().mean()
        
        # 거래량 변동성
        volume_volatility = volume_data.std().mean()
        
        # 유동성 점수
        liquidity_score = avg_volume / (volume_volatility + 1e-10)
        
        if liquidity_score > 1000:
            liquidity_risk = 'Low'
        elif liquidity_score > 100:
            liquidity_risk = 'Medium'
        else:
            liquidity_risk = 'High'
        
        return {
            'liquidity_risk': liquidity_risk,
            'avg_volume': avg_volume,
            'volume_volatility': volume_volatility,
            'liquidity_score': liquidity_score
        }
    
    def generate_risk_report(self, portfolio_returns: pd.Series,
                           benchmark_returns: pd.Series = None,
                           weights: np.ndarray = None,
                           returns_data: pd.DataFrame = None) -> Dict:
        """종합 리스크 리포트 생성"""
        report = {
            'portfolio_metrics': {},
            'benchmark_comparison': {},
            'risk_breakdown': {},
            'recommendations': []
        }
        
        # 포트폴리오 기본 지표
        if not portfolio_returns.empty:
            report['portfolio_metrics'] = {
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_returns),
                'sortino_ratio': self.calculate_sortino_ratio(portfolio_returns),
                'calmar_ratio': self.calculate_calmar_ratio(portfolio_returns),
                'var_95': self.calculate_var(portfolio_returns, 0.05),
                'cvar_95': self.calculate_cvar(portfolio_returns, 0.05),
                'max_drawdown': self.calculate_max_drawdown(portfolio_returns)
            }
        
        # 벤치마크 대비 분석
        if benchmark_returns is not None and not benchmark_returns.empty:
            beta = self.calculate_beta(portfolio_returns, benchmark_returns)
            information_ratio = self.calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            report['benchmark_comparison'] = {
                'beta': beta,
                'information_ratio': information_ratio,
                'excess_return': portfolio_returns.mean() - benchmark_returns.mean(),
                'tracking_error': (portfolio_returns - benchmark_returns).std()
            }
        
        # 포트폴리오 리스크 분석
        if weights is not None and returns_data is not None:
            report['risk_breakdown'] = self.calculate_portfolio_risk_metrics(weights, returns_data)
            
            # 집중도 리스크
            if len(weights) > 0:
                concentration = self.calculate_concentration_risk(weights)
                report['risk_breakdown'].update(concentration)
        
        # 권장사항 생성
        if not portfolio_returns.empty:
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = self.calculate_sharpe_ratio(portfolio_returns)
            
            if volatility > 0.3:
                report['recommendations'].append("Consider reducing portfolio volatility")
            if sharpe < 0.5:
                report['recommendations'].append("Improve risk-adjusted returns")
            if 'concentration_risk' in report.get('risk_breakdown', {}):
                if report['risk_breakdown']['concentration_risk'] == 'High':
                    report['recommendations'].append("Reduce portfolio concentration")
        
        return report
