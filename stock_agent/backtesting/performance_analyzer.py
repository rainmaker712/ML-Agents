import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class PerformanceAnalyzer:
    """성과 분석기"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% 무위험 수익률
    
    def analyze_performance(self, portfolio_values: List[float], 
                           dates: List[datetime],
                           benchmark_values: List[float] = None) -> Dict:
        """포트폴리오 성과 분석"""
        if not portfolio_values or len(portfolio_values) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # 수익률 계산
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # 기본 통계
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 리스크 지표
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        # 최대 낙폭
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # VaR 및 CVaR
        var_95 = self._calculate_var(returns, 0.05)
        cvar_95 = self._calculate_cvar(returns, 0.05)
        
        # 벤치마크 대비 분석
        benchmark_analysis = {}
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            benchmark_analysis = self._analyze_benchmark_comparison(
                portfolio_values, benchmark_values
            )
        
        # 월별/연도별 성과
        monthly_returns = self._calculate_monthly_returns(portfolio_values, dates)
        yearly_returns = self._calculate_yearly_returns(portfolio_values, dates)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'benchmark_analysis': benchmark_analysis,
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
            'return_statistics': self._calculate_return_statistics(returns)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """소르티노 비율 계산"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns.mean() - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        
        return excess_returns / downside_volatility if downside_volatility > 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """칼마 비율 계산"""
        if returns.empty:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """최대 낙폭 계산"""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """수익률로부터 최대 낙폭 계산"""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Value at Risk 계산"""
        if returns.empty:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Conditional Value at Risk 계산"""
        if returns.empty:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _analyze_benchmark_comparison(self, portfolio_values: List[float], 
                                    benchmark_values: List[float]) -> Dict:
        """벤치마크 대비 분석"""
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        benchmark_returns = pd.Series(benchmark_values).pct_change().dropna()
        
        # 공통 기간으로 정렬
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_aligned = portfolio_returns.iloc[:min_len]
        benchmark_aligned = benchmark_returns.iloc[:min_len]
        
        # 베타 계산
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # 알파 계산
        portfolio_annual_return = portfolio_aligned.mean() * 252
        benchmark_annual_return = benchmark_aligned.mean() * 252
        alpha = portfolio_annual_return - (self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate))
        
        # 정보 비율
        excess_returns = portfolio_aligned - benchmark_aligned
        tracking_error = excess_returns.std()
        information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
        
        # 상관관계
        correlation = portfolio_aligned.corr(benchmark_aligned)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'correlation': correlation,
            'excess_return': portfolio_annual_return - benchmark_annual_return,
            'tracking_error': tracking_error
        }
    
    def _calculate_monthly_returns(self, values: List[float], dates: List[datetime]) -> Dict:
        """월별 수익률 계산"""
        if not values or not dates or len(values) != len(dates):
            return {}
        
        df = pd.DataFrame({'value': values, 'date': dates})
        df.set_index('date', inplace=True)
        
        monthly_values = df.resample('M').last()
        monthly_returns = monthly_values['value'].pct_change().dropna()
        
        return {
            'returns': monthly_returns.to_dict(),
            'mean': monthly_returns.mean(),
            'std': monthly_returns.std(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min()
        }
    
    def _calculate_yearly_returns(self, values: List[float], dates: List[datetime]) -> Dict:
        """연도별 수익률 계산"""
        if not values or not dates or len(values) != len(dates):
            return {}
        
        df = pd.DataFrame({'value': values, 'date': dates})
        df.set_index('date', inplace=True)
        
        yearly_values = df.resample('Y').last()
        yearly_returns = yearly_values['value'].pct_change().dropna()
        
        return {
            'returns': yearly_returns.to_dict(),
            'mean': yearly_returns.mean(),
            'std': yearly_returns.std(),
            'best_year': yearly_returns.max(),
            'worst_year': yearly_returns.min()
        }
    
    def _calculate_return_statistics(self, returns: pd.Series) -> Dict:
        """수익률 통계 계산"""
        if returns.empty:
            return {}
        
        return {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min': returns.min(),
            'max': returns.max(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'win_rate': (returns > 0).mean()
        }
    
    def generate_performance_report(self, portfolio_values: List[float],
                                  dates: List[datetime],
                                  benchmark_values: List[float] = None,
                                  title: str = "Performance Report") -> Dict:
        """성과 리포트 생성"""
        analysis = self.analyze_performance(portfolio_values, dates, benchmark_values)
        
        if 'error' in analysis:
            return analysis
        
        # 성과 등급
        performance_grade = self._calculate_performance_grade(analysis)
        
        # 리스크 등급
        risk_grade = self._calculate_risk_grade(analysis)
        
        # 권장사항
        recommendations = self._generate_recommendations(analysis)
        
        return {
            'title': title,
            'analysis_date': datetime.now().isoformat(),
            'performance_grade': performance_grade,
            'risk_grade': risk_grade,
            'analysis': analysis,
            'recommendations': recommendations,
            'summary': self._generate_summary(analysis)
        }
    
    def _calculate_performance_grade(self, analysis: Dict) -> str:
        """성과 등급 계산"""
        sharpe_ratio = analysis.get('sharpe_ratio', 0)
        total_return = analysis.get('total_return', 0)
        
        if sharpe_ratio > 1.5 and total_return > 0.15:
            return 'A+'
        elif sharpe_ratio > 1.2 and total_return > 0.12:
            return 'A'
        elif sharpe_ratio > 1.0 and total_return > 0.10:
            return 'A-'
        elif sharpe_ratio > 0.8 and total_return > 0.08:
            return 'B+'
        elif sharpe_ratio > 0.6 and total_return > 0.06:
            return 'B'
        elif sharpe_ratio > 0.4 and total_return > 0.04:
            return 'B-'
        elif sharpe_ratio > 0.2 and total_return > 0.02:
            return 'C+'
        elif sharpe_ratio > 0 and total_return > 0:
            return 'C'
        else:
            return 'D'
    
    def _calculate_risk_grade(self, analysis: Dict) -> str:
        """리스크 등급 계산"""
        volatility = analysis.get('volatility', 0)
        max_drawdown = analysis.get('max_drawdown', 0)
        
        if volatility < 0.15 and max_drawdown < 0.1:
            return 'Low'
        elif volatility < 0.25 and max_drawdown < 0.2:
            return 'Medium'
        elif volatility < 0.35 and max_drawdown < 0.3:
            return 'High'
        else:
            return 'Very High'
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        sharpe_ratio = analysis.get('sharpe_ratio', 0)
        volatility = analysis.get('volatility', 0)
        max_drawdown = analysis.get('max_drawdown', 0)
        
        if sharpe_ratio < 0.5:
            recommendations.append("Consider improving risk-adjusted returns through better diversification or strategy optimization")
        
        if volatility > 0.3:
            recommendations.append("High volatility detected - consider adding defensive positions or reducing position sizes")
        
        if max_drawdown > 0.2:
            recommendations.append("Large drawdowns observed - consider implementing stop-loss strategies or reducing leverage")
        
        if analysis.get('benchmark_analysis', {}).get('alpha', 0) < -0.02:
            recommendations.append("Underperforming benchmark - consider reviewing strategy or rebalancing portfolio")
        
        return recommendations
    
    def _generate_summary(self, analysis: Dict) -> str:
        """요약 생성"""
        total_return = analysis.get('total_return', 0)
        sharpe_ratio = analysis.get('sharpe_ratio', 0)
        max_drawdown = analysis.get('max_drawdown', 0)
        
        summary = f"Portfolio achieved a total return of {total_return:.1%} with a Sharpe ratio of {sharpe_ratio:.2f}. "
        
        if max_drawdown > 0.1:
            summary += f"Maximum drawdown was {max_drawdown:.1%}, indicating significant downside risk. "
        else:
            summary += f"Maximum drawdown was {max_drawdown:.1%}, showing good downside protection. "
        
        if sharpe_ratio > 1.0:
            summary += "Risk-adjusted returns are strong."
        elif sharpe_ratio > 0.5:
            summary += "Risk-adjusted returns are moderate."
        else:
            summary += "Risk-adjusted returns need improvement."
        
        return summary
    
    def create_performance_charts(self, portfolio_values: List[float],
                                dates: List[datetime],
                                benchmark_values: List[float] = None,
                                save_path: str = None) -> Dict:
        """성과 차트 생성"""
        try:
            if not portfolio_values or not dates:
                return {'error': 'Insufficient data for charting'}
            
            # 차트 설정
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Portfolio Performance Analysis', fontsize=16)
            
            # 1. 누적 수익률 차트
            ax1 = axes[0, 0]
            cumulative_returns = np.array(portfolio_values) / portfolio_values[0] - 1
            ax1.plot(dates, cumulative_returns, label='Portfolio', linewidth=2)
            
            if benchmark_values and len(benchmark_values) == len(portfolio_values):
                benchmark_cumulative = np.array(benchmark_values) / benchmark_values[0] - 1
                ax1.plot(dates, benchmark_cumulative, label='Benchmark', linewidth=2, alpha=0.7)
            
            ax1.set_title('Cumulative Returns')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 일일 수익률 히스토그램
            ax2 = axes[0, 1]
            returns = pd.Series(portfolio_values).pct_change().dropna()
            ax2.hist(returns, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_title('Daily Returns Distribution')
            ax2.set_xlabel('Daily Return')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # 3. 롤링 샤프 비율
            ax3 = axes[1, 0]
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
            ax3.plot(dates[1:], rolling_sharpe, linewidth=2)
            ax3.set_title('Rolling Sharpe Ratio (252 days)')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
            
            # 4. 최대 낙폭
            ax4 = axes[1, 1]
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            ax4.fill_between(dates[1:], drawdown, 0, alpha=0.3, color='red')
            ax4.plot(dates[1:], drawdown, color='red', linewidth=1)
            ax4.set_title('Drawdown')
            ax4.set_ylabel('Drawdown')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return {'success': True, 'chart_path': save_path}
            
        except Exception as e:
            return {'error': f'Error creating charts: {str(e)}'}
    
    def compare_strategies_performance(self, strategy_results: Dict[str, Dict]) -> Dict:
        """전략 성과 비교"""
        comparison = {}
        
        for strategy_name, result in strategy_results.items():
            if result.get('success', False):
                comparison[strategy_name] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'volatility': result.get('volatility', 0)
                }
        
        if not comparison:
            return {'error': 'No successful strategy results to compare'}
        
        # 메트릭별 순위
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        rankings = {}
        
        for metric in metrics:
            if metric == 'max_drawdown':  # 낮을수록 좋음
                sorted_strategies = sorted(comparison.items(), key=lambda x: x[1][metric])
            else:  # 높을수록 좋음
                sorted_strategies = sorted(comparison.items(), key=lambda x: x[1][metric], reverse=True)
            
            rankings[metric] = [strategy[0] for strategy in sorted_strategies]
        
        # 종합 점수 계산
        strategy_scores = {}
        for strategy_name in comparison.keys():
            score = 0
            for metric in metrics:
                rank = rankings[metric].index(strategy_name) + 1
                if metric == 'max_drawdown':
                    score += (len(comparison) - rank + 1)  # 낮은 순위일수록 높은 점수
                else:
                    score += (len(comparison) - rank + 1)  # 높은 순위일수록 높은 점수
            strategy_scores[strategy_name] = score
        
        overall_ranking = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'comparison': comparison,
            'rankings': rankings,
            'overall_ranking': overall_ranking,
            'best_strategy': overall_ranking[0][0] if overall_ranking else None
        }
