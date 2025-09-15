from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_engine import PortfolioOptimizer, RecommendationEngine, RiskAnalyzer, MarketAnalyzer
from models import DatabaseManager, Portfolio, Stock
from services import DataService, PortfolioService

class AnalysisService:
    """분석 서비스"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.data_service = DataService(db_manager)
        self.portfolio_service = PortfolioService(db_manager)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.recommendation_engine = RecommendationEngine()
        self.risk_analyzer = RiskAnalyzer()
        self.market_analyzer = MarketAnalyzer()
    
    def analyze_portfolio_comprehensive(self, portfolio_id: int) -> Dict:
        """포트폴리오 종합 분석"""
        try:
            portfolio = self.portfolio_service.get_portfolio(portfolio_id)
            if not portfolio:
                return {'success': False, 'message': 'Portfolio not found'}
            
            # 성과 데이터 수집
            performance_data = {}
            for item in portfolio.items:
                hist_data = self.data_service.yfinance_collector.get_historical_data(
                    item.symbol, period='1y'
                )
                if not hist_data.empty:
                    performance_data[item.symbol] = hist_data['Close'].pct_change().dropna()
            
            if not performance_data:
                return {'success': False, 'message': 'Insufficient data for analysis'}
            
            returns_df = pd.DataFrame(performance_data)
            weights = np.array([item.weight for item in portfolio.items])
            
            # 1. 리스크 분석
            risk_analysis = self._analyze_portfolio_risk(returns_df, weights)
            
            # 2. 성과 분석
            performance_analysis = self._analyze_portfolio_performance(returns_df, weights)
            
            # 3. 다각화 분석
            diversification_analysis = self._analyze_portfolio_diversification(portfolio)
            
            # 4. 최적화 분석
            optimization_analysis = self._analyze_portfolio_optimization(returns_df, weights)
            
            # 5. 시장 대비 분석
            market_comparison = self._analyze_market_comparison(returns_df)
            
            return {
                'success': True,
                'portfolio_id': portfolio_id,
                'analysis_date': datetime.now().isoformat(),
                'risk_analysis': risk_analysis,
                'performance_analysis': performance_analysis,
                'diversification_analysis': diversification_analysis,
                'optimization_analysis': optimization_analysis,
                'market_comparison': market_comparison,
                'overall_grade': self._calculate_overall_grade(
                    risk_analysis, performance_analysis, diversification_analysis
                )
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error in comprehensive analysis: {str(e)}'}
    
    def _analyze_portfolio_risk(self, returns_df: pd.DataFrame, weights: np.ndarray) -> Dict:
        """포트폴리오 리스크 분석"""
        try:
            # 기본 리스크 지표
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            risk_metrics = self.risk_analyzer.calculate_portfolio_risk_metrics(weights, returns_df)
            
            # VaR 및 CVaR
            var_95 = self.risk_analyzer.calculate_var(portfolio_returns, 0.05)
            cvar_95 = self.risk_analyzer.calculate_cvar(portfolio_returns, 0.05)
            
            # 최대 낙폭
            max_dd = self.risk_analyzer.calculate_max_drawdown(portfolio_returns)
            
            # 베타 계산 (S&P 500 대비)
            sp500_data = self.data_service.yfinance_collector.get_historical_data('^GSPC', period='1y')
            if not sp500_data.empty:
                sp500_returns = sp500_data['Close'].pct_change().dropna()
                beta = self.risk_analyzer.calculate_beta(portfolio_returns, sp500_returns)
            else:
                beta = 1.0
            
            # 리스크 등급
            volatility = risk_metrics['portfolio_volatility']
            if volatility < 0.15:
                risk_grade = 'Low'
            elif volatility < 0.25:
                risk_grade = 'Medium'
            else:
                risk_grade = 'High'
            
            return {
                'volatility': volatility,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_dd['max_drawdown'],
                'sharpe_ratio': risk_metrics['portfolio_sharpe'],
                'sortino_ratio': risk_metrics['portfolio_sortino'],
                'beta': beta,
                'risk_grade': risk_grade,
                'risk_score': self._calculate_risk_score(volatility, max_dd['max_drawdown'], beta)
            }
            
        except Exception as e:
            return {'error': f'Risk analysis failed: {str(e)}'}
    
    def _analyze_portfolio_performance(self, returns_df: pd.DataFrame, weights: np.ndarray) -> Dict:
        """포트폴리오 성과 분석"""
        try:
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # 기본 성과 지표
            total_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = self.risk_analyzer.calculate_sharpe_ratio(portfolio_returns)
            
            # 벤치마크 대비 성과
            sp500_data = self.data_service.yfinance_collector.get_historical_data('^GSPC', period='1y')
            if not sp500_data.empty:
                sp500_returns = sp500_data['Close'].pct_change().dropna()
                benchmark_return = sp500_returns.mean() * 252
                excess_return = total_return - benchmark_return
                information_ratio = self.risk_analyzer.calculate_information_ratio(
                    portfolio_returns, sp500_returns
                )
            else:
                benchmark_return = 0
                excess_return = total_return
                information_ratio = 0
            
            # 성과 등급
            if total_return > 0.15:
                performance_grade = 'Excellent'
            elif total_return > 0.10:
                performance_grade = 'Good'
            elif total_return > 0.05:
                performance_grade = 'Fair'
            else:
                performance_grade = 'Poor'
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'benchmark_return': benchmark_return,
                'excess_return': excess_return,
                'information_ratio': information_ratio,
                'performance_grade': performance_grade,
                'performance_score': self._calculate_performance_score(
                    total_return, sharpe_ratio, excess_return
                )
            }
            
        except Exception as e:
            return {'error': f'Performance analysis failed: {str(e)}'}
    
    def _analyze_portfolio_diversification(self, portfolio: Portfolio) -> Dict:
        """포트폴리오 다각화 분석"""
        try:
            # 섹터 분포
            sector_allocation = portfolio.get_sector_allocation()
            
            # 집중도 분석
            weights = np.array([item.weight for item in portfolio.items])
            concentration_risk = self.risk_analyzer.calculate_concentration_risk(weights)
            
            # 다각화 점수
            num_positions = len(portfolio.items)
            max_weight = max(weights) if len(weights) > 0 else 0
            num_sectors = len(sector_allocation)
            
            diversification_score = self._calculate_diversification_score(
                num_positions, max_weight, num_sectors
            )
            
            # 다각화 등급
            if diversification_score >= 80:
                diversification_grade = 'Excellent'
            elif diversification_score >= 60:
                diversification_grade = 'Good'
            elif diversification_score >= 40:
                diversification_grade = 'Fair'
            else:
                diversification_grade = 'Poor'
            
            return {
                'num_positions': num_positions,
                'num_sectors': num_sectors,
                'max_weight': max_weight,
                'sector_allocation': sector_allocation,
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'diversification_grade': diversification_grade
            }
            
        except Exception as e:
            return {'error': f'Diversification analysis failed: {str(e)}'}
    
    def _analyze_portfolio_optimization(self, returns_df: pd.DataFrame, weights: np.ndarray) -> Dict:
        """포트폴리오 최적화 분석"""
        try:
            expected_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # 현재 포트폴리오 지표
            current_metrics = self.portfolio_optimizer.calculate_portfolio_metrics(
                weights, expected_returns.values, cov_matrix.values
            )
            
            # 최적화된 포트폴리오
            optimal_result = self.portfolio_optimizer.optimize_portfolio(
                expected_returns.values, cov_matrix.values
            )
            
            if optimal_result['success']:
                optimal_metrics = self.portfolio_optimizer.calculate_portfolio_metrics(
                    optimal_result['weights'], expected_returns.values, cov_matrix.values
                )
                
                # 개선 가능성
                return_improvement = optimal_metrics['expected_return'] - current_metrics['expected_return']
                volatility_improvement = current_metrics['volatility'] - optimal_metrics['volatility']
                sharpe_improvement = optimal_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio']
                
                return {
                    'current_metrics': current_metrics,
                    'optimal_metrics': optimal_metrics,
                    'return_improvement': return_improvement,
                    'volatility_improvement': volatility_improvement,
                    'sharpe_improvement': sharpe_improvement,
                    'optimization_potential': self._calculate_optimization_potential(
                        return_improvement, sharpe_improvement
                    )
                }
            else:
                return {
                    'current_metrics': current_metrics,
                    'optimization_failed': True,
                    'message': optimal_result.get('message', 'Optimization failed')
                }
                
        except Exception as e:
            return {'error': f'Optimization analysis failed: {str(e)}'}
    
    def _analyze_market_comparison(self, returns_df: pd.DataFrame) -> Dict:
        """시장 대비 분석"""
        try:
            # 시장 데이터 수집
            market_data = self.data_service.collect_market_data(['^GSPC', '^IXIC', '^VIX'])
            
            if not market_data['success']:
                return {'error': 'Failed to collect market data'}
            
            # 포트폴리오 수익률
            portfolio_returns = returns_df.mean(axis=1)
            
            # S&P 500 대비
            sp500_data = market_data['market_data'].get('^GSPC')
            if sp500_data is not None and not sp500_data.empty:
                sp500_returns = sp500_data['Close'].pct_change().dropna()
                
                # 공통 기간으로 정렬
                common_dates = portfolio_returns.index.intersection(sp500_returns.index)
                if len(common_dates) > 0:
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    sp500_aligned = sp500_returns.loc[common_dates]
                    
                    # 상관관계
                    correlation = portfolio_aligned.corr(sp500_aligned)
                    
                    # 베타
                    beta = self.risk_analyzer.calculate_beta(portfolio_aligned, sp500_aligned)
                    
                    # 정보 비율
                    information_ratio = self.risk_analyzer.calculate_information_ratio(
                        portfolio_aligned, sp500_aligned
                    )
                else:
                    correlation = 0
                    beta = 1
                    information_ratio = 0
            else:
                correlation = 0
                beta = 1
                information_ratio = 0
            
            # VIX 분석
            vix_data = market_data['market_data'].get('^VIX')
            if vix_data is not None and not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                avg_vix = vix_data['Close'].mean()
                vix_level = 'High' if current_vix > avg_vix * 1.2 else 'Low' if current_vix < avg_vix * 0.8 else 'Normal'
            else:
                current_vix = 20
                vix_level = 'Normal'
            
            return {
                'correlation_with_sp500': correlation,
                'beta': beta,
                'information_ratio': information_ratio,
                'current_vix': current_vix,
                'vix_level': vix_level,
                'market_sensitivity': self._calculate_market_sensitivity(beta, correlation)
            }
            
        except Exception as e:
            return {'error': f'Market comparison analysis failed: {str(e)}'}
    
    def generate_stock_recommendations(self, criteria: Dict) -> Dict:
        """주식 추천 생성"""
        try:
            # 기준 설정
            risk_tolerance = criteria.get('risk_tolerance', 'medium')
            investment_style = criteria.get('investment_style', 'balanced')
            sector_preference = criteria.get('sector_preference', [])
            max_stocks = criteria.get('max_stocks', 10)
            
            # 주식 유니버스 수집 (실제 구현에서는 데이터베이스에서 조회)
            stock_universe = self._get_stock_universe()
            
            # AI 추천 실행
            recommendations = self.recommendation_engine.recommend_stocks(
                stock_universe=stock_universe,
                risk_tolerance=risk_tolerance,
                investment_style=investment_style,
                sector_preference=sector_preference,
                max_stocks=max_stocks
            )
            
            return {
                'success': True,
                'recommendations': recommendations,
                'criteria_used': criteria
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error generating recommendations: {str(e)}'}
    
    def analyze_market_outlook(self) -> Dict:
        """시장 전망 분석"""
        try:
            # 시장 데이터 수집
            market_overview = self.data_service.get_market_overview()
            
            if not market_overview['success']:
                return {'success': False, 'message': 'Failed to get market overview'}
            
            # 시장 분석
            market_outlook = self.market_analyzer.generate_market_outlook(
                market_overview['market_data'].get('^GSPC'),
                market_overview['sector_data'],
                market_overview['market_sentiment'].get('articles', [])
            )
            
            return {
                'success': True,
                'market_outlook': market_outlook,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error analyzing market outlook: {str(e)}'}
    
    def _get_stock_universe(self) -> List[Dict]:
        """주식 유니버스 조회 (샘플 데이터)"""
        # 실제 구현에서는 데이터베이스에서 조회
        return [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'market_cap': 3000000000000,
                'pe_ratio': 25.5,
                'beta': 1.2,
                'volatility': 0.25,
                'current_price': 150.0
            },
            # ... 더 많은 주식 데이터
        ]
    
    def _calculate_risk_score(self, volatility: float, max_drawdown: float, beta: float) -> float:
        """리스크 점수 계산 (0-100)"""
        score = 100
        
        # 변동성 페널티
        if volatility > 0.3:
            score -= 30
        elif volatility > 0.2:
            score -= 20
        elif volatility > 0.15:
            score -= 10
        
        # 최대 낙폭 페널티
        if max_drawdown < -0.3:
            score -= 25
        elif max_drawdown < -0.2:
            score -= 15
        elif max_drawdown < -0.1:
            score -= 5
        
        # 베타 페널티
        if beta > 1.5:
            score -= 15
        elif beta > 1.2:
            score -= 10
        elif beta < 0.8:
            score -= 5
        
        return max(0, score)
    
    def _calculate_performance_score(self, total_return: float, sharpe_ratio: float, excess_return: float) -> float:
        """성과 점수 계산 (0-100)"""
        score = 50  # 기본 점수
        
        # 수익률 보너스
        if total_return > 0.2:
            score += 30
        elif total_return > 0.15:
            score += 25
        elif total_return > 0.10:
            score += 20
        elif total_return > 0.05:
            score += 10
        
        # 샤프 비율 보너스
        if sharpe_ratio > 1.5:
            score += 20
        elif sharpe_ratio > 1.0:
            score += 15
        elif sharpe_ratio > 0.5:
            score += 10
        
        # 초과수익 보너스
        if excess_return > 0.05:
            score += 10
        elif excess_return > 0.02:
            score += 5
        
        return min(100, score)
    
    def _calculate_diversification_score(self, num_positions: int, max_weight: float, num_sectors: int) -> float:
        """다각화 점수 계산 (0-100)"""
        score = 0
        
        # 포지션 수 점수 (40점)
        if num_positions >= 15:
            score += 40
        elif num_positions >= 10:
            score += 30
        elif num_positions >= 5:
            score += 20
        else:
            score += 10
        
        # 최대 가중치 점수 (30점)
        if max_weight <= 0.1:
            score += 30
        elif max_weight <= 0.15:
            score += 25
        elif max_weight <= 0.2:
            score += 20
        elif max_weight <= 0.3:
            score += 15
        else:
            score += 5
        
        # 섹터 수 점수 (30점)
        if num_sectors >= 8:
            score += 30
        elif num_sectors >= 5:
            score += 25
        elif num_sectors >= 3:
            score += 20
        else:
            score += 10
        
        return score
    
    def _calculate_optimization_potential(self, return_improvement: float, sharpe_improvement: float) -> str:
        """최적화 잠재력 계산"""
        if return_improvement > 0.05 and sharpe_improvement > 0.3:
            return 'High'
        elif return_improvement > 0.02 and sharpe_improvement > 0.1:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_market_sensitivity(self, beta: float, correlation: float) -> str:
        """시장 민감도 계산"""
        if beta > 1.3 and correlation > 0.8:
            return 'High'
        elif beta > 1.1 and correlation > 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_overall_grade(self, risk_analysis: Dict, performance_analysis: Dict, diversification_analysis: Dict) -> str:
        """종합 등급 계산"""
        risk_score = risk_analysis.get('risk_score', 50)
        performance_score = performance_analysis.get('performance_score', 50)
        diversification_score = diversification_analysis.get('diversification_score', 50)
        
        overall_score = (risk_score + performance_score + diversification_score) / 3
        
        if overall_score >= 85:
            return 'A+'
        elif overall_score >= 80:
            return 'A'
        elif overall_score >= 75:
            return 'A-'
        elif overall_score >= 70:
            return 'B+'
        elif overall_score >= 65:
            return 'B'
        elif overall_score >= 60:
            return 'B-'
        elif overall_score >= 55:
            return 'C+'
        elif overall_score >= 50:
            return 'C'
        else:
            return 'D'
