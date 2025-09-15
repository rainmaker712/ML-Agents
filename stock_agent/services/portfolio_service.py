from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import DatabaseManager, Portfolio, PortfolioItem, Stock
from ai_engine import PortfolioOptimizer, RecommendationEngine, RiskAnalyzer
from data_collectors import YFinanceCollector, AlphaVantageCollector

class PortfolioService:
    """포트폴리오 관리 서비스"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.recommendation_engine = RecommendationEngine()
        self.risk_analyzer = RiskAnalyzer()
        self.yfinance_collector = YFinanceCollector()
        self.alpha_vantage_collector = AlphaVantageCollector()
    
    def create_portfolio(self, name: str, description: str = "", 
                        risk_tolerance: str = "medium", 
                        target_return: float = None,
                        max_positions: int = 10) -> int:
        """포트폴리오 생성"""
        portfolio_id = self.db_manager.create_portfolio(
            name=name,
            description=description,
            risk_tolerance=risk_tolerance
        )
        return portfolio_id
    
    def get_portfolio(self, portfolio_id: int) -> Optional[Portfolio]:
        """포트폴리오 조회"""
        portfolio_data = self.db_manager.get_portfolios()
        for portfolio in portfolio_data:
            if portfolio.id == portfolio_id:
                items = self.db_manager.get_portfolio_items(portfolio_id)
                portfolio.items = [
                    PortfolioItem(
                        symbol=item.symbol,
                        weight=item.weight,
                        target_weight=item.target_weight,
                        current_price=item.current_price,
                        shares=item.shares,
                        cost_basis=item.cost_basis,
                        market_value=item.market_value,
                        unrealized_pnl=item.unrealized_pnl,
                        realized_pnl=item.realized_pnl
                    )
                    for item in items
                ]
                return portfolio
        return None
    
    def add_stock_to_portfolio(self, portfolio_id: int, symbol: str, 
                              weight: float, target_weight: float = None,
                              shares: float = None, cost_basis: float = None) -> bool:
        """포트폴리오에 주식 추가"""
        try:
            # 주식 정보 조회 및 저장
            stock_info = self.yfinance_collector.get_stock_info(symbol)
            if stock_info:
                self.db_manager.add_stock_info(symbol, stock_info)
            
            # 포트폴리오 아이템 추가
            self.db_manager.add_portfolio_item(
                portfolio_id=portfolio_id,
                symbol=symbol,
                weight=weight,
                target_weight=target_weight or weight,
                shares=shares,
                cost_basis=cost_basis
            )
            return True
        except Exception as e:
            print(f"Error adding stock to portfolio: {e}")
            return False
    
    def remove_stock_from_portfolio(self, portfolio_id: int, symbol: str) -> bool:
        """포트폴리오에서 주식 제거"""
        try:
            # 실제 구현에서는 데이터베이스에서 제거
            return True
        except Exception as e:
            print(f"Error removing stock from portfolio: {e}")
            return False
    
    def update_portfolio_weights(self, portfolio_id: int, 
                                weight_updates: Dict[str, float]) -> bool:
        """포트폴리오 가중치 업데이트"""
        try:
            for symbol, new_weight in weight_updates.items():
                self.db_manager.update_item_weight(portfolio_id, symbol, new_weight)
            return True
        except Exception as e:
            print(f"Error updating portfolio weights: {e}")
            return False
    
    def rebalance_portfolio(self, portfolio_id: int) -> Dict:
        """포트폴리오 리밸런싱"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {'success': False, 'message': 'Portfolio not found'}
        
        # 현재 가중치와 목표 가중치 비교
        rebalance_needed = False
        weight_changes = {}
        
        for item in portfolio.items:
            weight_diff = abs(item.weight - item.target_weight)
            if weight_diff > 0.05:  # 5% 이상 차이
                rebalance_needed = True
                weight_changes[item.symbol] = item.target_weight - item.weight
        
        if rebalance_needed:
            # 리밸런싱 실행
            for symbol, weight_change in weight_changes.items():
                new_weight = portfolio.items[0].weight + weight_change  # 간단한 예시
                self.update_portfolio_weights(portfolio_id, {symbol: new_weight})
        
        return {
            'success': True,
            'rebalance_needed': rebalance_needed,
            'weight_changes': weight_changes
        }
    
    def get_portfolio_performance(self, portfolio_id: int, 
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> Dict:
        """포트폴리오 성과 분석"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {}
        
        # 성과 데이터 수집
        performance_data = {}
        for item in portfolio.items:
            try:
                # 과거 데이터 조회
                hist_data = self.yfinance_collector.get_historical_data(
                    item.symbol, period='1y'
                )
                if not hist_data.empty:
                    returns = hist_data['Close'].pct_change().dropna()
                    performance_data[item.symbol] = returns
            except Exception as e:
                print(f"Error getting performance data for {item.symbol}: {e}")
                continue
        
        if not performance_data:
            return portfolio.get_performance_summary()
        
        # 포트폴리오 수익률 계산
        returns_df = pd.DataFrame(performance_data)
        weights = np.array([item.weight for item in portfolio.items])
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # 성과 지표 계산
        total_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = self.risk_analyzer.calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self.risk_analyzer.calculate_max_drawdown(portfolio_returns)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown['max_drawdown'],
            'portfolio_returns': portfolio_returns,
            'individual_returns': performance_data
        }
    
    def get_portfolio_risk_analysis(self, portfolio_id: int) -> Dict:
        """포트폴리오 리스크 분석"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {}
        
        # 성과 데이터 수집
        performance_data = {}
        for item in portfolio.items:
            try:
                hist_data = self.yfinance_collector.get_historical_data(
                    item.symbol, period='1y'
                )
                if not hist_data.empty:
                    returns = hist_data['Close'].pct_change().dropna()
                    performance_data[item.symbol] = returns
            except Exception as e:
                print(f"Error getting risk data for {item.symbol}: {e}")
                continue
        
        if not performance_data:
            return {}
        
        returns_df = pd.DataFrame(performance_data)
        weights = np.array([item.weight for item in portfolio.items])
        
        # 리스크 지표 계산
        risk_metrics = self.risk_analyzer.calculate_portfolio_risk_metrics(
            weights, returns_df
        )
        
        # 상관관계 분석
        correlation_matrix = self.risk_analyzer.calculate_correlation_matrix(returns_df)
        
        # 집중도 리스크
        concentration_risk = self.risk_analyzer.calculate_concentration_risk(weights)
        
        return {
            'risk_metrics': risk_metrics,
            'correlation_matrix': correlation_matrix,
            'concentration_risk': concentration_risk,
            'diversification_ratio': self.risk_analyzer.calculate_diversification_ratio(
                weights, returns_df.cov()
            )
        }
    
    def get_portfolio_recommendations(self, portfolio_id: int, 
                                    risk_tolerance: str = None,
                                    investment_style: str = None) -> Dict:
        """포트폴리오 개선 권장사항"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {}
        
        recommendations = []
        
        # 리스크 분석
        risk_analysis = self.get_portfolio_risk_analysis(portfolio_id)
        if risk_analysis:
            risk_metrics = risk_analysis.get('risk_metrics', {})
            
            # 변동성 권장사항
            volatility = risk_metrics.get('portfolio_volatility', 0)
            if volatility > 0.25:
                recommendations.append({
                    'type': 'risk',
                    'priority': 'high',
                    'message': f'포트폴리오 변동성이 높습니다 ({volatility:.1%}). 리스크를 줄이기 위해 안정적인 자산을 추가하세요.'
                })
            
            # 샤프 비율 권장사항
            sharpe_ratio = risk_metrics.get('portfolio_sharpe', 0)
            if sharpe_ratio < 0.5:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'message': f'샤프 비율이 낮습니다 ({sharpe_ratio:.2f}). 리스크 대비 수익률을 개선하세요.'
                })
        
        # 집중도 권장사항
        concentration_risk = risk_analysis.get('concentration_risk', {})
        max_weight = concentration_risk.get('max_weight', 0)
        if max_weight > 0.3:
            recommendations.append({
                'type': 'diversification',
                'priority': 'high',
                'message': f'단일 종목 비중이 높습니다 ({max_weight:.1%}). 다각화를 늘리세요.'
            })
        
        # 섹터 다각화 권장사항
        sector_allocation = portfolio.get_sector_allocation()
        if len(sector_allocation) < 3:
            recommendations.append({
                'type': 'diversification',
                'priority': 'medium',
                'message': '섹터 다각화가 부족합니다. 다양한 섹터의 종목을 추가하세요.'
            })
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'high_priority': len([r for r in recommendations if r['priority'] == 'high'])
        }
    
    def optimize_portfolio(self, portfolio_id: int, 
                          optimization_type: str = 'sharpe') -> Dict:
        """포트폴리오 최적화"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {'success': False, 'message': 'Portfolio not found'}
        
        # 성과 데이터 수집
        performance_data = {}
        symbols = []
        for item in portfolio.items:
            try:
                hist_data = self.yfinance_collector.get_historical_data(
                    item.symbol, period='1y'
                )
                if not hist_data.empty:
                    returns = hist_data['Close'].pct_change().dropna()
                    performance_data[item.symbol] = returns
                    symbols.append(item.symbol)
            except Exception as e:
                print(f"Error getting optimization data for {item.symbol}: {e}")
                continue
        
        if len(performance_data) < 2:
            return {'success': False, 'message': 'Insufficient data for optimization'}
        
        returns_df = pd.DataFrame(performance_data)
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # 최적화 실행
        if optimization_type == 'sharpe':
            result = self.portfolio_optimizer.optimize_portfolio(
                expected_returns.values,
                cov_matrix.values,
                risk_tolerance=portfolio.risk_tolerance
            )
        elif optimization_type == 'min_variance':
            result = self.portfolio_optimizer.optimize_minimum_variance(
                cov_matrix.values
            )
        elif optimization_type == 'risk_parity':
            result = self.portfolio_optimizer.optimize_risk_parity(
                cov_matrix.values
            )
        else:
            return {'success': False, 'message': 'Invalid optimization type'}
        
        if result['success']:
            # 최적화된 가중치를 포트폴리오에 적용
            optimal_weights = result['weights']
            weight_updates = {
                symbols[i]: optimal_weights[i] 
                for i in range(len(symbols))
            }
            
            self.update_portfolio_weights(portfolio_id, weight_updates)
            
            return {
                'success': True,
                'optimal_weights': weight_updates,
                'expected_return': result.get('expected_return', 0),
                'volatility': result.get('volatility', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0)
            }
        else:
            return {
                'success': False,
                'message': result.get('message', 'Optimization failed')
            }
    
    def get_portfolio_summary(self, portfolio_id: int) -> Dict:
        """포트폴리오 요약 정보"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {}
        
        # 기본 정보
        summary = portfolio.get_performance_summary()
        
        # 리스크 분석
        risk_analysis = self.get_portfolio_risk_analysis(portfolio_id)
        if risk_analysis:
            summary['risk_metrics'] = risk_analysis.get('risk_metrics', {})
            summary['concentration_risk'] = risk_analysis.get('concentration_risk', {})
        
        # 권장사항
        recommendations = self.get_portfolio_recommendations(portfolio_id)
        summary['recommendations'] = recommendations
        
        return summary
