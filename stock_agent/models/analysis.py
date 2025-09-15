from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from .portfolio import Portfolio
from .stock import Stock

@dataclass
class AnalysisResult:
    """분석 결과 클래스"""
    analysis_type: str
    symbol: str = None
    portfolio_id: int = None
    result_data: Dict = None
    confidence_score: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class PortfolioAnalyzer:
    """포트폴리오 분석 클래스"""
    
    def __init__(self):
        pass
    
    def analyze_portfolio_risk(self, portfolio: Portfolio, market_data: pd.DataFrame = None) -> AnalysisResult:
        """포트폴리오 리스크 분석"""
        risk_metrics = portfolio.calculate_risk_metrics(market_data)
        
        # 리스크 등급 결정
        volatility = risk_metrics['volatility']
        if volatility < 0.15:
            risk_grade = 'Low'
        elif volatility < 0.25:
            risk_grade = 'Medium'
        else:
            risk_grade = 'High'
        
        # 샤프 비율 평가
        sharpe_ratio = risk_metrics['sharpe_ratio']
        if sharpe_ratio > 1.0:
            sharpe_grade = 'Excellent'
        elif sharpe_ratio > 0.5:
            sharpe_grade = 'Good'
        elif sharpe_ratio > 0:
            sharpe_grade = 'Fair'
        else:
            sharpe_grade = 'Poor'
        
        result_data = {
            'risk_grade': risk_grade,
            'sharpe_grade': sharpe_grade,
            'metrics': risk_metrics,
            'recommendations': self._get_risk_recommendations(risk_metrics, portfolio.risk_tolerance)
        }
        
        confidence_score = min(0.9, max(0.1, 1.0 - abs(volatility - 0.2) / 0.2))
        
        return AnalysisResult(
            analysis_type='portfolio_risk',
            portfolio_id=portfolio.id,
            result_data=result_data,
            confidence_score=confidence_score
        )
    
    def analyze_portfolio_diversification(self, portfolio: Portfolio, stock_data: Dict[str, Stock] = None) -> AnalysisResult:
        """포트폴리오 다각화 분석"""
        sector_allocation = portfolio.get_sector_allocation()
        
        # 다각화 점수 계산
        num_sectors = len(sector_allocation)
        max_weight = max(sector_allocation.values()) if sector_allocation else 0
        
        # 섹터 수 기반 점수 (0-50점)
        sector_score = min(50, num_sectors * 5)
        
        # 최대 비중 기반 점수 (0-50점)
        concentration_score = max(0, 50 - (max_weight - 0.1) * 500)
        
        diversification_score = sector_score + concentration_score
        
        # 다각화 등급
        if diversification_score >= 80:
            grade = 'Excellent'
        elif diversification_score >= 60:
            grade = 'Good'
        elif diversification_score >= 40:
            grade = 'Fair'
        else:
            grade = 'Poor'
        
        result_data = {
            'diversification_score': diversification_score,
            'grade': grade,
            'sector_allocation': sector_allocation,
            'num_sectors': num_sectors,
            'max_sector_weight': max_weight,
            'recommendations': self._get_diversification_recommendations(sector_allocation, portfolio.risk_tolerance)
        }
        
        confidence_score = min(0.9, diversification_score / 100)
        
        return AnalysisResult(
            analysis_type='portfolio_diversification',
            portfolio_id=portfolio.id,
            result_data=result_data,
            confidence_score=confidence_score
        )
    
    def analyze_portfolio_performance(self, portfolio: Portfolio, benchmark_returns: pd.Series = None) -> AnalysisResult:
        """포트폴리오 성과 분석"""
        total_return = portfolio.get_total_return_percent()
        
        # 벤치마크 대비 성과
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_return = benchmark_returns.mean() * 252 * 100  # 연간 수익률로 변환
            excess_return = total_return - benchmark_return
        else:
            benchmark_return = 0
            excess_return = total_return
        
        # 성과 등급
        if total_return > 15:
            performance_grade = 'Excellent'
        elif total_return > 10:
            performance_grade = 'Good'
        elif total_return > 5:
            performance_grade = 'Fair'
        else:
            performance_grade = 'Poor'
        
        result_data = {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'performance_grade': performance_grade,
            'top_performers': self._get_top_performers(portfolio),
            'underperformers': self._get_underperformers(portfolio),
            'recommendations': self._get_performance_recommendations(total_return, excess_return)
        }
        
        confidence_score = min(0.9, max(0.1, abs(total_return) / 20))
        
        return AnalysisResult(
            analysis_type='portfolio_performance',
            portfolio_id=portfolio.id,
            result_data=result_data,
            confidence_score=confidence_score
        )
    
    def _get_risk_recommendations(self, risk_metrics: Dict, risk_tolerance: str) -> List[str]:
        """리스크 개선 권장사항"""
        recommendations = []
        
        volatility = risk_metrics['volatility']
        sharpe_ratio = risk_metrics['sharpe_ratio']
        max_drawdown = risk_metrics['max_drawdown']
        
        if volatility > 0.3:
            recommendations.append("Consider reducing portfolio volatility by adding more stable assets")
        
        if sharpe_ratio < 0.5:
            recommendations.append("Improve risk-adjusted returns by rebalancing portfolio")
        
        if max_drawdown < -0.2:
            recommendations.append("Consider adding defensive positions to reduce drawdown risk")
        
        if risk_tolerance == 'low' and volatility > 0.15:
            recommendations.append("Portfolio may be too risky for low risk tolerance")
        elif risk_tolerance == 'high' and volatility < 0.1:
            recommendations.append("Portfolio may be too conservative for high risk tolerance")
        
        return recommendations
    
    def _get_diversification_recommendations(self, sector_allocation: Dict, risk_tolerance: str) -> List[str]:
        """다각화 개선 권장사항"""
        recommendations = []
        
        if len(sector_allocation) < 3:
            recommendations.append("Consider adding stocks from different sectors for better diversification")
        
        max_weight = max(sector_allocation.values()) if sector_allocation else 0
        if max_weight > 0.4:
            recommendations.append("Reduce concentration in single sector (currently {:.1%})".format(max_weight))
        
        if risk_tolerance == 'low':
            recommendations.append("Consider adding defensive sectors like utilities and consumer staples")
        elif risk_tolerance == 'high':
            recommendations.append("Consider adding growth sectors like technology and healthcare")
        
        return recommendations
    
    def _get_performance_recommendations(self, total_return: float, excess_return: float) -> List[str]:
        """성과 개선 권장사항"""
        recommendations = []
        
        if total_return < 0:
            recommendations.append("Consider reviewing underperforming positions")
        
        if excess_return < -5:
            recommendations.append("Portfolio is underperforming benchmark significantly")
        
        if total_return > 20:
            recommendations.append("Consider taking some profits and rebalancing")
        
        return recommendations
    
    def _get_top_performers(self, portfolio: Portfolio) -> List[Dict]:
        """상위 성과 종목"""
        performers = []
        for item in portfolio.items:
            if item.get_return_percent() > 0:
                performers.append({
                    'symbol': item.symbol,
                    'return_percent': item.get_return_percent(),
                    'weight': item.weight
                })
        return sorted(performers, key=lambda x: x['return_percent'], reverse=True)[:3]
    
    def _get_underperformers(self, portfolio: Portfolio) -> List[Dict]:
        """하위 성과 종목"""
        underperformers = []
        for item in portfolio.items:
            if item.get_return_percent() < -5:
                underperformers.append({
                    'symbol': item.symbol,
                    'return_percent': item.get_return_percent(),
                    'weight': item.weight
                })
        return sorted(underperformers, key=lambda x: x['return_percent'])[:3]

class StockAnalyzer:
    """개별 주식 분석 클래스"""
    
    def __init__(self):
        pass
    
    def analyze_stock_fundamentals(self, stock: Stock) -> AnalysisResult:
        """주식 펀더멘털 분석"""
        financial_health_score = stock.get_financial_health_score()
        valuation_score = stock.get_valuation_score()
        
        # 종합 점수
        overall_score = (financial_health_score + valuation_score) / 2
        
        # 투자 등급
        if overall_score >= 80:
            grade = 'Strong Buy'
        elif overall_score >= 70:
            grade = 'Buy'
        elif overall_score >= 60:
            grade = 'Hold'
        elif overall_score >= 50:
            grade = 'Weak Hold'
        else:
            grade = 'Sell'
        
        result_data = {
            'overall_score': overall_score,
            'grade': grade,
            'financial_health_score': financial_health_score,
            'valuation_score': valuation_score,
            'strengths': self._get_stock_strengths(stock),
            'weaknesses': self._get_stock_weaknesses(stock),
            'recommendations': self._get_stock_recommendations(stock, overall_score)
        }
        
        confidence_score = min(0.9, overall_score / 100)
        
        return AnalysisResult(
            analysis_type='stock_fundamentals',
            symbol=stock.symbol,
            result_data=result_data,
            confidence_score=confidence_score
        )
    
    def analyze_stock_technical(self, stock_data: pd.DataFrame) -> AnalysisResult:
        """주식 기술적 분석"""
        if stock_data.empty:
            return AnalysisResult(
                analysis_type='stock_technical',
                result_data={'error': 'No data available'},
                confidence_score=0.0
            )
        
        latest = stock_data.iloc[-1]
        
        # 기술적 지표 분석
        technical_signals = []
        
        # RSI 분석
        if 'rsi' in stock_data.columns and not pd.isna(latest['rsi']):
            if latest['rsi'] > 70:
                technical_signals.append('RSI Overbought')
            elif latest['rsi'] < 30:
                technical_signals.append('RSI Oversold')
        
        # MACD 분석
        if 'macd' in stock_data.columns and 'macd_signal' in stock_data.columns:
            if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                if latest['macd'] > latest['macd_signal']:
                    technical_signals.append('MACD Bullish')
                else:
                    technical_signals.append('MACD Bearish')
        
        # 이동평균 분석
        if 'sma_20' in stock_data.columns and 'sma_50' in stock_data.columns:
            if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
                if latest['sma_20'] > latest['sma_50']:
                    technical_signals.append('SMA Bullish')
                else:
                    technical_signals.append('SMA Bearish')
        
        # 기술적 등급
        bullish_signals = sum(1 for signal in technical_signals if 'Bullish' in signal or 'Oversold' in signal)
        bearish_signals = sum(1 for signal in technical_signals if 'Bearish' in signal or 'Overbought' in signal)
        
        if bullish_signals > bearish_signals:
            technical_grade = 'Bullish'
        elif bearish_signals > bullish_signals:
            technical_grade = 'Bearish'
        else:
            technical_grade = 'Neutral'
        
        result_data = {
            'technical_grade': technical_grade,
            'signals': technical_signals,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'latest_data': latest.to_dict()
        }
        
        confidence_score = min(0.9, len(technical_signals) / 10)
        
        return AnalysisResult(
            analysis_type='stock_technical',
            result_data=result_data,
            confidence_score=confidence_score
        )
    
    def _get_stock_strengths(self, stock: Stock) -> List[str]:
        """주식 강점 분석"""
        strengths = []
        
        if stock.is_undervalued:
            strengths.append("Undervalued based on P/E ratio")
        
        if stock.is_growth_stock:
            strengths.append("Strong growth potential")
        
        if stock.is_value_stock:
            strengths.append("Good value investment")
        
        if stock.is_dividend_stock:
            strengths.append("Attractive dividend yield")
        
        if stock.get_financial_health_score() > 70:
            strengths.append("Strong financial health")
        
        if stock.beta < 0.8:
            strengths.append("Low volatility")
        
        return strengths
    
    def _get_stock_weaknesses(self, stock: Stock) -> List[str]:
        """주식 약점 분석"""
        weaknesses = []
        
        if stock.pe_ratio > 30:
            weaknesses.append("High P/E ratio")
        
        if stock.debt_to_equity > 1.0:
            weaknesses.append("High debt levels")
        
        if stock.profit_margin < 0.05:
            weaknesses.append("Low profit margins")
        
        if stock.beta > 1.5:
            weaknesses.append("High volatility")
        
        if stock.get_financial_health_score() < 50:
            weaknesses.append("Weak financial health")
        
        return weaknesses
    
    def _get_stock_recommendations(self, stock: Stock, overall_score: float) -> List[str]:
        """주식 투자 권장사항"""
        recommendations = []
        
        if overall_score >= 80:
            recommendations.append("Strong buy recommendation")
        elif overall_score >= 70:
            recommendations.append("Buy recommendation")
        elif overall_score < 50:
            recommendations.append("Consider selling or avoiding")
        
        if stock.is_undervalued and stock.get_financial_health_score() > 60:
            recommendations.append("Good value opportunity")
        
        if stock.is_dividend_stock and stock.risk_level == 'low':
            recommendations.append("Suitable for income-focused investors")
        
        if stock.is_growth_stock and stock.risk_level == 'high':
            recommendations.append("Suitable for growth-oriented investors")
        
        return recommendations
