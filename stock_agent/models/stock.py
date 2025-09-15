from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

@dataclass
class StockData:
    """주식 데이터 클래스"""
    symbol: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: Optional[float] = None
    
    # 기술적 지표
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_percent: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    williams_r: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    cci: Optional[float] = None
    obv: Optional[float] = None
    vwap: Optional[float] = None

@dataclass
class Stock:
    """주식 클래스"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: float
    forward_pe: float
    peg_ratio: float
    price_to_book: float
    dividend_yield: float
    beta: float
    year_high: float
    year_low: float
    avg_volume: int
    shares_outstanding: float
    enterprise_value: float
    revenue: float
    profit_margin: float
    return_on_equity: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    description: str
    current_price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.current_price == 0 and self.year_high > 0:
            self.current_price = (self.year_high + self.year_low) / 2
    
    @property
    def market_cap_formatted(self) -> str:
        """시가총액 포맷팅"""
        if self.market_cap >= 1e12:
            return f"${self.market_cap/1e12:.2f}T"
        elif self.market_cap >= 1e9:
            return f"${self.market_cap/1e9:.2f}B"
        elif self.market_cap >= 1e6:
            return f"${self.market_cap/1e6:.2f}M"
        else:
            return f"${self.market_cap:,.0f}"
    
    @property
    def revenue_formatted(self) -> str:
        """매출 포맷팅"""
        if self.revenue >= 1e12:
            return f"${self.revenue/1e12:.2f}T"
        elif self.revenue >= 1e9:
            return f"${self.revenue/1e9:.2f}B"
        elif self.revenue >= 1e6:
            return f"${self.revenue/1e6:.2f}M"
        else:
            return f"${self.revenue:,.0f}"
    
    @property
    def is_undervalued(self) -> bool:
        """저평가 여부 판단"""
        if self.pe_ratio <= 0 or self.forward_pe <= 0:
            return False
        return self.pe_ratio < 15 and self.forward_pe < 15
    
    @property
    def is_growth_stock(self) -> bool:
        """성장주 여부 판단"""
        if self.peg_ratio <= 0:
            return False
        return self.peg_ratio < 1.0 and self.pe_ratio > 20
    
    @property
    def is_value_stock(self) -> bool:
        """가치주 여부 판단"""
        if self.pe_ratio <= 0 or self.price_to_book <= 0:
            return False
        return self.pe_ratio < 15 and self.price_to_book < 2.0
    
    @property
    def is_dividend_stock(self) -> bool:
        """배당주 여부 판단"""
        return self.dividend_yield > 0.03  # 3% 이상
    
    @property
    def risk_level(self) -> str:
        """리스크 레벨 판단"""
        if self.beta <= 0:
            return "unknown"
        elif self.beta < 0.8:
            return "low"
        elif self.beta < 1.2:
            return "medium"
        else:
            return "high"
    
    def get_financial_health_score(self) -> float:
        """재무 건전성 점수 (0-100)"""
        score = 0
        
        # 수익성 지표 (30점)
        if self.profit_margin > 0.15:
            score += 15
        elif self.profit_margin > 0.10:
            score += 10
        elif self.profit_margin > 0.05:
            score += 5
        
        if self.return_on_equity > 0.20:
            score += 15
        elif self.return_on_equity > 0.15:
            score += 10
        elif self.return_on_equity > 0.10:
            score += 5
        
        # 유동성 지표 (25점)
        if self.current_ratio > 2.0:
            score += 15
        elif self.current_ratio > 1.5:
            score += 10
        elif self.current_ratio > 1.0:
            score += 5
        
        if self.quick_ratio > 1.5:
            score += 10
        elif self.quick_ratio > 1.0:
            score += 5
        
        # 부채 지표 (25점)
        if self.debt_to_equity < 0.3:
            score += 25
        elif self.debt_to_equity < 0.5:
            score += 20
        elif self.debt_to_equity < 0.7:
            score += 15
        elif self.debt_to_equity < 1.0:
            score += 10
        
        # 성장성 지표 (20점)
        if self.peg_ratio > 0 and self.peg_ratio < 1.0:
            score += 20
        elif self.peg_ratio > 0 and self.peg_ratio < 1.5:
            score += 15
        elif self.peg_ratio > 0 and self.peg_ratio < 2.0:
            score += 10
        
        return min(score, 100)
    
    def get_valuation_score(self) -> float:
        """밸류에이션 점수 (0-100)"""
        score = 0
        
        # P/E 비율 (40점)
        if self.pe_ratio > 0:
            if self.pe_ratio < 10:
                score += 40
            elif self.pe_ratio < 15:
                score += 30
            elif self.pe_ratio < 20:
                score += 20
            elif self.pe_ratio < 25:
                score += 10
        
        # P/B 비율 (30점)
        if self.price_to_book > 0:
            if self.price_to_book < 1.0:
                score += 30
            elif self.price_to_book < 1.5:
                score += 25
            elif self.price_to_book < 2.0:
                score += 20
            elif self.price_to_book < 3.0:
                score += 10
        
        # PEG 비율 (30점)
        if self.peg_ratio > 0:
            if self.peg_ratio < 0.5:
                score += 30
            elif self.peg_ratio < 1.0:
                score += 25
            elif self.peg_ratio < 1.5:
                score += 20
            elif self.peg_ratio < 2.0:
                score += 10
        
        return min(score, 100)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'market_cap_formatted': self.market_cap_formatted,
            'pe_ratio': self.pe_ratio,
            'forward_pe': self.forward_pe,
            'peg_ratio': self.peg_ratio,
            'price_to_book': self.price_to_book,
            'dividend_yield': self.dividend_yield,
            'beta': self.beta,
            'year_high': self.year_high,
            'year_low': self.year_low,
            'current_price': self.current_price,
            'change': self.change,
            'change_percent': self.change_percent,
            'avg_volume': self.avg_volume,
            'revenue': self.revenue,
            'revenue_formatted': self.revenue_formatted,
            'profit_margin': self.profit_margin,
            'return_on_equity': self.return_on_equity,
            'debt_to_equity': self.debt_to_equity,
            'current_ratio': self.current_ratio,
            'quick_ratio': self.quick_ratio,
            'is_undervalued': self.is_undervalued,
            'is_growth_stock': self.is_growth_stock,
            'is_value_stock': self.is_value_stock,
            'is_dividend_stock': self.is_dividend_stock,
            'risk_level': self.risk_level,
            'financial_health_score': self.get_financial_health_score(),
            'valuation_score': self.get_valuation_score()
        }
