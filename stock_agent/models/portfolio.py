from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class PortfolioItem:
    """포트폴리오 아이템 클래스"""
    symbol: str
    weight: float  # 포트폴리오 내 비중 (0-1)
    target_weight: float = None  # 목표 비중
    current_price: float = 0.0
    shares: float = 0.0
    cost_basis: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    added_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.target_weight is None:
            self.target_weight = self.weight
        if self.current_price > 0 and self.shares > 0:
            self.market_value = self.current_price * self.shares
            if self.cost_basis > 0:
                self.unrealized_pnl = self.market_value - self.cost_basis
    
    def update_price(self, new_price: float):
        """가격 업데이트"""
        self.current_price = new_price
        if self.shares > 0:
            self.market_value = self.current_price * self.shares
            if self.cost_basis > 0:
                self.unrealized_pnl = self.market_value - self.cost_basis
    
    def get_return_percent(self) -> float:
        """수익률 계산 (%)"""
        if self.cost_basis <= 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

@dataclass
class Portfolio:
    """포트폴리오 클래스"""
    id: int
    name: str
    description: str = ""
    risk_tolerance: str = "medium"  # low, medium, high
    target_return: float = 0.0
    max_positions: int = 10
    items: List[PortfolioItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_item(self, symbol: str, weight: float, target_weight: float = None, 
                shares: float = None, cost_basis: float = None):
        """포트폴리오 아이템 추가"""
        if len(self.items) >= self.max_positions:
            raise ValueError(f"Maximum positions ({self.max_positions}) exceeded")
        
        if sum(item.weight for item in self.items) + weight > 1.0:
            raise ValueError("Total weight cannot exceed 1.0")
        
        item = PortfolioItem(
            symbol=symbol,
            weight=weight,
            target_weight=target_weight or weight,
            shares=shares,
            cost_basis=cost_basis
        )
        self.items.append(item)
        self.updated_at = datetime.now()
    
    def remove_item(self, symbol: str):
        """포트폴리오 아이템 제거"""
        self.items = [item for item in self.items if item.symbol != symbol]
        self.updated_at = datetime.now()
    
    def update_item_weight(self, symbol: str, new_weight: float):
        """아이템 비중 업데이트"""
        for item in self.items:
            if item.symbol == symbol:
                item.weight = new_weight
                break
        self.updated_at = datetime.now()
    
    def rebalance(self):
        """포트폴리오 리밸런싱"""
        total_weight = sum(item.weight for item in self.items)
        if total_weight == 0:
            return
        
        # 비중 정규화
        for item in self.items:
            item.weight = item.weight / total_weight
    
    def get_total_value(self) -> float:
        """총 포트폴리오 가치"""
        return sum(item.market_value for item in self.items)
    
    def get_total_cost(self) -> float:
        """총 투자 원금"""
        return sum(item.cost_basis for item in self.items)
    
    def get_total_unrealized_pnl(self) -> float:
        """총 미실현 손익"""
        return sum(item.unrealized_pnl for item in self.items)
    
    def get_total_realized_pnl(self) -> float:
        """총 실현 손익"""
        return sum(item.realized_pnl for item in self.items)
    
    def get_total_pnl(self) -> float:
        """총 손익"""
        return self.get_total_unrealized_pnl() + self.get_total_realized_pnl()
    
    def get_total_return_percent(self) -> float:
        """총 수익률 (%)"""
        total_cost = self.get_total_cost()
        if total_cost <= 0:
            return 0.0
        return (self.get_total_pnl() / total_cost) * 100
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """섹터별 배분"""
        # 실제 구현에서는 각 주식의 섹터 정보가 필요
        # 여기서는 간단히 주식 심볼로 구분
        sector_allocation = {}
        for item in self.items:
            # 실제로는 데이터베이스에서 섹터 정보를 가져와야 함
            sector = "Unknown"  # 임시
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += item.weight
        return sector_allocation
    
    def get_top_holdings(self, n: int = 5) -> List[PortfolioItem]:
        """상위 보유 종목"""
        return sorted(self.items, key=lambda x: x.weight, reverse=True)[:n]
    
    def get_performance_summary(self) -> Dict:
        """성과 요약"""
        return {
            'total_value': self.get_total_value(),
            'total_cost': self.get_total_cost(),
            'total_pnl': self.get_total_pnl(),
            'total_return_percent': self.get_total_return_percent(),
            'unrealized_pnl': self.get_total_unrealized_pnl(),
            'realized_pnl': self.get_total_realized_pnl(),
            'num_positions': len(self.items),
            'sector_allocation': self.get_sector_allocation(),
            'top_holdings': [
                {
                    'symbol': item.symbol,
                    'weight': item.weight,
                    'market_value': item.market_value,
                    'return_percent': item.get_return_percent()
                }
                for item in self.get_top_holdings()
            ]
        }
    
    def calculate_risk_metrics(self, returns_data: pd.DataFrame = None) -> Dict:
        """리스크 지표 계산"""
        if returns_data is None or returns_data.empty:
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        # 포트폴리오 수익률 계산
        portfolio_returns = []
        for item in self.items:
            if item.symbol in returns_data.columns:
                stock_returns = returns_data[item.symbol].pct_change().dropna()
                weighted_returns = stock_returns * item.weight
                portfolio_returns.append(weighted_returns)
        
        if not portfolio_returns:
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        # 포트폴리오 수익률 합계
        portfolio_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
        
        # 변동성 (연간)
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 샤프 비율
        risk_free_rate = 0.02  # 2% 가정
        excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # CVaR (95%)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'risk_tolerance': self.risk_tolerance,
            'target_return': self.target_return,
            'max_positions': self.max_positions,
            'num_positions': len(self.items),
            'total_value': self.get_total_value(),
            'total_cost': self.get_total_cost(),
            'total_pnl': self.get_total_pnl(),
            'total_return_percent': self.get_total_return_percent(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'items': [
                {
                    'symbol': item.symbol,
                    'weight': item.weight,
                    'target_weight': item.target_weight,
                    'current_price': item.current_price,
                    'shares': item.shares,
                    'market_value': item.market_value,
                    'unrealized_pnl': item.unrealized_pnl,
                    'return_percent': item.get_return_percent()
                }
                for item in self.items
            ]
        }
