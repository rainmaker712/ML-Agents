import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """포트폴리오 추천 엔진"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def recommend_stocks(self, stock_universe: List[Dict], 
                        risk_tolerance: str = 'medium',
                        investment_style: str = 'balanced',
                        sector_preference: List[str] = None,
                        max_stocks: int = 10) -> Dict:
        """주식 추천"""
        
        # 필터링
        filtered_stocks = self._filter_stocks(
            stock_universe, risk_tolerance, investment_style, sector_preference
        )
        
        if not filtered_stocks:
            return {'recommendations': [], 'message': 'No stocks match criteria'}
        
        # 스코어링
        scored_stocks = self._score_stocks(filtered_stocks, risk_tolerance, investment_style)
        
        # 상위 종목 선택
        top_stocks = sorted(scored_stocks, key=lambda x: x['total_score'], reverse=True)[:max_stocks]
        
        # 포트폴리오 가중치 계산
        weights = self._calculate_portfolio_weights(top_stocks, risk_tolerance)
        
        return {
            'recommendations': top_stocks,
            'weights': weights,
            'total_stocks': len(top_stocks),
            'criteria': {
                'risk_tolerance': risk_tolerance,
                'investment_style': investment_style,
                'sector_preference': sector_preference
            }
        }
    
    def _filter_stocks(self, stock_universe: List[Dict], 
                      risk_tolerance: str, investment_style: str,
                      sector_preference: List[str] = None) -> List[Dict]:
        """주식 필터링"""
        filtered = []
        
        for stock in stock_universe:
            # 기본 필터링
            if not self._passes_basic_filters(stock):
                continue
            
            # 리스크 허용도 필터링
            if not self._passes_risk_filter(stock, risk_tolerance):
                continue
            
            # 투자 스타일 필터링
            if not self._passes_style_filter(stock, investment_style):
                continue
            
            # 섹터 선호도 필터링
            if sector_preference and stock.get('sector') not in sector_preference:
                continue
            
            filtered.append(stock)
        
        return filtered
    
    def _passes_basic_filters(self, stock: Dict) -> bool:
        """기본 필터 통과 여부"""
        # 시가총액 필터 (1억 달러 이상)
        market_cap = stock.get('market_cap', 0)
        if market_cap < 100_000_000:
            return False
        
        # 거래량 필터 (평균 거래량 10만주 이상)
        avg_volume = stock.get('avg_volume', 0)
        if avg_volume < 100_000:
            return False
        
        # P/E 비율 필터 (0 < P/E < 100)
        pe_ratio = stock.get('pe_ratio', 0)
        if pe_ratio <= 0 or pe_ratio > 100:
            return False
        
        return True
    
    def _passes_risk_filter(self, stock: Dict, risk_tolerance: str) -> bool:
        """리스크 필터 통과 여부"""
        beta = stock.get('beta', 1.0)
        volatility = stock.get('volatility', 0.2)
        
        if risk_tolerance == 'low':
            return beta < 0.8 and volatility < 0.2
        elif risk_tolerance == 'medium':
            return beta < 1.3 and volatility < 0.3
        else:  # high
            return True
    
    def _passes_style_filter(self, stock: Dict, investment_style: str) -> bool:
        """투자 스타일 필터 통과 여부"""
        pe_ratio = stock.get('pe_ratio', 0)
        peg_ratio = stock.get('peg_ratio', 0)
        dividend_yield = stock.get('dividend_yield', 0)
        
        if investment_style == 'value':
            return pe_ratio < 15 and peg_ratio < 1.5
        elif investment_style == 'growth':
            return pe_ratio > 15 and peg_ratio < 2.0
        elif investment_style == 'income':
            return dividend_yield > 0.03
        else:  # balanced
            return True
    
    def _score_stocks(self, stocks: List[Dict], risk_tolerance: str, investment_style: str) -> List[Dict]:
        """주식 스코어링"""
        scored_stocks = []
        
        for stock in stocks:
            score = self._calculate_stock_score(stock, risk_tolerance, investment_style)
            stock['total_score'] = score
            scored_stocks.append(stock)
        
        return scored_stocks
    
    def _calculate_stock_score(self, stock: Dict, risk_tolerance: str, investment_style: str) -> float:
        """개별 주식 점수 계산"""
        score = 0.0
        
        # 펀더멘털 점수 (40%)
        fundamental_score = self._calculate_fundamental_score(stock)
        score += fundamental_score * 0.4
        
        # 기술적 점수 (20%)
        technical_score = self._calculate_technical_score(stock)
        score += technical_score * 0.2
        
        # 밸류에이션 점수 (20%)
        valuation_score = self._calculate_valuation_score(stock)
        score += valuation_score * 0.2
        
        # 리스크 조정 점수 (20%)
        risk_adjusted_score = self._calculate_risk_adjusted_score(stock, risk_tolerance)
        score += risk_adjusted_score * 0.2
        
        return score
    
    def _calculate_fundamental_score(self, stock: Dict) -> float:
        """펀더멘털 점수 계산"""
        score = 0.0
        
        # 수익성 지표
        profit_margin = stock.get('profit_margin', 0)
        if profit_margin > 0.15:
            score += 25
        elif profit_margin > 0.10:
            score += 20
        elif profit_margin > 0.05:
            score += 15
        
        roe = stock.get('return_on_equity', 0)
        if roe > 0.20:
            score += 25
        elif roe > 0.15:
            score += 20
        elif roe > 0.10:
            score += 15
        
        # 성장성 지표
        revenue_growth = stock.get('revenue_growth', 0)
        if revenue_growth > 0.20:
            score += 25
        elif revenue_growth > 0.10:
            score += 20
        elif revenue_growth > 0.05:
            score += 15
        
        # 안정성 지표
        debt_to_equity = stock.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:
            score += 25
        elif debt_to_equity < 0.5:
            score += 20
        elif debt_to_equity < 0.7:
            score += 15
        
        return min(score, 100)
    
    def _calculate_technical_score(self, stock: Dict) -> float:
        """기술적 점수 계산"""
        score = 0.0
        
        # RSI 점수
        rsi = stock.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 25
        elif 20 <= rsi <= 80:
            score += 20
        else:
            score += 10
        
        # MACD 점수
        macd = stock.get('macd', 0)
        macd_signal = stock.get('macd_signal', 0)
        if macd > macd_signal:
            score += 25
        else:
            score += 15
        
        # 이동평균 점수
        sma_20 = stock.get('sma_20', 0)
        sma_50 = stock.get('sma_50', 0)
        current_price = stock.get('current_price', 0)
        
        if current_price > sma_20 > sma_50:
            score += 25
        elif current_price > sma_20:
            score += 20
        else:
            score += 10
        
        # 볼린저 밴드 점수
        bb_percent = stock.get('bb_percent', 0.5)
        if 0.2 <= bb_percent <= 0.8:
            score += 25
        elif 0.1 <= bb_percent <= 0.9:
            score += 20
        else:
            score += 15
        
        return min(score, 100)
    
    def _calculate_valuation_score(self, stock: Dict) -> float:
        """밸류에이션 점수 계산"""
        score = 0.0
        
        # P/E 비율 점수
        pe_ratio = stock.get('pe_ratio', 0)
        if 0 < pe_ratio < 15:
            score += 40
        elif 0 < pe_ratio < 20:
            score += 30
        elif 0 < pe_ratio < 25:
            score += 20
        else:
            score += 10
        
        # P/B 비율 점수
        pb_ratio = stock.get('price_to_book', 0)
        if 0 < pb_ratio < 1.5:
            score += 30
        elif 0 < pb_ratio < 2.0:
            score += 25
        elif 0 < pb_ratio < 3.0:
            score += 20
        else:
            score += 10
        
        # PEG 비율 점수
        peg_ratio = stock.get('peg_ratio', 0)
        if 0 < peg_ratio < 1.0:
            score += 30
        elif 0 < peg_ratio < 1.5:
            score += 25
        elif 0 < peg_ratio < 2.0:
            score += 20
        else:
            score += 10
        
        return min(score, 100)
    
    def _calculate_risk_adjusted_score(self, stock: Dict, risk_tolerance: str) -> float:
        """리스크 조정 점수 계산"""
        score = 0.0
        
        # 베타 점수
        beta = stock.get('beta', 1.0)
        if risk_tolerance == 'low':
            if beta < 0.8:
                score += 50
            elif beta < 1.0:
                score += 40
            else:
                score += 20
        elif risk_tolerance == 'medium':
            if 0.8 <= beta <= 1.2:
                score += 50
            elif 0.6 <= beta <= 1.4:
                score += 40
            else:
                score += 30
        else:  # high
            if beta > 1.2:
                score += 50
            elif beta > 1.0:
                score += 40
            else:
                score += 30
        
        # 변동성 점수
        volatility = stock.get('volatility', 0.2)
        if risk_tolerance == 'low':
            if volatility < 0.15:
                score += 50
            elif volatility < 0.20:
                score += 40
            else:
                score += 20
        elif risk_tolerance == 'medium':
            if 0.15 <= volatility <= 0.25:
                score += 50
            elif 0.10 <= volatility <= 0.30:
                score += 40
            else:
                score += 30
        else:  # high
            if volatility > 0.25:
                score += 50
            elif volatility > 0.20:
                score += 40
            else:
                score += 30
        
        return min(score, 100)
    
    def _calculate_portfolio_weights(self, stocks: List[Dict], risk_tolerance: str) -> Dict[str, float]:
        """포트폴리오 가중치 계산"""
        if not stocks:
            return {}
        
        # 점수 기반 가중치 계산
        total_score = sum(stock['total_score'] for stock in stocks)
        weights = {}
        
        for stock in stocks:
            symbol = stock['symbol']
            score = stock['total_score']
            weight = score / total_score if total_score > 0 else 1.0 / len(stocks)
            
            # 최대 가중치 제한
            max_weight = 0.3 if risk_tolerance == 'low' else 0.4
            weight = min(weight, max_weight)
            
            weights[symbol] = weight
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def recommend_sector_allocation(self, market_data: Dict[str, pd.DataFrame],
                                  risk_tolerance: str = 'medium') -> Dict:
        """섹터별 배분 추천"""
        sector_performance = {}
        
        for sector, data in market_data.items():
            if data.empty:
                continue
            
            returns = data['Close'].pct_change().dropna()
            sector_return = returns.mean() * 252  # 연간 수익률
            sector_volatility = returns.std() * np.sqrt(252)  # 연간 변동성
            sharpe_ratio = sector_return / sector_volatility if sector_volatility > 0 else 0
            
            sector_performance[sector] = {
                'return': sector_return,
                'volatility': sector_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        
        # 리스크 허용도에 따른 섹터 가중치
        if risk_tolerance == 'low':
            # 방어적 섹터 선호
            preferred_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
        elif risk_tolerance == 'high':
            # 성장 섹터 선호
            preferred_sectors = ['Technology', 'Communication Services', 'Consumer Discretionary']
        else:
            # 균형 잡힌 배분
            preferred_sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer Discretionary']
        
        # 섹터별 가중치 계산
        sector_weights = {}
        total_weight = 0
        
        for sector, performance in sector_performance.items():
            base_weight = 0.1  # 기본 가중치
            
            # 선호 섹터 보너스
            if sector in preferred_sectors:
                base_weight += 0.05
            
            # 성과 보너스
            if performance['sharpe_ratio'] > 1.0:
                base_weight += 0.05
            elif performance['sharpe_ratio'] > 0.5:
                base_weight += 0.02
            
            sector_weights[sector] = base_weight
            total_weight += base_weight
        
        # 가중치 정규화
        if total_weight > 0:
            sector_weights = {k: v/total_weight for k, v in sector_weights.items()}
        
        return {
            'sector_weights': sector_weights,
            'sector_performance': sector_performance,
            'recommended_sectors': preferred_sectors
        }
    
    def predict_stock_performance(self, stock_data: pd.DataFrame, 
                                 horizon_days: int = 30) -> Dict:
        """주식 성과 예측"""
        if stock_data.empty or len(stock_data) < 50:
            return {'prediction': 0, 'confidence': 0, 'error': 'Insufficient data'}
        
        # 특성 준비
        features = self._prepare_features(stock_data)
        target = stock_data['Close'].pct_change().shift(-horizon_days).dropna()
        
        if len(features) != len(target):
            min_len = min(len(features), len(target))
            features = features.iloc[:min_len]
            target = target.iloc[:min_len]
        
        if len(features) < 20:
            return {'prediction': 0, 'confidence': 0, 'error': 'Insufficient data for prediction'}
        
        # 모델 훈련
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # 여러 모델 앙상블
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        
        predictions = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions[name] = pred
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not predictions:
            return {'prediction': 0, 'confidence': 0, 'error': 'Model training failed'}
        
        # 앙상블 예측
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        avg_prediction = np.mean(ensemble_pred)
        
        # 신뢰도 계산
        confidence = min(0.9, max(0.1, 1.0 - np.std(ensemble_pred)))
        
        return {
            'prediction': avg_prediction,
            'confidence': confidence,
            'horizon_days': horizon_days,
            'model_predictions': {k: float(v[0]) for k, v in predictions.items()}
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """특성 준비"""
        features = pd.DataFrame()
        
        # 가격 특성
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['high_low_ratio'] = data['High'] / data['Low']
        features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # 기술적 지표
        if 'rsi' in data.columns:
            features['rsi'] = data['rsi']
        if 'macd' in data.columns:
            features['macd'] = data['macd']
        if 'sma_20' in data.columns:
            features['sma_20'] = data['sma_20']
        if 'bb_percent' in data.columns:
            features['bb_percent'] = data['bb_percent']
        
        # 이동평균 비율
        if 'Close' in data.columns and 'sma_20' in data.columns:
            features['price_sma20_ratio'] = data['Close'] / data['sma_20']
        if 'Close' in data.columns and 'sma_50' in data.columns:
            features['price_sma50_ratio'] = data['Close'] / data['sma_50']
        
        return features.fillna(0)
