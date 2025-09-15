import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MarketAnalyzer:
    """시장 분석 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def analyze_market_regime(self, market_data: pd.DataFrame) -> Dict:
        """시장 체제 분석"""
        if market_data.empty:
            return {'regime': 'Unknown', 'confidence': 0.0}
        
        # 시장 수익률 계산
        market_returns = market_data['Close'].pct_change().dropna()
        
        # 변동성 계산 (20일 롤링)
        volatility = market_returns.rolling(20).std()
        
        # 트렌드 분석 (50일 이동평균)
        sma_50 = market_data['Close'].rolling(50).mean()
        current_price = market_data['Close'].iloc[-1]
        sma_50_current = sma_50.iloc[-1]
        
        # 시장 체제 분류
        regime = self._classify_market_regime(
            market_returns.iloc[-1] if not market_returns.empty else 0,
            volatility.iloc[-1] if not volatility.empty else 0,
            current_price / sma_50_current if sma_50_current > 0 else 1
        )
        
        # 신뢰도 계산
        confidence = self._calculate_regime_confidence(volatility, market_returns)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility': volatility.iloc[-1] if not volatility.empty else 0,
            'trend_strength': (current_price / sma_50_current - 1) if sma_50_current > 0 else 0
        }
    
    def _classify_market_regime(self, return_value: float, volatility: float, trend_ratio: float) -> str:
        """시장 체제 분류"""
        if volatility > 0.03:  # 높은 변동성
            if return_value > 0.02:
                return 'Bull Market (High Vol)'
            elif return_value < -0.02:
                return 'Bear Market (High Vol)'
            else:
                return 'Volatile Market'
        else:  # 낮은 변동성
            if trend_ratio > 1.05:
                return 'Bull Market (Low Vol)'
            elif trend_ratio < 0.95:
                return 'Bear Market (Low Vol)'
            else:
                return 'Sideways Market'
    
    def _calculate_regime_confidence(self, volatility: pd.Series, returns: pd.Series) -> float:
        """체제 신뢰도 계산"""
        if volatility.empty or returns.empty:
            return 0.0
        
        # 변동성의 일관성
        vol_consistency = 1.0 - (volatility.std() / volatility.mean()) if volatility.mean() > 0 else 0.0
        
        # 수익률의 방향성
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        direction_consistency = abs(positive_days / total_days - 0.5) * 2 if total_days > 0 else 0.0
        
        return min(0.9, (vol_consistency + direction_consistency) / 2)
    
    def analyze_sector_rotation(self, sector_data: Dict[str, pd.DataFrame]) -> Dict:
        """섹터 로테이션 분석"""
        if not sector_data:
            return {'rotation': 'Unknown', 'leading_sectors': [], 'lagging_sectors': []}
        
        sector_performance = {}
        
        for sector, data in sector_data.items():
            if data.empty:
                continue
            
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 20:
                continue
            
            # 최근 20일 성과
            recent_performance = returns.tail(20).mean() * 252
            
            # 상대적 강도 (RSI 스타일)
            gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            rs = gains / losses if losses > 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            sector_performance[sector] = {
                'performance': recent_performance,
                'rsi': rsi,
                'volatility': returns.std() * np.sqrt(252)
            }
        
        if not sector_performance:
            return {'rotation': 'Unknown', 'leading_sectors': [], 'lagging_sectors': []}
        
        # 성과 기준 정렬
        sorted_sectors = sorted(sector_performance.items(), 
                              key=lambda x: x[1]['performance'], reverse=True)
        
        # 상위/하위 섹터
        n_sectors = len(sorted_sectors)
        top_n = max(1, n_sectors // 3)
        
        leading_sectors = [sector for sector, _ in sorted_sectors[:top_n]]
        lagging_sectors = [sector for sector, _ in sorted_sectors[-top_n:]]
        
        # 로테이션 패턴 분석
        rotation = self._analyze_rotation_pattern(sector_performance)
        
        return {
            'rotation': rotation,
            'leading_sectors': leading_sectors,
            'lagging_sectors': lagging_sectors,
            'sector_performance': sector_performance
        }
    
    def _analyze_rotation_pattern(self, sector_performance: Dict) -> str:
        """로테이션 패턴 분석"""
        performances = [data['performance'] for data in sector_performance.values()]
        
        if not performances:
            return 'Unknown'
        
        # 성과 분산
        performance_std = np.std(performances)
        performance_mean = np.mean(performances)
        
        # 변동성 분석
        volatilities = [data['volatility'] for data in sector_performance.values()]
        avg_volatility = np.mean(volatilities)
        
        if performance_std > 0.2:  # 높은 분산
            if performance_mean > 0.1:
                return 'Growth Rotation'
            else:
                return 'Defensive Rotation'
        elif avg_volatility > 0.25:  # 높은 변동성
            return 'Volatile Rotation'
        else:
            return 'Stable Market'
    
    def analyze_market_sentiment(self, news_data: List[Dict], 
                               price_data: pd.DataFrame = None) -> Dict:
        """시장 심리 분석"""
        if not news_data:
            return {'sentiment': 'Neutral', 'confidence': 0.0}
        
        # 뉴스 감정 분석
        sentiment_scores = []
        for news in news_data:
            sentiment = news.get('sentiment', 'neutral')
            score = news.get('sentiment_score', 0.5)
            
            if sentiment == 'positive':
                sentiment_scores.append(score)
            elif sentiment == 'negative':
                sentiment_scores.append(-score)
            else:
                sentiment_scores.append(0)
        
        avg_sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # 가격 모멘텀 분석
        price_momentum = 0
        if price_data is not None and not price_data.empty:
            returns = price_data['Close'].pct_change().dropna()
            if len(returns) >= 5:
                price_momentum = returns.tail(5).mean()
        
        # 종합 심리 점수
        combined_score = (avg_sentiment_score + price_momentum * 10) / 2
        combined_score = np.clip(combined_score, -1, 1)
        
        # 심리 분류
        if combined_score > 0.3:
            sentiment = 'Bullish'
        elif combined_score < -0.3:
            sentiment = 'Bearish'
        else:
            sentiment = 'Neutral'
        
        # 신뢰도 계산
        confidence = min(0.9, len(sentiment_scores) / 20)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'sentiment_score': combined_score,
            'news_sentiment': avg_sentiment_score,
            'price_momentum': price_momentum,
            'news_count': len(news_data)
        }
    
    def analyze_volatility_regime(self, price_data: pd.DataFrame) -> Dict:
        """변동성 체제 분석"""
        if price_data.empty:
            return {'regime': 'Unknown', 'volatility': 0.0}
        
        returns = price_data['Close'].pct_change().dropna()
        
        # 단기 변동성 (5일)
        short_vol = returns.tail(5).std() * np.sqrt(252)
        
        # 중기 변동성 (20일)
        medium_vol = returns.tail(20).std() * np.sqrt(252)
        
        # 장기 변동성 (60일)
        long_vol = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else medium_vol
        
        # 변동성 체제 분류
        if short_vol > long_vol * 1.5:
            regime = 'High Volatility'
        elif short_vol < long_vol * 0.7:
            regime = 'Low Volatility'
        else:
            regime = 'Normal Volatility'
        
        # 변동성 클러스터링 분석
        vol_clustering = self._analyze_volatility_clustering(returns)
        
        return {
            'regime': regime,
            'short_volatility': short_vol,
            'medium_volatility': medium_vol,
            'long_volatility': long_vol,
            'volatility_ratio': short_vol / long_vol if long_vol > 0 else 1,
            'clustering': vol_clustering
        }
    
    def _analyze_volatility_clustering(self, returns: pd.Series) -> Dict:
        """변동성 클러스터링 분석"""
        if len(returns) < 20:
            return {'clustering': 'Unknown', 'persistence': 0.0}
        
        # ARCH 효과 검정 (간단한 버전)
        squared_returns = returns**2
        autocorr = squared_returns.autocorr(lag=1)
        
        # 클러스터링 정도
        if autocorr > 0.1:
            clustering = 'High'
        elif autocorr > 0.05:
            clustering = 'Medium'
        else:
            clustering = 'Low'
        
        return {
            'clustering': clustering,
            'persistence': autocorr,
            'autocorrelation': squared_returns.autocorr(lag=5)
        }
    
    def analyze_correlation_structure(self, returns_data: pd.DataFrame) -> Dict:
        """상관관계 구조 분석"""
        if returns_data.empty:
            return {'structure': 'Unknown', 'diversification': 0.0}
        
        # 상관관계 행렬
        corr_matrix = returns_data.corr()
        
        # 평균 상관관계
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_correlation = upper_triangle.stack().mean()
        
        # 다각화 효과
        diversification_ratio = 1 - avg_correlation
        
        # 상관관계 구조 분석
        if avg_correlation > 0.7:
            structure = 'High Correlation'
        elif avg_correlation > 0.4:
            structure = 'Medium Correlation'
        else:
            structure = 'Low Correlation'
        
        # 주요 상관관계 쌍
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'pair': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr_value
                    })
        
        return {
            'structure': structure,
            'diversification': diversification_ratio,
            'avg_correlation': avg_correlation,
            'high_corr_pairs': high_corr_pairs,
            'correlation_matrix': corr_matrix
        }
    
    def analyze_market_stress(self, market_data: pd.DataFrame,
                            vix_data: pd.Series = None) -> Dict:
        """시장 스트레스 분석"""
        if market_data.empty:
            return {'stress_level': 'Unknown', 'indicators': {}}
        
        stress_indicators = {}
        
        # 가격 하락률
        returns = market_data['Close'].pct_change().dropna()
        recent_return = returns.tail(5).mean() * 252
        stress_indicators['price_decline'] = recent_return
        
        # 변동성 스파이크
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_spike = (volatility.iloc[-1] / volatility.mean() - 1) if not volatility.empty else 0
        stress_indicators['volatility_spike'] = vol_spike
        
        # 거래량 급증
        volume = market_data['Volume']
        volume_spike = (volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] - 1) if not volume.empty else 0
        stress_indicators['volume_spike'] = volume_spike
        
        # VIX 지수 (있는 경우)
        if vix_data is not None and not vix_data.empty:
            vix_level = vix_data.iloc[-1]
            stress_indicators['vix_level'] = vix_level
        else:
            vix_level = 20  # 기본값
            stress_indicators['vix_level'] = vix_level
        
        # 스트레스 점수 계산
        stress_score = 0
        
        if recent_return < -0.1:  # 10% 이상 하락
            stress_score += 3
        elif recent_return < -0.05:  # 5% 이상 하락
            stress_score += 2
        elif recent_return < 0:
            stress_score += 1
        
        if vol_spike > 0.5:  # 변동성 50% 이상 증가
            stress_score += 3
        elif vol_spike > 0.2:  # 변동성 20% 이상 증가
            stress_score += 2
        elif vol_spike > 0.1:
            stress_score += 1
        
        if volume_spike > 1.0:  # 거래량 100% 이상 증가
            stress_score += 2
        elif volume_spike > 0.5:  # 거래량 50% 이상 증가
            stress_score += 1
        
        if vix_level > 30:
            stress_score += 3
        elif vix_level > 25:
            stress_score += 2
        elif vix_level > 20:
            stress_score += 1
        
        # 스트레스 레벨 분류
        if stress_score >= 8:
            stress_level = 'Extreme'
        elif stress_score >= 6:
            stress_level = 'High'
        elif stress_score >= 4:
            stress_level = 'Medium'
        elif stress_score >= 2:
            stress_level = 'Low'
        else:
            stress_level = 'Normal'
        
        return {
            'stress_level': stress_level,
            'stress_score': stress_score,
            'indicators': stress_indicators
        }
    
    def generate_market_outlook(self, market_data: pd.DataFrame,
                              sector_data: Dict[str, pd.DataFrame] = None,
                              news_data: List[Dict] = None) -> Dict:
        """시장 전망 생성"""
        outlook = {
            'overall_outlook': 'Neutral',
            'confidence': 0.0,
            'key_factors': [],
            'recommendations': []
        }
        
        # 시장 체제 분석
        regime_analysis = self.analyze_market_regime(market_data)
        outlook['regime'] = regime_analysis['regime']
        outlook['confidence'] += regime_analysis['confidence'] * 0.3
        
        # 변동성 분석
        vol_analysis = self.analyze_volatility_regime(market_data)
        outlook['volatility_regime'] = vol_analysis['regime']
        
        # 섹터 로테이션 분석
        if sector_data:
            rotation_analysis = self.analyze_sector_rotation(sector_data)
            outlook['sector_rotation'] = rotation_analysis['rotation']
            outlook['confidence'] += 0.2
        
        # 시장 심리 분석
        if news_data:
            sentiment_analysis = self.analyze_market_sentiment(news_data, market_data)
            outlook['sentiment'] = sentiment_analysis['sentiment']
            outlook['confidence'] += sentiment_analysis['confidence'] * 0.2
        
        # 스트레스 분석
        stress_analysis = self.analyze_market_stress(market_data)
        outlook['stress_level'] = stress_analysis['stress_level']
        
        # 종합 전망 결정
        positive_factors = 0
        negative_factors = 0
        
        if regime_analysis['regime'].startswith('Bull'):
            positive_factors += 1
        elif regime_analysis['regime'].startswith('Bear'):
            negative_factors += 1
        
        if vol_analysis['regime'] == 'Low Volatility':
            positive_factors += 1
        elif vol_analysis['regime'] == 'High Volatility':
            negative_factors += 1
        
        if news_data:
            if sentiment_analysis['sentiment'] == 'Bullish':
                positive_factors += 1
            elif sentiment_analysis['sentiment'] == 'Bearish':
                negative_factors += 1
        
        if stress_analysis['stress_level'] in ['Normal', 'Low']:
            positive_factors += 1
        elif stress_analysis['stress_level'] in ['High', 'Extreme']:
            negative_factors += 1
        
        # 전망 결정
        if positive_factors > negative_factors + 1:
            outlook['overall_outlook'] = 'Bullish'
        elif negative_factors > positive_factors + 1:
            outlook['overall_outlook'] = 'Bearish'
        else:
            outlook['overall_outlook'] = 'Neutral'
        
        # 권장사항 생성
        if outlook['overall_outlook'] == 'Bullish':
            outlook['recommendations'].append("Consider increasing equity exposure")
        elif outlook['overall_outlook'] == 'Bearish':
            outlook['recommendations'].append("Consider reducing equity exposure and adding defensive positions")
        
        if vol_analysis['regime'] == 'High Volatility':
            outlook['recommendations'].append("High volatility environment - consider hedging strategies")
        
        if stress_analysis['stress_level'] in ['High', 'Extreme']:
            outlook['recommendations'].append("Market stress detected - monitor positions closely")
        
        return outlook
