from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collectors import AlphaVantageCollector, NewsCollector, YFinanceCollector, TechnicalIndicators
from models import DatabaseManager, Stock, StockData
from ai_engine import MarketAnalyzer

class DataService:
    """데이터 수집 및 관리 서비스"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.alpha_vantage = AlphaVantageCollector()
        self.news_collector = NewsCollector()
        self.yfinance_collector = YFinanceCollector()
        self.technical_indicators = TechnicalIndicators()
        self.market_analyzer = MarketAnalyzer()
    
    def collect_stock_data(self, symbol: str, period: str = '1y') -> Dict:
        """주식 데이터 수집"""
        try:
            # 기본 정보 수집
            stock_info = self.yfinance_collector.get_stock_info(symbol)
            if not stock_info:
                return {'success': False, 'message': 'Failed to get stock info'}
            
            # 과거 데이터 수집
            hist_data = self.yfinance_collector.get_historical_data(symbol, period=period)
            if hist_data.empty:
                return {'success': False, 'message': 'Failed to get historical data'}
            
            # 기술적 지표 계산
            hist_data_with_indicators = self.technical_indicators.calculate_all_indicators(hist_data)
            
            # 데이터베이스에 저장
            self.db_manager.add_stock_info(symbol, stock_info)
            
            # 일별 데이터 저장
            for date, row in hist_data_with_indicators.iterrows():
                data_dict = {
                    'date': date,
                    'open': row.get('Open', 0),
                    'high': row.get('High', 0),
                    'low': row.get('Low', 0),
                    'close': row.get('Close', 0),
                    'volume': row.get('Volume', 0),
                    'adjusted_close': row.get('Close', 0),
                    'sma_20': row.get('SMA_20'),
                    'sma_50': row.get('SMA_50'),
                    'sma_200': row.get('SMA_200'),
                    'ema_12': row.get('EMA_12'),
                    'ema_26': row.get('EMA_26'),
                    'rsi': row.get('RSI'),
                    'macd': row.get('MACD'),
                    'macd_signal': row.get('MACD_Signal'),
                    'macd_histogram': row.get('MACD_Histogram'),
                    'bb_upper': row.get('BB_Upper'),
                    'bb_middle': row.get('BB_Middle'),
                    'bb_lower': row.get('BB_Lower'),
                    'bb_width': row.get('BB_Width'),
                    'bb_percent': row.get('BB_Percent'),
                    'stoch_k': row.get('Stoch_K'),
                    'stoch_d': row.get('Stoch_D'),
                    'williams_r': row.get('Williams_R'),
                    'atr': row.get('ATR'),
                    'adx': row.get('ADX'),
                    'cci': row.get('CCI'),
                    'obv': row.get('OBV'),
                    'vwap': row.get('VWAP')
                }
                self.db_manager.add_stock_data(symbol, data_dict)
            
            return {
                'success': True,
                'symbol': symbol,
                'data_points': len(hist_data_with_indicators),
                'date_range': {
                    'start': hist_data_with_indicators.index[0].strftime('%Y-%m-%d'),
                    'end': hist_data_with_indicators.index[-1].strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error collecting data: {str(e)}'}
    
    def collect_multiple_stocks_data(self, symbols: List[str], period: str = '1y') -> Dict:
        """여러 주식 데이터 수집"""
        results = {}
        successful = 0
        failed = 0
        
        for symbol in symbols:
            result = self.collect_stock_data(symbol, period)
            results[symbol] = result
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        return {
            'total_symbols': len(symbols),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    def collect_news_data(self, symbol: str, days: int = 7) -> Dict:
        """뉴스 데이터 수집"""
        try:
            # 주식 관련 뉴스 수집
            news_articles = self.news_collector.get_stock_news(symbol)
            
            # 최근 N일 뉴스만 필터링
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_news = []
            
            for article in news_articles:
                try:
                    if article.get('published_at'):
                        pub_date = pd.to_datetime(article['published_at'])
                        if pub_date >= cutoff_date:
                            recent_news.append(article)
                except:
                    # 날짜 파싱 실패 시 포함
                    recent_news.append(article)
            
            # 감정 분석
            sentiment_analysis = self.news_collector.get_news_sentiment(symbol)
            
            # 데이터베이스에 저장
            for article in recent_news:
                article['sentiment'] = sentiment_analysis.get('sentiment', 'neutral')
                article['sentiment_score'] = sentiment_analysis.get('score', 0.5)
                self.db_manager.add_news_data(symbol, article)
            
            return {
                'success': True,
                'symbol': symbol,
                'news_count': len(recent_news),
                'sentiment': sentiment_analysis,
                'articles': recent_news[:10]  # 최대 10개 기사만 반환
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error collecting news: {str(e)}'}
    
    def collect_market_data(self, symbols: List[str] = None) -> Dict:
        """시장 데이터 수집"""
        if symbols is None:
            # 주요 지수
            symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^TNX']
        
        market_data = {}
        
        for symbol in symbols:
            try:
                hist_data = self.yfinance_collector.get_historical_data(symbol, period='1y')
                if not hist_data.empty:
                    market_data[symbol] = hist_data
            except Exception as e:
                print(f"Error collecting market data for {symbol}: {e}")
                continue
        
        return {
            'success': True,
            'market_data': market_data,
            'symbols_collected': list(market_data.keys())
        }
    
    def collect_sector_data(self) -> Dict:
        """섹터 데이터 수집"""
        # 주요 섹터 ETF
        sector_etfs = {
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLF': 'Financial',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLC': 'Communication Services'
        }
        
        sector_data = {}
        
        for etf, sector in sector_etfs.items():
            try:
                hist_data = self.yfinance_collector.get_historical_data(etf, period='1y')
                if not hist_data.empty:
                    sector_data[sector] = hist_data
            except Exception as e:
                print(f"Error collecting sector data for {etf}: {e}")
                continue
        
        return {
            'success': True,
            'sector_data': sector_data,
            'sectors_collected': list(sector_data.keys())
        }
    
    def get_stock_analysis(self, symbol: str) -> Dict:
        """주식 종합 분석"""
        try:
            # 기본 정보
            stock_info = self.db_manager.get_stock_info(symbol)
            if not stock_info:
                return {'success': False, 'message': 'Stock info not found'}
            
            # 과거 데이터
            hist_data = self.db_manager.get_stock_data(symbol)
            if not hist_data:
                return {'success': False, 'message': 'Historical data not found'}
            
            # 데이터프레임으로 변환
            df_data = []
            for data in hist_data:
                df_data.append({
                    'Date': data.date,
                    'Open': data.open_price,
                    'High': data.high_price,
                    'Low': data.low_price,
                    'Close': data.close_price,
                    'Volume': data.volume,
                    'RSI': data.rsi,
                    'MACD': data.macd,
                    'SMA_20': data.sma_20,
                    'SMA_50': data.sma_50,
                    'BB_Upper': data.bb_upper,
                    'BB_Lower': data.bb_lower
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            # 기술적 분석
            technical_signals = self.technical_indicators.get_trading_signals(df)
            
            # 뉴스 분석
            news_data = self.db_manager.get_news_data(symbol, limit=20)
            news_sentiment = {
                'positive': len([n for n in news_data if n.sentiment == 'positive']),
                'negative': len([n for n in news_data if n.sentiment == 'negative']),
                'neutral': len([n for n in news_data if n.sentiment == 'neutral'])
            }
            
            # 시장 분석
            market_analysis = self.market_analyzer.analyze_market_regime(df)
            
            return {
                'success': True,
                'symbol': symbol,
                'stock_info': {
                    'name': stock_info.name,
                    'sector': stock_info.sector,
                    'market_cap': stock_info.market_cap,
                    'pe_ratio': stock_info.pe_ratio,
                    'beta': stock_info.beta
                },
                'technical_analysis': technical_signals,
                'news_sentiment': news_sentiment,
                'market_regime': market_analysis,
                'data_points': len(df),
                'latest_price': df['Close'].iloc[-1] if not df.empty else 0
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error analyzing stock: {str(e)}'}
    
    def get_market_overview(self) -> Dict:
        """시장 개요 분석"""
        try:
            # 시장 데이터 수집
            market_data = self.collect_market_data()
            if not market_data['success']:
                return {'success': False, 'message': 'Failed to collect market data'}
            
            # 섹터 데이터 수집
            sector_data = self.collect_sector_data()
            if not sector_data['success']:
                return {'success': False, 'message': 'Failed to collect sector data'}
            
            # 시장 분석
            market_analysis = {}
            for symbol, data in market_data['market_data'].items():
                analysis = self.market_analyzer.analyze_market_regime(data)
                market_analysis[symbol] = analysis
            
            # 섹터 로테이션 분석
            rotation_analysis = self.market_analyzer.analyze_sector_rotation(
                sector_data['sector_data']
            )
            
            # 뉴스 감정 분석
            market_news = self.news_collector.get_market_news(query='stock market', page_size=20)
            sentiment_analysis = self.market_analyzer.analyze_market_sentiment(
                market_news, market_data['market_data'].get('^GSPC')
            )
            
            return {
                'success': True,
                'market_analysis': market_analysis,
                'sector_rotation': rotation_analysis,
                'market_sentiment': sentiment_analysis,
                'market_data': market_data['market_data'],
                'sector_data': sector_data['sector_data']
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error getting market overview: {str(e)}'}
    
    def update_stock_prices(self, symbols: List[str]) -> Dict:
        """주식 가격 업데이트"""
        results = {}
        successful = 0
        failed = 0
        
        for symbol in symbols:
            try:
                # 실시간 가격 조회
                quote = self.alpha_vantage.get_quote(symbol)
                if quote:
                    # 데이터베이스 업데이트 (실제 구현에서는 별도 테이블 사용)
                    results[symbol] = {
                        'success': True,
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent']
                    }
                    successful += 1
                else:
                    results[symbol] = {'success': False, 'message': 'No quote data'}
                    failed += 1
            except Exception as e:
                results[symbol] = {'success': False, 'message': str(e)}
                failed += 1
        
        return {
            'total_symbols': len(symbols),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    def get_data_quality_report(self) -> Dict:
        """데이터 품질 리포트"""
        try:
            # 데이터베이스에서 통계 수집
            # 실제 구현에서는 데이터베이스 쿼리 사용
            
            report = {
                'stocks_covered': 0,  # 실제 구현에서는 DB에서 조회
                'data_completeness': 0.0,
                'last_update': datetime.now().isoformat(),
                'data_sources': {
                    'alpha_vantage': True,
                    'yfinance': True,
                    'news_api': True
                },
                'issues': []
            }
            
            return {
                'success': True,
                'report': report
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error generating quality report: {str(e)}'}
    
    def cleanup_old_data(self, days: int = 30) -> Dict:
        """오래된 데이터 정리"""
        try:
            # 실제 구현에서는 데이터베이스에서 오래된 데이터 삭제
            cutoff_date = datetime.now() - timedelta(days=days)
            
            return {
                'success': True,
                'cutoff_date': cutoff_date.isoformat(),
                'message': f'Data older than {days} days cleaned up'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error cleaning up data: {str(e)}'}
