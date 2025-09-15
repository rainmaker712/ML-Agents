import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from config import Config

class AlphaVantageCollector:
    """Alpha Vantage API를 통한 주식 데이터 수집"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.ALPHA_VANTAGE_API_KEY
        self.base_url = Config.ALPHA_VANTAGE_BASE_URL
        self.rate_limit_delay = 12  # API rate limit: 5 calls per minute
        
    def get_quote(self, symbol: str) -> Dict:
        """실시간 주식 가격 정보 조회"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Global Quote' in data:
            quote = data['Global Quote']
            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': float(quote.get('10. change percent', 0).replace('%', '')),
                'volume': int(quote.get('06. volume', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'open': float(quote.get('02. open', 0)),
                'previous_close': float(quote.get('08. previous close', 0))
            }
        return {}
    
    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """일별 주식 데이터 조회"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            return df.sort_index()
        return pd.DataFrame()
    
    def get_company_overview(self, symbol: str) -> Dict:
        """회사 개요 정보 조회"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Symbol' in data:
            return {
                'symbol': data.get('Symbol'),
                'name': data.get('Name'),
                'sector': data.get('Sector'),
                'industry': data.get('Industry'),
                'market_cap': data.get('MarketCapitalization'),
                'pe_ratio': data.get('PERatio'),
                'dividend_yield': data.get('DividendYield'),
                'beta': data.get('Beta'),
                'description': data.get('Description')
            }
        return {}
    
    def get_earnings(self, symbol: str) -> Dict:
        """실적 정보 조회"""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'annualEarnings' in data:
            return {
                'annual_earnings': data.get('annualEarnings', []),
                'quarterly_earnings': data.get('quarterlyEarnings', [])
            }
        return {}
    
    def get_technical_indicators(self, symbol: str, function: str, interval: str = 'daily', 
                               time_period: int = 20) -> pd.DataFrame:
        """기술적 지표 조회"""
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # API 응답에서 데이터 추출
        key = list(data.keys())[1] if len(data.keys()) > 1 else None
        if key and isinstance(data[key], dict):
            df = pd.DataFrame(data[key]).T
            df.index = pd.to_datetime(df.index)
            return df.astype(float)
        return pd.DataFrame()
    
    def get_sma(self, symbol: str, time_period: int = 20) -> pd.DataFrame:
        """단순 이동평균 조회"""
        return self.get_technical_indicators(symbol, 'SMA', time_period=time_period)
    
    def get_ema(self, symbol: str, time_period: int = 20) -> pd.DataFrame:
        """지수 이동평균 조회"""
        return self.get_technical_indicators(symbol, 'EMA', time_period=time_period)
    
    def get_rsi(self, symbol: str, time_period: int = 14) -> pd.DataFrame:
        """RSI 조회"""
        return self.get_technical_indicators(symbol, 'RSI', time_period=time_period)
    
    def get_macd(self, symbol: str) -> pd.DataFrame:
        """MACD 조회"""
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': 'daily',
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Technical Analysis: MACD' in data:
            df = pd.DataFrame(data['Technical Analysis: MACD']).T
            df.index = pd.to_datetime(df.index)
            return df.astype(float)
        return pd.DataFrame()
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """여러 주식의 실시간 가격 정보 조회"""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_quote(symbol)
            time.sleep(self.rate_limit_delay)  # Rate limiting
        return quotes
