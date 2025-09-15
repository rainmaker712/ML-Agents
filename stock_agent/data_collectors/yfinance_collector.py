import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class YFinanceCollector:
    """Yahoo Finance를 통한 주식 데이터 수집"""
    
    def __init__(self):
        pass
    
    def get_stock_info(self, symbol: str) -> Dict:
        """주식 기본 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, period: str = '1y', 
                          interval: str = '1d') -> pd.DataFrame:
        """과거 주식 데이터 조회"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """배당 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            return dividends
        except Exception as e:
            print(f"Error getting dividends for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_splits(self, symbol: str) -> pd.DataFrame:
        """주식 분할 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            return splits
        except Exception as e:
            print(f"Error getting splits for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_earnings(self, symbol: str) -> Dict:
        """실적 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings
            quarterly_earnings = ticker.quarterly_earnings
            
            return {
                'annual_earnings': earnings.to_dict() if not earnings.empty else {},
                'quarterly_earnings': quarterly_earnings.to_dict() if not quarterly_earnings.empty else {}
            }
        except Exception as e:
            print(f"Error getting earnings for {symbol}: {e}")
            return {}
    
    def get_financials(self, symbol: str) -> Dict:
        """재무제표 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            financials = ticker.financials
            quarterly_financials = ticker.quarterly_financials
            balance_sheet = ticker.balance_sheet
            quarterly_balance_sheet = ticker.quarterly_balance_sheet
            cashflow = ticker.cashflow
            quarterly_cashflow = ticker.quarterly_cashflow
            
            return {
                'annual_financials': financials.to_dict() if not financials.empty else {},
                'quarterly_financials': quarterly_financials.to_dict() if not quarterly_financials.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'quarterly_balance_sheet': quarterly_balance_sheet.to_dict() if not quarterly_balance_sheet.empty else {},
                'cashflow': cashflow.to_dict() if not cashflow.empty else {},
                'quarterly_cashflow': quarterly_cashflow.to_dict() if not quarterly_cashflow.empty else {}
            }
        except Exception as e:
            print(f"Error getting financials for {symbol}: {e}")
            return {}
    
    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """애널리스트 추천 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_analyst_info(self, symbol: str) -> Dict:
        """애널리스트 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'target_mean_price': info.get('targetMeanPrice', 0),
                'target_high_price': info.get('targetHighPrice', 0),
                'target_low_price': info.get('targetLowPrice', 0),
                'recommendation_mean': info.get('recommendationMean', 0),
                'recommendation_key': info.get('recommendationKey', ''),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0)
            }
        except Exception as e:
            print(f"Error getting analyst info for {symbol}: {e}")
            return {}
    
    def get_major_holders(self, symbol: str) -> pd.DataFrame:
        """주요 주주 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            major_holders = ticker.major_holders
            return major_holders
        except Exception as e:
            print(f"Error getting major holders for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_institutional_holders(self, symbol: str) -> pd.DataFrame:
        """기관 투자자 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            institutional_holders = ticker.institutional_holders
            return institutional_holders
        except Exception as e:
            print(f"Error getting institutional holders for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """여러 주식의 데이터를 한 번에 조회"""
        try:
            tickers = yf.Tickers(' '.join(symbols))
            data = {}
            for symbol in symbols:
                data[symbol] = tickers.tickers[symbol].history(period=period)
            return data
        except Exception as e:
            print(f"Error getting multiple stocks data: {e}")
            return {}
    
    def get_sector_performance(self) -> pd.DataFrame:
        """섹터별 성과 조회"""
        try:
            # 주요 섹터 ETF들
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
            
            performance_data = []
            for etf, sector in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    info = ticker.info
                    hist = ticker.history(period='1y')
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        year_ago_price = hist['Close'].iloc[0]
                        ytd_return = ((current_price - year_ago_price) / year_ago_price) * 100
                        
                        performance_data.append({
                            'sector': sector,
                            'etf': etf,
                            'current_price': current_price,
                            'ytd_return': ytd_return,
                            'volume': hist['Volume'].iloc[-1]
                        })
                except Exception as e:
                    print(f"Error getting data for {etf}: {e}")
                    continue
            
            return pd.DataFrame(performance_data)
        except Exception as e:
            print(f"Error getting sector performance: {e}")
            return pd.DataFrame()
