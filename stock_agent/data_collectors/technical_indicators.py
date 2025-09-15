import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self):
        pass
    
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """단순 이동평균 계산"""
        return data.rolling(window=window).mean()
    
    def calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """지수 이동평균 계산"""
        return data.ewm(span=window).mean()
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """RSI 계산"""
        return ta.momentum.RSIIndicator(data, window=window).rsi()
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD 계산"""
        macd_indicator = ta.trend.MACD(data, window_fast=fast, window_slow=slow, window_sign=signal)
        return {
            'macd': macd_indicator.macd(),
            'macd_signal': macd_indicator.macd_signal(),
            'macd_histogram': macd_indicator.macd_diff()
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        bb_indicator = ta.volatility.BollingerBands(data, window=window, window_dev=std_dev)
        return {
            'upper_band': bb_indicator.bollinger_hband(),
            'middle_band': bb_indicator.bollinger_mavg(),
            'lower_band': bb_indicator.bollinger_lband(),
            'bb_width': bb_indicator.bollinger_wband(),
            'bb_percent': bb_indicator.bollinger_pband()
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """스토캐스틱 계산"""
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, 
                                                          window=k_window, smooth_window=d_window)
        return {
            'stoch_k': stoch_indicator.stoch(),
            'stoch_d': stoch_indicator.stoch_signal()
        }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           window: int = 14) -> pd.Series:
        """Williams %R 계산"""
        return ta.momentum.WilliamsRIndicator(high, low, close, lbp=window).williams_r()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     window: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        return ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     window: int = 14) -> Dict[str, pd.Series]:
        """ADX 계산"""
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=window)
        return {
            'adx': adx_indicator.adx(),
            'adx_pos': adx_indicator.adx_pos(),
            'adx_neg': adx_indicator.adx_neg()
        }
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     window: int = 20) -> pd.Series:
        """CCI (Commodity Channel Index) 계산"""
        return ta.trend.CCIIndicator(high, low, close, window=window).cci()
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV (On Balance Volume) 계산"""
        return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series) -> pd.Series:
        """VWAP (Volume Weighted Average Price) 계산"""
        # VWAP = (High + Low + Close) / 3 * Volume의 누적합 / Volume의 누적합
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series,
                          window1: int = 9, window2: int = 26, window3: int = 52) -> Dict[str, pd.Series]:
        """일목균형표 계산"""
        ichimoku_indicator = ta.trend.IchimokuIndicator(high, low, 
                                                       window1=window1, window2=window2, window3=window3)
        return {
            'conversion_line': ichimoku_indicator.ichimoku_conversion_line(),
            'base_line': ichimoku_indicator.ichimoku_base_line(),
            'span_a': ichimoku_indicator.ichimoku_a(),
            'span_b': ichimoku_indicator.ichimoku_b()
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 계산"""
        if df.empty or 'Close' not in df.columns:
            return df
        
        result_df = df.copy()
        
        # 기본 이동평균
        result_df['SMA_20'] = self.calculate_sma(df['Close'], 20)
        result_df['SMA_50'] = self.calculate_sma(df['Close'], 50)
        result_df['SMA_200'] = self.calculate_sma(df['Close'], 200)
        result_df['EMA_12'] = self.calculate_ema(df['Close'], 12)
        result_df['EMA_26'] = self.calculate_ema(df['Close'], 26)
        
        # RSI
        result_df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        macd_data = self.calculate_macd(df['Close'])
        result_df['MACD'] = macd_data['macd']
        result_df['MACD_Signal'] = macd_data['macd_signal']
        result_df['MACD_Histogram'] = macd_data['macd_histogram']
        
        # 볼린저 밴드
        if 'High' in df.columns and 'Low' in df.columns:
            bb_data = self.calculate_bollinger_bands(df['Close'])
            result_df['BB_Upper'] = bb_data['upper_band']
            result_df['BB_Middle'] = bb_data['middle_band']
            result_df['BB_Lower'] = bb_data['lower_band']
            result_df['BB_Width'] = bb_data['bb_width']
            result_df['BB_Percent'] = bb_data['bb_percent']
            
            # 스토캐스틱
            stoch_data = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
            result_df['Stoch_K'] = stoch_data['stoch_k']
            result_df['Stoch_D'] = stoch_data['stoch_d']
            
            # Williams %R
            result_df['Williams_R'] = self.calculate_williams_r(df['High'], df['Low'], df['Close'])
            
            # ATR
            result_df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
            
            # ADX
            adx_data = self.calculate_adx(df['High'], df['Low'], df['Close'])
            result_df['ADX'] = adx_data['adx']
            result_df['ADX_Pos'] = adx_data['adx_pos']
            result_df['ADX_Neg'] = adx_data['adx_neg']
            
            # CCI
            result_df['CCI'] = self.calculate_cci(df['High'], df['Low'], df['Close'])
            
            # 일목균형표
            ichimoku_data = self.calculate_ichimoku(df['High'], df['Low'], df['Close'])
            result_df['Ichimoku_Conversion'] = ichimoku_data['conversion_line']
            result_df['Ichimoku_Base'] = ichimoku_data['base_line']
            result_df['Ichimoku_Span_A'] = ichimoku_data['span_a']
            result_df['Ichimoku_Span_B'] = ichimoku_data['span_b']
        
        # 볼륨 지표
        if 'Volume' in df.columns:
            result_df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
            if 'High' in df.columns and 'Low' in df.columns:
                result_df['VWAP'] = self.calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return result_df
    
    def get_trading_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """거래 신호 생성"""
        if df.empty or len(df) < 50:
            return {'signal': 'neutral', 'strength': 'weak'}
        
        latest = df.iloc[-1]
        signals = []
        
        # RSI 신호
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            if latest['RSI'] > 70:
                signals.append('sell')
            elif latest['RSI'] < 30:
                signals.append('buy')
            else:
                signals.append('neutral')
        
        # MACD 신호
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal']:
                    signals.append('buy')
                else:
                    signals.append('sell')
        
        # 이동평균 신호
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
                if latest['SMA_20'] > latest['SMA_50']:
                    signals.append('buy')
                else:
                    signals.append('sell')
        
        # 볼린저 밴드 신호
        if 'BB_Percent' in df.columns and not pd.isna(latest['BB_Percent']):
            if latest['BB_Percent'] > 1:
                signals.append('sell')
            elif latest['BB_Percent'] < 0:
                signals.append('buy')
            else:
                signals.append('neutral')
        
        # 신호 집계
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count > sell_count:
            signal = 'buy'
            strength = 'strong' if buy_count >= 3 else 'medium'
        elif sell_count > buy_count:
            signal = 'sell'
            strength = 'strong' if sell_count >= 3 else 'medium'
        else:
            signal = 'neutral'
            strength = 'weak'
        
        return {
            'signal': signal,
            'strength': strength,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'total_signals': len(signals)
        }
