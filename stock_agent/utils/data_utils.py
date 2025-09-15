import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataUtils:
    """데이터 유틸리티 클래스"""
    
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   fill_missing: str = 'forward',
                   remove_outliers: bool = False,
                   outlier_method: str = 'iqr') -> pd.DataFrame:
        """데이터 정리"""
        cleaned_df = df.copy()
        
        # 중복 제거
        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
        
        # 결측값 처리
        if fill_missing == 'forward':
            cleaned_df = cleaned_df.fillna(method='ffill')
        elif fill_missing == 'backward':
            cleaned_df = cleaned_df.fillna(method='bfill')
        elif fill_missing == 'interpolate':
            cleaned_df = cleaned_df.interpolate()
        elif fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
        
        # 이상치 제거
        if remove_outliers:
            cleaned_df = DataUtils._remove_outliers(cleaned_df, outlier_method)
        
        return cleaned_df
    
    @staticmethod
    def _remove_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """이상치 제거"""
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df >= lower_bound) & (df <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df - df.mean()) / df.std())
            return df[z_scores < 3]
        else:
            return df
    
    @staticmethod
    def resample_data(df: pd.DataFrame, 
                     freq: str = 'D',
                     method: str = 'last') -> pd.DataFrame:
        """데이터 리샘플링"""
        if df.empty:
            return df
        
        if method == 'last':
            return df.resample(freq).last()
        elif method == 'first':
            return df.resample(freq).first()
        elif method == 'mean':
            return df.resample(freq).mean()
        elif method == 'sum':
            return df.resample(freq).sum()
        else:
            return df.resample(freq).last()
    
    @staticmethod
    def align_data(data_dict: Dict[str, pd.DataFrame],
                  method: str = 'inner') -> Dict[str, pd.DataFrame]:
        """여러 데이터프레임 정렬"""
        if not data_dict:
            return {}
        
        # 공통 인덱스 찾기
        common_index = None
        for df in data_dict.values():
            if common_index is None:
                common_index = df.index
            else:
                if method == 'inner':
                    common_index = common_index.intersection(df.index)
                elif method == 'outer':
                    common_index = common_index.union(df.index)
        
        # 데이터 정렬
        aligned_data = {}
        for symbol, df in data_dict.items():
            aligned_data[symbol] = df.reindex(common_index)
        
        return aligned_data
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, 
                         price_column: str = 'Close',
                         method: str = 'simple') -> pd.Series:
        """수익률 계산"""
        if price_column not in df.columns:
            raise ValueError(f"Column '{price_column}' not found in DataFrame")
        
        prices = df[price_column]
        
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, 
                           window: int = 20,
                           annualized: bool = True) -> pd.Series:
        """변동성 계산"""
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def calculate_correlation_matrix(data_dict: Dict[str, pd.DataFrame],
                                   price_column: str = 'Close') -> pd.DataFrame:
        """상관관계 행렬 계산"""
        returns_data = {}
        
        for symbol, df in data_dict.items():
            if price_column in df.columns:
                returns = DataUtils.calculate_returns(df, price_column)
                returns_data[symbol] = returns
        
        if not returns_data:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()
    
    @staticmethod
    def calculate_rolling_statistics(df: pd.DataFrame,
                                   column: str,
                                   window: int = 20,
                                   statistics: List[str] = None) -> pd.DataFrame:
        """롤링 통계 계산"""
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max']
        
        result_df = df[[column]].copy()
        
        for stat in statistics:
            if stat == 'mean':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).mean()
            elif stat == 'std':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).std()
            elif stat == 'min':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).min()
            elif stat == 'max':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).max()
            elif stat == 'median':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).median()
            elif stat == 'skew':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).skew()
            elif stat == 'kurt':
                result_df[f'{column}_{stat}_{window}'] = df[column].rolling(window).kurt()
        
        return result_df
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame,
                        column: str,
                        method: str = 'iqr',
                        threshold: float = 1.5) -> pd.Series:
        """이상치 탐지"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        data = df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame,
                                price_column: str = 'Close',
                                windows: List[int] = None) -> pd.DataFrame:
        """이동평균 계산"""
        if windows is None:
            windows = [5, 10, 20, 50, 200]
        
        if price_column not in df.columns:
            raise ValueError(f"Column '{price_column}' not found in DataFrame")
        
        result_df = df[[price_column]].copy()
        
        for window in windows:
            result_df[f'MA_{window}'] = df[price_column].rolling(window).mean()
            result_df[f'EMA_{window}'] = df[price_column].ewm(span=window).mean()
        
        return result_df
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        result_df = df.copy()
        
        if 'Close' not in df.columns:
            return result_df
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result_df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        result_df['MACD'] = ema_12 - ema_26
        result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9).mean()
        result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
        
        # 볼린저 밴드
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        result_df['BB_Upper'] = sma_20 + (std_20 * 2)
        result_df['BB_Lower'] = sma_20 - (std_20 * 2)
        result_df['BB_Middle'] = sma_20
        
        return result_df
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
        """성과 지표 계산"""
        if returns.empty:
            return {}
        
        # 기본 통계
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 리스크 지표
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR (95%)
        var_95 = np.percentile(returns, 5)
        
        # CVaR (95%)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    @staticmethod
    def create_benchmark_data(start_date: str,
                             end_date: str,
                             symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """벤치마크 데이터 생성"""
        if symbols is None:
            symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        
        # 실제 구현에서는 API를 통해 데이터를 가져옴
        # 여기서는 샘플 데이터 생성
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        benchmark_data = {}
        for symbol in symbols:
            # 랜덤 가격 데이터 생성 (실제 구현에서는 실제 데이터 사용)
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(date_range)))
            
            df = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.01, len(date_range))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.02, len(date_range)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(date_range)))),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(date_range))
            }, index=date_range)
            
            benchmark_data[symbol] = df
        
        return benchmark_data
    
    @staticmethod
    def validate_data(df: pd.DataFrame,
                     required_columns: List[str] = None,
                     min_rows: int = 1) -> Dict[str, Any]:
        """데이터 유효성 검사"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 기본 검사
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("DataFrame is empty")
            return validation_result
        
        if len(df) < min_rows:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"DataFrame has fewer than {min_rows} rows")
        
        # 필수 컬럼 검사
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
        
        # 데이터 타입 검사
        for col in df.columns:
            if df[col].dtype == 'object':
                # 문자열 컬럼에서 숫자로 변환 가능한지 확인
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    validation_result['warnings'].append(f"Column '{col}' contains non-numeric data")
        
        # 결측값 검사
        missing_data = df.isnull().sum()
        if missing_data.any():
            validation_result['warnings'].append(f"Missing data found: {missing_data[missing_data > 0].to_dict()}")
        
        # 중복 검사
        if df.duplicated().any():
            validation_result['warnings'].append("Duplicate rows found")
        
        return validation_result
    
    @staticmethod
    def export_data(df: pd.DataFrame,
                   file_path: str,
                   format: str = 'csv',
                   **kwargs) -> bool:
        """데이터 내보내기"""
        try:
            if format.lower() == 'csv':
                df.to_csv(file_path, **kwargs)
            elif format.lower() == 'excel':
                df.to_excel(file_path, **kwargs)
            elif format.lower() == 'json':
                df.to_json(file_path, **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    @staticmethod
    def load_data(file_path: str,
                 format: str = 'csv',
                 **kwargs) -> pd.DataFrame:
        """데이터 불러오기"""
        try:
            if format.lower() == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif format.lower() == 'excel':
                return pd.read_excel(file_path, **kwargs)
            elif format.lower() == 'json':
                return pd.read_json(file_path, **kwargs)
            elif format.lower() == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
