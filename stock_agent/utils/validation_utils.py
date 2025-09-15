from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date
import re

class ValidationUtils:
    """유효성 검사 유틸리티 클래스"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """주식 심볼 유효성 검사"""
        if not isinstance(symbol, str):
            return False
        
        # 기본 길이 검사 (1-10자)
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # 영문자, 숫자, 점(.), 하이픈(-)만 허용
        if not re.match(r'^[A-Za-z0-9.\-]+$', symbol):
            return False
        
        return True
    
    @staticmethod
    def validate_date(date_input: Union[str, datetime, date]) -> bool:
        """날짜 유효성 검사"""
        try:
            if isinstance(date_input, str):
                pd.to_datetime(date_input)
            elif isinstance(date_input, (datetime, date)):
                pass
            else:
                return False
            return True
        except:
            return False
    
    @staticmethod
    def validate_numeric(value: Any, min_value: float = None, 
                        max_value: float = None) -> bool:
        """숫자 유효성 검사"""
        try:
            num_value = float(value)
            
            if min_value is not None and num_value < min_value:
                return False
            
            if max_value is not None and num_value > max_value:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_percentage(value: Any) -> bool:
        """퍼센트 값 유효성 검사 (0-100)"""
        return ValidationUtils.validate_numeric(value, 0, 100)
    
    @staticmethod
    def validate_weight(value: Any) -> bool:
        """가중치 유효성 검사 (0-1)"""
        return ValidationUtils.validate_numeric(value, 0, 1)
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> Tuple[bool, str]:
        """포트폴리오 가중치 유효성 검사"""
        if not isinstance(weights, dict):
            return False, "Weights must be a dictionary"
        
        if not weights:
            return False, "Weights cannot be empty"
        
        # 개별 가중치 검사
        for symbol, weight in weights.items():
            if not ValidationUtils.validate_symbol(symbol):
                return False, f"Invalid symbol: {symbol}"
            
            if not ValidationUtils.validate_weight(weight):
                return False, f"Invalid weight for {symbol}: {weight}"
        
        # 총 가중치 검사
        total_weight = sum(weights.values())
        if not ValidationUtils.validate_numeric(total_weight, 0.99, 1.01):
            return False, f"Total weight must be approximately 1.0, got: {total_weight}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: List[str] = None,
                          min_rows: int = 1) -> Tuple[bool, str]:
        """데이터프레임 유효성 검사"""
        if not isinstance(df, pd.DataFrame):
            return False, "Input must be a pandas DataFrame"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if len(df) < min_rows:
            return False, f"DataFrame must have at least {min_rows} rows"
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_returns_data(returns: Union[pd.Series, np.ndarray]) -> Tuple[bool, str]:
        """수익률 데이터 유효성 검사"""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if not isinstance(returns, np.ndarray):
            return False, "Returns must be a pandas Series or numpy array"
        
        if len(returns) == 0:
            return False, "Returns data is empty"
        
        # NaN 값 검사
        if np.any(np.isnan(returns)):
            return False, "Returns data contains NaN values"
        
        # 무한대 값 검사
        if np.any(np.isinf(returns)):
            return False, "Returns data contains infinite values"
        
        # 극단적 값 검사 (일일 수익률이 100%를 초과하는 경우)
        if np.any(np.abs(returns) > 1.0):
            return False, "Returns data contains extreme values (>100%)"
        
        return True, "Valid"
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """가격 데이터 유효성 검사"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        is_valid, message = ValidationUtils.validate_dataframe(df, required_columns)
        
        if not is_valid:
            return False, message
        
        # 가격 데이터 검사
        for col in ['Open', 'High', 'Low', 'Close']:
            if not ValidationUtils._validate_price_column(df[col]):
                return False, f"Invalid price data in column: {col}"
        
        # OHLC 논리 검사
        if not ValidationUtils._validate_ohlc_logic(df):
            return False, "Invalid OHLC logic (High < Low or prices < 0)"
        
        # 거래량 검사
        if not ValidationUtils._validate_volume(df['Volume']):
            return False, "Invalid volume data (negative values)"
        
        return True, "Valid"
    
    @staticmethod
    def _validate_price_column(series: pd.Series) -> bool:
        """가격 컬럼 유효성 검사"""
        # NaN 값 검사
        if series.isnull().any():
            return False
        
        # 음수 값 검사
        if (series < 0).any():
            return False
        
        # 무한대 값 검사
        if np.isinf(series).any():
            return False
        
        return True
    
    @staticmethod
    def _validate_ohlc_logic(df: pd.DataFrame) -> bool:
        """OHLC 논리 검사"""
        # High >= Low 검사
        if not (df['High'] >= df['Low']).all():
            return False
        
        # High >= Open, Close 검사
        if not (df['High'] >= df['Open']).all():
            return False
        if not (df['High'] >= df['Close']).all():
            return False
        
        # Low <= Open, Close 검사
        if not (df['Low'] <= df['Open']).all():
            return False
        if not (df['Low'] <= df['Close']).all():
            return False
        
        return True
    
    @staticmethod
    def _validate_volume(series: pd.Series) -> bool:
        """거래량 유효성 검사"""
        # 음수 값 검사
        if (series < 0).any():
            return False
        
        # NaN 값 검사
        if series.isnull().any():
            return False
        
        return True
    
    @staticmethod
    def validate_api_key(api_key: str, api_type: str = None) -> bool:
        """API 키 유효성 검사"""
        if not isinstance(api_key, str):
            return False
        
        if len(api_key) < 10:
            return False
        
        # API 타입별 검사
        if api_type == 'alpha_vantage':
            return len(api_key) >= 16 and api_key.isalnum()
        elif api_type == 'news_api':
            return len(api_key) >= 32 and api_key.isalnum()
        elif api_type == 'openai':
            return api_key.startswith('sk-') and len(api_key) >= 20
        
        return True
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """이메일 유효성 검사"""
        if not isinstance(email, str):
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """전화번호 유효성 검사"""
        if not isinstance(phone, str):
            return False
        
        # 숫자, 하이픈, 괄호, 공백만 허용
        cleaned = re.sub(r'[^\d\-\(\)\s]', '', phone)
        digits_only = re.sub(r'[^\d]', '', cleaned)
        
        # 10-15자리 숫자
        return 10 <= len(digits_only) <= 15
    
    @staticmethod
    def validate_risk_tolerance(risk_tolerance: str) -> bool:
        """리스크 허용도 유효성 검사"""
        valid_levels = ['low', 'medium', 'high', 'Low', 'Medium', 'High']
        return risk_tolerance in valid_levels
    
    @staticmethod
    def validate_investment_style(style: str) -> bool:
        """투자 스타일 유효성 검사"""
        valid_styles = ['value', 'growth', 'balanced', 'income', 
                       'Value', 'Growth', 'Balanced', 'Income']
        return style in valid_styles
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """시간 프레임 유효성 검사"""
        valid_timeframes = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y']
        return timeframe in valid_timeframes
    
    @staticmethod
    def validate_sector(sector: str) -> bool:
        """섹터 유효성 검사"""
        valid_sectors = [
            'Technology', 'Healthcare', 'Financial', 'Consumer Discretionary',
            'Consumer Staples', 'Energy', 'Industrials', 'Materials',
            'Real Estate', 'Utilities', 'Communication Services'
        ]
        return sector in valid_sectors
    
    @staticmethod
    def validate_portfolio_name(name: str) -> bool:
        """포트폴리오 이름 유효성 검사"""
        if not isinstance(name, str):
            return False
        
        if len(name) < 1 or len(name) > 100:
            return False
        
        # 특수문자 제한
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
            return False
        
        return True
    
    @staticmethod
    def validate_analysis_type(analysis_type: str) -> bool:
        """분석 유형 유효성 검사"""
        valid_types = [
            'portfolio_risk', 'portfolio_performance', 'portfolio_diversification',
            'stock_fundamentals', 'stock_technical', 'market_analysis'
        ]
        return analysis_type in valid_types
    
    @staticmethod
    def validate_optimization_type(optimization_type: str) -> bool:
        """최적화 유형 유효성 검사"""
        valid_types = ['sharpe', 'min_variance', 'risk_parity', 'max_return']
        return optimization_type in valid_types
    
    @staticmethod
    def validate_strategy_name(strategy_name: str) -> bool:
        """전략 이름 유효성 검사"""
        if not isinstance(strategy_name, str):
            return False
        
        if len(strategy_name) < 1 or len(strategy_name) > 50:
            return False
        
        # 영문자, 숫자, 언더스코어만 허용
        if not re.match(r'^[a-zA-Z0-9_]+$', strategy_name):
            return False
        
        return True
    
    @staticmethod
    def validate_parameter_ranges(parameter_ranges: Dict[str, List]) -> Tuple[bool, str]:
        """파라미터 범위 유효성 검사"""
        if not isinstance(parameter_ranges, dict):
            return False, "Parameter ranges must be a dictionary"
        
        if not parameter_ranges:
            return False, "Parameter ranges cannot be empty"
        
        for param_name, param_values in parameter_ranges.items():
            if not isinstance(param_name, str):
                return False, f"Parameter name must be a string: {param_name}"
            
            if not isinstance(param_values, list):
                return False, f"Parameter values must be a list for {param_name}"
            
            if len(param_values) == 0:
                return False, f"Parameter values cannot be empty for {param_name}"
            
            # 숫자 값 검사
            for value in param_values:
                if not ValidationUtils.validate_numeric(value):
                    return False, f"Invalid parameter value for {param_name}: {value}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_file_path(file_path: str, 
                          allowed_extensions: List[str] = None) -> bool:
        """파일 경로 유효성 검사"""
        if not isinstance(file_path, str):
            return False
        
        if len(file_path) == 0:
            return False
        
        # 파일 확장자 검사
        if allowed_extensions:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension not in allowed_extensions:
                return False
        
        return True
    
    @staticmethod
    def validate_json_data(json_data: Any) -> bool:
        """JSON 데이터 유효성 검사"""
        try:
            import json
            if isinstance(json_data, str):
                json.loads(json_data)
            return True
        except:
            return False
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """URL 유효성 검사"""
        if not isinstance(url, str):
            return False
        
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_confidence_level(confidence_level: float) -> bool:
        """신뢰도 레벨 유효성 검사"""
        return ValidationUtils.validate_numeric(confidence_level, 0, 1)
    
    @staticmethod
    def validate_threshold(threshold: float) -> bool:
        """임계값 유효성 검사"""
        return ValidationUtils.validate_numeric(threshold, 0, 1)
    
    @staticmethod
    def validate_lookback_period(period: int) -> bool:
        """룩백 기간 유효성 검사"""
        if not isinstance(period, int):
            return False
        
        return 1 <= period <= 1000
    
    @staticmethod
    def validate_window_size(window: int) -> bool:
        """윈도우 크기 유효성 검사"""
        if not isinstance(window, int):
            return False
        
        return 2 <= window <= 1000
    
    @staticmethod
    def validate_confidence_interval(confidence_interval: Tuple[float, float]) -> bool:
        """신뢰구간 유효성 검사"""
        if not isinstance(confidence_interval, tuple) or len(confidence_interval) != 2:
            return False
        
        lower, upper = confidence_interval
        
        if not ValidationUtils.validate_numeric(lower, 0, 1):
            return False
        
        if not ValidationUtils.validate_numeric(upper, 0, 1):
            return False
        
        return lower < upper
