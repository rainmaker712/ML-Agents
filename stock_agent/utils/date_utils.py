from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np

class DateUtils:
    """날짜 유틸리티 클래스"""
    
    @staticmethod
    def get_trading_days(start_date: Union[str, datetime, date],
                        end_date: Union[str, datetime, date],
                        exclude_weekends: bool = True,
                        exclude_holidays: bool = True) -> List[datetime]:
        """거래일 목록 생성"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 전체 날짜 범위 생성
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 주말 제외
        if exclude_weekends:
            date_range = date_range[date_range.weekday < 5]
        
        # 휴일 제외 (간단한 버전)
        if exclude_holidays:
            # 주요 미국 휴일 (간단한 버전)
            holidays = DateUtils._get_us_holidays(start_date.year, end_date.year)
            date_range = date_range[~date_range.isin(holidays)]
        
        return date_range.tolist()
    
    @staticmethod
    def _get_us_holidays(start_year: int, end_year: int) -> List[datetime]:
        """미국 휴일 목록 (간단한 버전)"""
        holidays = []
        
        for year in range(start_year, end_year + 1):
            # 신년
            holidays.append(datetime(year, 1, 1))
            
            # 마틴 루터 킹 주니어 데이 (1월 셋째 월요일)
            mlk_day = DateUtils._get_nth_weekday(year, 1, 3, 0)  # 0 = 월요일
            holidays.append(mlk_day)
            
            # 대통령의 날 (2월 셋째 월요일)
            presidents_day = DateUtils._get_nth_weekday(year, 2, 3, 0)
            holidays.append(presidents_day)
            
            # 메모리얼 데이 (5월 마지막 월요일)
            memorial_day = DateUtils._get_last_weekday(year, 5, 0)
            holidays.append(memorial_day)
            
            # 독립기념일
            holidays.append(datetime(year, 7, 4))
            
            # 노동절 (9월 첫째 월요일)
            labor_day = DateUtils._get_nth_weekday(year, 9, 1, 0)
            holidays.append(labor_day)
            
            # 콜럼버스 데이 (10월 둘째 월요일)
            columbus_day = DateUtils._get_nth_weekday(year, 10, 2, 0)
            holidays.append(columbus_day)
            
            # 재향군인의 날
            holidays.append(datetime(year, 11, 11))
            
            # 추수감사절 (11월 넷째 목요일)
            thanksgiving = DateUtils._get_nth_weekday(year, 11, 4, 3)  # 3 = 목요일
            holidays.append(thanksgiving)
            
            # 크리스마스
            holidays.append(datetime(year, 12, 25))
        
        return holidays
    
    @staticmethod
    def _get_nth_weekday(year: int, month: int, n: int, weekday: int) -> datetime:
        """특정 월의 n번째 요일 찾기"""
        # 해당 월의 첫째 날
        first_day = datetime(year, month, 1)
        
        # 첫째 날의 요일 (0=월요일, 6=일요일)
        first_weekday = first_day.weekday()
        
        # 목표 요일까지의 일수 계산
        days_to_target = (weekday - first_weekday) % 7
        
        # n번째 해당 요일 계산
        target_date = first_day + timedelta(days=days_to_target + (n - 1) * 7)
        
        return target_date
    
    @staticmethod
    def _get_last_weekday(year: int, month: int, weekday: int) -> datetime:
        """특정 월의 마지막 요일 찾기"""
        # 다음 달 첫째 날
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        # 이전 날부터 역산
        last_day = next_month - timedelta(days=1)
        
        # 마지막 요일까지 역산
        days_back = (last_day.weekday() - weekday) % 7
        target_date = last_day - timedelta(days=days_back)
        
        return target_date
    
    @staticmethod
    def get_business_days_between(start_date: Union[str, datetime, date],
                                end_date: Union[str, datetime, date]) -> int:
        """두 날짜 사이의 영업일 수 계산"""
        trading_days = DateUtils.get_trading_days(start_date, end_date)
        return len(trading_days)
    
    @staticmethod
    def add_business_days(start_date: Union[str, datetime, date],
                         days: int) -> datetime:
        """영업일 추가"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        current_date = start_date
        added_days = 0
        
        while added_days < days:
            current_date += timedelta(days=1)
            # 주말이 아닌 경우에만 카운트
            if current_date.weekday() < 5:
                added_days += 1
        
        return current_date
    
    @staticmethod
    def get_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
        """분기 시작/종료 날짜"""
        quarter_starts = {
            1: datetime(year, 1, 1),
            2: datetime(year, 4, 1),
            3: datetime(year, 7, 1),
            4: datetime(year, 10, 1)
        }
        
        quarter_ends = {
            1: datetime(year, 3, 31),
            2: datetime(year, 6, 30),
            3: datetime(year, 9, 30),
            4: datetime(year, 12, 31)
        }
        
        return quarter_starts[quarter], quarter_ends[quarter]
    
    @staticmethod
    def get_year_dates(year: int) -> Tuple[datetime, datetime]:
        """연도 시작/종료 날짜"""
        return datetime(year, 1, 1), datetime(year, 12, 31)
    
    @staticmethod
    def get_month_dates(year: int, month: int) -> Tuple[datetime, datetime]:
        """월 시작/종료 날짜"""
        start_date = datetime(year, month, 1)
        
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        return start_date, end_date
    
    @staticmethod
    def get_week_dates(year: int, week: int) -> Tuple[datetime, datetime]:
        """주 시작/종료 날짜 (ISO 주 기준)"""
        # 해당 연도의 1월 1일
        jan_1 = datetime(year, 1, 1)
        
        # 1월 1일이 목요일 이전이면 이전 주부터 시작
        if jan_1.weekday() < 3:  # 0=월요일, 3=목요일
            week_start = jan_1 - timedelta(days=jan_1.weekday() + 3)
        else:
            week_start = jan_1 + timedelta(days=4 - jan_1.weekday())
        
        # 목표 주 계산
        target_week_start = week_start + timedelta(weeks=week - 1)
        target_week_end = target_week_start + timedelta(days=6)
        
        return target_week_start, target_week_end
    
    @staticmethod
    def is_trading_day(date: Union[str, datetime, date]) -> bool:
        """거래일 여부 확인"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 주말 확인
        if date.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
        
        # 휴일 확인 (간단한 버전)
        holidays = DateUtils._get_us_holidays(date.year, date.year)
        if date in holidays:
            return False
        
        return True
    
    @staticmethod
    def get_next_trading_day(date: Union[str, datetime, date]) -> datetime:
        """다음 거래일"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        next_date = date + timedelta(days=1)
        
        while not DateUtils.is_trading_day(next_date):
            next_date += timedelta(days=1)
        
        return next_date
    
    @staticmethod
    def get_previous_trading_day(date: Union[str, datetime, date]) -> datetime:
        """이전 거래일"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        prev_date = date - timedelta(days=1)
        
        while not DateUtils.is_trading_day(prev_date):
            prev_date -= timedelta(days=1)
        
        return prev_date
    
    @staticmethod
    def get_trading_days_in_month(year: int, month: int) -> List[datetime]:
        """특정 월의 거래일 목록"""
        start_date, end_date = DateUtils.get_month_dates(year, month)
        return DateUtils.get_trading_days(start_date, end_date)
    
    @staticmethod
    def get_trading_days_in_quarter(year: int, quarter: int) -> List[datetime]:
        """특정 분기의 거래일 목록"""
        start_date, end_date = DateUtils.get_quarter_dates(year, quarter)
        return DateUtils.get_trading_days(start_date, end_date)
    
    @staticmethod
    def get_trading_days_in_year(year: int) -> List[datetime]:
        """특정 연도의 거래일 목록"""
        start_date, end_date = DateUtils.get_year_dates(year)
        return DateUtils.get_trading_days(start_date, end_date)
    
    @staticmethod
    def format_date(date: Union[str, datetime, date], 
                   format_str: str = '%Y-%m-%d') -> str:
        """날짜 포맷팅"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        return date.strftime(format_str)
    
    @staticmethod
    def parse_date(date_str: str, 
                  format_str: str = None) -> datetime:
        """날짜 문자열 파싱"""
        if format_str:
            return datetime.strptime(date_str, format_str)
        else:
            return pd.to_datetime(date_str)
    
    @staticmethod
    def get_date_range(start_date: Union[str, datetime, date],
                      end_date: Union[str, datetime, date],
                      freq: str = 'D') -> List[datetime]:
        """날짜 범위 생성"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        return pd.date_range(start=start_date, end=end_date, freq=freq).tolist()
    
    @staticmethod
    def get_relative_date(base_date: Union[str, datetime, date],
                         years: int = 0,
                         months: int = 0,
                         days: int = 0,
                         weeks: int = 0) -> datetime:
        """상대 날짜 계산"""
        if isinstance(base_date, str):
            base_date = pd.to_datetime(base_date)
        
        # 년/월 추가
        if years != 0 or months != 0:
            # 월 계산
            total_months = base_date.month + months + (years * 12)
            new_year = base_date.year + (total_months - 1) // 12
            new_month = ((total_months - 1) % 12) + 1
            
            # 일 계산 (월말 처리)
            try:
                new_day = base_date.day
                new_date = datetime(new_year, new_month, new_day)
            except ValueError:
                # 월말인 경우 마지막 날로 설정
                new_date = datetime(new_year, new_month, 1) - timedelta(days=1)
        else:
            new_date = base_date
        
        # 일/주 추가
        new_date += timedelta(days=days + (weeks * 7))
        
        return new_date
    
    @staticmethod
    def get_age_in_days(start_date: Union[str, datetime, date],
                       end_date: Union[str, datetime, date] = None) -> int:
        """날짜 간 일수 차이"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        return (end_date - start_date).days
    
    @staticmethod
    def get_age_in_business_days(start_date: Union[str, datetime, date],
                               end_date: Union[str, datetime, date] = None) -> int:
        """날짜 간 영업일 차이"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        return DateUtils.get_business_days_between(start_date, end_date)
    
    @staticmethod
    def is_end_of_month(date: Union[str, datetime, date]) -> bool:
        """월말 여부"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        next_day = date + timedelta(days=1)
        return date.month != next_day.month
    
    @staticmethod
    def is_end_of_quarter(date: Union[str, datetime, date]) -> bool:
        """분기말 여부"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        return date.month in [3, 6, 9, 12] and DateUtils.is_end_of_month(date)
    
    @staticmethod
    def is_end_of_year(date: Union[str, datetime, date]) -> bool:
        """연말 여부"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        return date.month == 12 and DateUtils.is_end_of_month(date)
    
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
