from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
from config import Config

Base = declarative_base()

class StockData(Base):
    """주식 데이터 테이블"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    adjusted_close = Column(Float)
    
    # 기술적 지표
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    bb_percent = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    williams_r = Column(Float)
    atr = Column(Float)
    adx = Column(Float)
    cci = Column(Float)
    obv = Column(Float)
    vwap = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class StockInfo(Base):
    """주식 기본 정보 테이블"""
    __tablename__ = 'stock_info'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(200))
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    price_to_book = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    year_high = Column(Float)
    year_low = Column(Float)
    avg_volume = Column(Integer)
    shares_outstanding = Column(Float)
    enterprise_value = Column(Float)
    revenue = Column(Float)
    profit_margin = Column(Float)
    return_on_equity = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    description = Column(Text)
    last_updated = Column(DateTime, default=datetime.utcnow)

class NewsData(Base):
    """뉴스 데이터 테이블"""
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    title = Column(String(500))
    description = Column(Text)
    url = Column(String(1000))
    published_at = Column(DateTime)
    source = Column(String(100))
    content = Column(Text)
    sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Portfolio(Base):
    """포트폴리오 테이블"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    risk_tolerance = Column(String(20))  # low, medium, high
    target_return = Column(Float)
    max_positions = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PortfolioItem(Base):
    """포트폴리오 아이템 테이블"""
    __tablename__ = 'portfolio_items'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    weight = Column(Float, nullable=False)  # 포트폴리오 내 비중 (0-1)
    target_weight = Column(Float)  # 목표 비중
    current_price = Column(Float)
    shares = Column(Float)
    cost_basis = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    added_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AnalysisResult(Base):
    """분석 결과 테이블"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, nullable=False, index=True)
    analysis_type = Column(String(50))  # portfolio_analysis, risk_analysis, etc.
    result_data = Column(Text)  # JSON 형태로 저장
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """테이블 생성"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """세션 반환"""
        return self.SessionLocal()
    
    def add_stock_data(self, symbol: str, data: dict):
        """주식 데이터 추가"""
        session = self.get_session()
        try:
            stock_data = StockData(
                symbol=symbol,
                date=data.get('date'),
                open_price=data.get('open'),
                high_price=data.get('high'),
                low_price=data.get('low'),
                close_price=data.get('close'),
                volume=data.get('volume'),
                adjusted_close=data.get('adjusted_close'),
                sma_20=data.get('sma_20'),
                sma_50=data.get('sma_50'),
                sma_200=data.get('sma_200'),
                ema_12=data.get('ema_12'),
                ema_26=data.get('ema_26'),
                rsi=data.get('rsi'),
                macd=data.get('macd'),
                macd_signal=data.get('macd_signal'),
                macd_histogram=data.get('macd_histogram'),
                bb_upper=data.get('bb_upper'),
                bb_middle=data.get('bb_middle'),
                bb_lower=data.get('bb_lower'),
                bb_width=data.get('bb_width'),
                bb_percent=data.get('bb_percent'),
                stoch_k=data.get('stoch_k'),
                stoch_d=data.get('stoch_d'),
                williams_r=data.get('williams_r'),
                atr=data.get('atr'),
                adx=data.get('adx'),
                cci=data.get('cci'),
                obv=data.get('obv'),
                vwap=data.get('vwap')
            )
            session.add(stock_data)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding stock data: {e}")
        finally:
            session.close()
    
    def get_stock_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None):
        """주식 데이터 조회"""
        session = self.get_session()
        try:
            query = session.query(StockData).filter(StockData.symbol == symbol)
            if start_date:
                query = query.filter(StockData.date >= start_date)
            if end_date:
                query = query.filter(StockData.date <= end_date)
            return query.order_by(StockData.date).all()
        finally:
            session.close()
    
    def add_stock_info(self, symbol: str, info: dict):
        """주식 기본 정보 추가/업데이트"""
        session = self.get_session()
        try:
            existing = session.query(StockInfo).filter(StockInfo.symbol == symbol).first()
            if existing:
                for key, value in info.items():
                    setattr(existing, key, value)
                existing.last_updated = datetime.utcnow()
            else:
                stock_info = StockInfo(symbol=symbol, **info)
                session.add(stock_info)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding stock info: {e}")
        finally:
            session.close()
    
    def get_stock_info(self, symbol: str):
        """주식 기본 정보 조회"""
        session = self.get_session()
        try:
            return session.query(StockInfo).filter(StockInfo.symbol == symbol).first()
        finally:
            session.close()
    
    def add_news_data(self, symbol: str, news: dict):
        """뉴스 데이터 추가"""
        session = self.get_session()
        try:
            news_data = NewsData(
                symbol=symbol,
                title=news.get('title'),
                description=news.get('description'),
                url=news.get('url'),
                published_at=news.get('published_at'),
                source=news.get('source'),
                content=news.get('content'),
                sentiment=news.get('sentiment'),
                sentiment_score=news.get('sentiment_score')
            )
            session.add(news_data)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding news data: {e}")
        finally:
            session.close()
    
    def get_news_data(self, symbol: str, limit: int = 50):
        """뉴스 데이터 조회"""
        session = self.get_session()
        try:
            return session.query(NewsData).filter(NewsData.symbol == symbol)\
                .order_by(NewsData.published_at.desc()).limit(limit).all()
        finally:
            session.close()
    
    def create_portfolio(self, name: str, description: str = None, risk_tolerance: str = 'medium'):
        """포트폴리오 생성"""
        session = self.get_session()
        try:
            portfolio = Portfolio(
                name=name,
                description=description,
                risk_tolerance=risk_tolerance
            )
            session.add(portfolio)
            session.commit()
            return portfolio.id
        except Exception as e:
            session.rollback()
            print(f"Error creating portfolio: {e}")
            return None
        finally:
            session.close()
    
    def get_portfolios(self):
        """포트폴리오 목록 조회"""
        session = self.get_session()
        try:
            return session.query(Portfolio).all()
        finally:
            session.close()
    
    def add_portfolio_item(self, portfolio_id: int, symbol: str, weight: float, 
                          target_weight: float = None, shares: float = None, cost_basis: float = None):
        """포트폴리오 아이템 추가"""
        session = self.get_session()
        try:
            item = PortfolioItem(
                portfolio_id=portfolio_id,
                symbol=symbol,
                weight=weight,
                target_weight=target_weight or weight,
                shares=shares,
                cost_basis=cost_basis
            )
            session.add(item)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding portfolio item: {e}")
        finally:
            session.close()
    
    def get_portfolio_items(self, portfolio_id: int):
        """포트폴리오 아이템 조회"""
        session = self.get_session()
        try:
            return session.query(PortfolioItem).filter(PortfolioItem.portfolio_id == portfolio_id).all()
        finally:
            session.close()
    
    def save_analysis_result(self, portfolio_id: int, analysis_type: str, result_data: dict, confidence_score: float = None):
        """분석 결과 저장"""
        session = self.get_session()
        try:
            import json
            result = AnalysisResult(
                portfolio_id=portfolio_id,
                analysis_type=analysis_type,
                result_data=json.dumps(result_data),
                confidence_score=confidence_score
            )
            session.add(result)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving analysis result: {e}")
        finally:
            session.close()
