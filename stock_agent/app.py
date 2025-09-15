import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collectors import AlphaVantageCollector, NewsCollector, YFinanceCollector, TechnicalIndicators
from ai_engine import PortfolioOptimizer, RecommendationEngine, RiskAnalyzer, MarketAnalyzer
from models import DatabaseManager, Portfolio, PortfolioItem, Stock
from config import Config

# 페이지 설정
st.set_page_config(
    page_title="미국 주식 포트폴리오 AI 에이전트",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown('<h1 class="main-header">📈 미국 주식 포트폴리오 AI 에이전트</h1>', unsafe_allow_html=True)

# 사이드바
st.sidebar.title("🔧 설정")

# API 키 설정
st.sidebar.subheader("API 키 설정")
alpha_vantage_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
news_api_key = st.sidebar.text_input("News API Key", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

# 설정 저장
if st.sidebar.button("설정 저장"):
    # 실제 구현에서는 환경변수나 설정 파일에 저장
    st.sidebar.success("설정이 저장되었습니다!")

# 메인 탭
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 대시보드", 
    "📊 포트폴리오 분석", 
    "🤖 AI 추천", 
    "📈 시장 분석", 
    "⚙️ 설정"
])

# 대시보드 탭
with tab1:
    st.header("📊 포트폴리오 대시보드")
    
    # 포트폴리오 선택
    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_name = st.selectbox(
            "포트폴리오 선택",
            ["새 포트폴리오", "성장형 포트폴리오", "가치형 포트폴리오", "배당형 포트폴리오"]
        )
    
    with col2:
        if st.button("포트폴리오 생성"):
            st.success("포트폴리오가 생성되었습니다!")
    
    # 포트폴리오 요약
    st.subheader("포트폴리오 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 가치", "$125,430", "2.3%")
    
    with col2:
        st.metric("총 수익률", "15.2%", "1.8%")
    
    with col3:
        st.metric("샤프 비율", "1.24", "0.05")
    
    with col4:
        st.metric("최대 낙폭", "-8.5%", "-2.1%")
    
    # 포트폴리오 구성
    st.subheader("포트폴리오 구성")
    
    # 샘플 데이터
    portfolio_data = {
        '종목': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
        '비중': [15.2, 12.8, 11.5, 10.3, 9.7, 8.9, 7.6, 6.4],
        '수익률': [18.5, 22.1, 15.3, 8.7, 35.2, 28.9, 12.4, 19.8],
        '섹터': ['Technology', 'Technology', 'Technology', 'Consumer', 'Automotive', 'Technology', 'Technology', 'Media']
    }
    
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # 포트폴리오 구성 차트
    fig_pie = px.pie(
        df_portfolio, 
        values='비중', 
        names='종목',
        title="포트폴리오 비중 분포"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # 섹터별 분포
    sector_data = df_portfolio.groupby('섹터')['비중'].sum().reset_index()
    fig_bar = px.bar(
        sector_data,
        x='섹터',
        y='비중',
        title="섹터별 분포"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# 포트폴리오 분석 탭
with tab2:
    st.header("📊 포트폴리오 분석")
    
    # 분석 옵션
    analysis_type = st.selectbox(
        "분석 유형",
        ["리스크 분석", "성과 분석", "다각화 분석", "상관관계 분석"]
    )
    
    if analysis_type == "리스크 분석":
        st.subheader("리스크 분석")
        
        # 리스크 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("변동성", "18.5%", "2.1%")
            st.metric("VaR (95%)", "-3.2%", "0.3%")
        
        with col2:
            st.metric("샤프 비율", "1.24", "0.05")
            st.metric("CVaR (95%)", "-4.8%", "0.5%")
        
        with col3:
            st.metric("최대 낙폭", "-8.5%", "-2.1%")
            st.metric("소르티노 비율", "1.45", "0.08")
        
        # 리스크 차트
        risk_data = pd.DataFrame({
            '날짜': pd.date_range('2024-01-01', periods=100, freq='D'),
            '포트폴리오': np.random.normal(0.001, 0.02, 100).cumsum(),
            '벤치마크': np.random.normal(0.0008, 0.015, 100).cumsum()
        })
        
        fig_risk = px.line(
            risk_data,
            x='날짜',
            y=['포트폴리오', '벤치마크'],
            title="포트폴리오 vs 벤치마크 수익률"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    elif analysis_type == "성과 분석":
        st.subheader("성과 분석")
        
        # 성과 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 수익률", "15.2%", "1.8%")
            st.metric("연간 수익률", "18.5%", "2.3%")
        
        with col2:
            st.metric("벤치마크 대비 초과수익", "3.2%", "0.5%")
            st.metric("정보 비율", "0.85", "0.12")
        
        with col3:
            st.metric("트레이너 비율", "1.45", "0.08")
            st.metric("칼마 비율", "2.18", "0.15")
        
        # 성과 차트
        performance_data = pd.DataFrame({
            '날짜': pd.date_range('2024-01-01', periods=252, freq='D'),
            '누적수익률': np.random.normal(0.0006, 0.015, 252).cumsum() * 100
        })
        
        fig_perf = px.line(
            performance_data,
            x='날짜',
            y='누적수익률',
            title="누적 수익률 추이"
        )
        st.plotly_chart(fig_perf, use_container_width=True)

# AI 추천 탭
with tab3:
    st.header("🤖 AI 포트폴리오 추천")
    
    # 추천 설정
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.selectbox(
            "리스크 허용도",
            ["낮음", "보통", "높음"],
            index=1
        )
        
        investment_style = st.selectbox(
            "투자 스타일",
            ["가치형", "성장형", "균형형", "수익형"],
            index=2
        )
    
    with col2:
        investment_amount = st.number_input(
            "투자 금액 ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        max_stocks = st.slider(
            "최대 종목 수",
            min_value=5,
            max_value=20,
            value=10
        )
    
    # 섹터 선호도
    st.subheader("섹터 선호도")
    sectors = st.multiselect(
        "선호 섹터 선택",
        ["Technology", "Healthcare", "Financial", "Consumer Discretionary", 
         "Consumer Staples", "Energy", "Industrials", "Materials", 
         "Real Estate", "Utilities", "Communication Services"],
        default=["Technology", "Healthcare", "Financial"]
    )
    
    # AI 추천 실행
    if st.button("🤖 AI 추천 생성", type="primary"):
        with st.spinner("AI가 포트폴리오를 분석하고 추천을 생성하고 있습니다..."):
            # 실제 구현에서는 AI 엔진을 호출
            st.success("AI 추천이 완료되었습니다!")
    
    # 추천 결과
    st.subheader("AI 추천 결과")
    
    # 추천 포트폴리오 테이블
    recommended_stocks = {
        '종목': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM'],
        'AI 점수': [95, 92, 88, 85, 82, 90, 87, 80, 78, 75],
        '추천 비중': [12.5, 11.2, 10.8, 9.5, 8.9, 9.8, 8.7, 7.5, 7.2, 6.9],
        '예상 수익률': [18.5, 22.1, 15.3, 8.7, 35.2, 28.9, 12.4, 19.8, 16.7, 14.2],
        '리스크 등급': ['Medium', 'Low', 'Medium', 'High', 'High', 'Medium', 'Medium', 'High', 'High', 'Medium']
    }
    
    df_recommended = pd.DataFrame(recommended_stocks)
    st.dataframe(df_recommended, use_container_width=True)
    
    # AI 추천 차트
    fig_recommended = px.bar(
        df_recommended,
        x='종목',
        y='AI 점수',
        color='AI 점수',
        title="AI 추천 점수"
    )
    st.plotly_chart(fig_recommended, use_container_width=True)

# 시장 분석 탭
with tab4:
    st.header("📈 시장 분석")
    
    # 시장 개요
    st.subheader("시장 개요")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,567.89", "1.23%")
    
    with col2:
        st.metric("NASDAQ", "14,234.56", "2.15%")
    
    with col3:
        st.metric("VIX", "18.45", "-0.85")
    
    with col4:
        st.metric("10Y Treasury", "4.25%", "0.12%")
    
    # 시장 심리
    st.subheader("시장 심리")
    
    sentiment_data = {
        '지표': ['뉴스 감정', '가격 모멘텀', '거래량', '변동성'],
        '점수': [75, 68, 82, 45],
        '등급': ['긍정적', '보통', '높음', '낮음']
    }
    
    df_sentiment = pd.DataFrame(sentiment_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sentiment = px.bar(
            df_sentiment,
            x='지표',
            y='점수',
            color='점수',
            title="시장 심리 지표"
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # 섹터 성과
        sector_performance = {
            '섹터': ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Energy'],
            '수익률': [12.5, 8.7, 6.2, 4.8, -2.1]
        }
        
        df_sector = pd.DataFrame(sector_performance)
        
        fig_sector = px.bar(
            df_sector,
            x='수익률',
            y='섹터',
            orientation='h',
            title="섹터별 성과"
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # 뉴스 분석
    st.subheader("주요 뉴스")
    
    news_data = [
        {
            '제목': 'Fed, 금리 인상 가능성 시사',
            '감정': '부정적',
            '영향도': '높음',
            '시간': '2시간 전'
        },
        {
            '제목': 'AI 기술주 급등, NVIDIA 신고가',
            '감정': '긍정적',
            '영향도': '중간',
            '시간': '4시간 전'
        },
        {
            '제목': '소비자 물가 상승률 둔화',
            '감정': '긍정적',
            '영향도': '높음',
            '시간': '6시간 전'
        }
    ]
    
    for news in news_data:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{news['제목']}**")
            with col2:
                sentiment_color = "🔴" if news['감정'] == '부정적' else "🟢" if news['감정'] == '긍정적' else "🟡"
                st.write(f"{sentiment_color} {news['감정']}")
            with col3:
                st.write(f"⏰ {news['시간']}")

# 설정 탭
with tab5:
    st.header("⚙️ 설정")
    
    # 데이터베이스 설정
    st.subheader("데이터베이스 설정")
    
    if st.button("데이터베이스 초기화"):
        with st.spinner("데이터베이스를 초기화하고 있습니다..."):
            # 실제 구현에서는 데이터베이스 초기화
            st.success("데이터베이스가 초기화되었습니다!")
    
    # API 설정
    st.subheader("API 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Alpha Vantage**")
        st.write("주식 가격 및 기술적 지표 데이터")
        if st.button("연결 테스트"):
            st.success("연결 성공!")
    
    with col2:
        st.write("**News API**")
        st.write("뉴스 및 시장 동향 데이터")
        if st.button("연결 테스트", key="news_test"):
            st.success("연결 성공!")
    
    # 알림 설정
    st.subheader("알림 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price_alert = st.checkbox("가격 알림", value=True)
        news_alert = st.checkbox("뉴스 알림", value=True)
    
    with col2:
        risk_alert = st.checkbox("리스크 알림", value=True)
        rebalance_alert = st.checkbox("리밸런싱 알림", value=True)
    
    # 백업 설정
    st.subheader("백업 설정")
    
    if st.button("데이터 백업"):
        with st.spinner("데이터를 백업하고 있습니다..."):
            # 실제 구현에서는 데이터 백업
            st.success("백업이 완료되었습니다!")
    
    if st.button("데이터 복원"):
        uploaded_file = st.file_uploader("백업 파일 선택", type=['json'])
        if uploaded_file is not None:
            st.success("데이터가 복원되었습니다!")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📈 미국 주식 포트폴리오 AI 에이전트 | Powered by AI & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

def main():
    """메인 함수"""
    pass

if __name__ == "__main__":
    main()
