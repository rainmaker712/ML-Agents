# 📈 미국 주식 포트폴리오 AI 에이전트

미국 주식 시장의 포트폴리오를 제안하고 관리하는 종합적인 AI 에이전트입니다. 실시간 데이터 수집, AI 기반 분석, 백테스팅, 리스크 관리 등 포트폴리오 관리에 필요한 모든 기능을 제공합니다.

## ✨ 주요 기능

### 🔍 데이터 수집
- **실시간 주식 데이터**: Alpha Vantage, Yahoo Finance API를 통한 실시간 가격 정보
- **뉴스 분석**: News API를 통한 시장 뉴스 수집 및 감정 분석
- **기술적 지표**: 20여 종의 기술적 지표 자동 계산
- **경제 지표**: FRED API를 통한 경제 지표 수집

### 🤖 AI 분석 엔진
- **포트폴리오 최적화**: 샤프 비율, 최소 분산, 리스크 패리티 등 다양한 최적화 방법
- **AI 추천 시스템**: 머신러닝 기반 주식 추천 및 포트폴리오 구성
- **리스크 분석**: VaR, CVaR, 최대 낙폭, 상관관계 분석
- **시장 분석**: 시장 체제, 섹터 로테이션, 변동성 분석

### 📊 백테스팅 & 성과 분석
- **전략 백테스팅**: 다양한 투자 전략의 과거 성과 검증
- **워크 포워드 분석**: 시간에 따른 전략 성과 분석
- **몬테카를로 시뮬레이션**: 확률적 시나리오 분석
- **성과 지표**: 샤프 비율, 소르티노 비율, 칼마 비율 등

### 🎯 포트폴리오 관리
- **다중 포트폴리오**: 여러 포트폴리오 동시 관리
- **리밸런싱**: 자동/수동 포트폴리오 리밸런싱
- **리스크 모니터링**: 실시간 리스크 지표 추적
- **알림 시스템**: 가격, 리스크, 뉴스 알림

### 🌐 웹 인터페이스
- **직관적 대시보드**: 포트폴리오 현황 한눈에 보기
- **인터랙티브 차트**: Plotly 기반 동적 차트
- **실시간 업데이트**: 자동 데이터 갱신
- **반응형 디자인**: 모바일/태블릿 지원

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/stock-agent.git
cd stock-agent

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정

```bash
# 초기 설정 실행
python run.py --mode setup

# .env 파일에서 API 키 설정
# ALPHA_VANTAGE_API_KEY=your_key_here
# NEWS_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

### 3. 실행

```bash
# 웹 앱 실행
python run.py

# 또는 직접 실행
streamlit run app.py
```

## 📋 API 키 설정

다음 API 키들을 `.env` 파일에 설정해야 합니다:

### Alpha Vantage (무료)
- [Alpha Vantage](https://www.alphavantage.co/support/#api-key)에서 무료 API 키 발급
- 주식 가격 및 기술적 지표 데이터 제공

### News API (무료)
- [News API](https://newsapi.org/register)에서 무료 API 키 발급
- 뉴스 데이터 및 감정 분석 제공

### OpenAI (유료)
- [OpenAI](https://platform.openai.com/api-keys)에서 API 키 발급
- AI 분석 및 추천 기능 제공

## 🏗️ 프로젝트 구조

```
stock_agent/
├── app.py                          # Streamlit 웹 앱
├── run.py                          # 실행 스크립트
├── config.py                       # 설정 파일
├── requirements.txt                # 의존성 목록
├── README.md                       # 프로젝트 문서
│
├── data_collectors/                # 데이터 수집 모듈
│   ├── alpha_vantage_collector.py  # Alpha Vantage API
│   ├── news_collector.py          # 뉴스 수집
│   ├── yfinance_collector.py      # Yahoo Finance
│   └── technical_indicators.py    # 기술적 지표
│
├── ai_engine/                      # AI 분석 엔진
│   ├── portfolio_optimizer.py     # 포트폴리오 최적화
│   ├── recommendation_engine.py   # 추천 시스템
│   ├── risk_analyzer.py           # 리스크 분석
│   └── market_analyzer.py         # 시장 분석
│
├── models/                         # 데이터 모델
│   ├── database.py                # 데이터베이스 모델
│   ├── portfolio.py               # 포트폴리오 모델
│   ├── stock.py                   # 주식 모델
│   └── analysis.py                # 분석 결과 모델
│
├── services/                       # 비즈니스 로직
│   ├── portfolio_service.py       # 포트폴리오 서비스
│   ├── data_service.py            # 데이터 서비스
│   ├── analysis_service.py        # 분석 서비스
│   └── notification_service.py    # 알림 서비스
│
├── backtesting/                    # 백테스팅
│   ├── backtest_engine.py         # 백테스팅 엔진
│   ├── strategy_tester.py         # 전략 테스터
│   └── performance_analyzer.py    # 성과 분석
│
└── utils/                          # 유틸리티
    ├── data_utils.py              # 데이터 유틸리티
    ├── date_utils.py              # 날짜 유틸리티
    ├── math_utils.py              # 수학 유틸리티
    └── validation_utils.py        # 유효성 검사
```

## 🔧 사용법

### 웹 인터페이스

1. **대시보드**: 포트폴리오 현황 및 주요 지표 확인
2. **포트폴리오 분석**: 리스크, 성과, 다각화 분석
3. **AI 추천**: AI 기반 주식 추천 및 포트폴리오 구성
4. **시장 분석**: 시장 동향 및 섹터 분석
5. **설정**: API 키 및 알림 설정

### 프로그래밍 인터페이스

```python
from services import PortfolioService, DataService
from ai_engine import RecommendationEngine

# 포트폴리오 서비스 초기화
portfolio_service = PortfolioService()

# 포트폴리오 생성
portfolio_id = portfolio_service.create_portfolio(
    name="성장형 포트폴리오",
    risk_tolerance="medium"
)

# 주식 추가
portfolio_service.add_stock_to_portfolio(
    portfolio_id=portfolio_id,
    symbol="AAPL",
    weight=0.2
)

# AI 추천
recommendation_engine = RecommendationEngine()
recommendations = recommendation_engine.recommend_stocks(
    stock_universe=stock_data,
    risk_tolerance="medium",
    max_stocks=10
)
```

## 📊 지원하는 분석 기능

### 기술적 분석
- 이동평균 (SMA, EMA)
- RSI, MACD, 볼린저 밴드
- 스토캐스틱, Williams %R
- ATR, ADX, CCI
- 일목균형표

### 펀더멘털 분석
- P/E, P/B, PEG 비율
- ROE, ROA, 부채비율
- 매출 성장률, 순이익률
- 배당 수익률

### 리스크 분석
- VaR, CVaR
- 최대 낙폭 (Max Drawdown)
- 샤프 비율, 소르티노 비율
- 베타, 상관관계 분석

### 포트폴리오 최적화
- 샤프 비율 최대화
- 최소 분산 포트폴리오
- 리스크 패리티
- 효율적 프론티어

## 🧪 테스트

```bash
# 전체 테스트 실행
python run.py --mode test

# pytest 사용
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=stock_agent --cov-report=html
```

## 📈 성능 최적화

- **비동기 데이터 수집**: 여러 API 동시 호출
- **캐싱**: Redis를 통한 데이터 캐싱
- **데이터베이스 최적화**: 인덱싱 및 쿼리 최적화
- **메모리 관리**: 대용량 데이터 처리 최적화

## 🔒 보안

- **API 키 암호화**: 환경변수를 통한 안전한 키 관리
- **데이터 검증**: 입력 데이터 유효성 검사
- **SQL 인젝션 방지**: ORM 사용
- **HTTPS**: 웹 인터페이스 보안 통신

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원

- **이슈 리포트**: [GitHub Issues](https://github.com/your-username/stock-agent/issues)
- **문서**: [Wiki](https://github.com/your-username/stock-agent/wiki)
- **이메일**: contact@stockagent.com

## 🙏 감사의 말

- [Alpha Vantage](https://www.alphavantage.co/) - 무료 주식 데이터 API
- [News API](https://newsapi.org/) - 뉴스 데이터 API
- [Yahoo Finance](https://finance.yahoo.com/) - 금융 데이터
- [Streamlit](https://streamlit.io/) - 웹 앱 프레임워크
- [Plotly](https://plotly.com/) - 인터랙티브 차트

---

**⚠️ 면책 조항**: 이 도구는 교육 및 연구 목적으로만 사용되어야 합니다. 투자 결정은 본인의 책임 하에 이루어져야 하며, 이 도구의 결과를 투자 조언으로 해석하지 마세요.