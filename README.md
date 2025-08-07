# 📚 논문 요약 플랫폼

PDF 논문을 업로드하거나 URL을 입력하면 AI가 자동으로 요약을 생성하는 플랫폼입니다.

## ✨ 주요 기능

- **PDF 파일 업로드**: 로컬 PDF 파일을 업로드하여 요약 생성
- **PDF URL 입력**: PDF URL을 입력하여 요약 생성
- **AI 기반 요약**: LLM을 활용한 구조화된 논문 요약
- **커스텀 프롬프트**: `app/paper_prompt.md` 파일을 통한 프롬프트 커스터마이징
- **다양한 인터페이스**: Streamlit 웹 인터페이스 및 명령줄 도구 제공

## 🚀 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

**참고**: 이 프로젝트는 OpenAI Python SDK 1.0.0+ 버전을 사용합니다. 이전 버전과 호환되지 않습니다.

### 2. 환경 변수 설정

환경 변수를 설정하거나 `.env` 파일을 생성하세요:

#### OpenAI API 사용 (기본)
```bash
# LLM 제공자 설정
export LLM_PROVIDER=openai

# OpenAI API 키 설정
export OPENAI_API_KEY="your_openai_api_key_here"
```

#### 로컬 LLM 서버 사용
```bash
# LLM 제공자 설정
export LLM_PROVIDER=local

# 로컬 LLM 서버 설정
export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
export LOCAL_LLM_API_KEY="your_token"
export LOCAL_LLM_MODEL="your_model_path"
```

### 3. 실행 방법

#### Streamlit 웹 인터페이스 실행

```bash
streamlit run app/streamlit_app.py
```
웹 브라우저에서 `http://localhost:8501`로 접속하세요.

#### 명령줄 도구 실행

```bash
python -m app.summarize_and_post <PDF 파일 경로 또는 PDF 링크>
```

**예시:**
```bash
python -m app.summarize_and_post app/original_paper/upcycling.pdf
```

## 📁 프로젝트 구조

```
ML-Agents/
├── app/                    # 애플리케이션 소스 코드
│   ├── __init__.py         # 앱 초기화
│   ├── llm_client.py       # LLM 클라이언트 (OpenAI/로컬)
│   ├── prompt_loader.py    # 프롬프트 로더 유틸리티
│   ├── streamlit_app.py    # Streamlit 웹 인터페이스
│   ├── summarize_and_post.py # 명령줄 도구
│   ├── paper_prompt.md     # 논문 요약 프롬프트
│   ├── requirements.txt    # Python 의존성
│   ├── core/               # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── config.py       # 설정 관리
│   │   └── services.py     # PDF 처리 및 요약 서비스
│   ├── summaries/          # 생성된 요약 파일 저장소
│   └── original_paper/     # 원본 논문 파일
├── requirements.txt        # Python 의존성
├── README.md               # 프로젝트 문서
└── .gitignore              # Git 무시 파일
```

## 📝 프롬프트 커스터마이징

`app/paper_prompt.md` 파일을 수정하여 요약 생성 프롬프트를 커스터마이징할 수 있습니다. 파일이 없거나 로드할 수 없는 경우 기본 프롬프트가 사용됩니다.

## 🤝 기여하기

1. 이 저장소를 포크하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
