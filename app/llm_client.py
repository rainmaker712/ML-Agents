import os
from typing import Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
from app.prompt_loader import load_prompt_from_file, get_default_prompt
from app.core.config import settings

# 환경변수 로드
load_dotenv()


class LLMClient:
    """로컬 LLM과 OpenAI API를 모두 지원하는 클라이언트"""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.client = None
        self.model_config = {}

        if self.provider == "local":
            self._init_local_client()
        else:
            self._init_openai_client()

    def _init_openai_client(self):
        """OpenAI 클라이언트 초기화"""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_config = {
            "model": settings.OPENAI_MODEL,
            "max_tokens": settings.OPENAI_MAX_TOKENS,
            "temperature": settings.OPENAI_TEMPERATURE,
        }

    def _init_local_client(self):
        """로컬 LLM 클라이언트 초기화"""
        if not all([settings.LOCAL_LLM_BASE_URL, settings.LOCAL_LLM_MODEL]):
            raise ValueError(
                "로컬 LLM 설정이 완전하지 않습니다. LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL을 확인해주세요."
            )

        self.client = OpenAI(
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key=settings.LOCAL_LLM_API_KEY,
        )

        self.model_config = {
            "model": settings.LOCAL_LLM_MODEL,
            "max_tokens": settings.LOCAL_LLM_MAX_TOKENS,
            "temperature": settings.LOCAL_LLM_TEMPERATURE,
        }

    def create_chat_completion(
        self, messages: list, **kwargs
    ) -> ChatCompletion:
        """채팅 완성 생성"""
        if not self.client:
            raise ValueError("LLM 클라이언트가 초기화되지 않았습니다.")

        config = self.model_config.copy()
        config.update(kwargs)

        try:
            response = self.client.chat.completions.create(
                messages=messages, **config
            )
            return response
        except Exception as e:
            raise Exception(f"LLM 요청 실패: {str(e)}")

    def summarize_paper_text(self, paper_text: str) -> str:
        """논문 텍스트를 paper_prompt.md 기반으로 요약합니다."""
        prompt_template = (
            load_prompt_from_file(settings.PROMPT_FILE) or get_default_prompt()
        )

        prompt = prompt_template.replace(
            "{{PAPER_TEXT_HERE}}", paper_text[: settings.MAX_PROMPT_LENGTH]
        )

        response = self.create_chat_completion(
            [{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def get_provider_info(self) -> Dict[str, str]:
        """현재 사용 중인 LLM 제공자 정보 반환"""
        info = {
            "provider": self.provider,
            "model": self.model_config.get("model"),
            "max_tokens": str(self.model_config.get("max_tokens")),
            "temperature": str(self.model_config.get("temperature")),
        }
        if self.provider == "local":
            info["base_url"] = settings.LOCAL_LLM_BASE_URL
        return info