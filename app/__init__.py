from .llm_client import LLMClient

def get_llm_client() -> LLMClient:
    """LLM 클라이언트 인스턴스 반환"""
    return LLMClient()
