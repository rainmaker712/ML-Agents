import os
from typing import Optional
from app.core.config import settings

def load_prompt_from_file(file_path: Optional[str] = None) -> Optional[str]:
    if file_path is None:
        file_path = settings.PROMPT_FILE
    """마크다운 파일에서 프롬프트를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"프롬프트 파일 로드 중 오류 발생: {e}")
        return None

def get_default_prompt() -> str:
    """기본 프롬프트를 반환합니다."""
    return """You are an expert AI research paper analyst. Your task is to read the provided text from an academic paper and generate a detailed, structured summary in Korean Markdown format.

**Output Constraints:**
- The summary must be in Korean.
- For technical terms or concepts that are ambiguous in Korean, keep the original English term. (e.g., "Transformer", "Attention is All You Need")
- The total length should be at least 1,500 characters to ensure detail.
- All citation marks like [1], (Kim et al., 2024) must be removed.
- The output must be a single block of Markdown code.

Based on the content above, please generate the summary following this exact Markdown structure:

# [Paper Title]

- **Authors**: [List of Authors]
- **Link**: [PDF Link, if available]

## 1. 연구 목적 (Purpose)
- 이 연구가 해결하고자 하는 핵심 문제는 무엇인가?
- 연구를 통해 달성하고자 하는 구체적인 목표는 무엇인가?

## 2. 배경 및 관련 연구 (Background)
- 이 연구가 어떤 기존 연구들 위에 세워졌는가?
- 기존 방법론들의 한계점은 무엇이며, 이 논문은 이를 어떻게 극복하려 하는가?

## 3. 제안 방법론 (Methodology)
- 제안하는 모델, 아키텍처, 알고리즘의 핵심 아이디어를 단계별로 상세히 설명하라.
- 수식이나 핵심적인 메커니즘이 있다면 자세히 서술하라.

## 4. 실험 설정 (Experimental Setup)
- 어떤 데이터셋을 사용했는가?
- 평가지표(metrics)는 무엇을 사용했는가?
- 비교를 위해 사용된 베이스라인 모델들은 무엇인가?

## 5. 주요 결과 및 분석 (Results & Analysis)
- 실험 결과(성능)를 정량적으로 제시하고, 베이스라인과 비교하여 설명하라.
- 결과가 왜 그렇게 나왔는지에 대한 저자의 심층적인 분석을 요약하라.

## 6. 결론 및 시사점 (Conclusion & Implications)
- 이 연구의 최종 결론은 무엇인가?
- 이 연구 결과가 학계나 산업에 어떤 영향을 미칠 수 있는가? (시사점)

## 7. 한계점 및 향후 연구 방향 (Limitations & Future Work)
- 저자가 직접 언급한 연구의 한계점은 무엇인가?
- 이를 바탕으로 제시된 향후 연구 방향은 무엇인가?

## 8. 주요 Figure 및 Table 요약
- **(Figure 1)**: [Figure 1에 대한 설명]
- **(Table 1)**: [Table 1에 대한 설명]
- (논문에 포함된 다른 주요 Figure/Table에 대해서도 위와 같이 요약)

## 9. Appendix 요약
- Appendix에 담긴 내용 중 본문 이해에 도움이 될 만한 추가 정보나 실험이 있다면 요약하라.

Here is the content of the paper:
<paper_content>
{{PAPER_TEXT_HERE}}
</paper_content>""" 