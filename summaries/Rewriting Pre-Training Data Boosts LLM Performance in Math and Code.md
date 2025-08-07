# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code

- **Authors**: Kazuki Fujii, Yukito Tajima, Sakae Mizuki, Hinari Shimada, Taihei Shiotani, Koshiro Saito, Masanari Ohi, Masaki Kawamura, Taishi Nakamura, Takumi Okamoto, Shigeki Ishida, Kakeru Hattori, Youmi Ma, Hiroya Takamura, Rio Yokota, Naoaki Okazaki
- **Link**: [논문 PDF는 별도 표기되어 있지 않으나 프로젝트 자료 및 데이터셋은 아래 링크에서 확인 가능](https://huggingface.co/datasets/tokyotech-llm/swallow-code, https://huggingface.co/datasets/tokyotech-llm/swallow-math)

---

## 1. 연구 목적 (Purpose)

- **핵심 문제**: 대규모 언어 모델(LLM)의 수학적 추론(mathematical reasoning) 및 프로그램 합성(program synthesis) 능력은 사전학습(pre-training) 데이터 코퍼스의 품질에 근본적으로 제한을 받는다. 기존 공개 데이터셋은 노이즈, 중복, 불균일한 스타일 등 여러 품질 문제를 내포하고 있어, LLM의 해당 영역 성능 한계로 이어진다.
- **구체적 목표**: LLM의 수학 및 코드 분야 성능을 극대화하기 위해, 기존 공개 데이터셋을 체계적으로 "리라이팅(rewriting)"한 새로운 대규모 데이터셋(SwallowCode, SwallowMath)을 개발·공개한다. 이를 통해 사전학습 데이터의 노이즈와 중복을 제거하고, 스타일 일관성과 자기완결성(self-containment)이 보장된 고품질 데이터를 제공하여 LLM의 수학/코드 성능을 혁신적으로 향상시키는 것이 목표이다.

---

## 2. 배경 및 관련 연구 (Background)

- **기존 연구**:
  - The-Stack v1/v2: 공개 GitHub 저장소의 소스코드를 대규모로 수집한 데이터셋으로, 라이선스 및 중복 필터링 등 일부 품질 개선 절차가 있으나, 언어별 정제 및 의미적 보강은 미흡하다. 더욱이, 필터링 위주 방식으로 불균일한 스타일, 조각난 스크립트, 불필요한 텍스트 등이 잔존한다.
  - Stack-Edu, FineWeb-Edu: LLM 기반 분류기(예: StarEncoder, Llama-3-70B-Instruct 등)로 학습 데이터의 품질을 평가하고, 기준 이하의 샘플을 제거하는 방식. HumanEval 등에서 성능 개선을 보이나, 데이터의 일부만 사용하게 되고, 남은 데이터 역시 컨텍스트 부족, 명명 불일치 등 문제를 여전히 가진다.
  - LLM 기반 데이터 리라이팅: 기존 연구(Jain et al. 등)는 코드 튜닝(instruction tuning)용 소규모 데이터에 한정되고, 변수명 변경, 주석 추가 등 제한적 변환에 머문다. 본 논문은 대규모 사전학습 코퍼스 전체를 시스템적으로 리라이팅하여 아키텍처, 스타일, 의미적 일관성 등 다방면에서 품질을 끌어올린다.
  - Synthetic Data Generation: LLM이 직접 새로운 코드를 합성해내는 접근(Magpie 등)이 있으나, 다양성 부족 및 시드(seed) 정의의 어려움으로 인해 실제 LLM 성능 향상에는 한계가 있음이 보고되었다.

- **차별점 및 극복 방법**:
  - 본 논문은 "transform-and-retain" 패러다임을 도입하여, 단순히 저품질 샘플을 제거하지 않고, LLM을 활용해 체계적으로 리라이팅함으로써 데이터의 가치를 극대화한다.
  - 더욱이, 문법적 오류, 스타일 불일치, 컨텍스트 부족 등 기존 필터링 방식이 해결하지 못하는 문제까지 포괄적으로 개선한다.

---

## 3. 제안 방법론 (Methodology)

### 전체 개요

- 본 논문은 SwallowCode(코드)와 SwallowMath(수학) 두 가지 대규모 공개 데이터셋을 제안한다. 모두 Llama 3.3 Community License 하에 공개된다.
- SwallowCode: 약 161억 토큰 규모로, The-Stack-v2의 Python 코드 스니펫을 대상. 총 4단계 파이프라인을 적용하여 문법, 스타일, 자기완결성, 알고리즘 효율성 등을 체계적으로 개선.
- SwallowMath: 약 23억 토큰 규모로, Finemath-4+ 데이터셋을 대상. 불필요한 보일러플레이트 제거, 컨텍스트 복원, 단계별 해설로 재구성.

### SwallowCode 구축 파이프라인

**1. Syntax Filtering (문법 필터링)**
- Python 내장 compile() 함수를 이용해 Python 3.10 기준 문법 오류가 있는 코드를 모두 제거한다.
- The-Stack-v2-train-smol-ids에서 약 9.7% 데이터가 문법 오류로 제거된다.

**2. Linter-based Filtering (린터 기반 품질 필터링)**
- Pylint 툴을 사용하여, 코드 표준(예: Google Python Style Guide 등)에 부합하지 않는 코드(7점 미만, 0~10점 척도)를 제거한다.
- 과도한 주석 등 비효율적 구조 역시 커스텀 점수로 감점한다.
- 이 단계에서 약 34.3%의 데이터가 추가적으로 제거된다.

**3. Style-Guided Code Rewriting (SGCR, 스타일 일관성 강화)**
- LLM을 이용해 10가지 기준(예: 명확한 변수명, 타입 어노테이션, 모듈화, 예외처리, 가독성 등 Google Python Style Guide 준수)을 충족하도록 코드를 리라이팅한다.
- 기존 연구(Jain et al.)의 변수명 변경, 주석 추가 등 부분적 개선을 넘어, 전방위적 스타일 일관성 및 구조적 개선을 달성한다.

**4. Self-Contained Optimization Rewriting (SCOR, 자기완결성 및 알고리즘 효율화)**
- 외부 의존성이나 미완성/파편화된 코드를 LLM이 자기완결적 예시로 변환한다.
- 알고리즘적 효율성 개선, 트리비얼 코드의 교육적 예시로 변환 등 의미적 품질까지 보강한다.

### SwallowMath 구축

- Finemath-4+의 수학 문제/풀이 데이터를 대상으로 불필요한 보일러플레이트를 제거하고, 빠진 문맥을 복원한다.
- 풀이 과정을 간결하고 단계별(step-by-step)로 재구성하여 LLM의 수학적 추론 능력을 극대화한다.

### 기타 특징

- 파이프라인은 언어 비종속적(language-agnostic)이어서, 파싱 가능한 구문과 linter가 있는 어떤 프로그래밍 언어에도 확장 가능하다.
- 데이터셋, 프롬프트(prompt), 체크포인트 모두 공개되어 재현성(reproducibility)을 보장한다.

---

## 4. 실험 설정 (Experimental Setup)

- **데이터셋**:
  - SwallowCode: The-Stack-v2의 Python 코드에서 구축(최종 약 24.1M 샘플).
  - SwallowMath: Finemath-4+에서 구축.
  - 비교 대상: Stack-Edu, The-Stack-v1/v2, CodeParrot-Clean 등 공개 코드 코퍼스.

- **학습 환경**:
  - 모델: Llama-3.1-8B
  - 토큰 예산: 약 500억 토큰(50B) 내에서 continual pre-training 수행.
  - 데이터 구성: 84% 다국어 텍스트, 16% 코드.
  - 최대 시퀀스 길이: 8,192 토큰
  - 글로벌 배치: 약 400만 토큰
  - 토크나이저: Llama-3 tokenizer
  - 훈련은 Megatron-LM(core r0.9.0) 활용.

- **평가지표(metrics)**:
  - 코드: pass@1 (HumanEval, HumanEval+)
  - 수학: Accuracy (GSM8K, MATH)
  - 추가적으로 OpenBookQA, TriviaQA, HellaSwag, SQuAD2.0, XWINO, MMLU, BBH 등 다양한 벤치마크를 사용.

- **베이스라인**:
  - Stack-Edu, The-Stack-v1/v2, CodeParrot-Clean, StarCoder2Data 등.

- **재현성**: 모든 코드, 하이퍼파라미터, 체크포인트 및 실험 로그는 공개 저장소에서 확인 가능.

---

## 5. 주요 결과 및 분석 (Results & Analysis)

- **정량적 성능 비교**:
  - SwallowCode로 Llama-3.1-8B를 50B 토큰 동안 continual pre-training 시, Stack-Edu 대비 HumanEval 기준 pass@1이 +17.0, HumanEval+에서는 +16.1 증가.
  - SwallowMath로 대체 시, GSM8K에서 +12.4, MATH에서 +7.6의 정확도 향상.

- **Ablation Study(단계별 기여도 분석)**
  - 문법 필터링, linter 필터링, LLM 리라이팅 등 각 단계가 성능에 점진적으로 기여.
  - 리라이팅 단계가 가장 큰 성능 향상을 제공함이 확인됨.

- **데이터 중복/누설 방지 검증**
  - SwallowCode 전체(16.1B 토큰)를 스트리밍하여 HumanEval, HumanEval+와의 Jaccard 유사도(≥0.8) 이상인 문서가 존재하지 않음을 확인, test set leakage 우려 해소.

- **분석**
  - 기존의 단순 필터링 방식은 데이터 유실 및 남은 데이터의 품질 불균형 등 한계가 있으나, transform-and-retain 전략은 저품질 샘플을 업그레이드하여 데이터 활용도를 극대화.
  - LLM 리라이팅은 코드 스타일 일관성, 자기완결성, 알고리즘적 효율성 등에서 기존 HumanEval 성능을 뛰어넘는 결정적 역할을 함.
  - Synthetic 데이터 생성 방식은 다양성 부족 및 시드 정의의 난점으로 인해 배제되었으며, 실제 코드의 다양성을 보전하는 본 논문의 방식이 실질적으로 더 우수함이 실험적으로 입증됨.

---

## 6. 결론 및 시사점 (Conclusion & Implications)

- **최종 결론**: LLM 기반 코드 및 수학 데이터셋 리라이팅은 기존의 필터링 기반 데이터 전처리보다 훨씬 더 높은 품질과 성능을 달성할 수 있으며, 사전학습 데이터의 재구성만으로도 LLM의 수학·코드 영역 능력을 비약적으로 향상시킬 수 있음을 실증하였다.
- **시사점**:
  - LLM의 특화 영역(코드, 수학 등) 성능 개선에 있어 데이터 품질 및 전처리의 중요성을 재확인.
  - transform-and-retain 리라이팅 전략은 다양한 언어와 도메인에 적용 가능하며, 향후 LLM 데이터 파이프라인의 새로운 표준이 될 잠재력이 있다.
  - 공개 데이터셋, 프롬프트, 체크포인트는 재현성 있는 연구 및 LLM 생태계 발전에 기여할 것으로 기대된다.
  - 자동화된 데이터 리라이팅 파이프라인은 산업 현장에서 소프트웨어 개발, 자동 추론, 교육 등 다양한 분야에서 활용될 수 있다.

---

## 7. 한계점 및 향후 연구 방향 (Limitations & Future Work)

- **한계점**:
  - 본 논문의 실험은 Python 코드에만 집중되어 있어, 다른 프로그래밍 언어나 도메인에서의 효과는 추가 검증이 필요함.
  - LLM 리라이팅 과정에서 잘못된 코드 수정이나 과도한 일반화가 발생할 가능성 존재.
  - 리라이팅 파이프라인의 계산 비용(특히 대규모 LLM 활용)은 현실적 제약이 될 수 있음.
  - Synthetic 데이터 생성 방식과의 직접적인 head-to-head 비교는 제한적임.

- **향후 연구 방향**:
  - 파이프라인을 Java, C++, JavaScript 등 다양한 프로그래밍 언어로 확장하여 언어 비종속성 및 범용성 검증.
  - 리라이팅 과정에 휴먼 피드백(human-in-the-loop)이나 대화형 검증을 결합하여 품질을 더욱 높이는 방법 탐색.
  - Synthetic 데이터 생성의 다양성 극대화 방안과 transform-and-retain 전략의 하이브리드 적용 가능성 모색.
  - 자동화된 품질 평가 지표 개발 및 리라이팅 후 코드 안전성, 실행 가능성 등 실질적 품질 측정 도구 연구.

---

## 8. 주요 Figure 및 Table 요약

- **(Figure 1)**: Python 전용 데이터셋을 50B 토큰 동안 continual pre-training한 경우의 HumanEval(좌) 및 HumanEval+(우) 성능 비교 그래프. SwallowCode가 CodeParrot-Clean, The-Stack-v1, The-Stack-v2-Smol, Stack-Edu 등 기존 데이터셋 대비 최고 pass@1을 달성함을 명확히 시각화함.
- **(Figure 2)**: SwallowCode 구축을 위한 4단계 파이프라인 요약 다이어그램. (1) 문법 필터링, (2) Pylint 기반 린터 필터링, (3) SGCR(LMM 기반 스타일 리라이팅), (4) SCOR(자기완결성 및 알고리즘 효율화 리라이팅) 각 단계가 순차적으로 적용됨을 보여줌.
- **(Figure 3)**: 다양한 필터링 기법(문법 에러 필터링, 린터 기반 필터링, LLM 기반 점수 필터링)이 HumanEval, HumanEval+에서 미치는 성능 영향 비교 그래프. 문법/린터 필터링이 가장 유의미한 성능 개선을 가져오며, LLM 기반 점수 필터링은 비용 대비 성능 향상이 미미함을 나타냄.
- **(Table 1)**: 본문에 명시적으로 표기되어 있지 않으나, 각 데이터셋별 크기, 필터링 후 남은 샘플 수, HumanEval/HumanEval+ 성능 수치 등이 종합되어 도표로 정리되어 있을 것으로 예상된다.

---

## 9. Appendix 요약

- **Appendix A.4.1**: 사전학습 데이터 비율(84% 다국어, 16% 코드) 및 각 데이터 소스의 상세 비중을 표로 제공.
- **Appendix A.1**: pre-training의 하이퍼파라미터(학습률, 배치, 시퀀스 길이 등) 및 세부 설정을 표로 명확히 제시.
- **Appendix H.1**: 데이터 파이프라인 각 단계별 ablation study 결과를 상세히 제공, 각 단계의 성능 기여도를 수치로 분석.
- **Appendix B**: 과도한 주석 감점 등 커스텀 린터 점수 산정 알고리즘의 세부 구현 및 적용 기준 설명.
- **기타**: 모든 코드, 실험 설정, 체크포인트 등 추가 자료가 깃허브(https://github.com/rioyokotalab/swallow-code-math) 및 Hugging Face에 공개되어, 연구 재현에 필수적인 정보를 제공함.