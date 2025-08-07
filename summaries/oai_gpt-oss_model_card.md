# gpt-oss-120b & gpt-oss-20b Model Card

- **Authors**: OpenAI
- **Link**: (공식 PDF 링크는 제공되지 않음)

---

## 1. 연구 목적 (Purpose)

OpenAI는 gpt-oss-120b와 gpt-oss-20b라는 두 개의 오픈 가중치(Open-Weight) Reasoning 모델을 공개하였다. 이 논문의 주요 목적은 다음과 같다.

- **오픈소스 커뮤니티를 위한 강력한 reasoning 및 tool use가 가능한 대형 언어모델 제공**: 기존의 폐쇄형(proprietary) 모델과 달리, 누구나 사용할 수 있도록 Apache 2.0 라이선스 하에 모델을 공개하고, gpt-oss 사용 정책을 명시함.
- **에이전트적(agentic) 워크플로우 지원**: 이 모델들은 강력한 명령어(instruction) 수행 능력, 웹서치와 Python 코드 실행 등 다양한 툴 사용, 그리고 reasoning effort(추론 강도) 조절 기능을 지원한다.
- **모델 안전성에 대한 새로운 접근**: 오픈 모델의 배포로 인해 발생할 수 있는 안전성 위험을 진단하고, 개발자 및 기업이 추가적인 보호조치를 취할 필요성을 강조한다.
- **모델의 성능 및 안전성 한계 검증**: gpt-oss-120b가 Biological & Chemical, Cyber, AI Self-Improvement 세 가지 주요 위험 범주에서 ‘High capability’ 기준에 미치지 못함을 확인하고, 적대적(adversarial) 파인튜닝 시에도 동일한 결과임을 검증한다.
- **AI 생태계의 전체적 안전 기준 제고**: 오픈 모델의 투명한 공개와 평가를 통해 AI 생태계의 안전 기준을 높이고자 하는 OpenAI의 의지를 재확인한다.

---

## 2. 배경 및 관련 연구 (Background)

gpt-oss-120b 및 gpt-oss-20b 모델은 GPT-2, GPT-3 아키텍처와 Mixture-of-Experts(MoE) Transformer 구조를 기반으로 개발되었다. 기존 연구와의 관련성을 요약하면 다음과 같다.

- **기존 GPT 계열 모델의 한계**: GPT-2, GPT-3 등 기존 대형 언어모델은 폐쇄형으로 공개되지 않거나, 오픈소스 버전(예: LLaMA, Falcon 등)이 reasoning과 tool use에서 제한적이었다.
- **Mixture-of-Experts(MoE) 트랜스포머**: 파라미터 효율성과 연산 속도를 높이기 위한 MoE 구조는 최근 다양한 논문에서 연구됐으나, 대규모 reasoning 및 tool use에 최적화된 오픈 가중치 모델은 드물었다.
- **안전성 이슈**: 오픈 모델은 악의적 fine-tuning 등의 위험이 존재하며, 기존 연구들은 주로 폐쇄형 모델의 안전성에 집중했다. 본 논문은 오픈 모델의 위험을 체계적으로 검증하고자 한다.
- **기존 방법론의 한계 극복**: 기존 모델들은 reasoning effort 조절, Harmony Chat Format 등의 고차원 기능을 충분히 제공하지 못했다. 본 논문은 이러한 기능적 한계를 기술적으로 극복하고자 한다.

---

## 3. 제안 방법론 (Methodology)

### 3.1 모델 아키텍처 및 파라미터

- **모델 구조**: gpt-oss-120b와 gpt-oss-20b는 autoregressive Mixture-of-Experts (MoE) Transformer로, 각각 36층(120b), 24층(20b)으로 구성된다.
- **파라미터 수**: 120b 모델은 116.8B total, 5.1B active parameters(각 토큰별 forward pass), 20b 모델은 20.9B total, 3.6B active parameters를 가진다.
- **세부 계층**:
  - MLP, Attention, Embedding/Unembedding으로 세분화하였으며, MoE 블록 내 120b는 128 experts, 20b는 32 experts를 사용.
  - Residual stream dimension은 2880, Root Mean Square Normalization(RMSNorm), Pre-LN placement 적용(GPT-2와 유사).
  - 각 Attention 블록은 banded window와 fully dense pattern을 번갈아 사용(128 token bandwidth), Grouped Query Attention(GQA) 적용.

### 3.2 Quantization

- **MXFP4 양자화**: MoE weight 파라미터(전체의 90% 이상)를 4.25bit(MXFP4)로 양자화하여, 120b 모델은 80GB GPU 하나에, 20b 모델은 16GB 메모리에서도 구동 가능하게 함.
- **체크포인트 크기**: 120b는 60.8GiB, 20b는 12.8GiB.

### 3.3 Tokenizer

- **o200k_harmony Tokenizer**: BPE 방식의 tokenizer로, 기존 GPT-4o, o4-mini 등에서 사용된 o200k tokenizer를 확장. Harmony chat format 지원을 위한 특수 토큰 포함, 전체 201,088 토큰.

### 3.4 Pretraining

- **데이터셋**: 텍스트 전용, 수조(trillions) 단위의 토큰 규모, STEM, 코딩, 일반상식에 중점. 유해 내용(특히 생명과학/화학 위험 관련)은 필터링(CBRN 필터 활용).
- **학습 환경**: NVIDIA H100 GPU, PyTorch 및 Triton 커널, Flash Attention 알고리즘 활용. 120b는 210만 H100-hours, 20b는 약 10배 적은 시간 소요.

### 3.5 Post-Training

- **Chain-of-Thought(CoT) RL**: OpenAI o3와 유사하게 CoT RL 기법으로 reasoning, 문제해결, 툴 사용법을 추가로 학습시킴.
- **Harmony Chat Format**: 시스템, 개발자, 유저, 어시스턴트, 툴 등 역할 기반 계층 구조 도입. 대화 채널(channel)로 메시지의 가시성과 기능 명확화(예: 분석, 코멘터리, 최종 답변 등).
- **Variable Effort Reasoning**: 시스템 프롬프트에 ‘Reasoning: low/medium/high’ 삽입해 reasoning 강도 조절. CoT 길이에 따라 정밀도 조절 가능.
- **Agentic Tool Use**: 웹 브라우징 도구, Python 실행(Jupyter 환경), 임의의 개발자 함수 등 다양한 도구 사용법 학습.

---

## 4. 실험 설정 (Experimental Setup)

### 4.1 데이터셋

- **Pretraining**: 수조 단위 텍스트(일반, STEM, 코딩 등), 유해/위험 데이터 사전 필터링.
- **평가**: AIME(경시 수학), GPQA Diamond(박사과정 과학), HLE(전문가 수준 질의응답), MMLU(대학수준 시험), Codeforces(경진 프로그래밍), SWE-Bench Verified(소프트웨어 엔지니어링), Tau-Bench Retail(함수 호출) 등 다양한 벤치마크.

### 4.2 평가지표 및 베이스라인

- **Metrics**: 정확도(Accuracy), Function Calling 성공률, 코드 문제 해결 능력, reasoning chain 길이별 성능 등.
- **비교 모델**: OpenAI의 o3, o3-mini, o4-mini 등과 직접 비교.
- **특이점**: reasoning 수준(저·중·고)별로 CoT+Answer 길이에 따른 성능 변화도 분석.

---

## 5. 주요 결과 및 분석 (Results & Analysis)

### 5.1 정량적 성능

- **Reasoning/지식**: gpt-oss-120b는 OpenAI o3-mini를 능가하며, o4-mini에 근접한 정확도 달성. 20b 모델도 크기 대비 경쟁력 있음(6배 작은데도 성능 격차 크지 않음).
  - 예: AIME 98.7%, GPQA Diamond 99.5%(120b 기준)
- **코딩/툴 사용**: Codeforces, SWE-Bench, Tau-Bench 등에서 120b가 o3-mini를 상회, o4-mini 수준 근접.
- **Reasoning effort 조절**: reasoning level을 높일수록 CoT+답변 길이 증가, 정확도도 스무스하게 상승(단, 과도하게 길어질 경우 diminishing return).
- **도구 사용**: 브라우저, 파이썬, 함수 호출 등 복합적 agentic tool 사용 문제도 우수하게 해결.

### 5.2 심층 분석

- **모델 크기와 성능**: MoE 구조, Flash Attention, Grouped Query Attention 등 최신 기술을 도입하여 모델 크기 대비 효율성이 매우 높음.
- **안전성 평가**: Biological & Chemical, Cyber, AI Self-Improvement 위험 범주에서 ‘High capability’ 기준 미달. 심지어 적대적 파인튜닝(악의적 fine-tuning) 시에도 threshold를 넘지 못함.
- **기존 공개 모델과의 차별점**: 대부분의 평가에서 기존 오픈 모델의 default 성능이 gpt-oss-120b의 adversarially fine-tuned 성능에 근접. 즉, gpt-oss-120b의 공개가 생물학/화학적 위험의 경계를 대폭 앞당기지는 않음.

---

## 6. 결론 및 시사점 (Conclusion & Implications)

- **최종 결론**: gpt-oss-120b와 20b는 강력한 reasoning, tool use, 에이전트 기능을 갖춘 오픈 가중치 모델로, 다양한 벤치마크에서 상위 성능을 보인다. 그러나 ‘High risk’ capability는 미달, 안전성 측면에서 추가 보호장치가 필요함.
- **학계/산업적 시사점**:
  - 오픈소스 LLM의 안전성 기준을 제시하고, 다양한 에이전트형 응용에서 즉시 활용 가능함.
  - 모델 자체의 안전성 한계로 인해, 실제 배포시 개발자·기업 수준에서 추가적인 보호조치가 필수임을 강조.
  - Harmony Chat Format, Variable Effort Reasoning 등 고차원 기능이 차세대 LLM 워크플로우 표준으로 자리잡을 가능성.

---

## 7. 한계점 및 향후 연구 방향 (Limitations & Future Work)

- **한계점**:
  - 오픈 공개 특성상, 악의적 fine-tuning 또는 목적 외 사용에 대해 OpenAI가 직접 개입하여 차단할 수 없음.
  - 생화학, 사이버, 자기개선(AI Self-Improvement) 등 고위험 분야에서 ‘High capability’에 도달하지 못함.
  - Harmony Chat Format 등 고급 기능의 올바른 사용법이 필수적이며, 잘못 적용시 성능 저하 가능.
- **향후 방향**:
  - 외부 안전 전문가(External Safety Expert)와의 협력을 통한 지속적 Red Teaming 및 평가.
  - 개발자·기업의 시스템별 맞춤 보호장치 구현 가이드 제공.
  - 향후 더 대규모 데이터, 멀티모달 기능, 강화된 안전성 훈련 등 후속 연구 예고.

---

## 8. 주요 Figure 및 Table 요약

- **(Figure 1)**:
  - gpt-oss-120b 및 20b의 reasoning, factuality, tool use 성능을 o3, o3-mini, o4-mini와 비교.
  - 120b는 o3-mini를 능가하고, o4-mini와 상당히 근접함. 20b도 크기에 비해 경쟁력 있음.
- **(Table 1)**:
  - 각 모델(120b, 20b)의 파라미터 수, active parameter, checkpoint size 등 상세 스펙 제시.
  - 120b는 116.8B total, 5.13B active, 60.8GiB; 20b는 20.9B total, 3.61B active, 12.8GiB.
- **(Figure 2)**:
  - 코드 및 툴 사용 성능 비교: Codeforces, SWE-Bench, Tau-Bench에서 120b의 우수성 확인.
- **(Figure 3)**:
  - Reasoning level(저·중·고)에 따른 CoT+답변 길이와 정확도 변화. reasoning effort 증가에 따라 성능이 점진적으로 향상됨을 시각화.

---

## 9. Appendix 요약

- **Harmony Chat Format 사용 예제**: Appendix Table 17, 18에 입력/출력 예제 제공. 올바른 multi-turn 대화 관리법, reasoning trace 제거 방법 등 상세 지침 수록.
- **추천/비추천 방안**: Appendix 2에는 도입된 추천 정책과 도입하지 않은 정책 목록이 명시되어, 실제 배포시 참고 가능.
- **추가 실험 및 안전성 평가**: Appendix 1, 2에 외부 안전 전문가의 평가 피드백과, 다양한 adversarial red teaming 결과 요약.
- **도구 사용 및 reasoning 세부 옵션**: 개발자 기능(Function Calling 등) 구성, Harmony 포맷의 다양한 활용 방법 등 구체적 가이드 제공.

---

**요약**  
gpt-oss-120b 및 gpt-oss-20b는 오픈소스 커뮤니티에 강력한 reasoning과 tool use 기능을 제공하는 대규모 LLM으로, 최신 MoE Transformer 구조, 다양한 agentic 워크플로우, 안전성 평가 및 투명한 공개를 특징으로 한다. 모델의 기본 성능은 상위권이지만, 안전성 및 배포 시 주의가 필요하며, 향후 연구 및 생태계 발전에 중요한 이정표가 될 것으로 기대된다.