# VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo

- **Authors**: Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, Zhi Zhang, Xin Liu (모두 ByteDanceSeed 소속, *Equal Contribution, †Corresponding authors)
- **Link**: [https://github.com/ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni)

## 1. 연구 목적 (Purpose)
- 본 논문은 최근 Large Language Model(LLM)의 발전에 따라 텍스트, 이미지, 오디오 등 다양한 모달리티를 동시에 이해하고 생성할 수 있는 omni-modal LLM의 필요성이 커지고 있으나, 이들 모델을 효율적이고 확장성 있게 학습시키는 것이 여전히 큰 도전 과제임을 지적한다.
- 기존의 모델 학습 프레임워크들은 다양한 모달리티를 동시에 처리하는 구조를 지원하지 못하거나, 모델 정의와 분산 병렬화 로직이 강하게 얽혀 있어 대규모 omni-modal LLM의 end-to-end 학습에 한계가 있다.
- 이에 본 연구는 새로운 모델 중심 분산 학습 레시피(Recipe)와 유연한 구성 인터페이스를 제공하는 VeOmni 프레임워크를 제안하여, 다양한 모달리티를 효율적이고 확장성 있게 통합/훈련할 수 있는 환경을 제공하는 것을 목표로 한다.

## 2. 배경 및 관련 연구 (Background)
- 최근 LLM 분야에서는 GPT-4o, BAGEL 등 텍스트 기반에서 이미지, 오디오 등 멀티모달로 발전이 가속화되고 있다. 대표적 멀티모달 태스크로는 시각적 질의응답(visual question answering), 이미지 생성, 멀티모달 추론 등이 있다.
- 기존의 멀티모달/omni-modal LLM들은 각 모달리티별로 사전학습(pre-trained) 네트워크를 결합하여 언어모델을 중심으로 시각, 오디오 등 인코더 및 디코더를 연결하는 구조를 많이 채택한다.
- 그러나 대다수의 기존 학습 프레임워크(예: Megatron-LM, Colossal-AI, NeMo, DistMM, DistTrain 등)는 텍스트 중심 또는 any-to-text 태스크에 최적화되어 있으며, any-to-any 형태의 완전한 omni-modal 학습은 지원하지 않는다. 이로 인해 모델 구조와 분산 병렬화 로직이 강하게 결합되어 유연성 및 확장성이 떨어지고, 엔지니어링 비용이 높다.
- 최근 일부 연구에서는 통신(communication)과 계산(computation) 분리를 시도하였으나, 여전히 레이어간 의존성과 확장성 부족 문제가 남아 있다.

## 3. 제안 방법론 (Methodology)
- VeOmni는 모델 중심(model-centric) 분산 학습 레시피와 모듈화된 아키텍처를 바탕으로 다양한 모달리티를 손쉽게 결합·확장할 수 있는 프레임워크를 제공한다.
- **경량화된 옴니모달 커스터마이징**: Encoder, Foundation, Decoder 세 가지 모듈로 완전히 분리된 구조를 제공. 각 모듈은 HuggingFace의 PreTrainedModel 클래스를 상속하며, 입력·출력 모달리티에 맞는 mixin을 통해 통합된다. 예를 들어 encoder는 lm_encode 함수로 원천 데이터를 토큰 임베딩으로 변환해 foundation model에 삽입하며, decoder 역시 lm_encode와 lm_head 등 함수로 타겟 토큰 및 최종 출력을 생성한다.
- **분산 학습 레시피 Zoo**: Fully Sharded Data Parallelism(FSDP), Sequence Parallelism(SP), Expert Parallelism(EP) 등 다양한 분산 전략을 모듈화하여, 모델 블록별로 병렬화 전략을 자유롭게 조합할 수 있다. 2D/3D 병렬화(FSDP+SP, FSDP+SP+EP 등)를 지원하며, 이를 통해 dense 또는 MoE(Mixture of Experts) 모델 등 다양한 구조에 최적화된 학습이 가능하다.
- **Plug-and-Play 아키텍처**: 새로운 모달리티의 인코더·디코더를 최소한의 코드 변경으로 손쉽게 추가할 수 있으며, 시스템 레벨 연산과 모델별 연산이 완전히 분리되어 있다.
- **비침투적(Non-intrusive) 분산 학습 API**: 분산 병렬화 코드를 모델 정의와 분리, 개발자는 모델 아키텍처에만 집중할 수 있다.

## 4. 실험 설정 (Experimental Setup)
- 실험에서는 7B에서 72B 파라미터의 다양한 규모의 omni-modal LLM을 대상으로, 8~128 GPU까지 분산 학습 환경에서 VeOmni의 효율성과 확장성을 검증하였다.
- 구체적인 데이터셋, 평가지표(metrics), 베이스라인 모델 등은 본문에 상세히 명시되어 있지 않으나, omni-modal 상황에서 다양한 모달리티(텍스트, 이미지, 오디오 등)를 통합한 상황을 가정한다.
- 비교 대상으로는 기존 멀티모달 분산 학습 프레임워크(Megatron-LM, Colossal-AI, DistMM, DistTrain, Optimus 등)가 언급된다.

## 5. 주요 결과 및 분석 (Results & Analysis)
- VeOmni를 활용하여 30B 파라미터의 omni-modal MoE 모델을 128개의 GPU에서 2,800 tokens/sec/GPU 이상의 처리량으로 학습 가능하며, 160K 이상의 context length까지 확장 학습이 가능함을 보였다.
- 다양한 분산 전략(FSDP, HSDP, SP, EP 등)을 조합하였을 때, 기존 프레임워크 대비 로드 불균형(load imbalance) 및 확장성(scalability) 문제를 효과적으로 해결하였으며, 엔지니어링 오버헤드도 크게 줄였다.
- 특히, 모델 정의와 분산 병렬화의 분리가 새로운 모달리티 통합, ultra-long sequence 학습, 대규모 MoE 모델 확장 등에서 높은 유연성과 효율성을 입증하였다.

## 6. 결론 및 시사점 (Conclusion & Implications)
- 본 논문은 omni-modal LLM 학습의 병목이었던 확장성, 유연성, 엔지니어링 오버헤드 문제를 모델 중심 분산 레시피와 완전 모듈화 구조로 해결할 수 있음을 제시하였다.
- VeOmni는 향후 다양한 모달리티의 통합, 대규모 LLM의 효율적 학습, 새로운 멀티모달 태스크 개발 등에서 학계 및 산업계 모두에 큰 파급효과를 미칠 수 있다.
- 특히, 맞춤형 모달리티 통합과 초대규모 분산 학습이 필요한 차세대 인공지능 시스템 개발에 중요한 인프라가 될 수 있다.

## 7. 한계점 및 향후 연구 방향 (Limitations & Future Work)
- 본문에서는 명시적으로 한계점이 상세히 언급되진 않았으나, 초대규모 모델 및 ultra-long context 환경에서의 운영, 다양한 실제 모달리티(음성, 비디오 등) 적용에 대한 추가 실험 필요성이 내포되어 있다.
- 향후 연구에서는 더욱 다양하고 복잡한 모달리티 지원, 최적화된 하드웨어 연동, 실시간/온라인 학습 시나리오 등에 대한 연구가 기대된다.

## 8. 주요 Figure 및 Table 요약
- **(Figure 1)**: 기존 멀티모달 학습 프레임워크와 VeOmni의 구조적 차이 비교. 기존 프레임워크는 모델 정의와 병렬 로직이 강하게 결합되어 있으나, VeOmni는 모델-시스템 분리를 통해 다양한 병렬화 전략을 자유롭게 조합할 수 있음을 시각적으로 설명.
- **(Figure 2)**: Encoder, Foundation, Decoder의 완전 분리 구조와 각 모듈이 어떻게 plug-and-play 방식으로 결합되는지, 그리고 각 모듈의 함수(lm_encode, lm_head, lm_embed 등)가 어떻게 통합되는지에 대한 다이어그램.
- **(Table 1)**: 본문에 직접 표로 언급된 내용은 없으나, Figure에서 각 분산 전략(예: FSDP, HSDP, SP, EP 등)의 적용 위치와 모듈별 역할 분리가 강조됨.

## 9. Appendix 요약
- Appendix B.3에서는 학습 및 추론 과정에서 각 모듈(Encoder, Foundation, Decoder)이 어떻게 상호작용하는지, 함수 호출 순서 및 데이터 흐름에 대한 상세 프로토콜을 제공한다.
- Appendix B.5에서는 FSDP1, FSDP2 등 다양한 버전의 Fully Sharded Data Parallel 전략의 구현 세부사항과, VeOmni에서 이를 어떻게 통합 및 구성할 수 있는지에 대한 추가 정보를 제공한다.
- 그 외에도, ultra-long sequence 학습, 다양한 모달리티 조합 실험 등 본문 이해에 도움이 되는 실험 설정 및 구현 팁이 포함되어 있다.
