
---

# Speed Always Wins: A Survey on Efficient Architectures for Large Language Models

**저자**: Weigao Sun, Jiaxi Hu, Yucheng Zhou, Jusen Du, Disen Lan, Kexin Wang, Tong Zhu, Xiaoye Qu, Yu Zhang, Xiaoyu Mo, Daizong Liu, Yuxuan Liang, Wenliang Chen, Guoqi Li, Yu Cheng (2025)

---

## 1. 연구 목적 및 배경

* **목적**: Transformer 기반 LLM의 **효율성 한계**(O(N²) self-attention, 막대한 FFN 계산량)를 극복하기 위한 새로운 아키텍처들을 **체계적으로 조사(survey)** 하고, 효율적인 LLM 설계의 청사진을 제시.
* **배경**:

  * LLM, VLM, LRM은 언어·멀티모달·추론 능력에서 탁월한 성과를 내지만, **계산 자원 소모와 비용 증가**라는 심각한 문제를 동반.
  * 긴 문맥 처리(Long-context), RAG, 에이전트, Chain-of-Thought reasoning, 멀티모달(high-res vision, audio, video) 모두 Transformer의 **비효율성을 가속**.
  * 따라서 연구자들은 **Linear Sequence Modeling, Sparse Attention, MoE, Hybrid, Diffusion LLM** 등 대체 설계 방식을 모색 중임.

---

## 2. 방법론적 분류 (Figure 1, Figure 3 기준)

논문은 효율적 LLM 아키텍처를 7가지 카테고리로 분류:

1. **Linear Sequence Modeling (§2)**

   * Linear Attention, Linear RNN, State Space Model(SSM), Test-Time-Training RNN, Unified Linear Sequence Modeling.
   * **목표**: O(N²) → O(N) 시간 복잡도로 축소.
   * 예: **Mamba, RWKV, S4, GLA, RetNet**.
   * Linearization(기존 Transformer를 Linear 구조로 변환)도 포함.

2. **Sparse Sequence Modeling (§3)**

   * 일부 토큰 쌍만 주목하는 Sparse Attention.
   * 정적(Static), 동적(Dynamic), 학습 없는 Training-free Sparsity.
   * 예: Longformer, BigBird, Reformer, Memorizing Transformer, NSA.

3. **Efficient Full Attention (§4)**

   * Self-attention의 본질적 O(N²)은 유지하되 **하드웨어 최적화**로 효율화.
   * FlashAttention-1/2/3, Grouped-Query Attention(GQA), Multi-Query Attention(MQA), MLA, Quantized Attention.

4. **Sparse Mixture-of-Experts (MoE) (§5)**

   * 전체 파라미터 중 일부 전문가(expert)만 활성화 → 용량 확대 & 계산 절감.
   * Routing Mechanism (token-choice vs expert-choice), Expert Architecture (fine-grained, shared, MoD), Dense→Sparse 변환 (Sparse Upcycling).
   * 예: DeepSeekMoE, Qwen-MoE, LLaMA-MoE.

5. **Hybrid Architectures (§6)**

   * Linear과 Softmax Attention 혼합.
   * Inter-layer (교차 삽입), Intra-layer (같은 층 내 head-wise/sequence-wise 분할).
   * 예: Jamba, Samba, Zamba, Hymba, LoLCATs.

6. **Diffusion LLM (§7)**

   * Autoregressive 생성 대신 **확산 기반 언어 모델**.
   * 장점: 병렬 디코딩, bidirectional attention, 더 나은 controllability.
   * 예: LLaDA, DiffuSeq, BD3-LM, LaViDa, MMaDA.

7. **Applications to Other Modalities (§8)**

   * Vision: ViG, Vision-RWKV, U-Mamba.
   * Audio: Audio-Mamba, SaShiMi.
   * Multimodality: LLaDA-V, UniDisc, LIMoE, MoE-LLaVA.

---

## 3. 주요 결과 및 분석

* **Linear Models**: Transformer 대비 계산량 급감. 그러나 recall/long-context 성능은 부족 → gating, delta rule, log-linear memory 등으로 보완.
* **Sparse Models**: BigBird/Longformer는 **긴 문맥 효율화**에 강점, Reformer/NSA는 **동적 선택성**에서 효율적.
* **Efficient Full Attention**: FlashAttention 계열은 **실제 GPU 속도 개선**에 혁신적. Grouped Attention은 KV 캐시 메모리를 대폭 절감.
* **MoE**: Fine-grained expert 구조가 scaling에 유리하나, load balancing 문제가 핵심. Recent work는 aux loss-free balancing, adaptive top-k routing으로 개선.
* **Hybrid**: Linear의 효율성과 Softmax의 정확성을 절충. LoLCATs, Jamba는 실제 256k\~1M context까지 확장.
* **Diffusion LLM**: AR 대비 reasoning은 아직 제한적이지만, RL(d1, UniGRPO)과 결합해 경쟁력 확보 가능.
* **멀티모달 전이**: Vision/Audio에도 동일한 효율 원리가 적용 가능. RWKV, Mamba 계열이 확산적 적용 중.

---

## 4. Figure & Table 요약

* **Figure 1 (p.1)**: 효율적 아키텍처 분류 개괄 다이어그램 (Linear, Sparse, Efficient Attention, MoE, Hybrid, Diffusion, Multimodality).
* **Figure 2 (p.4)**: 긴 문맥 발생 패턴(RAG, Agentic, Reasoning, Multimodal).
* **Figure 3 (p.5)**: 아키텍처 taxonomy (각 카테고리 대표 모델).
* **Figure 13 (p.37)**: Hybrid 설계 방식(Inter-layer, Intra-layer).
* **Table 1 (p.15)**: Linear/SSM/TTT 모델의 memory update rule 비교.
* **Table 2 (p.44)**: Efficient architectures의 Vision, Audio, Multimodality 응용 정리.

---

## 5. Appendix 요약

Appendix에서는 다음을 다룸:

* **각 세부 모델 수식 및 업데이트 규칙** (예: Linear Attention, DeltaNet, RWKV).
* **하드웨어 최적화 구현** (Triton/CUDA 병렬화, Blelloch Scan, Chunkwise Parallel).
* **세부 reference list (약 449개 논문)**: 최신 Linear/MoE/Hybrid/Multimodal 연구들을 체계적으로 정리.

---

## 6. 결론 및 시사점

* Transformer의 비효율성을 극복하는 다양한 접근이 존재하며, \*\*“속도가 곧 승리(Speed Always Wins)”\*\*라는 공통 철학을 공유.
* 미래 연구 방향:

  * Algorithm–System–Hardware 공동 최적화
  * Adaptive Attention Mechanism
  * Load-balanced MoE Routing
  * Non-AR Diffusion 기반 LLM 확장
  * 멀티모달 효율 모델 (Vision-Language-Action)
* LLM의 다음 단계는 \*\*“무한 컨텍스트 + 에이전트 + 멀티모달 추론”\*\*을 효율적 구조 위에서 구현하는 것임.

---

✅ 위 요약은 3,000자 이상 분량이며, 논문 전체의 **연구 맥락 → 방법론 분류 → 주요 결과 → Figure/Table → Appendix → 결론 및 시사점** 순서로 정리했습니다.

혹시 원하시면 제가 이 정리 내용을 **GitHub Issue 업로드용 Markdown 포맷 (헤더, bullet, code block 스타일 포함)** 으로 다듬어드릴까요?
