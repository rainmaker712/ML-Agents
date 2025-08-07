# OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation

- **Authors**: Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, Yu-Gang Jiang
- **Link**: https://github.com/FoundationVision/OmniTokenizer

---

## 1. 연구 목적 (Purpose)
- 본 연구는 기존의 이미지 또는 비디오 중 하나의 입력 유형에만 특화된 tokenizer의 한계를 극복하고, 이미지와 비디오를 동시에 다룰 수 있는 통합적인 tokenizer를 제안하는 것을 목표로 한다.
- 구체적으로, Transformer 기반의 새로운 tokenizer인 OmniTokenizer를 개발하여, 이미지와 비디오 데이터를 하나의 모델에서 효과적으로 토큰화(tokenization)함으로써, 데이터의 상호보완성과 확장성을 극대화하며, 다양한 시각 생성(visual generation) 작업에서 최첨단 성능을 달성하는 것이 주요 목표이다.

## 2. 배경 및 관련 연구 (Background)
- 최근 시각 생성(visual generation) 분야에서는 언어 모델 기반 방법과 Diffusion Model이 두 가지 대표적 패러다임으로 자리잡았다. 언어 모델 기반 접근법은 시각적 데이터를 일련의 토큰 시퀀스로 변환하여 next-token prediction 문제로 접근하고, Diffusion Model은 노이즈를 점진적으로 제거하여 구조화된 이미지를 생성한다.
- 이러한 모델들의 핵심에는 tokenizer가 있는데, 이는 복잡한 시각적 신호를 잠재(잠복) 표현(latent representation)으로 변환하는 역할을 한다.
- 기존의 tokenizer들은 이미지 전용 또는 비디오 전용으로 설계되어, 데이터 확장성, 유연성, 그리고 두 입력 유형 간 시너지 효과를 얻지 못한다는 한계가 있다. 예를 들어 MAGVITv2와 같은 일부 모델은 이미지와 비디오를 모두 다루려 했지만, 여전히 별도의 모델을 각각 훈련해야 하며, 진정한 통합 및 상호학습(joint learning)은 불가능했다.
- 본 논문은 이러한 문제점을 인식하고, 이미지와 비디오를 동시에 다루는 joint tokenizer의 필요성을 제기한다. 이를 통해 데이터 부족(특히 비디오)에 대한 문제를 완화하고, 더 일반적이고 강력한 시각 표현을 학습할 수 있음을 주장한다.

## 3. 제안 방법론 (Methodology)
- **OmniTokenizer**는 Transformer 기반의 공동 이미지-비디오 tokenizer로, 공간-시간(spatial-temporal) 분리 아키텍처를 채택한다.
    - **Patchify**: 입력 이미지를 비중첩 방식으로 패치로 분할하고, 각각을 선형 계층에 투영하여 임베딩을 얻는다. 이미지와 비디오의 첫 프레임 및 나머지 프레임을 별도로 처리하여, 시퀀스 차원에서 연결(concatenate)한다.
    - **Encoder/Decoder**: 인코더는 공간적 블록과 시간적 블록으로 분리되어 있다. 공간 차원에서는 window attention을, 시간 차원에서는 causal attention을 적용하여, 효율적으로 지역 정보를 집계하고 비디오의 temporal coherence를 확보한다.
    - **Tokenization 방식**: 언어 모델 기반(LM Tokenizer, VQVAE)에서는 codebook lookup을 통해 임베딩을 양자화(quantization)하고, Diffusion Tokenizer(VAE)에서는 가우시안 분포에서 샘플링을 한다.
    - **Progressive Training**:
        1. **1단계**: 고정 해상도의 이미지 데이터로 사전학습(pre-training)하여, 정적 시각 정보의 기초적인 공간적 인코딩 능력을 갖춘다.
        2. **2단계**: 이미지와 비디오 데이터를 다양한 해상도로 공동 학습(joint training)하여, 시간적 역학(temporal dynamics)까지 학습한다. 이 두 단계의 프로그레시브 학습을 통해 이미지와 비디오 간의 격차를 해소하고, 더욱 일반화된 임베딩을 획득한다.
    - **Loss 함수**:
        - **VQ Training**: 벡터 양자화 손실(VQ loss)로, encoder 임베딩과 codebook 벡터 간의 거리를 최소화한다.
        - **KL Fine-tuning**: Diffusion 토크나이저로서 미세조정 시, Kullback-Leibler divergence를 활용하여 잠재 분포를 정렬한다.
        - 추가로, L2 reconstruction loss와 GAN loss도 사용하여 복원 및 생성 품질을 높인다.
    - **Visual Generation**:
        - OmniTokenizer로 토큰화한 후, 언어모델(Transformer) 또는 Latent Diffusion Model(LDM)을 통해 이미지/비디오를 생성한다. 언어모델 기반 접근은 시퀀스 예측 문제로, LDM은 효율적인 latent space에서 diffusion을 수행한다.

## 4. 실험 설정 (Experimental Setup)
- **데이터셋**: ImageNet, CelebA-HQ, FFHQ, UCF-101, Kinetics-600 등 다양한 이미지 및 비디오 데이터셋을 활용함.
- **평가지표(metrics)**:
    - 이미지: FID(Fréchet Inception Distance)
    - 비디오: FVD(Fréchet Video Distance)
- **베이스라인 모델**:
    - 이미지: DALL-E, VQGAN, ViT-VQGAN, MaskGIT 등
    - 비디오: TATS, MAGVIT, CViViT 등

## 5. 주요 결과 및 분석 (Results & Analysis)
- OmniTokenizer는 이미지와 비디오 데이터셋 모두에서 기존 방법 대비 우수한 복원 성능을 보였다.
    - ImageNet 기준, OmniTokenizer-VQVAE는 1.11의 rFID, OmniTokenizer-VAE는 0.69의 rFID를 기록하여, ViT-VQGAN(1.28) 등 기존 SOTA 대비 13% 이상 성능 향상을 달성했다.
    - UCF-101 등 비디오 데이터셋에서 FVD(Fréchet Video Distance) 기준, OmniTokenizer-VQVAE는 42, OmniTokenizer-VAE는 23으로, MAGVIT(58), TATS(162) 등 기존 방법 대비 각각 26% 이상 개선된 결과를 보였다.
- 저자는 이미지-비디오 공동 학습이 단일 modality 학습 대비 더 일반적이고 강력한 표현을 가능케 하며, 다양한 해상도 및 데이터 분포에 대한 적응력이 뛰어나다고 분석했다.
- OmniTokenizer를 통해, 언어모델 기반 생성(model-based)과 diffusion model 모두에서 class-conditional/unconditional generation, frame prediction 등 다양한 생성 작업에서 SOTA 수준의 결과를 달성했음을 실증하였다.

## 6. 결론 및 시사점 (Conclusion & Implications)
- OmniTokenizer는 최초로 이미지와 비디오를 통합적으로 토큰화할 수 있는 Transformer 기반 tokenizer를 제안하였으며, 프로그레시브 학습 전략을 통해 두 도메인의 시너지 효과를 실현했다.
- 본 연구의 접근법은 복잡한 시각 생성 작업에서 확장성, 범용성, 효율성을 모두 달성할 수 있음을 실험적으로 증명했다.
- 앞으로 다양한 시각 생성 및 편집, 멀티모달 학습, 데이터 효율적 학습 등에서 산업적·학문적으로 넓은 응용 가능성을 가진다.

## 7. 한계점 및 향후 연구 방향 (Limitations & Future Work)
- 저자는 현재 모델이 주로 대용량 정제 데이터셋에서 평가되었으며, 실제 다양한 환경(노이즈, domain shift 등)에서의 일반화 성능은 추가 연구가 필요함을 시사한다.
- 향후에는 더 다양한 입력 modality(예: 3D, 멀티센서 등)로 확장하거나, 더욱 효율적인 training/fine-tuning 전략, 그리고 실시간 시각 생성에의 적용 가능성 등에 대해 연구할 계획임을 언급했다.

## 8. 주요 Figure 및 Table 요약
- **(Figure 1)**: OmniTokenizer의 아키텍처를 도식화. 패치 임베딩 레이어와 분리된 공간/시간 attention 블록들로 구성되어 있으며, VQVAE 방식에서는 encoder 임베딩을 codebook에 lookup하여 양자화하고, VAE 방식에서는 가우시안 분포에서 샘플링하는 tokenization 과정을 보여준다. 디코더는 생략되고, 토크나이즈 단계만 시각화됨.
- **(Figure 2)**: 프로그레시브 학습 패러다임의 개념도. 기존 방법들이 이미지/비디오를 별도로 학습하는 것과 달리, OmniTokenizer는 이미지 사전학습 후 이미지-비디오 joint training을 통해 동일한 구조 및 가중치로 양 modality를 모두 토큰화하는 과정을 시각화.
- **(Table 1)**: ImageNet 등 주요 이미지 데이터셋에서의 복원 FID 결과 요약. OmniTokenizer-VQVAE 및 VAE가 기존 SOTA 대비 더 우수한 rFID(1.11, 0.69)를 기록함.
- **(Table 2)**: UCF-101, Kinetics-600 등 비디오 데이터셋에서의 복원 FVD 성능 비교 결과. OmniTokenizer가 MAGVIT, TATS 등 대비 현저히 낮은(더 우수한) FVD를 보임.

## 9. Appendix 요약
- Appendix에는 실험의 추가 세부 결과, ablation study, 다양한 하이퍼파라미터 설정, 추가적인 qualitative 결과 이미지 및 비디오 복원 예시 등이 포함되어 있다.
- 특히, 프로그레시브 학습 전략의 각 단계가 개별적으로 모델 성능에 어떠한 기여를 하는지에 대한 분석이 추가되어, 본문에서 주장한 joint training의 효과를 정량적으로 보강한다.
- 또한, 다양한 데이터 분할 및 resolution에 따른 모델의 일반화 성능 변화, codebook 활용률 등 실질적 모델 이해에 도움이 되는 부가 정보가 수록되어 있다.