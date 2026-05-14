# Monotonic TTS - Stage 2: Karras Diffusion (EDM) Implementation Plan

본 문서는 Karras et al. (2022)의 "Elucidating the Design Space of Diffusion-Based Generative Models" 논문을 바탕으로 한 멜-스펙트로그램 정제 모델(Stage 2)의 상세 구현 계획을 담고 있습니다.

## 1. 핵심 아키텍처 개요
Stage 1(`MonotonicTTSSynthesizer`)의 정렬된 특징량(`aligned_feats`)을 조건(Conditioning)으로 하여, Gaussian Noise로부터 고품질 멜-스펙트로그램을 생성하는 Karras Diffusion 프레임워크를 구축합니다.

---

## 2. 수식 명세 (EDM 기반)

### 2.1 Preconditioning 및 Normalization ($D_\theta$)
네트워크 $F_\theta$ (Unet)가 평균 0, 표준편차 1인 정규화된 공간에서 동작하도록 $D_\theta$를 다음과 같이 정의합니다:

$$x_{norm} = x - \mu_{data}$$
$$D_\theta(x; \sigma) = [c_{skip}(\sigma)x_{norm} + c_{out}(\sigma)F_\theta(c_{in}(\sigma)x_{norm}; c_{noise}(\sigma))] + \mu_{data}$$

*   **기준 통계치 (Measured):**
    *   $\sigma_{data} = 2.0990$
    *   $\mu_{data} = -4.9307$
*   **입력 스케일링 ($c_{in}$):** $1 / \sqrt{\sigma^2 + \sigma_{data}^2}$
*   **출력 스케일링 ($c_{out}$):** $\sigma \cdot \sigma_{data} / \sqrt{\sigma^2 + \sigma_{data}^2}$
*   **스킵 연결 ($c_{skip}$):** $\sigma_{data}^2 / (\sigma^2 + \sigma_{data}^2)$
*   **노이즈 레벨 임베딩 ($c_{noise}$):** $\frac{1}{4} \ln(\sigma)$

### 2.2 학습 손실 함수 (Training Loss)
$$Loss = \lambda(\sigma) \| D_\theta(x+\epsilon; \sigma) - x \|^2$$
$$\lambda(\sigma) = \frac{\sigma^2 + \sigma_{data}^2}{(\sigma \cdot \sigma_{data})^2}$$

*   **학습용 노이즈 샘플링 (Log-normal):**
    *   $P_{mean} = -1.2$
    *   $P_{std} = 1.2$
    *   $\ln(\sigma) \sim N(P_{mean}, P_{std}^2)$

### 2.3 샘플링 하이퍼파라미터 (Inference)
추론 시 Heun's method 및 Stochastic sampling에 사용되는 표준 설정값입니다.

*   **노이즈 스케줄 (Polynomial Schedule):**
    *   $\sigma_{max} = 80.0$ (최대 노이즈 레벨)
    *   $\sigma_{min} = 0.002$ (최종 노이즈 레벨)
    *   $\rho = 7.0$ (스케줄 곡률, EDM 표준값)
    *   $N = 35$ (기본 샘플링 스텝 수)
*   **확률적 샘플링 (Stochasticity - Algorithm 2):**
    *   $S_{churn} = 40.0$ (노이즈 주입 강도)
    *   $S_{tmin} = 0.05$ (Churn 적용 시작점)
    *   $S_{tmax} = 50.0$ (Churn 적용 종료점)
    *   $S_{noise} = 1.003$ (추가 노이즈 스케일)
*   **가이던스 (CFG):**
    *   $w = 1.0$ (기본 Guidance Scale)

---

## 3. 클래스 명세

### 3.1 `KarrasScoreEstimator`
EDM Preconditioning, Centering, 그리고 Classifier-Free Guidance(CFG)를 통합 처리합니다.
- **주요 기능:**
    - 입력 $x$에서 $\mu_{data}$를 감산 (Centering).
    - $\sigma$에 따른 $c$ 계수들 계산.
    - `forward_diffusion`: 가우시안 노이즈 주입 및 학습용 $\sigma$ 샘플링 통합.
    - `reverse_diffusion`: EDM 수식 기반의 디노이징 수행.

### 3.2 `KarrasSampler`
Heun's 2nd order method를 사용하여 노이즈를 제거하는 결정론적/확률적 샘플러입니다.
- **주요 기능:**
    - `get_sigmas()`: Polynomial schedule ($\rho=7$) 생성.
    - `sample()`: Heun step 반복 및 Churn 주입(확률적 모드) 처리.

### 3.3 `KarrasTTSSynthesizer` (Top-level)
Stage 1 모델과 Stage 2 Diffusion을 하나로 묶는 통합 시스템입니다.
- **내부 모듈:**
    - `self.syn_backbone`: `MonotonicTTSSynthesizer` (Stage 1)
    - `self.estimator`: `KarrasScoreEstimator` (Stage 2)
- **Forward (Training):**
    - Stage 1에서 MAS(Hard Path) 특징량 추출.
    - Diffusion Loss 계산용 예측값 반환.
- **Inference:**
    - Stage 1 정렬 예측 -> Stage 2 Diffusion 정제 -> Vocoder 연쇄 실행.

---

## 4. 구현 로드맵

1.  **Phase 0 (준비):** `tts/preprocess/compute_stats.py`를 사용하여 데이터셋 통계 산출 (완료).
2.  **Phase 1 (코어 구현):** EDM 수식 기반의 `score_estimator`, `denoiser_net`, `unet_backbone`, `sampler` 모듈 구현 및 검증 (완료).
3.  **Phase 2 (통합):** `KarrasTTSSynthesizer` 클래스 및 Stage 1(`syn_backbone`) 연동 로직 완성 (완료).
4.  **Phase 3 (학습 준비):** Stage 2 전용 Config 구축 및 `BaseTrainer` 추상화를 통한 훈련 프레임워크 고도화 (완료).
5.  **Phase 4 (학습 구현):** `tts/train/train_second.py` 작성 및 EDM 기반 학습 루프 구축 (완료).
6.  **Phase 5 (검증):** 전체 추론 파이프라인 테스트 및 결과 품질 최적화 (진행 예정).

---

## 6. 남은 과제 (To-Do)

### Phase 5: 최종 검증 및 최적화
- [ ] **U-Net 4의 배수 길이 대응**: `T_mel`이 4의 배수가 아닐 경우 발생할 수 있는 Down/Up sampling 불일치 해결 (Padding 로직 추가).
- [ ] 통합 추론 테스트 실행: `train_second.py`를 통한 실제 학습 시도 및 텐서보드 결과 분석.
- [ ] 하이퍼파라미터 미세 조정 (Learning Rate 스케줄, Churn 파라미터 등).

---

## 5. 변경 이력 (2026-04-03)
- [x] **훈련 프레임워크 고도화**: `BaseTrainer` 추상 클래스 도입으로 Stage 1/2 공통 학습 로직 통합 및 로깅 강화.
- [x] **Stage 2 손실 계산 정규화**: 패딩을 제외한 유효 프레임 기준 평균 손실 계산 적용.
- [x] **EDM 논문 정합성 검토**: Karras et al. (2022)의 EDM 수식(Preconditioning, Weighting, Sampling) 구현 완료 확인.
- [x] **Stage 2 전용 학습 스크립트 완성**: `tts/train/train_second.py` 구현.
