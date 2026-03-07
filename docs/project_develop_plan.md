# CSCI: Cross-Spectral Coherence Imputation
## 모델 상세 설계 문서 (개발용)

---

## 0. 문제 정의 및 목표

### 입력/출력 정의

```
훈련 시:
  입력: X ∈ R^(B × T × N)
        B = 배치 크기
        T = 입력 시퀀스 길이 (예: 96)
        N = 전체 변수 수 (예: 20)
  출력: Ŷ ∈ R^(B × H × N)
        H = 예측 horizon (예: 12)

추론 시:
  입력: X_obs ∈ R^(B × T × K)   K < N (관측 변수만)
        mask ∈ {0,1}^N           관측 여부 마스크
  출력: Ŷ ∈ R^(B × H × N)
```

### 핵심 설계 원칙

```
1. 수치 복원 없음: iFFT 결과를 백본에 넣지 않음
2. 주파수 임베딩 직접 주입: Spectral Projector → 백본 입력
3. 불확실성 전파: σ를 백본 어텐션에 반영
4. 백본 최소 수정: 입력 레이어만 교체, 내부 구조 유지
```

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         CSCI 전체 구조                           │
│                                                                   │
│  입력: X_obs [B×T×K]  +  mask [N]                               │
│           │                                                       │
│    ┌──────▼──────────────────────────────────┐                  │
│    │         Module 1: Spectral Encoder       │                  │
│    │  FFT(X_obs) → Ṽ_obs [B×F×K] (복소수)   │                  │
│    └──────────────────┬──────────────────────┘                  │
│                        │                                          │
│    ┌──────────────────▼──────────────────────┐                  │
│    │    Module 2: Cross-Spectral Estimator    │                  │
│    │  학습된 S [F×N×N] + Ṽ_obs              │                  │
│    │  → Wiener Filter → Ṽ_miss [B×F×(N-K)]  │                  │
│    │  → σ_miss [B×F×(N-K)]  (불확실성)       │                  │
│    └──────────────────┬──────────────────────┘                  │
│                        │                                          │
│    ┌──────────────────▼──────────────────────┐                  │
│    │       Module 3: Spectral Projector       │                  │
│    │  Ṽ_obs  → Linear → e_obs  [B×T×K×d]   │                  │
│    │  Ṽ_miss → Linear → e_miss [B×T×(N-K)×d]│                  │
│    │  σ_miss를 e_miss에 반영                  │                  │
│    └──────────────────┬──────────────────────┘                  │
│                        │                                          │
│    ┌──────────────────▼──────────────────────┐                  │
│    │       Module 4: Backbone (MTGNN)         │                  │
│    │  [e_obs ; e_miss] → 예측값 Ŷ           │                  │
│    └─────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module 1: Spectral Encoder

### 역할
시간 도메인 시계열 → 주파수 도메인 복소수 표현

### 입출력
```
입력: X_obs  ∈ R^(B × T × K)
출력: Ṽ_obs  ∈ C^(B × F × K)   F = T//2 + 1 (단측 스펙트럼)
```

### 구현

```python
import torch
import torch.nn as nn

class SpectralEncoder(nn.Module):
    def __init__(self, T: int, norm: str = 'ortho'):
        super().__init__()
        self.T = T
        self.norm = norm
        self.F = T // 2 + 1  # 단측 주파수 수

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, K] 실수
        return: [B, F, K] 복소수
        """
        # dim=1: 시간 축 방향으로 FFT
        V = torch.fft.rfft(x, dim=1, norm=self.norm)  # [B, F, K]
        return V
```

### 설계 선택 이유
- `rfft` (실수 입력 FFT): 입력이 실수이므로 단측 스펙트럼만 계산 → 메모리 절반
- `norm='ortho'`: 에너지 보존 정규화 → 학습 안정성

---

## 3. Module 2: Cross-Spectral Estimator

### 역할
1. 훈련 시 전체 변수 간 교차 스펙트럼 행렬 S 학습
2. 추론 시 Wiener 필터로 결측 변수 주파수 표현 추정 + 불확실성 계산

### 핵심 수식

**교차 스펙트럼 행렬 (주파수 f에서):**
```
S(f) ∈ C^(N×N)

S_ij(f) = E[Ṽ_i(f) · conj(Ṽ_j(f))]
         ≈ (1/n_batch) Σ Ṽ_i(f) · conj(Ṽ_j(f))
```

**Wiener 필터 추정:**
```
Ṽ_miss(f) = S_miss,obs(f) · [S_obs,obs(f) + λI]^(-1) · Ṽ_obs(f)

notation:
  S_miss,obs: 결측-관측 블록 [(N-K) × K]
  S_obs,obs:  관측-관측 블록 [K × K]
  λ:          정규화 파라미터 (역행렬 안정화)
```

**불확실성 추정 (Wiener 필터의 잔여 분산):**
```
σ_miss(f) = diag(S_miss,miss(f)) - diag(S_miss,obs(f) · [S_obs,obs(f)]^(-1) · S_obs,miss(f))

직관: 전체 분산 - 관측 변수로 설명된 분산 = 설명 못한 불확실성
```

### 구현

```python
class CrossSpectralEstimator(nn.Module):
    def __init__(self, N: int, F: int, lambda_reg: float = 1e-3):
        super().__init__()
        self.N = N
        self.F = F
        self.lambda_reg = lambda_reg

        # 학습 파라미터: 교차 스펙트럼 행렬 (실수부, 허수부 분리)
        # S(f) = S_real(f) + i * S_imag(f), 대칭 구조 강제
        self.S_real = nn.Parameter(torch.eye(N).unsqueeze(0).repeat(F, 1, 1))
        # [F, N, N]
        self.S_imag = nn.Parameter(torch.zeros(F, N, N))
        # [F, N, N]

    def get_S(self) -> torch.Tensor:
        """
        S(f) 복소수 행렬 반환, 에르미트 대칭 강제
        에르미트 대칭: S_ij = conj(S_ji) → 물리적으로 타당한 스펙트럼
        """
        S_real_sym = (self.S_real + self.S_real.transpose(-1, -2)) / 2
        S_imag_antisym = (self.S_imag - self.S_imag.transpose(-1, -2)) / 2
        S = torch.complex(S_real_sym, S_imag_antisym)  # [F, N, N]
        return S

    def forward(
        self,
        V_obs: torch.Tensor,   # [B, F, K] 복소수
        obs_idx: list,         # 관측 변수 인덱스 [K]
        miss_idx: list,        # 결측 변수 인덱스 [N-K]
    ) -> tuple:
        """
        return:
          V_miss_hat: [B, F, N-K] 추정된 결측 주파수 표현
          sigma:      [B, F, N-K] 불확실성 (실수)
        """
        B, F, K = V_obs.shape
        M = len(miss_idx)
        S = self.get_S()  # [F, N, N]

        # 블록 분리
        S_oo = S[:, obs_idx, :][:, :, obs_idx]    # [F, K, K]
        S_mo = S[:, miss_idx, :][:, :, obs_idx]   # [F, M, K]
        S_mm = S[:, miss_idx, :][:, :, miss_idx]  # [F, M, M]

        # 정규화된 역행렬: [S_oo + λI]^(-1)
        reg = self.lambda_reg * torch.eye(K, device=V_obs.device).unsqueeze(0)
        S_oo_reg = S_oo + reg  # [F, K, K]
        S_oo_inv = torch.linalg.inv(S_oo_reg)  # [F, K, K]

        # Wiener 필터: W = S_mo · S_oo_inv → [F, M, K]
        W = torch.bmm(S_mo, S_oo_inv)  # [F, M, K]

        # 결측 주파수 추정: Ṽ_miss = W · Ṽ_obs
        # V_obs: [B, F, K] → [B, F, K, 1]
        V_obs_expand = V_obs.unsqueeze(-1)  # [B, F, K, 1]
        W_expand = W.unsqueeze(0).expand(B, -1, -1, -1)  # [B, F, M, K]
        V_miss_hat = torch.bmm(
            W_expand.view(B*F, M, K),
            V_obs_expand.view(B*F, K, 1)
        ).view(B, F, M)  # [B, F, M]

        # 불확실성: σ = diag(S_mm) - diag(W · S_om)
        S_om = S_mo.conj().transpose(-1, -2)  # [F, K, M]
        WS_om = torch.bmm(W, S_om)  # [F, M, M]
        # 대각 성분만 추출
        diag_S_mm = torch.diagonal(S_mm, dim1=-2, dim2=-1)   # [F, M]
        diag_WS_om = torch.diagonal(WS_om, dim1=-2, dim2=-1)  # [F, M]
        sigma = (diag_S_mm - diag_WS_om).real  # [F, M] 실수
        sigma = sigma.unsqueeze(0).expand(B, -1, -1)  # [B, F, M]
        sigma = torch.clamp(sigma, min=0)  # 음수 방지

        return V_miss_hat, sigma

    def update_S(self, V_full: torch.Tensor):
        """
        훈련 시 S 업데이트용 손실 계산 (EMA 방식으로 대체 가능)
        V_full: [B, F, N] 전체 변수 주파수 표현
        """
        B, F, N = V_full.shape
        # 외적: V_i · conj(V_j)
        S_batch = torch.bmm(
            V_full.view(B*F, N, 1),
            V_full.conj().view(B*F, 1, N)
        ).view(B, F, N, N)
        return S_batch.mean(0)  # [F, N, N] 배치 평균
```

---

## 4. Module 3: Spectral Projector

### 역할
주파수 도메인 복소수 표현 → 백본이 이해하는 임베딩 벡터
관측 변수(시간 도메인)와 결측 변수(주파수 도메인)를 동일한 임베딩 공간으로 변환

### 핵심 아이디어

```
관측 변수 경로:
  x_obs [B, T, K] → 기존 Linear 임베딩 → e_obs [B, T, K, d]

결측 변수 경로:
  V_miss_hat [B, F, M] (복소수)
  → 실수부/허수부 분리: [B, F, M, 2]
  → σ 결합: [B, F, M, 3]  ← 불확실성 함께 전달
  → Linear + 정규화 → e_miss [B, T', M, d]
  (T' = T로 맞추기 위해 주파수 차원 보간)
```

### 구현

```python
class SpectralProjector(nn.Module):
    def __init__(self, F: int, T: int, d_model: int):
        """
        F: 주파수 수 (T//2 + 1)
        T: 시간 시퀀스 길이
        d_model: 임베딩 차원
        """
        super().__init__()
        self.F = F
        self.T = T
        self.d_model = d_model

        # 주파수 특성 → 임베딩 변환
        # 입력: [실수부, 허수부, σ] = 3채널
        self.freq_to_emb = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 주파수 차원(F) → 시간 차원(T) 변환
        # F와 T 차원이 다를 수 있으므로 선형 보간
        self.freq_to_time = nn.Linear(F, T)

        # 불확실성 스케일링 파라미터 (학습 가능)
        self.sigma_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        V_miss: torch.Tensor,   # [B, F, M] 복소수
        sigma: torch.Tensor,    # [B, F, M] 실수
    ) -> tuple:
        """
        return:
          e_miss: [B, T, M, d] 임베딩
          attn_bias: [B, M] 어텐션 바이어스 (σ 기반)
        """
        B, F, M = V_miss.shape

        # 실수부/허수부 분리 + σ 결합
        real_part = V_miss.real  # [B, F, M]
        imag_part = V_miss.imag  # [B, F, M]
        sigma_scaled = sigma * self.sigma_scale  # [B, F, M]

        # 3채널로 합치기: [B, F, M, 3]
        freq_features = torch.stack(
            [real_part, imag_part, sigma_scaled], dim=-1
        )

        # 주파수 특성 → 임베딩: [B, F, M, d]
        e_freq = self.freq_to_emb(freq_features)

        # 주파수 차원(F) → 시간 차원(T): [B, T, M, d]
        e_freq = e_freq.permute(0, 2, 3, 1)  # [B, M, d, F]
        e_time = self.freq_to_time(e_freq)    # [B, M, d, T]
        e_miss = e_time.permute(0, 3, 1, 2)  # [B, T, M, d]

        # 어텐션 바이어스: 주파수 평균 σ → [B, M]
        # 값이 클수록 어텐션에서 덜 주목
        attn_bias = sigma.mean(dim=1)  # [B, M] 주파수 평균

        return e_miss, attn_bias
```

---

## 5. Module 4: 수정된 백본 입력 레이어

### MTGNN 기준 수정 범위

```
MTGNN 원본 입력 레이어:
  x [B, T, N] → Linear → h [B, T, N, d]

수정 후:
  x_obs [B, T, K]       → Linear         → e_obs  [B, T, K, d]
  V_miss_hat [B, F, M]  → SpectralProj   → e_miss [B, T, M, d]
  index_restore()로 원래 N개 순서로 재배열
  → h [B, T, N, d]
```

### 구현

```python
class CSCIInputLayer(nn.Module):
    def __init__(
        self,
        N: int,          # 전체 변수 수
        F: int,          # 주파수 수
        T: int,          # 시간 길이
        d_model: int,    # 임베딩 차원
    ):
        super().__init__()
        self.N = N

        # 관측 변수 경로: 시간 도메인 → 임베딩 (기존과 동일)
        self.obs_embedding = nn.Linear(1, d_model)

        # 결측 변수 경로: 주파수 도메인 → 임베딩
        self.spectral_projector = SpectralProjector(F, T, d_model)

    def forward(
        self,
        x_obs: torch.Tensor,       # [B, T, K]
        V_miss_hat: torch.Tensor,  # [B, F, M] 복소수
        sigma: torch.Tensor,       # [B, F, M]
        obs_idx: list,             # 관측 변수 인덱스
        miss_idx: list,            # 결측 변수 인덱스
    ) -> tuple:
        """
        return:
          h: [B, T, N, d] 전체 임베딩 (관측 + 결측 합쳐진)
          attn_bias: [N] 어텐션 바이어스
        """
        B, T, K = x_obs.shape
        d = self.obs_embedding.out_features

        # 관측 변수 임베딩
        e_obs = self.obs_embedding(x_obs.unsqueeze(-1))  # [B, T, K, d]

        # 결측 변수 임베딩
        e_miss, attn_bias_miss = self.spectral_projector(V_miss_hat, sigma)
        # e_miss: [B, T, M, d]

        # 원래 N개 순서로 재조립
        h = torch.zeros(B, T, self.N, d, device=x_obs.device)
        h[:, :, obs_idx, :] = e_obs
        h[:, :, miss_idx, :] = e_miss

        # 전체 어텐션 바이어스 (관측=0, 결측=σ)
        attn_bias = torch.zeros(B, self.N, device=x_obs.device)
        attn_bias[:, miss_idx] = attn_bias_miss

        return h, attn_bias
```

### 어텐션 바이어스 적용 (백본 내부)

```python
# MTGNN 또는 Transformer의 어텐션 계산 시:
# 기존: attn_weight = softmax(Q·K^T / sqrt(d))
# 수정: attn_weight = softmax(Q·K^T / sqrt(d) - attn_bias)
#                                                    ↑
#                              σ가 클수록 어텐션 점수 낮아짐

def modified_attention(Q, K, V, attn_bias=None):
    scores = torch.bmm(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    if attn_bias is not None:
        scores = scores - attn_bias.unsqueeze(1)  # 브로드캐스트
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

---

## 6. 훈련 전략

### 6-1. 마스킹 커리큘럼 (핵심)

85% 결측까지 대응하려면 점진적 학습이 필수입니다.

```
Epoch 1-30:   결측률 0~30%   (기초 관계 학습)
Epoch 31-60:  결측률 0~60%   (중간 결맞음 학습)
Epoch 61-90:  결측률 0~85%   (극단 상황 대응)
Epoch 91-:    결측률 랜덤 균등 샘플링

구현:
def get_mask_ratio(epoch, max_epoch=120, max_ratio=0.85):
    if epoch < max_epoch * 0.25:
        return 0.30
    elif epoch < max_epoch * 0.50:
        return 0.60
    elif epoch < max_epoch * 0.75:
        return 0.85
    else:
        # 균등 샘플링
        return torch.FloatTensor(1).uniform_(0, max_ratio).item()
```

### 6-2. 계층적 전파 (85% 대응의 핵심)

관측 변수와 직접 결맞음이 낮은 변수는 중간 변수를 거쳐 복원합니다.

```
Round 1: 관측 변수 → 결맞음 높은 결측 변수 복원
Round 2: 관측 + Round1 복원 → 나머지 결측 변수 복원

구현:
def hierarchical_propagation(V_obs, obs_idx, miss_idx, S, n_rounds=2):
    current_obs = V_obs.clone()
    current_obs_idx = list(obs_idx)
    remaining_miss = list(miss_idx)

    for round_i in range(n_rounds):
        # 현재 관측 변수들과 결맞음 계산
        coherence = compute_coherence(S, current_obs_idx, remaining_miss)
        # coherence: [F, len(remaining_miss)]

        # 결맞음 임계값 이상인 변수 선택
        threshold = 0.3 - round_i * 0.1  # 라운드마다 임계값 완화
        high_coh_mask = coherence.mean(0) > threshold  # [len(remaining_miss)]

        if not high_coh_mask.any():
            break

        # 결맞음 높은 변수 복원
        easy_miss = [remaining_miss[i] for i in high_coh_mask.nonzero()]
        V_easy = wiener_filter(current_obs, current_obs_idx, easy_miss, S)

        # 복원된 변수를 관측에 추가 (σ로 가중)
        current_obs = torch.cat([current_obs, V_easy], dim=-1)
        current_obs_idx.extend(easy_miss)
        remaining_miss = [i for i in remaining_miss if i not in easy_miss]

    return current_obs, current_obs_idx, remaining_miss
```

### 6-3. 손실 함수

```python
class CSCILoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # 예측 손실 가중치
        self.beta  = beta   # 스펙트럼 정렬 손실 가중치
        self.gamma = gamma  # 불확실성 보정 손실 가중치

    def forecast_loss(self, y_pred, y_true):
        """MSE 예측 손실"""
        return nn.MSELoss()(y_pred, y_true)

    def spectral_alignment_loss(self, V_miss_hat, V_miss_true):
        """
        추정된 주파수 표현이 실제와 얼마나 가까운가
        (훈련 시에만 계산 가능 - y_miss를 알 때)
        """
        # 진폭 정렬
        amp_loss = nn.MSELoss()(V_miss_hat.abs(), V_miss_true.abs())
        # 위상 정렬 (코사인 유사도)
        phase_pred = V_miss_hat / (V_miss_hat.abs() + 1e-8)
        phase_true = V_miss_true / (V_miss_true.abs() + 1e-8)
        phase_loss = 1 - (phase_pred * phase_true.conj()).real.mean()
        return amp_loss + phase_loss

    def uncertainty_calibration_loss(self, sigma, V_miss_hat, V_miss_true):
        """
        σ가 실제 오차와 비례하는지 보정
        실제 오차 크면 σ도 커야 함
        """
        actual_error = (V_miss_hat - V_miss_true).abs().mean(dim=1)  # [B, M]
        sigma_mean = sigma.mean(dim=1)  # [B, M]
        return nn.MSELoss()(sigma_mean, actual_error.detach())

    def forward(self, y_pred, y_true, V_miss_hat, V_miss_true, sigma):
        L_fc  = self.forecast_loss(y_pred, y_true)
        L_sp  = self.spectral_alignment_loss(V_miss_hat, V_miss_true)
        L_uc  = self.uncertainty_calibration_loss(sigma, V_miss_hat, V_miss_true)

        total = self.alpha * L_fc + self.beta * L_sp + self.gamma * L_uc
        return total, {'forecast': L_fc, 'spectral': L_sp, 'uncertainty': L_uc}
```

---

## 7. 전체 모델 클래스

```python
class CSCI(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        N        = config['N']         # 전체 변수 수
        T        = config['T']         # 입력 시퀀스 길이
        H        = config['H']         # 예측 horizon
        d_model  = config['d_model']   # 임베딩 차원
        F        = T // 2 + 1          # 주파수 수
        lambda_r = config.get('lambda_reg', 1e-3)
        n_rounds = config.get('n_rounds', 2)

        self.N = N
        self.T = T
        self.n_rounds = n_rounds

        # Module 1: Spectral Encoder
        self.spectral_encoder = SpectralEncoder(T)

        # Module 2: Cross-Spectral Estimator
        self.cs_estimator = CrossSpectralEstimator(N, F, lambda_r)

        # Module 3+4: CSCI 입력 레이어
        self.input_layer = CSCIInputLayer(N, F, T, d_model)

        # 예측 백본 (MTGNN 또는 다른 백본)
        # 입력 레이어만 교체했으므로 내부는 그대로
        self.backbone = build_backbone(config)  # MTGNN 등

    def forward(
        self,
        x_obs: torch.Tensor,  # [B, T, K]
        obs_idx: list,
        miss_idx: list,
    ) -> tuple:

        # Step 1: FFT
        V_obs = self.spectral_encoder(x_obs)  # [B, F, K]

        # Step 2: 계층적 전파 + Wiener 필터
        V_miss_hat, sigma = self.cs_estimator(V_obs, obs_idx, miss_idx)
        # V_miss_hat: [B, F, M], sigma: [B, F, M]

        # Step 3: Spectral Projector → 임베딩 통합
        h, attn_bias = self.input_layer(
            x_obs, V_miss_hat, sigma, obs_idx, miss_idx
        )  # h: [B, T, N, d]

        # Step 4: 백본 예측
        y_pred = self.backbone(h, attn_bias)  # [B, H, N]

        return y_pred, V_miss_hat, sigma
```

---

## 8. 하이퍼파라미터 설계 가이드

| 파라미터 | 추천값 | 설명 |
|---|---|---|
| `d_model` | 64 또는 128 | 임베딩 차원 |
| `lambda_reg` | 1e-3 ~ 1e-2 | Wiener 필터 정규화 |
| `n_rounds` | 2 | 계층적 전파 라운드 수 |
| `alpha` (손실) | 0.6 | 예측 손실 비중 |
| `beta` (손실) | 0.3 | 스펙트럼 정렬 비중 |
| `gamma` (손실) | 0.1 | 불확실성 보정 비중 |
| 커리큘럼 단계 | 3단계 | 30% → 60% → 85% |
| `coherence_threshold` | 0.3 → 0.2 | 계층 전파 임계값 |

---

## 9. 기존 모델 대비 포지셔닝

```
방법          | 수치복원 | iFFT | 불확실성 | 백본재활용 | 위상활용 범위
-------------|--------|------|---------|----------|----------
VIDA         | O      | O    | X       | O        | 변수 내부
GIMCC        | O      | O    | X       | O        | 없음(가속용)
GinAR        | X      | X    | X       | X        | 없음
CSCI (제안)  | X      | X    | O       | O (입력층만 수정) | 변수 간 (Cross)
```

---

## 10. 개발 체크리스트

### Phase 1: 단위 테스트
- [ ] SpectralEncoder: FFT/iFFT 가역성 확인
- [ ] CrossSpectralEstimator: S 행렬 에르미트 대칭 확인
- [ ] Wiener 필터: 완전 관측 시 완벽 복원 확인 (K=N)
- [ ] SpectralProjector: 출력 차원 [B, T, M, d] 확인
- [ ] CSCIInputLayer: 변수 인덱스 재조립 정확성 확인

### Phase 2: 통합 테스트
- [ ] 결측률 0%, 30%, 60%, 85%에서 순전파 통과
- [ ] 역전파 그래디언트 흐름 확인
- [ ] 손실 함수 각 항목 수치 확인

### Phase 3: 실험
- [ ] 베이스라인: 단순 0 채움 + MTGNN
- [ ] 비교군 1: VIDA
- [ ] 비교군 2: GIMCC
- [ ] 제안 모델: CSCI (Spectral Projector 포함)
- [ ] Ablation: Spectral Projector 없이 iFFT 사용 버전과 비교