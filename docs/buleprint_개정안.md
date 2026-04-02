# CVFA Blueprint: 국내 학회 초록 + 포스터 제출용

> 작성일: 2026-04-02
> 원래 이름: CSCI (Cross-Spectral Coherence Imputation)
> 제안 이름: **CVFA** (Cross-Variate Frequency Attention for Variable Subset Forecasting)

---

## 1. 핵심 한 줄 요약

> **주파수 도메인에서 학습 가능한 변수 간 상호작용 행렬을 통해,
> 관측되지 않은 변수의 시계열을 closed-form attention으로 복원하고 예측한다.**

---

## 2. 문제 정의: Variable Subset Forecasting (VSF)

### 2.1 무엇이 문제인가

다변량 시계열 예측에서, 훈련 시에는 N개 변수 모두 관측 가능하지만
추론 시에는 일부(K < N)만 관측 가능한 상황.

```
훈련: X ∈ R^{B × T × N}  →  Ŷ ∈ R^{B × H × N}     (모든 변수 관측)
추론: X_obs ∈ R^{B × T × K}  →  Ŷ ∈ R^{B × H × N}  (K개만 관측, M = N-K개 결측)
```

### 2.2 기존 접근의 한계

기존 VSF 방법들(FDW, VIDA, GinAR 등)은 **시간 도메인**에서 결측 변수를 복원하거나,
결측에 강건한 표현을 학습한다. 이 접근들의 공통 약점:

- **2단계 오차 증폭**: 먼저 결측을 복원(impute)한 뒤 예측하면, 복원 오차가 예측 오차로 전이됨
- **변수 간 주파수 관계 무시**: 시간 도메인에서의 상관만 활용하여, 변수 간 주기적 동조(coherence)를 포착하지 못함

### 2.3 우리의 질문

> "변수 간 관계를 **주파수 도메인**에서 모델링하면,
> 결측 변수를 더 정확하게 복원할 수 있지 않을까?"

---

## 3. 제안 방법: CVFA

### 3.1 전체 파이프라인

```
x_obs [B,T,K]
    │
    ▼
┌─────────────────────┐
│  ① Frequency        │  rFFT
│     Tokenizer       │  x_obs → V_obs ∈ ℂ^{B×F×K}
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  ② Cross-Variate Frequency Attention   │  ★ 핵심 모듈
│                                         │
│  학습 가능한 상호작용 행렬:             │
│    S(f) = U(f)·U(f)^H + diag(d(f))    │
│                                         │
│  Closed-form attention:                 │
│    W(f) = S_mo(f) · [S_oo(f) + λI]⁻¹  │
│    V̂_miss = W(f) · V_obs(f)            │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  ③ Inverse FFT      │  [V_obs ; V̂_miss] → iFFT → x̂_full ∈ R^{B×T×N}
│     Reconstruction   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ④ Forecast         │  기존 백본(MTGNN 등) 수정 없이 사용
│     Backbone        │  x̂_full → Ŷ ∈ R^{B×H×N}
└─────────────────────┘
```

### 3.2 핵심 모듈 상세: Cross-Variate Frequency Attention

#### 상호작용 행렬 S(f)

```
S(f) = U(f) · U(f)^H + diag(d(f))

- U(f) ∈ ℂ^{N×r}: 주파수 f에서의 저랭크 변수 간 상호작용 패턴
- d(f) ∈ R₊^N: 변수별 잔차 분산 (softplus로 양수 보장)
- r ≪ N: 저랭크로 파라미터 효율성 확보
- Hermitian PSD 자동 보장: S = UU^H + diag(d) 구조
```

#### 이 행렬의 해석 (두 가지 읽기)

| 관점 | S(f)의 의미 | W(f)의 의미 |
|------|------------|------------|
| 신호처리 | 교차 스펙트럼 밀도 행렬 (CSD) | Wiener filter (MMSE 추정) |
| **ML (채택)** | **변수 간 주파수별 상호작용 행렬** | **Closed-form cross-variate attention** |

두 읽기 모두 수학적으로 동일하지만, ML 관점이 기존 시계열 문헌과 자연스럽게 연결된다.

#### 왜 "attention"이라 부를 수 있나

일반적인 attention:
```
Attn(Q, K, V) = softmax(Q·K^T / √d) · V
```

CVFA의 구조:
```
CVFA(miss, obs, V_obs) = S_mo · S_oo⁻¹ · V_obs
                          ↑         ↑
                     query-key    value
                     interaction
```

차이점: softmax 대신 행렬 역원(S_oo⁻¹)을 사용하여 **closed-form 해**를 얻음.
장점: 학습 불안정성이 적고, MMSE 최적성이 수학적으로 보장됨.

### 3.3 학습 목표

```
L_total = α · L_forecast + β · L_spectral

L_forecast:  MAE(Ŷ, Y)                          — 예측 정확도
L_spectral:  MSE(|V̂_miss|, |V_miss|)            — 주파수 진폭 정합
           + (1 - cos_sim(phase(V̂), phase(V)))  — 주파수 위상 정합
```

σ (불확실성)와 L_uncertainty는 **제거**.
→ S(f) 학습 안정화 후 ablation으로 재검토.

### 3.4 설계 원칙 (기존 대비 변경)

| 원칙 | 기존 CSCI | CVFA (제안) |
|------|----------|-------------|
| 복원 방식 | 주파수 임베딩 직접 주입 (no iFFT) | **iFFT로 시계열 원복** |
| 불확실성 | σ → confidence gating | **제거 (모니터링만)** |
| 백본 수정 | 입력층 교체 (d_model channels) | **수정 없음 (1ch 그대로)** |
| Loss 구성 | 3-term (fc + spectral + uncertainty) | **2-term (fc + spectral)** |
| 모듈 수 | 4 (Enc + Est + Proj + Head) | **3 (Tok + CVFA + iFFT)** |

---

## 4. 이론적 배경과 관련 연구

### 4.1 주파수 도메인 시계열 학습 — "왜 FFT인가"

CVFA의 첫 번째 전제: *시계열의 주파수 표현이 학습에 더 효과적이다*

| 논문 | 학회 | CVFA와의 연결 |
|------|------|--------------|
| **FreTS** (Yi et al.) | NeurIPS 2023 | FFT → 주파수 도메인에서 채널 간 MLP. **CVFA의 가장 직접적 선행**: 변수 간 상호작용을 주파수 도메인에서 학습한다는 핵심 아이디어가 동일. FreTS는 MLP, CVFA는 학습 가능한 행렬의 closed-form 해를 사용하는 것이 차이점 |
| **FITS** (Xu et al.) | ICLR 2024 (Spotlight) | rFFT → complex-valued linear layer → iFFT. 10k 파라미터만으로 SOTA급 성능. "주파수 도메인 조작만으로 충분하다"는 CVFA의 전제를 실증적으로 뒷받침 |
| **FilterNet** (Yi et al.) | NeurIPS 2024 | 학습 가능한 주파수 필터로 시계열 패턴 추출. CVFA의 W(f) = S_mo·S_oo⁻¹은 본질적으로 "학습된 cross-variate frequency filter". FilterNet의 single-variate filter를 multi-variate로 확장한 것으로 위치 가능 |
| **FEDformer** (Zhou et al.) | ICML 2022 | Fourier 기저에서 attention 수행. 주파수 도메인 attention의 효과를 입증한 선행 |
| **FreDF** (Wang et al.) | ICLR 2025 | 주파수 도메인 학습 목표가 label correlation을 제거하여 추정 편향을 줄임. 주파수 도메인에서의 loss (L_spectral)의 이론적 정당성 |

**인용 전략**: FreTS를 주요 비교 대상으로, FITS와 FilterNet을 방법론적 근거로, FEDformer와 FreDF를 일반적 배경으로 인용.

### 4.2 변수 간 상호작용 모델링 — "왜 cross-variate인가"

CVFA의 두 번째 전제: *변수 간 관계를 명시적으로 모델링해야 결측에 대응할 수 있다*

| 논문 | 학회 | CVFA와의 연결 |
|------|------|--------------|
| **FourierGNN** (Yi et al.) | NeurIPS 2023 | "Hypervariate graph"에서 Fourier 공간의 행렬 곱으로 변수 간 상호작용 모델링. CVFA의 S(f)·V_obs 연산과 수학적으로 가장 유사한 구조 |
| **StemGNN** (Cao et al.) | NeurIPS 2020 | GFT(변수 간) + DFT(시간)를 동시에 spectral domain에서 학습. "주파수 도메인에서 변수 간 관계를 학습한다"는 방향의 원조 |
| **iTransformer** (Liu et al.) | ICLR 2024 | variate dimension에 attention 적용 (inverted Transformer). CVFA도 variate 축에 attention하지만 주파수 도메인에서 수행 |
| **CrossFormer** (Zhang & Yan) | ICLR 2023 | cross-variate attention을 명시적으로 도입. 시간 도메인에서의 cross-variate attention이 CVFA의 시간 도메인 대응물 |

**인용 전략**: "기존 cross-variate 모델링은 시간 도메인에서 수행(iTransformer, CrossFormer). 주파수 도메인에서의 시도(StemGNN, FourierGNN)는 예측에만 활용. CVFA는 이를 **결측 변수 복원**에 적용한 첫 시도"로 포지셔닝.

### 4.3 저랭크 상호작용 행렬 — "왜 UU^H + diag(d)인가"

CVFA의 S(f) = UU^H + diag(d)는 Factor Analysis / PPCA의 주파수 도메인 확장.

| 논문/교재 | 출처 | CVFA와의 연결 |
|-----------|------|--------------|
| **PPCA** (Tipping & Bishop) | JRSS-B 1999 | 공분산 행렬을 WW^T + σ²I로 분해. CVFA의 S(f) = UU^H + diag(d)와 구조적으로 동일. 이 분해의 통계적 정당성(MLE 해)을 제공 |
| **Cao, Lindquist & Picci** | IEEE TAC 2023 | 저랭크 스펙트럼 밀도 행렬을 가진 벡터 프로세스의 식별 이론. CVFA의 S(f)가 저랭크인 것의 이론적 근거: 실세계 다변량 시계열은 소수의 잠재 요인으로 설명됨 |
| **ImputeFormer** (Nie et al.) | KDD 2024 | Low-rank inductive bias를 Transformer에 도입하여 시공간 결측 복원. "저랭크 구조가 결측 환경에서 일반화에 유리하다"는 실증 근거 |

**인용 전략**: PPCA를 수학적 기원으로, Cao et al.을 스펙트럼 도메인 확장으로, ImputeFormer를 ML적 실증으로 인용.

### 4.4 결측 시계열 분석 — "VSF 기존 연구"

| 논문 | 학회 | CVFA와의 관계 |
|------|------|--------------|
| **FDW** (Chauhan et al.) | KDD 2022 | VSF 문제를 처음 정의. 비관측 변수 처리를 위해 weighting 기반 접근 |
| **VIDA** (이름미정) | KDD 2025 | 2단계 (impute → forecast) VSF. CVFA의 직접 비교 대상 |
| **GinAR** (이름미정) | KDD 2024 | 최초의 end-to-end VSF. Graph 기반. CVFA와 다른 E2E 접근 |
| **GRIN** (Cini et al.) | ICLR 2022 | GNN 기반 다변량 시계열 결측 복원. 공간적 관계 활용한 imputation의 대표 연구 |
| **FGTI** (Yang et al.) | NeurIPS 2024 | 주파수 인식 생성 모델로 시계열 결측 복원. 주파수 도메인 결측 복원의 최신 연구 |

**인용 전략**: FDW/VIDA/GinAR는 문제 정의와 비교 대상으로, GRIN/FGTI는 방법론적 관련 연구로 인용.

### 4.5 고전적 기초 (간략 인용)

| 출처 | 역할 |
|------|------|
| **Brillinger (2001)** *Time Series: Data Analysis and Theory* | V_miss = S_mo·S_oo⁻¹·V_obs 공식의 수학적 출처. 교차 스펙트럼 조건부 추정 이론 |
| **DWDN** (Dong et al., NeurIPS 2020) | Wiener deconvolution을 learnable feature space에서 수행. "고전 필터 + 학습 가능 파라미터" 프레임워크의 선례 |
| **Kendall & Gal (2017)** NeurIPS | Heteroscedastic aleatoric uncertainty 학습 (향후 σ 재활성화 시 인용) |

---

## 5. CVFA의 포지셔닝

### 5.1 기존 연구 대비 위치

```
                        시간 도메인          주파수 도메인
                     ┌──────────────┐    ┌──────────────┐
  예측만            │ iTransformer  │    │ FreTS        │
  (전체 변수)       │ PatchTST      │    │ FITS         │
                     │ CrossFormer   │    │ FilterNet    │
                     └──────────────┘    └──────────────┘

                     ┌──────────────┐    ┌──────────────┐
  결측 복원         │ GRIN          │    │ FGTI         │
  (imputation)      │ BRITS         │    │              │
                     │ SAITS         │    │              │
                     └──────────────┘    └──────────────┘

                     ┌──────────────┐    ┌──────────────┐
  VSF               │ FDW, VIDA     │    │              │
  (복원+예측)       │ GinAR, GIMCC  │    │  ★ CVFA ★    │
                     └──────────────┘    └──────────────┘
```

**CVFA는 "주파수 도메인 × VSF" 교차점을 차지하는 최초의 방법.**

### 5.2 novelty 정리

1. **주파수 도메인 cross-variate attention**: 변수 간 관계를 주파수별로 분해하여 학습
   (기존: 시간 도메인에서 전체 주파수를 뭉뚱그려 처리)
2. **Closed-form 최적 추정**: softmax attention과 달리, MMSE 최적해가 보장됨
3. **저랭크 상호작용 행렬**: N²F 파라미터를 2NrF로 압축하면서 PSD 자동 보장
4. **주파수 도메인 VSF**: 이 조합이 처음

---

## 6. 초록 (안)

### 제목
주파수 도메인 변수 간 어텐션을 활용한 부분 관측 다변량 시계열 예측

### 영문 제목
Cross-Variate Frequency Attention for Variable Subset Forecasting

### 본문 (300자 내외)

다변량 시계열 예측에서 추론 시 일부 변수만 관측 가능한 Variable Subset
Forecasting(VSF) 문제는 실용적으로 중요하지만, 기존 접근법들은 시간 도메인에서의
결측 복원에 의존하여 변수 간 주파수 관계를 활용하지 못한다. 본 연구에서는 주파수
도메인에서 변수 간 상호작용을 모델링하는 Cross-Variate Frequency Attention(CVFA)을
제안한다. CVFA는 학습 가능한 저랭크 상호작용 행렬 S(f) = UU^H + diag(d)를 각
주파수 대역별로 구성하고, 이를 기반으로 관측 변수의 스펙트럼으로부터 결측 변수의
스펙트럼을 closed-form attention으로 추정한다. 추정된 스펙트럼은 역 FFT를 통해
시계열로 복원되어 기존 예측 백본에 직접 입력된다. 제안 방법은 최근 주파수 도메인
시계열 학습 연구(FreTS, FITS, FilterNet)의 성공을 VSF 문제로 확장하며, 교차
스펙트럼 조건부 추정 이론에 기반한 수학적 최적성을 갖는다. 교통 및 에너지 데이터셋
에서의 실험을 통해 기존 VSF 방법 대비 성능을 비교 분석한다.

**키워드**: 다변량 시계열, 주파수 도메인 학습, 변수 부분 관측, 어텐션 메커니즘

---

## 7. 포스터 구성안

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│  [제목]  주파수 도메인 변수 간 어텐션을 활용한                    │
│          부분 관측 다변량 시계열 예측                               │
│                                                                   │
├───────────────────┬─────────────────┬───────────────────────────┤
│                   │                 │                             │
│  ① 문제 정의     │  ② 왜 주파수?  │  ③ CVFA 아키텍처           │
│                   │                 │                             │
│  - VSF 문제 설명  │  - 시간 vs 주파수│  - 전체 파이프라인 그림    │
│  - 추론 시 변수   │    비교 그림    │  - S(f) → W(f) → V̂_miss  │
│    부분 관측 상황  │  - FreTS/FITS  │    → iFFT → backbone       │
│                   │    성공 사례    │                             │
├───────────────────┴─────────────────┤                             │
│                                     │  ④ 핵심 수식               │
│  ⑤ Cross-Variate Frequency         │                             │
│     Attention 메커니즘 시각화       │  S(f) = UU^H + diag(d)    │
│                                     │  W(f) = S_mo · S_oo⁻¹     │
│  (앞서 그린 attention weight        │  V̂_miss = W · V_obs       │
│   두께 다이어그램 활용)             │                             │
│                                     │  "두 가지 읽기" 박스       │
├─────────────────────────────────────┴───────────────────────────┤
│                                                                   │
│  ⑥ 실험 결과                        ⑦ 관련 연구 포지셔닝        │
│                                                                   │
│  - Main Table (METR-LA, Solar 등)   - 2×3 매트릭스 그림          │
│  - vs FDW, VIDA, GinAR              - (시간/주파수 × 예측/복원/VSF)│
│  - Missing rate 별 성능 곡선         - CVFA = 주파수×VSF 교차점   │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ⑧ 결론 + 향후 연구                                              │
│                                                                   │
│  - 주파수 도메인 VSF의 가능성 확인                                │
│  - 향후: 불확실성(σ) 활용, hierarchical propagation 등            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 실험 계획

### 8.1 데이터셋
- **METR-LA** (교통, N=207): VIDA와 동일 세팅으로 비교
- **Solar-Energy** (에너지, N=137): VSF 표준 벤치마크
- (선택) PEMS-BAY, ECG

### 8.2 비교 대상
- **FDW** (KDD 2022): VSF 원조
- **VIDA** (KDD 2025): 현 SOTA, 2단계 방식
- **GinAR** (KDD 2024): E2E VSF
- **Zero-fill baseline**: 결측=0으로 처리 후 예측

### 8.3 Missing rate
- 30%, 50%, 70% (VIDA 세팅 따름)

### 8.4 핵심 ablation
- **S(f) rank**: r = {4, 8, 16, 32, N//10}
- **λ_reg**: {0.01, 0.1, 0.5}
- **L_spectral 유무**: α만 vs α+β
- (향후) **σ 활용 유무**: iFFT 전에 σ로 주파수 가중

### 8.5 평가 지표
- MAE, RMSE (VIDA와 동일)
- (선택) 복원 정확도: 주파수 진폭/위상 오차

---

## 9. 관련 연구 BibTeX

```bibtex
% === 주파수 도메인 시계열 학습 ===

@inproceedings{yi2023frets,
  title={Frequency-domain {MLP}s are More Effective Learners in Time Series Forecasting},
  author={Yi, Kun and Zhang, Qi and Fan, Wei and Wang, Shoujin and Wang, Pengyang and He, Hui and An, Ning and Lian, Defu and Cao, Longbing and Niu, Zhendong},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{xu2024fits,
  title={{FITS}: Modeling Time Series with 10k Parameters},
  author={Xu, Zhijian and Zeng, Ailing and Xu, Qiang},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{yi2024filternet,
  title={FilterNet: Harnessing Frequency Filters for Time Series Forecasting},
  author={Yi, Kun and Zhang, Qi and Fan, Wei and He, Hui and Hu, Liang and Wang, Pengyang and An, Ning},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{zhou2022fedformer,
  title={{FEDformer}: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting},
  author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle={ICML},
  year={2022}
}

@inproceedings{wang2025fredf,
  title={Fre{DF}: Learning to Forecast in the Frequency Domain},
  author={Wang, Hao and Pan, Licheng and Chen, Zhichao and Yang, Degui and Zhang, Sen and Yang, Yifei and Liu, Xinggao and Li, Haoxuan and Tao, Dacheng},
  booktitle={ICLR},
  year={2025}
}

% === 변수 간 상호작용 모델링 ===

@inproceedings{yi2023fouriergnn,
  title={{FourierGNN}: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective},
  author={Yi, Kun and Zhang, Qi and Fan, Wei and He, Hui and Hu, Liang and Wang, Pengyang and An, Ning and Cao, Longbing and Niu, Zhendong},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{cao2020stemgnn,
  title={Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting},
  author={Cao, Defu and Wang, Yujing and Duan, Juanyong and Zhang, Ce and Zhu, Xia and Huang, Congrui and Tong, Yunhai and Xu, Bixiong and Bai, Jing and Tong, Jie and Zhang, Qi},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{liu2024itransformer,
  title={i{T}ransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{zhang2023crossformer,
  title={Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting},
  author={Zhang, Yunhao and Yan, Junchi},
  booktitle={ICLR},
  year={2023}
}

% === 저랭크 모델링 ===

@article{tipping1999ppca,
  title={Probabilistic Principal Component Analysis},
  author={Tipping, Michael E and Bishop, Christopher M},
  journal={Journal of the Royal Statistical Society: Series B},
  volume={61},
  number={3},
  pages={611--622},
  year={1999}
}

@article{cao2023lowrank,
  title={Modeling of Low Rank Time Series},
  author={Cao, Wenqi and Lindquist, Anders and Picci, Giorgio},
  journal={IEEE Transactions on Automatic Control},
  volume={68},
  number={12},
  pages={7270--7285},
  year={2023}
}

@inproceedings{nie2024imputeformer,
  title={ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation},
  author={Nie, Tong and Qin, Guoyang and Ma, Wei and Mei, Yuewen and Sun, Jian},
  booktitle={KDD},
  year={2024}
}

% === VSF 기존 연구 ===

@inproceedings{chauhan2022fdw,
  title={FDW: A Deep Learning Approach for Forecasting with Dynamic Variable Subsets},
  author={Chauhan, Aditya and others},
  booktitle={KDD},
  year={2022}
}

% VIDA, GinAR, GIMCC — 정확한 서지정보는 각 논문에서 확인 필요

% === 시계열 결측 복원 ===

@inproceedings{cini2022grin,
  title={Filling the G\_ap\_s: Multivariate Time Series Imputation by Graph Neural Networks},
  author={Cini, Andrea and Marisca, Ivan and Alippi, Cesare},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{yang2024fgti,
  title={Frequency-aware Generative Models for Multivariate Time Series Imputation},
  author={Yang, Xinyu and Sun, Yu and Yuan, Xiaojie and Chen, Xinyang},
  booktitle={NeurIPS},
  year={2024}
}

% === 고전적 기초 ===

@book{brillinger2001time,
  title={Time Series: Data Analysis and Theory},
  author={Brillinger, David R},
  year={2001},
  publisher={SIAM}
}

@inproceedings{dong2020dwdn,
  title={Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring},
  author={Dong, Jiangxin and Roth, Stefan and Schiele, Bernt},
  booktitle={NeurIPS},
  year={2020}
}
```

---

## 10. 향후 확장 (포스터 "Future Work" 섹션용)

제거한 요소들은 버리는 게 아니라, **S(f) 학습이 안정화된 후 순차적으로 검증**:

1. **불확실성(σ) 재활성화**: σ로 iFFT 전 주파수 성분 가중 → 불확실한 주파수 억제
2. **Coherence-guided iterative attention**: 높은 coherence 변수부터 순차 복원
3. **Embedding 모드 재비교**: iFFT vs 직접 임베딩 ablation (S(f) 안정화 후)
4. **Cross-variate attention 시각화**: S(f)의 학습된 패턴 분석 (t-SNE, heatmap)

---

## 11. 교수님 설득 포인트 요약

1. **관련 연구가 튼튼해짐**: FreTS, FITS, FilterNet, FourierGNN 등 1급 학회 논문 10편 이상이 직접 연결됨
2. **포지셔닝이 명확함**: "주파수 도메인 × VSF" 교차점 = 빈자리
3. **수학은 그대로**: S(f), W(f) 공식 변경 없음. 이름과 서사만 ML 친화적으로 전환
4. **단순해짐**: 모듈 4→3, 하이퍼파라미터 7→4, Loss 3→2
5. **확장성 있음**: σ, hierarchical, embedding 모드를 향후 ablation으로 추가 가능

실험 조건
1. 데이터셋 및 분할 비율 (Datasets & Data Split)
표준 벤치마크 데이터셋: VSF 연구에서 가장 보편적으로 사용되는 4대 표준 데이터셋은 METR-LA (교통 속도, 207개 변수), SOLAR (태양광 출력, 137개 변수), TRAFFIC (교통 점유율, 862개 변수), ECG5000 (의료 심전도, 140개 변수) 입니다
. 모델의 일반화 성능을 입증하기 위해 최소 이 4개 중 3~4개를 포함하는 것이 좋습니다.
데이터 분할 비율 (Split Ratio): 모든 표준 연구는 전체 데이터를 시간순으로 **학습(Training) 70%, 검증(Validation) 10%, 테스트(Testing) 20%**의 비율로 분할하여 사용합니다
.
2. 입력 및 예측 구간 길이 (Lookback & Horizon Length)
최근 장기 예측(Long-term)을 다루는 일부 연구를 제외하고, VSF 문제의 표준 설정은 과거 관측 길이(Lookback Window, L)와 미래 예측 길이(Horizon Window, Q)를 모두 12로 고정하는 것입니다
.
따라서 X∈R 
N×12×D
  를 입력받아  
Y
^
 ∈R 
N×12×D
  를 출력하도록 맞추어야 합니다.
3. 결측률 및 관측률 설정 (Missing Rate / Observation Rate)
표준 기준점 (Default Setting): 추론(Inference) 시 **전체 변수의 단 15%만 관측되고 85%가 결측되는 상황(Missing rate = 85%)**을 모델 성능 평가의 메인 세팅으로 삼습니다
.
추가 실험 (Ablation): 모델의 강건성을 증명하기 위해 관측률을 5%, 15%, 25% 로 다르게 설정하거나
, 결측률을 25%, 50%, 75%, 90%로 변화시키며 성능 저하 폭을 측정하는 실험을 추가합니다
.
4. 무작위 마스킹 및 다중 실행 (Randomness & Repeated Runs)
VSF 환경에서는 "어떤 변수가 살아남느냐"에 따라 모델 성능이 크게 요동칠 수 있으므로, 특정 결측 패턴에 대한 과적합을 막기 위한 엄격한 다중 실행이 필수입니다.
100번의 마스킹: 추론 시 평가를 수행할 때, 테스트 세트에 대해 관측 변수 부분집합 S를 **무작위로 100번 샘플링(Randomly sampled 100 times)**하여 평가합니다
.
다중 모델 학습 (Mean & Std): 무작위 초기화에 따른 불확실성을 상쇄하기 위해, **모델을 처음부터 10번 반복 학습(10 runs/models)한 뒤, 그 결과의 평균(Mean)과 표준편차(Standard Deviation)**를 논문 표(Table)에 기재하는 것이 완벽한 표준입니다
. (연산량이 부담될 경우 GinAR처럼 최소 5번의 시드를 사용해야 합니다
).
5. 평가 지표 (Evaluation Metrics: ObsMAE / ObsRMSE)
일반적인 시계열 예측과 달리, VSF에서는 테스트 시점에 주어지지 않은 결측 변수(N∖S)의 예측 오차는 평가하지 않습니다.
반드시 테스트 시점에 살아남은 관측 변수 집합(S)에 대해서만 오차를 계산하는 ObsMAE와 ObsRMSE를 사용해야 공정한 비교가 됩니다
.
6. 데이터 정규화 (Normalization)
입력 데이터는 학습 데이터(Training set)에서 구한 전체 스칼라 평균(μ)과 표준편차(σ)를 이용한 **Z-score 정규화 (X 
input
​
 =(X 
input
​
 −μ)/σ)**를 공통적으로 적용합니다
.


기여 구조 제안
1. 해석 가능한 주파수별 변수 관계 발견CVFA만의 고유 산출물. S(f) heatmap, coherence 시각화 등. "이 모델이 뭘 학습했는지 열어볼 수 있다"
2. 새로운 관점 제시"주파수 도메인 × VSF"라는 빈 교차점을 처음 탐색. 완성도보다 방향성이 기여
3. 효율성S(f)의 파라미터: 2×F×N×r + F×N. METR-LA 기준 약 ~50K. VIDA 대비 수십~수백 분의 1. "이 적은 파라미터로 이 정도 성능"
