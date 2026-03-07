# VIDA 레퍼런스 프로젝트 분석

> **논문**: "Imputation via Domain Adaptation: Rethinking Variable Subset Forecasting from Knowledge Transfer"
> **소스코드**: `../vida-vsf/VIDA/`

---

## 1. 프로젝트 구조

```
vida-vsf/VIDA/
├── main_vida.py                 # 메인 진입점 (3-stage 파이프라인)
├── train_multi_step.py          # baseline 훈련 스크립트 (forecaster만 단독 학습)
├── trainer.py                   # DATrainer_v2 (VIDA), Trainer (baseline)
├── util.py                      # DataLoader, metrics, scaler, subset 유틸리티
├── generate_training_data.py    # 원시 데이터 → npz 전처리
├── models/
│   ├── vida.py                  # tf_encoder (SpectralConv1d + TCN), tf_decoder
│   └── loss.py                  # SinkhornDistance, MMD, CORAL, HoMM 등
├── forecasters/
│   ├── net.py                   # MTGNN (gtnet) — 주요 forecaster
│   ├── layer.py                 # graph_constructor, mixprop, dilated_inception
│   ├── ASTGCN.py
│   ├── MSTGCN.py
│   └── TGCN.py
├── scripts/
│   └── main_vida.sh             # 실험 실행 쉘 스크립트
└── requirements.txt
```

---

## 2. 핵심 아이디어

Variable Subset Forecasting (VSF) 문제를 **Domain Adaptation**으로 해결:
- **Source domain**: 전체 변수가 관측된 시계열
- **Target domain**: 일부 변수(subset S, 15%)만 관측된 시계열
- Encoder-Decoder가 target domain의 결측 변수를 복원(imputation)한 뒤, 사전 학습된 forecaster로 예측

---

## 3. 3-Stage 훈련 파이프라인

### Stage 1: Forecaster Training

| 항목 | 내용 |
|------|------|
| **목적** | 시계열 예측기(MTGNN 등)를 전체 변수로 학습 |
| **학습 대상** | Forecaster만 |
| **Epochs** | 100 |
| **Loss** | `masked_mae` (curriculum learning 적용) |
| **Optimizer** | Adam (lr=0.001, wd=0.0001) |

```python
# trainer.py — train_forecaster()
output = self.forecaster(input, idx=idx, args=args)
predict = self.scaler.inverse_transform(output.transpose(1,3))
loss = masked_mae(predict[:,:,:,:task_level], real[:,:,:,:task_level], 0.0)  # CL
```

- **Curriculum Learning**: `step_size1`마다 예측 horizon을 1씩 확장 (1→2→...→12)
- 학습 완료 후 best model (val loss 기준) 저장 및 로드

### Stage 2: Source Pre-training

| 항목 | 내용 |
|------|------|
| **목적** | Encoder+Decoder가 full→subset 복원 능력 학습 |
| **학습 대상** | Encoder, Decoder (Forecaster **frozen**) |
| **Epochs** | 70 (`pre_epochs`) |
| **Early Stopping** | patience=20 |

```python
# trainer.py — pretrain()
feat_src, out_s = self.encoder(input)              # 전체 변수 인코딩
src_recons = self.decoder(feat_src, out_s)          # 복원
src_recons[:, :, idx_subset, :] = input[:, :, idx_subset, :]  # 관측값 보존

# Loss 1: 복원 후 forecasting 결과 vs 실제값 (subset S에 대해)
pred_result = self.forecaster(src_recons, idx=idx)
loss_spfc = masked_mae(predict_subset, real_subset, 0.0)

# Loss 2: 복원 후 forecasting vs 원본 forecasting (self-supervised consistency)
hidden_state = self.forecaster(input, idx=idx)      # 원본 입력 forecasting
loss_spc = masked_mae(predict_all, hidden_state, 0.0)

loss = 0.9 * loss_spfc + 0.1 * loss_spc
```

- `idx_subset`은 전체 노드 중 15%를 랜덤 선택 (`get_idx_subset_from_idx_all_nodes`)
- 복원된 데이터에서 관측된 위치는 원본값으로 **대체** (skip-connection 역할)

### Stage 3: Domain Alignment

| 항목 | 내용 |
|------|------|
| **목적** | Source(전체)와 Target(부분) 간 feature 분포 정렬 |
| **학습 대상** | Encoder, Decoder (Forecaster, Encoder_src **frozen**) |
| **Epochs** | 30 (`align_epochs`) |
| **Early Stopping** | patience=20 |

```python
# trainer.py — alignment()
src_x = input.clone()                                           # 전체 변수
trg_x = zero_out_remaining_input(input.clone(), idx_subset)     # subset만 남김

feat_src, _ = self.encoder_src(src_x)    # frozen source encoder
feat_trg, out_t = self.encoder(trg_x)    # trainable target encoder

# Sinkhorn Distance로 feature 분포 정렬
dr, _, _ = self.sink(feat_src, feat_trg)
loss_align = dr

# 복원 후 forecasting loss
trg_recons = self.decoder(feat_trg, out_t)
trg_recons[:, :, idx_subset, :] = trg_x[:, :, idx_subset, :]
output = self.forecaster(trg_recons, idx=idx)
loss_fafc = masked_mae(predict_subset, real_subset, 0.0)

loss = 0.9 * loss_fafc + 0.1 * loss_align
```

- `encoder_src`는 Stage 2에서 학습된 encoder의 deep copy (frozen)
- `encoder` (target)는 Stage 2 weight로 초기화 후 alignment 학습

---

## 4. 모델 아키텍처

### 4.1 Encoder (`tf_encoder`)

입력 `(batch, in_dim, num_nodes, seq_len)` → reshape → `(batch, in_dim*num_nodes, seq_len)`

**Frequency Branch** (`SpectralConv1d`):
```
input → cos(x) → FFT → low-freq modes 선택 (modes=6)
      → complex multiplication (learnable weights)
      → amplitude + phase 추출 → concat → (batch, 2*modes)
```
- `modes1 = seq_len // 2 = 6`: 저주파 모드 수
- amplitude와 phase를 분리하여 주파수 특징 표현

**Time Branch** (`TCN`):
```
input → Conv1d(dilated, causal) → BN → ReLU → Conv1d → residual connection
      → Conv1d(dilation=2) → BN → ReLU → Conv1d → residual connection
      → 마지막 timestep 추출 → (batch, seq_len)
```
- 2-layer TCN with residual connections
- kernel_size=17, dilation=[1, 2]
- `mid_channels`: 데이터셋별 최적화 (64~512)

**최종 출력**:
```python
f = normalize(concat(ef, et))  # (batch, 2*modes + seq_len) = (batch, 24)
```

### 4.2 Decoder (`tf_decoder`)

```python
x_low  = BN(IFFT(out_ft, n=12))                    # 저주파로 시계열 복원
x_high = ReLU(BN(ConvTranspose1d(et)))              # 시간 특징으로 고주파 복원
y = x_low + x_high                                   # (batch, 1, num_nodes, seq_len)
```

### 4.3 Forecaster (`gtnet` = MTGNN)

- Graph-based spatio-temporal 예측 모델
- Adaptive adjacency matrix 학습 (`graph_constructor`)
- Dilated inception + Graph convolution (mixprop)
- 입력: `(batch, in_dim, num_nodes, seq_len)` → 출력: `(batch, pred_len, num_nodes, 1)`

### 4.4 Alignment Loss (`SinkhornDistance`)

- Wasserstein Optimal Transport의 Sinkhorn approximation
- `eps=1e-3`, `max_iter=1000`, `reduction='sum'`
- Source/Target encoder 출력 간 분포 거리를 최소화

---

## 5. 데이터 파이프라인

### 5.1 데이터셋

| 데이터셋 | 변수 수 | 도메인 | Raw 스케일링 | Adjacency |
|----------|---------|--------|-------------|-----------|
| METR-LA | 207 | 교통 속도 | 없음 | adj_mx.pkl |
| SOLAR | 137 | 태양에너지 | 없음 | 없음 (adaptive) |
| TRAFFIC | 862 | 교통량 | ×1000 | 없음 (adaptive) |
| ECG5000 | 140 | 심전도 | ×10 | 없음 (adaptive) |

### 5.2 전처리 (`generate_training_data.py`)

```python
# 슬라이딩 윈도우 생성
x_offsets = np.arange(-11, 1, 1)    # 과거 12 timesteps
y_offsets = np.arange(1, 13, 1)     # 미래 12 timesteps

# METR-LA만 time_of_day 피처 추가 (in_dim=2)
if ds_name == "metr-la":
    add_time_in_day = True  # 하루 중 시간을 0~1로 인코딩

# 분할: 전체 윈도우 생성 후 순서대로 분할
x_train = x[:num_train]          # 70%
x_val   = x[num_train:num_train+num_val]  # 10%
x_test  = x[-num_test:]          # 20%
```

**주의**: 윈도우 생성 **후** 분할하므로, val/test 초반 윈도우가 train 데이터를 포함할 수 있음 (경계 leakage)

### 5.3 정규화

```python
# util.py — 윈도우된 train input의 첫 번째 feature 채널로 global mean/std 계산
scaler = StandardScaler(
    mean=data['x_train'][..., 0].mean(),
    std=data['x_train'][..., 0].std()
)
# x만 정규화, y는 raw 유지
# 평가 시: pred = inverse_transform(model_output), real = raw_y
```

### 5.4 DataLoader (`DataLoaderM`)

```python
# 커스텀 데이터 로더 (PyTorch DataLoader 미사용)
# - 마지막 샘플 반복으로 batch_size 배수 맞춤
# - shuffle: 인덱스 기반 permutation
# - get_iterator(): generator 방식
```

---

## 6. VSF 실험 프로토콜

### 6.1 마스킹 방식

```python
# 비관측 변수를 0으로 채움 (전체 N개 노드 유지)
def zero_out_remaining_input(testx, idx_current_nodes, device):
    zero_val_mask = torch.ones_like(testx).bool()
    zero_val_mask[:, :, idx_current_nodes, :] = False
    inps = testx.masked_fill_(zero_val_mask, 0.0)
    return inps
```

### 6.2 실험 세팅

| Setting | 설명 | 관련 인자 |
|---------|------|----------|
| **Standard** | 15% 랜덤 subset, 100 random splits | `lb=ub=15, split_runs=100` |
| **Partial** | subset S만으로 inference (mask_remaining) | `mask_remaining=True, epochs=0` |
| **Oracle** | 전체로 학습, subset으로 평가 | `do_full_set_oracle=True` |
| **S apriori** | 사전 정의 subset으로 학습+평가 | `predefined_S=True` |

### 6.3 Inference 흐름 (Standard Setting)

```python
for split_run in range(100):  # 100 랜덤 subset
    idx_current_nodes = get_node_random_idx_split(num_nodes, lb=15, ub=15)  # ~15%

    for x, y in test_loader:
        testx = zero_out_remaining_input(testx, idx_current_nodes)  # 마스킹

        feat_trg, out_t = encoder(testx)           # 인코딩
        trg_recons = decoder(feat_trg, out_t)       # 복원
        trg_recons[:, :, idx_current_nodes, :] = testx[:, :, idx_current_nodes, :]  # 관측값 보존

        preds = forecaster(trg_recons)              # 예측
        preds = preds[:, idx_current_nodes, :]      # subset에 대해서만 평가

    # horizon별 MAE, RMSE 계산
```

### 6.4 다중 실행 구조

```
runs (모델 학습 반복, default=10)
  └── random_node_idx_split_runs (추론 시 랜덤 subset, default=100)
       └── horizon 1~12 각각 MAE, RMSE 계산

# 최종 보고: 전체 run×split에 대한 mean ± std
```

---

## 7. 평가 메트릭

```python
# util.py — null_val=0.0으로 마스킹 (값이 0인 지점 제외)
def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)  # 정규화
    loss = torch.abs(preds - labels) * mask
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))

# 최종 보고 형식:
# horizon별: MAE ± std, RMSE ± std
# 전체 평균: Final MAE ± std, Final RMSE ± std
```

---

## 8. 하이퍼파라미터 정리

### 공통 설정

| 항목 | 값 |
|------|-----|
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| batch_size | 64 |
| grad_clip | 5.0 |
| seed | 3407 |
| seq_in_len / seq_out_len | 12 / 12 |
| fourier_modes | 6 (= seq_len // 2) |
| TCN kernel_size | 17 |
| dropout | 0.3 (forecaster), 0.0 (TCN) |
| patience | 20 |

### Loss 가중치

| 항목 | 값 | 설명 |
|------|-----|------|
| w_spfc | 0.9 | Source pre-training forecast loss |
| w_spc | 0.1 | Self-supervised prediction consistency |
| w_fafc | 0.9 | Feature alignment forecast loss |
| w_align | 0.1 | Sinkhorn alignment loss |

### 데이터셋별 설정

| 데이터셋 | step_size1 | mid_channels | in_dim |
|----------|-----------|-------------|--------|
| METR-LA | 2500 | 128 | 2 (값 + time_of_day) |
| SOLAR | 2500 | 512 | 1 |
| TRAFFIC | 1000 | 256 | 1 |
| ECG5000 | 400 | 64 | 1 |

### 모델 저장 경로

```
saved_models/{model_name}/{dataset}/seed{seed}/
  ├── forecaster{expid}_{runid}_seed{seed}.pth
  ├── pre_encoder{expid}_{runid}_seed{seed}.pth
  ├── pre_decoder{expid}_{runid}_seed{seed}.pth
  ├── align_encoder{expid}_{runid}_seed{seed}.pth
  └── align_decoder{expid}_{runid}_seed{seed}.pth
```

---

## 9. 실험 실행 방법

### 쉘 스크립트 (`scripts/main_vida.sh`)

```bash
for dataset in ECG5000 SOLAR TRAFFIC METR-LA; do
    # 데이터셋별 step_size1, mid_channels 자동 설정
    nohup python -u ../main_vida.py \
        --data '../data/'$dataset \
        --model_name MTGNN \
        --epochs 100 --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --w_fc 1.0 --w_pc 0.5 --w_align 0.5 \
        --patience 20 --mid_channels $mid_channels \
        --step_size1 $step_size1 \
        > ../logs/main_exp/${dataset}_${model_name}_exp${expid}.out 2>&1 &
done
```

### Inference Only (학습 없이 평가)

```bash
# Partial setting
python train_multi_step.py --epochs 0 --mask_remaining True \
    --random_node_idx_split_runs 100

# Oracle setting
python train_multi_step.py --epochs 0 --do_full_set_oracle true \
    --full_set_oracle_lower_limit 15 --full_set_oracle_upper_limit 15
```

---

## 10. 보고된 성능 (100-run test, 15% subset)

| Dataset | MAE ± std | RMSE ± std |
|---------|-----------|------------|
| METR-LA | 3.36 ± 0.07 | — |
| Traffic | 11.40 ± 0.61 | — |
| ECG5000 | 3.22 ± 0.50 | — |
| Solar | 2.33 ± 0.39 | — |

---

## 11. VIDA의 구조적 한계 분석 (CSCI 관점)

VIDA의 접근법을 주파수 도메인 활용 관점에서 분석하면 다음과 같은 한계가 존재한다.

### 11.1 주파수 활용의 제한성

| 항목 | VIDA | 한계 |
|------|------|------|
| **FFT 사용 목적** | 스펙트럼 임베딩 생성 (저주파 특징 추출) | 주파수 정보를 **변수 간 관계**에 활용하지 않음 |
| **위상(Phase) 활용** | 변수 **내부** 위상 `p_i` (intra-variable) | 변수 **간** 위상차 `∠C_ij` (inter-variable) 미활용 |
| **변수 간 관계** | 독립 처리 (각 변수 별도 인코딩) | 교차 스펙트럼 결맞음(Coherence) 활용 없음 |

```
VIDA의 SpectralConv1d:
  input → cos(x) → FFT → 저주파 modes만 선택 → amplitude + phase

  핵심: 각 변수를 개별적으로 처리하여 주파수 특징 추출
       변수 i와 변수 j 사이의 lead-lag 관계 (∠C_ij) 는 활용하지 않음
```

### 11.2 2-Stage 오차 증폭 문제

VIDA는 본질적으로 **2-Stage** 파이프라인이다:

```
Stage A: Encoder-Decoder → iFFT → 수치 복원 x̂_miss(t)
Stage B: 복원된 시계열 → Forecaster → 예측 Ŷ

문제:
  1. iFFT로 수치 단일값만 전달 → 불확실성 정보 소실
  2. 복원 오차가 Forecaster에 그대로 전파·증폭
  3. Forecaster는 복원값을 "진짜 관측"으로 인식 (보간 신뢰도 구분 불가)
```

- Decoder가 `x_low + x_high`로 시계열을 **수치 복원**한 뒤 Forecaster에 입력
- 복원의 **불확실성(σ)**이 Forecaster에 전달되지 않음
- 관측값 보존 (`trg_recons[:, :, idx, :] = testx[:, :, idx, :]`)으로 부분 완화하지만, 결측 변수의 복원 오차는 그대로 잔존

### 11.3 Domain Adaptation의 한계

```
Sinkhorn Distance:
  - Source(전체 변수) encoder 출력과 Target(부분 변수) encoder 출력의 분포 정렬
  - 전역적(global) 분포 매칭이므로, 개별 변수 수준의 결맞음 관계를 포착하기 어려움
  - 연산 복잡도: max_iter=1000의 반복 최적화 필요
```

### 11.4 VIDA vs CSCI 방법론 비교

| 항목 | VIDA | CSCI (제안) |
|------|------|------------|
| **분류** | 2-Stage (보간 → 예측) | 임베딩 직접 주입 (iFFT 없음) |
| **수치 복원** | O (iFFT로 시계열 복원) | X (주파수 임베딩 직접 전달) |
| **불확실성 전파** | X (σ 없음) | O (Wiener 잔여분산 → 어텐션 바이어스) |
| **위상 활용** | 변수 내부 (intra) | 변수 간 (inter, ∠C_ij) |
| **변수 간 관계** | 독립 처리 | 교차 스펙트럼 행렬 S [F×N×N] |
| **백본 재활용** | O (Forecaster 그대로) | O (입력층만 수정) |
| **오차 증폭** | 구조적으로 존재 | 구조적으로 제거 |
| **이론적 근거** | Domain Adaptation | Wiener 필터 (MMSE 최적) |

---

## 12. 벤치마킹 시 주요 참고사항

### 12.1 실험 재현을 위한 체크리스트

1. **METR-LA는 in_dim=2**: time_of_day 피처가 성능에 큰 기여 (교통 주기성)
2. **데이터 분할**: 윈도우 생성 후 분할 (경계 leakage 있으나 영향 미미)
3. **정규화**: x만 정규화, y는 raw 유지 → 평가 시 pred만 inverse_transform
4. **마스킹**: 비관측 변수를 0으로 채움 (obs_mask 방식이 아닌 zero-fill)
5. **복원 후 관측값 보존**: `trg_recons[:, :, idx, :] = testx[:, :, idx, :]`
6. **평가 대상**: 관측 센서(subset S)에 대해서만 메트릭 계산
7. **Curriculum Learning**: step_size에 따라 예측 horizon을 점진적으로 확장
8. **Early Stopping**: Stage 2, 3 모두 patience=20 적용

### 12.2 CSCI 실험 설계 시 VIDA 프로토콜 준수 사항

VIDA와의 공정 비교를 위해 다음 프로토콜을 동일하게 유지해야 한다:

| 항목 | VIDA 프로토콜 | CSCI 적용 |
|------|-------------|----------|
| **결측률** | 85% (15% 관측) | 동일 |
| **마스킹 단위** | 변수(variate) 단위 | 동일 |
| **Eval 100-run** | run당 고정 마스크, 전체 test에 동일 적용 | 동일 |
| **메트릭** | masked_mae, masked_rmse (null_val=0.0) | 동일 |
| **보고 형식** | horizon별 mean ± std | 동일 |
| **데이터셋** | METR-LA, SOLAR, TRAFFIC, ECG5000 | 동일 + PEMS-BAY, ETTh1 추가 |
| **Forecaster** | MTGNN (주), ASTGCN, MSTGCN, TGCN | MTGNN (주) |

### 12.3 CSCI가 검증해야 할 VIDA 대비 차별점

```
H1: 교차 스펙트럼 결맞음 (C_ij) 활용이 VIDA의 독립 인코딩보다 복원 품질 우수
    → 측정: 스펙트럼 복원 MSE (진폭/위상 분리)

H2: Spectral Projector (iFFT 없음 + σ 전달)가 VIDA의 2-Stage보다 예측 성능 우수
    → 측정: 예측 MAE/MSE, Ablation (iFFT 버전 vs Projector 버전)

H3: 계층적 전파(Hierarchical Propagation)가 극단 결측(85%)에서 단일 라운드 대비 우수
    → 측정: 결측률 70~85% 구간 MAE/MSE
```
