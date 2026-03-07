

## 1. 데이터 전처리 (Preprocessing)

| 항목 | VIDA | COMET (ours) | 일치? |
|------|------|-------------|-------|
| **Raw 스케일링** | Traffic ×1000, ECG ×10, Electricity ÷1000 | Traffic ×1000, ECG ×10 | O |
| **METR-LA 로딩** | `pd.read_hdf()` → DataFrame | `pd.read_hdf()` → numpy | O |
| **Solar/Traffic 로딩** | CSV, delimiter="," | `np.loadtxt(delimiter=",")` | O |

### 코드 참조

**VIDA** (`generate_training_data.py:75-80`):
```python
if args.ds_name == "traffic":
    df = df * 1000
if args.ds_name == "ECG":
    df = df * 10
if args.ds_name.lower() == "electricity":
    df = df / 1000
```

## 2. 데이터 분할 (Data Split)

| 항목 | VIDA | COMET | 일치? |
|------|------|-------|-------|
| **비율** | train=70%, val=10%, test=20% | train=70%, val=10%, test=20% | O |
| **분할 시점** | 윈도우 생성 **후** 분할 | Raw 데이터 **먼저** 분할 → 각 split에서 window 생성 | **X** |
| **경계 leakage** | val/test 첫 윈도우가 train 마지막 데이터 포함 가능 | 각 split이 독립적 (leakage 없음) | **X** |

### 상세 설명

**VIDA**: 전체 raw 데이터에서 sliding window를 먼저 생성한 뒤, 윈도우 단위로 70/10/20 분할.
```python
# generate_training_data.py
x, y = generate_graph_seq2seq_io_data(df, x_offsets, y_offsets)  # 전체 데이터로 윈도우 생성
num_train = round(num_samples * 0.7)
num_test = round(num_samples * 0.2)
x_train, y_train = x[:num_train], y[:num_train]
x_test, y_test = x[-num_test:], y[-num_test:]
```

**영향**: VIDA 방식에서는 val/test 초반 윈도우의 input이 train 마지막 데이터를 포함할 수 있어, 경계 부근 성능이 약간 높아질 수 있음. 단, 전체 test set 대비 영향은 미미.

---

## 3. 정규화 (Normalization)

| 항목 | VIDA | COMET | 일치? |
|------|------|-------|-------|
| **Scaler 타입** | Global (단일 mean/std) | Global (`global_scaler=True`) | O |
| **fit 대상** | `x_train[..., 0]` (윈도우된 train input, 중복 포함) | `train_raw` (raw train 데이터, 중복 없음) | **X** |
| **정규화 대상** | **x만** 정규화, y는 raw 유지 | **x, y 모두** 정규화 | **X** |
| **Eval 역변환** | `pred = inverse(model_out)`, `real = raw_y` | `pred = inverse(pred)`, `real = inverse(y)` | 수학적 동치 |
| **RevIN** | 없음 | 비활성 (`use_revin=False`) | O |

### 상세 설명

**VIDA scaler** (`util.py:165`):
```python
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
# x_train shape: (num_samples, 12, num_nodes, input_dim)
# x_train[..., 0] shape: (num_samples, 12, num_nodes)
# → 전체 윈도우(중복 포함) 기반 global mean/std 계산
```

**정규화 대상 차이**:
- VIDA: x만 정규화 → 모델 출력을 inverse_transform하여 raw y와 비교
- COMET: x, y 모두 정규화 → 모델 출력과 y 모두 inverse_transform 후 비교
- 수학적으로 동치이나, float32 round-trip (transform → inverse_transform)에 의한 미세한 수치 오차 가능

---

## 4. 입력 특성 (Input Features)

| 항목 | VIDA | COMET | 일치? |
|------|------|-------|-------|
| **METR-LA** | **2채널**: 값 + time_of_day | **1채널**: 값만 | **X !!** |
| **Solar/Traffic/ECG** | 1채널: 값만 | 1채널: 값만 | O |
| **seq_len (input)** | 12 | 12 | O |
| **pred_len (output)** | 12 | 12 | O |

### 상세 설명

**VIDA** (`generate_training_data.py:92-95`):
```python
if args.ds_name == "metr-la":
    add_time_in_day = True   # METR-LA만 time-of-day 피처 추가
else:
    add_time_in_day = False
```

time-of-day 피처는 하루 중 시간을 0~1 사이 값으로 인코딩한 것으로, 교통 데이터의 주기적 패턴(출퇴근 시간 등)을 직접적으로 제공합니다.


**영향**: **매우 높음**. 교통 데이터는 시간대별 패턴이 매우 강하므로, time-of-day 피처가 예측 성능에 큰 기여를 합니다. METR-LA에서 COMET이 VIDA 대비 열세인 주요 원인 중 하나로 추정.

---

## 5. Missing Rate & 마스킹 프로토콜

| 항목 | VIDA | COMET | 일치? |
|------|------|-------|-------|
| **Missing rate** | 85% (lb=ub=15, 즉 15% 관측) | 85% (`missing_rate=0.85`) | O |
| **마스킹 방식** | 비관측 센서를 **0으로 채움** (전체 N개 유지) | `obs_mask` boolean: 관측 센서만 사용 | **X** |
| **마스크 단위** | 센서 단위 (variate-level) | 센서 단위 (variate-level) | O |
| **학습 시** | 매 epoch/batch마다 랜덤 마스크 | 매 sample마다 독립 랜덤 마스크 | 유사 |
| **Eval (100-run)** | Run당 고정 센서 마스크, 전체 test에 동일 적용 | Run당 고정 센서 마스크 (VIDA 프로토콜에 맞춤) | O |

### VIDA 마스킹 상세

```python
# util.py:217-224 — 관측할 센서 인덱스 선택
def get_node_random_idx_split(args, num_nodes, lb, ub):
    count_percent = np.random.choice(np.arange(lb, ub+1), size=1)[0]  # lb=ub=15 → 항상 15%
    count = math.ceil(num_nodes * (count_percent / 100))
    current_node_idxs = np.random.choice(np.arange(num_nodes), size=count, replace=False)
    return current_node_idxs

# util.py:227-231 — 비관측 센서를 0으로 채움
def zero_out_remaining_input(testx, idx_current_nodes, device):
    zero_val_mask = torch.ones_like(testx).bool()
    zero_val_mask[:, :, idx_current_nodes, :] = False
    inps = testx.masked_fill_(zero_val_mask, 0.0)
    return inps
```

**VIDA eval 프로토콜** (`main_vida.py:602-689`):
- 100개 run, 각 run마다 고정된 센서 인덱스 선택 (15% = ~31개 for METR-LA)
- 같은 마스크를 해당 run의 **모든 test sample**에 동일 적용
- VIDA는 imputation(복원) 후 관측 센서의 값을 **원본으로 대체** (line 640):
  ```python
  trg_recons[:, :, idx_current_nodes, :] = testx[:, :, idx_current_nodes, :]
  ```
- 메트릭은 **관측 센서에 대해서만** 계산 (line 648-649):
  ```python
  preds = preds[:, idx_current_nodes, :]
  ```
---

## 6. 메트릭 (Metrics)

| 항목 | VIDA | COMET | 일치? |
|------|------|-------|-------|
| **MAE 수식** | `masked_mae(pred, real, null_val=0.0)` | `abs(pred-true) * valid_mask / count` | 수학적 동치 |
| **null_val 마스킹** | `labels != 0.0` (inverse 후 비교) | `y_raw != 0.0` (raw 데이터로 비교) | 미세 차이 |
| **보고 방식** | `mean(per-horizon MAE)` ± `mean(per-horizon std)` | 동일 | O |
| **Eval 대상** | test set, 관측 센서만 | test set, 관측 센서만 | O |

### VIDA masked_mae 구현

```python
# util.py:196-207
def masked_mae(preds, labels, null_val=np.nan):
    mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)  # 정규화: mask값을 1/mean(mask)로 스케일링
    loss = torch.abs(preds - labels) * mask
    return torch.mean(loss)
    # 수학적으로: sum(|pred-true| * mask) / sum(mask) 과 동치
```

두 구현은 **수학적으로 동일한 결과**를 산출합니다.

### null_val 마스킹 차이

- VIDA: inverse_transform 후의 `labels != 0.0`으로 비교 → float32 변환 오차로 원래 0인 값이 미세하게 비-0이 될 수 있음
- COMET: raw 데이터(`y_raw`)의 `!= 0.0`으로 비교 → 정확한 0 판별
- 영향: METR-LA에서는 실질적 차이 거의 없음 (0값이 드묾)

---

## 7. 모델 구조 & 학습 설정

| 항목 | VIDA | COMET |
|------|------|-------|
| **아키텍처** | Encoder-Decoder + MTGNN forecaster | CI-Mamba → AsymEncoder → Codebook → RestoreDecoder → ts_decoder → MTGNN |
| **파라미터 수** | 데이터셋별 상이 (METR-LA: ~3-5M, Solar: ~27M) | ~1.3M (d_model=128 기준) |
| **Optimizer** | Adam | Adam |
| **Learning rate** | 0.001 | 0.001 |
| **Weight decay** | 0.0001 | 0.0001 |
| **Batch size** | 64 | 32 (METR-LA), 8 (Traffic) |
| **Epochs** | 100 (forecaster) + 100 (alignment) = **200 total** | **100** (3-stage curriculum) |
| **Patience** | 20 | 15 |
| **Loss** | MAE (masked, null_val=0.0) | MAE (denorm, null_val=0.0) |
| **Grad clip** | 5.0 | 5.0 |
| **Seed** | 3407 (1개) | 789 (1개) |
| **AMP** | 미사용 | bf16 mixed precision |

### VIDA 데이터셋별 모델 크기

```bash
# main_vida.sh
ECG5000:  mid_channels=64
METR-LA:  mid_channels=128
TRAFFIC:  mid_channels=256
SOLAR:    mid_channels=512
```

## 8. 학습 커리큘럼

### VIDA: 2-Phase 학습

1. **Phase 1** (Forecaster Training, 100 epochs): MTGNN forecaster를 oracle 데이터로 학습
2. **Phase 2** (Domain Alignment, 100 epochs): Encoder-Decoder로 missing → full 복원 학습, forecaster fine-tune


## 9. 핵심 차이점 요약 (성능 영향도순)

| 순위 | 차이점 | 영향도 | VIDA 유리? | 비고 |
|------|--------|--------|-----------|------|
| **1** | **METR-LA time-of-day 피처** | **높음** | Yes | 교통 데이터의 시간 주기성 직접 활용 |
| **2** | **모델 파라미터 수** (VIDA ~3-5M vs COMET ~1.3M) | **높음** | Yes | 데이터셋별 최적 크기 사용 |
| **3** | **총 학습량** (VIDA 200 epochs vs COMET 100 epochs) | **중간** | Yes | VIDA가 2배 더 학습 |
| **4** | **데이터 분할 방식** (윈도우 후 vs 전) | **중간** | Yes | 경계 leakage로 약간 유리 |
| **5** | **Batch size** (VIDA 64 vs COMET 32) | **낮음** | 상황별 | 큰 batch = 안정적 gradient |
| **6** | **Patience** (VIDA 20 vs COMET 15) | **낮음** | Yes | 더 오래 탐색 가능 |
| **7** | **Scaler fit 대상** (윈도우 vs raw) | **미미** | - | 실질적 차이 무시 가능 |

---

## 10. 현재 성능 비교 (2026-03-07 기준)

| Dataset | COMET (val best) | COMET (100-run test) | VIDA (100-run test) | 차이 |
|---------|-----------------|---------------------|--------------------|----|
| **METR-LA** | 3.25 | 3.66 ± 0.25 | **3.36 ± 0.07** | +8.9% |
| **Traffic** | - | 14.55 (single eval) | **11.40 ± 0.61** | +27.6% |
| **ECG5000** | 3.51 | - | **3.22 ± 0.50** | +9.0% |
| **Solar** | 2.45 | - | **2.33 ± 0.39** | +5.2% |

---

## 11. 개선 방안 (향후)

1. **METR-LA time-of-day 피처 추가**: 가장 직접적인 성능 개선 예상
2. **데이터 분할 방식 VIDA 정렬**: 윈도우 생성 후 분할로 변경
3. **모델 크기 데이터셋별 최적화**: d_model, n_layers를 데이터셋에 맞게 조정
4. **학습 epoch 증가**: patience=20, epochs=150+ 시도
5. **Seed 탐색**: 여러 seed로 학습하여 최적 seed 선택
