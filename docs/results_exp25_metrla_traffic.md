# Exp25: METR-LA + TRAFFIC 10-seed (B+A 채택안)

채택 설정: `obs_only_loss=True`, `gamma_loss=0.1` (consistency loss)
실행: ExpID 25, seeds 0-9, 100 random splits, GPU 1 (A30)

## METR-LA (10 seeds)

| Seed | ObsMAE | ObsRMSE |
|------|--------|---------|
| 0 | 3.267 | 6.585 |
| 1 | 3.320 | 6.696 |
| 2 | 3.356 | 6.760 |
| 3 | 3.315 | 6.665 |
| 4 | 3.314 | 6.640 |
| 5 | 3.306 | 6.629 |
| 6 | 3.332 | 6.684 |
| 7 | 3.298 | 6.685 |
| 8 | 3.313 | 6.731 |
| 9 | 3.258 | 6.594 |
| **Mean ± Std** | **3.308 ± 0.027** | **6.667 ± 0.054** |

## TRAFFIC (10 seeds)

| Seed | ObsMAE | ObsRMSE |
|------|--------|---------|
| 0 | 10.893 | 26.721 |
| 1 | 10.844 | 26.647 |
| 2 | 10.919 | 26.828 |
| 3 | 10.828 | 26.639 |
| 4 | 10.774 | 26.634 |
| 5 | 10.920 | 26.914 |
| 6 | 10.837 | 26.558 |
| 7 | 11.042 | 27.001 |
| 8 | 11.018 | 27.528 |
| 9 | 11.079 | 27.324 |
| **Mean ± Std** | **10.915 ± 0.096** | **26.879 ± 0.319** |

## Ablation 비교 (METR-LA, single-seed)

| 설정 | MAE | vs Baseline |
|------|-----|-------------|
| Baseline (ExpID 16) | 3.321 | - |
| ExpA (obs_only_loss) | 3.310 | -0.3% |
| ExpB (gamma=0.1) | 3.284 | -1.1% |
| **B+A combo (gamma=0.1 + obs_only)** | **3.267** | **-1.6%** |
| B+A gamma=0.05 | 3.284 | -1.1% |
| B+A gamma=0.2 | 3.283 | -1.1% |

## 관찰
- METR-LA 10-seed 평균 (3.308)은 single-seed best (3.267)보다 약 1.3% 높음 → seed variance 존재
- TRAFFIC도 채택안으로 안정적인 결과
- consistency loss는 gamma 값 0.05/0.1/0.2에 robust, obs_only_loss와 결합 시 시너지

## 실행 정보
- 스크립트: `scripts/run_metrla_traffic_10seeds_gpu1.sh`
- 로그: `/data/sheda7788/VSF_CSCI/logs/metrla_traffic_10seeds_gpu1.log`
- 결과 JSON: `/data/sheda7788/VSF_CSCI/results/summary/cvfa_{METR-LA,TRAFFIC}_exp25_run0_seed{0-9}_summary.json`
