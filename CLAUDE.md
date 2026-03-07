# CSCI 프로젝트 개발 지침

## 프로젝트 개요
- **프로젝트명**: CSCI (Cross-Spectral Coherence Imputation)
- **목표**: Variable Subset Forecasting(VSF) 문제를 교차 스펙트럼 결맞음 기반으로 해결
- **레퍼런스 모델**: VIDA (`../vida-vsf/VIDA/`)

## 핵심 원칙

### 1. VIDA 기반 실험/평가 프로토콜 준수
- 전체 프로젝트 구조, 실험 방식, 평가 방식은 VIDA를 참조한다.
- VIDA는 연구 결과와 직접 비교할 대상이므로, **실험 환경의 공정성**이 최우선이다.
- 데이터 전처리, 분할, 정규화, 마스킹, 메트릭 계산 방식을 VIDA와 동일하게 맞출 것.
- 관련 문서: `docs/vida_reference_analysis.md`, `docs/experiment_comparison_vida.md`

### 2. 서버 환경 및 리소스 전략
- **GPU**: GTX 1080 × 3대 (VRAM 제한적)
- 리소스가 충분하지 않으므로, 3대 GPU를 **모두 활용**하는 병렬 실행 전략을 항상 고려할 것.
- 데이터셋/실험을 GPU별로 분배하여 동시 실행.
- 메모리 효율을 고려한 배치 사이즈, mixed precision 등을 적극 활용.

### 3. 모델 설계 문서 참조
- **연구 설계서**: `docs/CSCI_Research_Design.docx` (이론적 배경, 가설, 실험 설계)
- **개발 가이드**: `docs/project_develop_plan.md` (모듈별 구현 코드, 하이퍼파라미터, 체크리스트)
- 구현 시 반드시 위 문서의 설계를 따를 것.

### 4. 구현 품질 주의사항
- **버그, 파라미터 실수, 파라미터 전달 누락** 등의 오류가 빈번하게 발생하므로 항상 주의할 것.
- 모듈 간 인터페이스(입출력 shape, dtype, device)를 명시적으로 검증할 것.
- 새로운 모듈 구현 후 반드시 단위 테스트로 shape/값 검증을 수행할 것.
- argparse 인자 추가 시 default 값, type, 실제 전달 여부를 반드시 확인할 것.

### 5. 백그라운드 실험 실행
- 모든 실험은 **서버 세션이 끊겨도 계속 실행**되도록 할 것.
- `nohup ... &` 또는 `tmux`/`screen` 세션을 활용.
- 로그는 반드시 파일로 리다이렉트하여 저장.
