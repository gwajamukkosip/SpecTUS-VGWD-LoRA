# SpecTUS VGWD 연구 맥락 및 누적 실험 기록

마지막 갱신: 2026-08-30 (KST)

이 문서는 이 저장소에서 수행한 VGWD 관련 실험의 근거와 결론을 누적하는 연구 원장이다. 아래 수치는 설정 파일, 체크포인트, 예측 결과, 평가 로그 또는 분석 산출물에서 직접 확인된 값만 기록했다.

## 기록 원칙

- 기존 기록은 삭제하거나 과거 결론에 맞추어 덮어쓰지 않는다. 새 실험은 실행일 기준으로 아래 연대기에 추가한다.
- `validation`으로 정한 선택과 `test`에서 보고한 최종 성능을 구분한다.
- 실패·중단·유의하지 않은 결과도 삭제하지 않는다.
- 같은 지표 이름이라도 정의가 다르면 별도로 기록한다. 특히 strict exact SMILES, connectivity exact, fingerprint-perfect를 혼용하지 않는다.
- 아직 실행하지 않은 아이디어는 실험 결과처럼 기록하지 않는다.
- 새 실험을 추가할 때에는 날짜, 연구 질문, 변경 변수, 고정 조건, seed, checkpoint 선택 규칙, validation 결과, test 사용 여부, 통계 검정, 산출물 경로를 함께 남긴다.
- 이 문서가 이후 모델 선택에 사용될 수 있으므로, test를 열람한 사실과 사후 분석 여부를 숨기지 않는다.

## 1. 연구 목적과 데이터 분할

### 연구 목적

SpecTUS는 GC-EI 질량 스펙트럼을 입력받아 분자 구조의 SMILES를 생성하는 encoder-decoder Transformer이다. 이 프로젝트의 현재 연구 목적은 사전학습된 `SpecTUS_pretrained`를 VGWD 실험 스펙트럼에 parameter-efficient fine-tuning하여, 제한된 GPU 메모리 환경에서 다음 성능을 높이는 것이다.

- strict exact structure Top-1 및 Top-k 정확도
- Morgan fingerprint Tanimoto 유사도
- 분자식 일치율
- seed 변화에 대한 재현성

현재 주 연구 질문은 rsLoRA에서 rank 증가 자체가 성능 향상에 기여하는지, 아니면 rank와 함께 바뀐 scaling이 주된 원인인지 구분하는 것이다.

### 데이터 분할

| 데이터 | train | validation | test | 비고 |
|---|---:|---:|---:|---|
| 초기 `data/vgwd` | 5,224 | 644 | 675 | 초기 비정제 분할. 당시 test 평가에서 674개가 사용됨 |
| 정제 `data/vgwd_clean` 원본 | 5,202 | 666 | 675 | 이후 주 실험에 사용 |
| 정제 데이터 중 `m/z <= 500` 사용 가능 | 5,197 | 664 | 675 | 제외: train 5개, validation 2개, test 0개 |

주 validation 파일은 `data/vgwd_clean/valid_filtered_mz500.jsonl`이며 664개 샘플이다. 주 test는 675개 샘플이다.

누수 감사에서 사용한 split 크기는 train 5,202, filtered validation 664, test 675이다. 이 분할은 exact molecule이나 spectrum hash 기준으로는 교차 중복이 없지만 scaffold-disjoint 분할은 아니다. 자세한 내용은 2026-08-28 누수 감사와 9절에 기록한다.

## 2. 공통 모델·학습·평가 설정

### 기반 모델

- 초기 체크포인트: `checkpoints/SpecTUS_pretrained`
- tokenizer: `tokenizer/tokenizer_mf10M.model`
- 최대 sequence length: 200
- 최대 입력 질량: `max_mz = 500`
- encoder/decoder layer: 각각 12
- attention head: 16
- FFN dimension: 4,096
- encoder/decoder embedding: separate

### PEFT 학습의 공통 조건

별도 표기가 없는 VGWD clean PEFT 실험은 다음 조건을 공유한다.

- 4,000 training steps
- AdamW, learning rate `1e-4`, warmup 200 steps
- per-device batch 1, gradient accumulation 4, effective batch 4
- validation batch 2, fp16
- 500 step마다 evaluation 및 checkpoint 저장
- adapter dropout 0.05
- 기본 target module: `q_proj`, `v_proj`

### 실제 실행된 adapter 설정

| 방법 | rank | alpha | rsLoRA 유효 scale `alpha/sqrt(rank)` | target module | 비고 |
|---|---:|---:|---:|---|---|
| LoRA | 8 | 16 | 해당 없음 | q/v | 초기 baseline |
| LoRA | 16 | 32 | 해당 없음 | q/v | standard LoRA 확장 |
| DoRA | 8 | 16 | 해당 없음 | q/v | 저장된 adapter 설정 기준 |
| AdaLoRA | init 16, target 8 | 32 | 해당 없음 | q/v | `tinit=200`, `tfinal=500`, `deltaT=10`, `beta=0.85`, `orth_reg_weight=0.5` |
| rsLoRA | 16 | 32 | 8.000 | q/v | r16 기준 모델 |
| rsLoRA | 16 | 32 | 8.000 | q/k/v/out | validation-only all-attention screen |
| rsLoRA | 16 | 45 | 11.250 | q/v | scaling ablation의 high-scale 조건 |
| rsLoRA | 32 | 45 | 7.955 | q/v | scaling ablation의 low-scale 조건 |
| rsLoRA | 32 | 64 | 11.314 | q/v | 기존 test-evaluated incumbent |
| rsLoRA | 32 | 64 | 11.314 | q/k/v/out | validation에서 선택된 all-attention 후보 |

`q_proj`, `k_proj`, `v_proj`, `out_proj`를 모두 학습하는 all-attention target 실험은 r16과 r32에서 validation-only로 실행했다. attention projection 외 FFN 포함 실험은 아직 실행하지 않았으므로 성능에 관한 결론이 없다.

### 평가 정의

- strict exact: 예측과 정답의 canonical isomeric SMILES가 동일한 경우이다. stereochemistry 차이를 오답으로 본다.
- connectivity exact: stereochemistry를 제거한 연결 구조가 같은 경우이다. reranker 탐색 일부에서만 사용했다.
- fingerprint-perfect: Morgan Tanimoto가 1인 경우이다. strict exact와 동일한 지표가 아니다.
- Morgan: radius 2 fingerprint의 Tanimoto similarity이다.
- 최종 주 추론 설정: deterministic Beam 10, return sequence 10, max length 200.

## 3. 날짜순 실험 기록

### 2026-08-08 — 초기 VGWD baseline 및 LoRA r8

#### 초기 비정제 VGWD

초기 `data/vgwd` test 평가에서 674개 샘플이 평가되었다.

| 모델 | strict Top-1 | strict Top-10 | Morgan Top-1 | Morgan best-10 | formula Top-1 | formula any-10 |
|---|---:|---:|---:|---:|---:|---:|
| pretrained, Beam 10 | 0/674 (0%) | 0/674 (0%) | 0.147331 | 0.202409 | 0.4451% | 1.3353% |
| LoRA r8/alpha16 q/v, step 4000 | 111/674 (16.4688%) | 365/674 (54.1543%) | 0.560521 | 0.804571 | 48.8131% | 82.6409% |

관련 경로:

- `configs/finetune_vgwd_lora_full.yaml`
- `configs/predict_vgwd_lora.yaml`
- `predictions/vgwd_original_full/`
- `predictions/vgwd_lora_full/`

#### 정제 VGWD clean

정제 split의 test 675개 결과는 다음과 같다.

| 모델 | strict Top-1 | strict Top-10 | Morgan Top-1 | Morgan best-10 | formula Top-1 | formula any-10 |
|---|---:|---:|---:|---:|---:|---:|
| pretrained, Beam 10 | 0/675 (0%) | 2/675 (0.2963%) | 0.141221 | 0.194774 | 0.8889% | 2.8148% |
| LoRA r8/alpha16 q/v, seed 42, step 4000 | 103/675 (15.2593%) | 359/675 (53.1852%) | 0.548615 | 0.808306 | 49.7778% | 85.1852% |

LoRA r8의 validation Morgan은 step 4000에서 0.507734였고 이 checkpoint가 사용되었다.

관련 경로:

- `configs/finetune_vgwd_clean_lora.yaml`
- `predictions/vgwd_clean_original/`
- `predictions/vgwd_clean_lora/`

해석: pretrained 모델보다 domain fine-tuning의 이득이 매우 컸다. 다만 이 초기 단계에서 test를 여러 adapter 비교에 사용했으므로, 이 test를 이후 선택 과정과 완전히 독립된 pristine test라고 부를 수 없다.

### 2026-08-12 — VGWD clean adapter screening

모든 결과는 test 675개, seed 42, Beam 10이다.

| 모델 | 사용 checkpoint | strict Top-1 | strict Top-10 | Morgan Top-1 | Morgan best-10 | formula Top-1 | formula any-10 | step 4000 validation Morgan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA r16/alpha32 | 4000 | 124 (18.3704%) | 424 (62.8148%) | 0.587632 | 0.854974 | 54.0741% | 87.8519% | 0.570743 |
| DoRA r8/alpha16 | 4000 | 102 (15.1111%) | 376 (55.7037%) | 0.550809 | 0.820994 | 49.6296% | 84.2963% | 0.519269 |
| AdaLoRA init16→target8/alpha32 | `4000-repaired` | 47 (6.9630%) | 172 (25.4815%) | 0.443331 | 0.646462 | 32.4444% | 68.4444% | 0.396293 |
| rsLoRA r16/alpha32 | 4000 | 187 (27.7037%) | 516 (76.4444%) | 0.662471 | 0.914465 | 64.1481% | 93.4815% | 0.637767 |

관련 경로:

- `configs/finetune_vgwd_clean_lora_r16.yaml`
- `configs/finetune_vgwd_clean_dora_r8.yaml`
- `configs/finetune_vgwd_clean_adalora.yaml`
- `configs/finetune_vgwd_clean_rslora_r16.yaml`
- `predictions/vgwd_clean_lora_r16/`
- `predictions/vgwd_clean_dora_r8/`
- `predictions/vgwd_clean_adalora/`
- `predictions/vgwd_clean_rslora_r16/`

결정: 이 screening에서는 rsLoRA r16이 가장 좋은 결과를 보여 후속 rank 연구의 기준 모델로 채택되었다. DoRA와 AdaLoRA는 현재 경로에서 중단했다. 단, 이 순위는 동일 VGWD test를 열람한 사후 비교이므로 완전히 독립적인 모델 선택 증거로 과대해석하지 않는다.

### 2026-08-27 — train-only spectral analog reranker

rsLoRA r16 validation 후보를 대상으로, train molecule의 Morgan 근접 이웃에서 얻은 spectral cosine을 생성 log-probability와 결합하는 reranker를 평가했다. 탐색 범위는 이웃 수 1/3/5/10, similarity power 1/2, linear/sqrt mode, alpha 0.1/0.25/0.5/0.75/1/1.5/2였으며 5-fold structure-grouped cross-validation을 사용했다. 이 분석의 exact 기준은 strict stereo exact가 아니라 connectivity exact이다.

- validation baseline Top-1: 176/664 = 26.5060%
- validation oracle hit-10: 513/664 = 77.2590%
- 전체 validation에 맞춘 최고 설정: `k=10`, `power=1`, `sqrt`, `alpha=0.5`
- 해당 설정 Top-1: 182/664 = 27.4096%, +6개, +0.9036 percentage point
- promoted 27, demoted 21, McNemar `p=0.470879`
- structure-grouped OOF Top-1: 171/664 = 25.7530%, baseline 대비 -5개
- OOF promoted 15, demoted 20
- OOF gain 95% CI: [-2.5602, +0.9036] percentage point
- OOF McNemar `p=0.499560`

결정: 전체 validation에 직접 튜닝한 수치의 소폭 상승은 OOF에서 재현되지 않았으므로 중단했다. test에는 적용하지 않았다.

관련 경로:

- `analysis/spectral_reranker/analog_reranker_config.json`
- `analysis/spectral_reranker/run_train_analog_reranker.py`
- `analysis/spectral_reranker/analog_reranker_grid_results.csv`
- `analysis/spectral_reranker/analog_reranker_cv_results.csv`

### 2026-08-27 — rsLoRA r32/alpha64 개발과 validation 선택

설정은 rsLoRA r32/alpha64, q/v, dropout 0.05, seed 42이다. 학습 파라미터는 4,718,592 / 358,736,896 = 1.315%였고, 4,000 steps 학습에 약 48분 55초가 걸렸다.

#### Validation 학습 곡선

| step | loss | Morgan | exact molecule | formula | canonical validity |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.25127 | 0.45678 | 0.06777 | 0.35693 | 0.92470 |
| 1000 | 0.19654 | 0.54677 | 0.13855 | 0.45783 | 0.92319 |
| 1500 | 0.16145 | 0.60346 | 0.20482 | 0.51807 | 0.94277 |
| 2000 | 0.15453 | 0.60784 | 0.19880 | 0.53916 | 0.93825 |
| 2500 | 0.12967 | 0.66983 | 0.30271 | 0.63404 | 0.93976 |
| 3000 | 0.12397 | 0.67382 | 0.29367 | 0.64307 | 0.94127 |
| 3500 | 0.11791 | **0.70029** | **0.34187** | 0.68976 | 0.93675 |
| 4000 | 0.11327 | 0.69403 | **0.34187** | 0.71988 | 0.93976 |

checkpoint 3500은 validation Morgan 최고값을 기준으로 선택했다.

#### Validation beam 선택

| beam | strict Top-1 | strict any-k | Morgan Top-1 | Morgan best-k | 결과 |
|---:|---:|---:|---:|---:|---|
| 10 | 212/664 (31.9277%) | 542/664 (81.6265%) | 0.692692 | 0.930903 | 최종 선택 |
| 20 | 203/664 (30.5723%) | 579/664 (87.1988%) | 0.684703 | 0.952012 | 완료, Top-1 하락 |
| 30 | 부분 412개 | 해당 없음 | 해당 없음 | 해당 없음 | 긴 decoder batch에서 CUBLAS 실패; 비교에서 제외 |
| 50 | 부분 62개 | 해당 없음 | 해당 없음 | 해당 없음 | 긴 decoder batch에서 CUBLAS 실패; 비교에서 제외 |

Beam 10은 validation Top-1이 더 높고 메모리 사용량이 낮으며 Top-10도 강했기 때문에 선택했다. Beam 20의 더 높은 후보 회수율은 확인되었지만 최종 주 프로토콜에는 채택하지 않았다.

#### 선택 고정 후 test 결과

checkpoint 3500, Beam 10으로 test 675개를 한 번의 주 평가 단위로 보고했다.

| 지표 | 결과 |
|---|---:|
| strict Top-1 | 250/675 = 37.0370% |
| strict Top-10 | 574/675 = 85.0370% |
| fingerprint-perfect Top-1 | 270/675 = 40.0000% |
| fingerprint-perfect Top-10 | 582/675 = 86.2222% |
| Morgan Top-1 | 0.722644 |
| Morgan best-10 | 0.943691 |
| formula Top-1 | 70.6667% |
| formula any-10 | 94.6667% |

seed 42의 기존 rsLoRA r16/alpha32와 비교하면 strict Top-1은 +9.3333 pp, Top-10은 +8.5926 pp였다.

strict cumulative Top-k는 다음과 같다.

| 모델 | k1 | k2 | k3 | k4 | k5 | k6 | k7 | k8 | k9 | k10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r16 | 27.704 | 44.000 | 52.000 | 58.963 | 63.259 | 66.815 | 70.963 | 74.370 | 75.704 | 76.444 |
| r32 | 37.037 | 53.333 | 62.963 | 68.593 | 73.185 | 77.778 | 81.333 | 83.259 | 84.444 | 85.037 |

관련 경로:

- `analysis/rslora_r32_experiment_summary.md`
- `config_runners/run_vgwd_clean_rslora_r32.sh`
- `configs/finetune_vgwd_clean_rslora_r32.yaml`
- `configs/predict_vgwd_clean_rslora_r32_valid_beam10.yaml`
- `configs/predict_vgwd_clean_rslora_r32_valid_beam20.yaml`
- `configs/predict_vgwd_clean_rslora_r32_test_beam10.yaml`
- `checkpoints/vgwd_clean_rslora_r32/2026-08-27-10_23_07_vgwd_clean_rslora_r32/checkpoint-3500/`
- `predictions/vgwd_clean_rslora_r32_test_beam10/`

결정: 현재 주 모델은 rsLoRA r32/alpha64 q/v, checkpoint 3500, Beam 10이다.

### 2026-08-27 — r16 대 r32 paired test 분석

seed 42에서 같은 675개 test 샘플을 paired comparison했다. 아래 CI는 20,000회 sample bootstrap이다.

| 지표 | r16 | r32 | 차이 | 95% CI | paired 검정 |
|---|---:|---:|---:|---:|---:|
| strict Top-1 | 27.7037% | 37.0370% | +9.3333 pp | [5.4815, 13.1852] pp | McNemar `p=5.36783e-06`, promoted 126, demoted 63 |
| strict Top-10 | 76.4444% | 85.0370% | +8.5926 pp | [5.9259, 11.2593] pp | McNemar `p=1.86215e-09`, promoted 77, demoted 19 |
| Morgan Top-1 | 0.662471 | 0.722644 | +0.060174 | [0.041113, 0.079486] | paired bootstrap |
| Morgan best-10 | 0.914465 | 0.943691 | +0.029226 | [0.019281, 0.039374] | paired bootstrap |
| formula Top-1 | 64.1481% | 70.6667% | +6.5185 pp | [2.6667, 10.3704] pp | McNemar `p=0.0010559` |
| formula Top-10 | 93.4815% | 94.6667% | +1.1852 pp | [-0.2963, 2.6667] pp | McNemar `p=0.18493` |

strict Top-1과 Top-10의 Benjamini-Hochberg 보정 q-value는 각각 `7.1571e-06`, `4.9657e-09`였다. formula Top-10 차이는 유의하지 않았다.

r32 Beam 10의 miss 유형은 formula-only 49개, low-similarity generation miss 31개, high-similarity non-exact 13개, connectivity/stereo mismatch 5개, fingerprint-perfect but non-exact 3개였다.

관련 경로:

- `analysis/analyze_r16_vs_r32_paper.py`
- `analysis/r16_vs_r32_paper/paper_analysis_report.md`
- `analysis/r16_vs_r32_paper/paper_analysis_summary.json`
- `analysis/audit_exact_topk.py`

통계적 주의: 이 최초 표의 sample bootstrap과 McNemar 검정은 675개 spectrum row를 독립 단위처럼 취급한다. 동일 구조의 반복 측정이 있으므로 분자 구조 수준의 독립 추론으로는 낙관적일 수 있다. 이후 scaling ablation에서는 canonical structure cluster bootstrap을 추가했다.

### 2026-08-27 — rsLoRA r32 seed 재현성

seed별로 모델 선택을 다시 하지 않고 checkpoint 3500과 Beam 10을 고정했다.

| seed | strict Top-1 | strict Top-10 | Morgan Top-1 | Morgan best-10 |
|---:|---:|---:|---:|---:|
| 42 | 250/675 (37.0370%) | 574/675 (85.0370%) | 0.722644 | 0.943691 |
| 123 | 242/675 (35.8519%) | 573/675 (84.8889%) | 0.718024 | 0.939629 |
| 2026 | 240/675 (35.5556%) | 552/675 (81.7778%) | 0.719549 | 0.936978 |

| 지표 | 3-seed mean | SD | range | t 기반 95% CI |
|---|---:|---:|---:|---:|
| strict Top-1 | 36.1481% | 0.7839 pp | 35.5556–37.0370% | [34.2008, 38.0955]% |
| strict Top-10 | 83.9012% | 1.8405 pp | 81.7778–85.0370% | [79.3293, 88.4732]% |
| Morgan Top-1 | 0.720072 | 0.002354 | 0.718024–0.722644 | [0.714224, 0.725921] |
| Morgan best-10 | 0.940099 | 0.003381 | 0.936978–0.943691 | [0.931701, 0.948498] |

결론: r32의 성능 방향은 세 seed에서 재현되었다. 그러나 seed 수가 3개뿐이라 t interval은 매우 불확실하며 모집단 seed 분산을 정밀하게 추정한 것으로 해석하지 않는다.

관련 경로:

- `analysis/r32_seed_reproducibility/`
- `analysis/r32_seed_reproducibility/seed_reproducibility_report.md`

### 2026-08-27~28 — r16과 r32의 same-seed 재현성

seed 123과 2026에서 r16과 r32를 같은 seed끼리 직접 비교했다. 두 rank 모두 checkpoint 3500, Beam 10을 고정했으며 seed별 validation 최고 checkpoint를 사후 선택하지 않았다. seed 42 모델은 이 matched-repeat 프로토콜 이전의 역사적 실행이므로 이 2-seed 요약에는 포함하지 않았다.

| seed | rank | strict Top-1 | strict Top-10 | Morgan Top-1 | Morgan best-10 |
|---:|---:|---:|---:|---:|---:|
| 123 | r16 | 190/675 (28.1481%) | 504/675 (74.6667%) | 0.661095 | 0.909590 |
| 123 | r32 | 242/675 (35.8519%) | 573/675 (84.8889%) | 0.718024 | 0.939629 |
| 2026 | r16 | 178/675 (26.3704%) | 499/675 (73.9259%) | 0.657271 | 0.904556 |
| 2026 | r32 | 240/675 (35.5556%) | 552/675 (81.7778%) | 0.719549 | 0.936978 |

| seed | strict Top-1 gain r32-r16 | McNemar | strict Top-10 gain | McNemar |
|---:|---:|---:|---:|---:|
| 123 | +7.7037 pp | `p=5.36477e-05`, promoted 107, demoted 55 | +10.2222 pp | `p=4.22241e-13`, promoted 83, demoted 14 |
| 2026 | +9.1852 pp | `p=2.25875e-06`, promoted 116, demoted 54 | +7.8519 pp | `p=3.19613e-09`, promoted 68, demoted 15 |

2-seed 요약:

- r16 Top-1: mean 27.2593%, SD 1.2571 pp
- r32 Top-1: mean 35.7037%, SD 0.2095 pp
- mean Top-1 gain: +8.4444 pp, seed별 범위 +7.7037~+9.1852 pp
- r16 Top-10: mean 74.2963%, SD 0.5238 pp
- r32 Top-10: mean 83.3333%, SD 2.1999 pp
- mean Top-10 gain: +9.0370 pp, seed별 범위 +7.8519~+10.2222 pp

관련 경로:

- `analysis/rslora_rank_matched_seeds/`
- `analysis/rslora_rank_matched_seeds/matched_seed_report.md`

결론: 관찰된 r32 우위는 seed 123과 2026에서도 같은 방향으로 반복되었다. 다만 McNemar p-value는 각 seed 안의 sample-level paired difference에 관한 것이며, seed 모집단에 대한 유의성 검정은 아니다.

### 2026-08-28 — learned cross-modal reranker

r32 Beam 10 validation 664개 후보에 learned cross-modal reranker를 적용했다. test prediction은 만들지 않았고, structure-grouped 5-fold OOF 및 model seed 1/7/21을 사용했다.

- baseline strict Top-1: 212/664 = 31.9277%
- reranked strict Top-1: 213/664 = 32.0783%
- gain: +1개, +0.1506 pp
- promoted 9, demoted 8
- McNemar `p=1.0`
- bootstrap 95% CI: [-1.0542, +1.3554] pp
- oracle Top-10: 542/664

결정: `stop_no_robust_oof_gain`. test를 사용하지 않고 중단했으며, reranker가 개선되었다고 주장하지 않는다.

관련 경로:

- `analysis/learned_candidate_reranker/run_crossmodal_reranker.py`
- `analysis/learned_candidate_reranker/results/`

### 2026-08-28 — VGWD split 데이터 누수 감사

train 5,202, filtered validation 664, test 675를 비교했다.

#### Split 간 직접 중복

| 비교 | exact structure overlap | connectivity overlap | spectrum hash overlap |
|---|---:|---:|---:|
| train-validation | 0 | 0 | 0 |
| train-test | 0 | 0 | 0 |
| validation-test | 0 | 0 | 0 |

exact molecule, stereochemistry를 제거한 connectivity, 정규화한 spectrum hash 기준의 직접적인 교차 split 중복은 발견되지 않았다.

#### Scaffold overlap

| 비교 | 공유 scaffold | 뒤쪽 split에서 영향을 받는 row |
|---|---:|---:|
| train-validation | 34 | 141/664 (21.2349%) |
| train-test | 25 | 133/675 (19.7037%) |
| validation-test | 17 | 119/675 (17.6296%) |

따라서 이 데이터는 scaffold-disjoint가 아니다.

#### Split 내부 반복 구조

| split | exact unique / rows | exact duplicate groups | duplicate-group rows | connectivity unique |
|---|---:|---:|---:|---:|
| train | 3,953/5,202 | 967 | 2,216 | 3,947 |
| validation | 492/664 | 133 | 305 | 491 |
| test | 494/675 | 138 | 319 | 494 |

split 내부 spectrum hash 중복은 발견되지 않았다. 같은 구조에 서로 다른 스펙트럼이 여러 개 존재하기 때문에 spectrum row를 완전히 독립적인 molecule observation으로 간주하면 불확실성을 과소평가할 수 있다.

관련 경로:

- `analysis/audit_vgwd_split_leakage.py`
- `analysis/vgwd_split_leakage/`
- `analysis/vgwd_split_leakage/report.md`

### 2026-08-28 — rank × scaling 2×2 ablation

rank와 rsLoRA scaling을 분리하기 위해 다음 4개 조건을 seed 123과 2026에서 비교했다. checkpoint 3500과 Beam 10을 모든 조건에서 고정했다.

`alpha=45`는 rank 16에서 scale 11.250, rank 32에서 scale 7.955이므로 두 scale target이 수학적으로 완전히 동일하지는 않다. paired target scale에서 약 0.6% 차이가 남는다.

| rank | alpha | 유효 scale | seed | strict Top-1 | strict Top-10 |
|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 8.000 | 123 | 190/675 (28.1481%) | 504/675 (74.6667%) |
| 16 | 32 | 8.000 | 2026 | 178/675 (26.3704%) | 499/675 (73.9259%) |
| 16 | 45 | 11.250 | 123 | 201/675 (29.7778%) | 526/675 (77.9259%) |
| 16 | 45 | 11.250 | 2026 | 201/675 (29.7778%) | 520/675 (77.0370%) |
| 32 | 45 | 7.955 | 123 | 223/675 (33.0370%) | 559/675 (82.8148%) |
| 32 | 45 | 7.955 | 2026 | 218/675 (32.2963%) | 553/675 (81.9259%) |
| 32 | 64 | 11.314 | 123 | 242/675 (35.8519%) | 573/675 (84.8889%) |
| 32 | 64 | 11.314 | 2026 | 240/675 (35.5556%) | 552/675 (81.7778%) |

#### 두 seed의 평균 paired effect

| contrast | Top-1 mean effect | seed SD | Top-10 mean effect | seed SD |
|---|---:|---:|---:|---:|
| rank 16→32, low scale | +5.4074 pp | 0.7333 pp | +8.0741 pp | 0.1048 pp |
| rank 16→32, high scale | +5.9259 pp | 0.2095 pp | +5.8519 pp | 1.5713 pp |
| scale low→high, rank 16 | +2.5185 pp | 1.2571 pp | +3.1852 pp | 0.1048 pp |
| scale low→high, rank 32 | +3.0370 pp | 0.3143 pp | +0.9630 pp | 1.5713 pp |

#### Canonical-structure cluster bootstrap와 paired McNemar

아래 CI는 같은 canonical structure의 반복 spectrum을 하나의 cluster로 함께 재표집했다.

| contrast | seed | 지표 | 차이 | structure-cluster bootstrap 95% CI | McNemar p |
|---|---:|---|---:|---:|---:|
| rank, low scale | 123 | Top-1 | +4.8889 pp | [0.9103, 8.7152] | 0.007656 |
| rank, low scale | 2026 | Top-1 | +5.9259 pp | [2.6393, 9.2678] | 0.0005715 |
| rank, low scale | 123 | Top-10 | +8.1481 pp | [5.4135, 10.9467] | 1.957e-09 |
| rank, low scale | 2026 | Top-10 | +8.0000 pp | [5.2315, 10.8927] | 3.206e-09 |
| rank, high scale | 123 | Top-1 | +6.0741 pp | [2.2726, 10.1620] | 0.0016469 |
| rank, high scale | 2026 | Top-1 | +5.7778 pp | [1.8018, 9.5588] | 0.0020249 |
| rank, high scale | 123 | Top-10 | +6.9630 pp | [4.4247, 9.5312] | 3.808e-08 |
| rank, high scale | 2026 | Top-10 | +4.7407 pp | [2.1645, 7.4134] | 0.0003778 |
| scaling, rank 16 | 123 | Top-1 | +1.6296 pp | [-2.0866, 5.3733] | 0.3998 |
| scaling, rank 16 | 2026 | Top-1 | +3.4074 pp | [0.1439, 6.6479] | 0.05049 |
| scaling, rank 16 | 123 | Top-10 | +3.2593 pp | [0.8559, 5.8559] | 0.01277 |
| scaling, rank 16 | 2026 | Top-10 | +3.1111 pp | [0.3007, 5.8914] | 0.02565 |
| scaling, rank 32 | 123 | Top-1 | +2.8148 pp | [-0.7206, 6.4027] | 0.1014 |
| scaling, rank 32 | 2026 | Top-1 | +3.2593 pp | [-0.4432, 6.9465] | 0.07345 |
| scaling, rank 32 | 123 | Top-10 | +2.0741 pp | [-0.2954, 4.4844] | 0.08143 |
| scaling, rank 32 | 2026 | Top-10 | -0.1481 pp | [-2.7497, 2.4062] | 1.0 |

결론:

- rank 32의 이점은 low-scale과 high-scale 모두에서, 그리고 두 seed 모두에서 같은 방향으로 나타났다.
- scaling 증가는 rank 증가보다 효과가 작고 일관성이 낮았다. 특히 rank 32 Top-10은 seed 2026에서 -0.1481 pp였다.
- 따라서 원래 r16/alpha32 대 r32/alpha64 차이를 scaling만의 효과로 설명할 수 없다. 이 실험 범위에서는 rank 증가가 별도의 성능 기여를 보였다.
- seed가 2개뿐이므로 이 표의 McNemar와 cluster bootstrap은 각 seed 안의 test sample/structure 차이에 관한 근거이며, seed 모집단 수준의 강한 통계적 일반화는 아니다.
- 이 ablation 자체가 기존 test를 사용했으므로 논문에서는 고정 VGWD benchmark에 대한 사후/확인적 분석으로 투명하게 표시해야 한다. 이후 새 설계 선택은 validation 또는 새 외부 holdout에서 수행해야 한다.

관련 경로:

- `config_runners/run_vgwd_clean_rslora_scaling_ablation.sh`
- `analysis/rslora_scaling_ablation/`
- `analysis/rslora_scaling_ablation/report.md`
- `analysis/rslora_scaling_ablation/paired_comparisons.csv`
- `analysis/rslora_scaling_ablation/run_metrics.csv`

### 2026-08-29~30 — rank × scaling ablation을 총 5 matched seeds로 확장

#### 사전 고정과 실행 범위

2026-08-29에 결과를 생성하기 전에 기존 primary seeds 123/2026에 추가할 seeds를 7/314/1729로 고정하고 `analysis/rslora_scaling_ablation_5seeds/protocol.md`에 기록했다. seed 42는 과거에 실행한 r32 결과이므로 matched 2×2 primary 분석에는 넣지 않았다. checkpoint 3500, deterministic Beam 10, test 675 rows, 네 조건 r16/alpha32·r16/alpha45·r32/alpha45·r32/alpha64를 모든 seed에서 동일하게 적용했으며, 실행 후 seed를 제외하지 않았다.

2026-08-30에 새 seeds 7/314/1729의 12개 학습과 예측이 모두 완료되었다. 각 실행에 checkpoint-3500 하나, 675행의 완전한 prediction 하나, `num_samples=675`인 strict exact 감사 결과 하나가 있음을 확인했다.

#### 새로 추가된 12개 test 결과

| seed | r16/a32 Top-1 / Top-10 | r16/a45 Top-1 / Top-10 | r32/a45 Top-1 / Top-10 | r32/a64 Top-1 / Top-10 |
|---:|---:|---:|---:|---:|
| 7 | 189/675 (28.0000%) / 504/675 (74.6667%) | 214/675 (31.7037%) / 508/675 (75.2593%) | 236/675 (34.9630%) / 548/675 (81.1852%) | 231/675 (34.2222%) / 560/675 (82.9630%) |
| 314 | 202/675 (29.9259%) / 507/675 (75.1111%) | 212/675 (31.4074%) / 528/675 (78.2222%) | 226/675 (33.4815%) / 542/675 (80.2963%) | 227/675 (33.6296%) / 555/675 (82.2222%) |
| 1729 | 191/675 (28.2963%) / 498/675 (73.7778%) | 208/675 (30.8148%) / 530/675 (78.5185%) | 223/675 (33.0370%) / 555/675 (82.2222%) | 255/675 (37.7778%) / 568/675 (84.1481%) |

#### 총 5 seeds의 paired effect

아래 95% CI는 다섯 training seed의 paired gain에 대한 t interval이다. 각 seed 안의 McNemar 검정이나 structure-cluster bootstrap과 분석 단위가 다르다.

| contrast | 지표 | 평균 이득 | seed SD | seed 범위 | 양수 seed | seed-t 95% CI |
|---|---|---:|---:|---:|---:|---:|
| r16/a32 → r32/a64, 원 비교 | Top-1 | +7.2593 pp | 2.3750 pp | +3.7037~+9.4815 pp | 5/5 | [+4.3103, +10.2082] pp |
| r16/a32 → r32/a64, 원 비교 | Top-10 | +8.7704 pp | 1.4568 pp | +7.1111~+10.3704 pp | 5/5 | [+6.9615, +10.5793] pp |
| rank 16→32, low scale | Top-1 | +5.2148 pp | 1.2890 pp | +3.5556~+6.9630 pp | 5/5 | [+3.6143, +6.8153] pp |
| rank 16→32, low scale | Top-10 | +7.2593 pp | 1.3779 pp | +5.1852~+8.4444 pp | 5/5 | [+5.5484, +8.9701] pp |
| rank 16→32, high scale | Top-1 | +4.7111 pp | 2.1834 pp | +2.2222~+6.9630 pp | 5/5 | [+2.0001, +7.4221] pp |
| rank 16→32, high scale | Top-10 | +5.8074 pp | 1.5303 pp | +4.0000~+7.7037 pp | 5/5 | [+3.9073, +7.7075] pp |
| scale low→high, rank 16 | Top-1 | +2.5481 pp | 1.0070 pp | +1.4815~+3.7037 pp | 5/5 | [+1.2978, +3.7985] pp |
| scale low→high, rank 16 | Top-10 | +2.9630 pp | 1.4926 pp | +0.5926~+4.7407 pp | 5/5 | [+1.1097, +4.8162] pp |
| scale low→high, rank 32 | Top-1 | +2.0444 pp | 2.2744 pp | -0.7407~+4.7407 pp | 4/5 | [-0.7797, +4.8685] pp |
| scale low→high, rank 32 | Top-10 | +1.5111 pp | 0.9335 pp | -0.1481~+2.0741 pp | 4/5 | [+0.3521, +2.6701] pp |

#### 결론과 제한

- scaling을 낮게 맞춘 비교와 높게 맞춘 비교 모두에서 rank 32의 Top-1·Top-10 이득이 5/5 seeds에서 양수였다. 두 rank contrast의 seed-t 95% CI도 0보다 높아, 이 데이터와 고정 프로토콜에서는 rank 증가가 scaling과 별개의 기여를 한다는 근거가 2-seed 분석보다 강해졌다.
- 원래 채택 비교인 r16/alpha32→r32/alpha64도 Top-1과 Top-10 모두 5/5 seeds에서 양수였다.
- scaling 증가는 rank 16에서는 5/5 seeds에서 양수였지만, rank 32에서는 Top-1과 Top-10이 각각 4/5 seeds에서만 양수였다. 특히 rank 32 Top-1 seed-t 95% CI는 0을 포함하므로 scaling 증가의 이득을 일관된 효과로 단정하지 않는다.
- n=5는 이전 n=2보다 낫지만 여전히 작은 training-seed 표본이다. t interval은 불확실하며, 모든 seed에 대한 보장이나 seed 모집단의 정밀한 분산 추정으로 표현하지 않는다.
- 동일한 VGWD test를 다시 사용했으므로 이 확장은 training-seed robustness를 평가한 것이다. 외부 데이터, unseen scaffold, 또는 untouched final test에 대한 일반화 검증은 아니다.
- alpha 정수 제약으로 paired low/high scale은 약 0.6% 차이가 남아 완벽히 동일하지 않다.

관련 경로:

- `analysis/rslora_scaling_ablation_5seeds/protocol.md`
- `analysis/rslora_scaling_ablation_5seeds/report.md`
- `analysis/rslora_scaling_ablation_5seeds/run_metrics.csv`
- `analysis/rslora_scaling_ablation_5seeds/paired_comparisons.csv`
- `analysis/rslora_scaling_ablation_5seeds/aggregate_effects.csv`
- `analysis/rslora_scaling_ablation_5seeds/summary.json`
- `analysis/generate_rslora_scaling_seed_configs.py`
- `analysis/summarize_rslora_scaling_ablation.py`
- `config_runners/run_vgwd_clean_rslora_scaling_ablation.sh`
- `configs/finetune_vgwd_clean_rslora_r16_seed7.yaml`, `configs/finetune_vgwd_clean_rslora_r16_seed314.yaml`, `configs/finetune_vgwd_clean_rslora_r16_seed1729.yaml`
- `configs/finetune_vgwd_clean_rslora_r16_alpha45_seed7.yaml`, `configs/finetune_vgwd_clean_rslora_r16_alpha45_seed314.yaml`, `configs/finetune_vgwd_clean_rslora_r16_alpha45_seed1729.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_alpha45_seed7.yaml`, `configs/finetune_vgwd_clean_rslora_r32_alpha45_seed314.yaml`, `configs/finetune_vgwd_clean_rslora_r32_alpha45_seed1729.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_seed7.yaml`, `configs/finetune_vgwd_clean_rslora_r32_seed314.yaml`, `configs/finetune_vgwd_clean_rslora_r32_seed1729.yaml`

### 2026-08-30 — r16 all-attention projection validation-only screen

#### 사전 고정과 실행 범위

rank와 scaling을 고정한 상태에서 학습 대상 attention projection을 넓히는 효과를 확인했다. 기준군은 rsLoRA r16/alpha32의 `q_proj`, `v_proj`이고 실험군은 동일한 r16/alpha32에서 `q_proj`, `k_proj`, `v_proj`, `out_proj`를 학습했다. SpecTUS는 BART 계열이므로 출력 projection의 실제 모듈명은 `o_proj`가 아니라 `out_proj`이다. 이 이름들은 encoder self-attention, decoder self-attention, decoder cross-attention에 존재하는 모든 같은 이름의 projection에 적용된다.

- 결과 생성 전에 seeds 7/123/2026, checkpoint 3500, deterministic Beam 10, strict Top-1 primary endpoint를 고정했다.
- 사전 진행 규칙은 Top-1 평균 이득이 양수이고 3개 중 최소 2개 seed에서 Top-1 이득이 양수이면 r32 all-attention validation screen으로 진행하는 것이었다.
- 기존 q/v checkpoint는 재학습하지 않고 같은 seed의 checkpoint 3500에서 다시 validation Beam 10으로 추론했다.
- 실험군 smoke test에서 q/k/v/out projection이 각각 36개, 총 144개 LoRA 대상에 연결되고 trainable parameter가 4,718,592개임을 확인했다.
- 평가에는 filtered validation 664개만 사용했고 VGWD test prediction은 생성하지 않았다.

#### 개별 validation 결과

| seed | q/v Top-1 | all-attention Top-1 | 이득 | q/v Top-10 | all-attention Top-10 | 이득 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 165/664 (24.8494%) | 225/664 (33.8855%) | +9.0361 pp | 489/664 (73.6446%) | 550/664 (82.8313%) | +9.1867 pp |
| 123 | 182/664 (27.4096%) | 218/664 (32.8313%) | +5.4217 pp | 493/664 (74.2470%) | 541/664 (81.4759%) | +7.2289 pp |
| 2026 | 163/664 (24.5482%) | 216/664 (32.5301%) | +7.9819 pp | 461/664 (69.4277%) | 547/664 (82.3795%) | +12.9518 pp |

#### seed 집계와 paired validation 통계

| 지표 | 평균 이득 | seed SD | seed 범위 | 양수 seed | seed-t 95% CI |
|---|---:|---:|---:|---:|---:|
| strict Top-1 | +7.4799 pp | 1.8588 pp | +5.4217~+9.0361 pp | 3/3 | [+2.8624, +12.0974] pp |
| strict Top-10 | +9.7892 pp | 2.9086 pp | +7.2289~+12.9518 pp | 3/3 | [+2.5638, +17.0146] pp |

아래 CI는 validation 내부의 같은 canonical structure 반복을 하나의 cluster로 함께 재표집했다. McNemar와 cluster bootstrap은 각 seed 안의 validation sample/structure 차이에 관한 분석이며 seed-level 검정이 아니다.

| seed | 지표 | structure-cluster bootstrap 95% CI | exact McNemar p |
|---:|---|---:|---:|
| 7 | Top-1 | [+4.8062, +13.3942] pp | 5.55812e-06 |
| 123 | Top-1 | [+1.7908, +9.2006] pp | 0.00437094 |
| 2026 | Top-1 | [+4.0283, +11.9449] pp | 2.82309e-05 |
| 7 | Top-10 | [+6.2027, +12.3311] pp | 4.39403e-10 |
| 123 | Top-10 | [+4.7196, +9.8509] pp | 8.05550e-09 |
| 2026 | Top-10 | [+9.7412, +16.3266] pp | 3.48792e-17 |

#### 실행 중 교정, 결론과 제한

- 첫 q/v seed 7 예측 후 감사기가 prediction 664행과 원본 `valid.jsonl` 666행의 불일치를 발견하여 실행을 중단했다. 추론 파이프라인이 기존 `m/z <= 500` 규칙으로 2행을 제외한다는 것을 확인하고, 감사·통계 라벨 경로를 프로젝트의 기존 주 validation 파일 `valid_filtered_mz500.jsonl` 664행으로 교정했다. 이는 결과에 따른 sample 선택이 아니라 문서 1절에 이미 정해진 validation population과의 경로 정합성 수정이며, 수정 전 잘못된 666행 통계는 생성되지 않았다.
- all-attention은 primary Top-1과 secondary Top-10 모두 3/3 seeds에서 q/v보다 높았다. 사전에 정한 r32 all-attention validation screen 진행 기준을 통과했다.
- 이 결과는 r16에서 target module 확장을 계속 검토할 근거이며 현재 r32/alpha64 q/v 주 모델을 즉시 교체하는 test 근거는 아니다. 다음 단계는 r32/alpha64 all-attention을 동일한 validation-only 원칙으로 비교하는 것이다.
- seed가 3개뿐이고 같은 validation이 모델 선택에 사용되었으므로 t interval은 탐색적이다. 외부 일반화나 최종 test 성능 향상을 뜻하지 않는다.
- 이 screen에서는 VGWD test를 열람하거나 생성하지 않았다. 향후 r32/FFN 선택도 validation에서만 수행하고 최종 확인에는 잠금 holdout 또는 외부 데이터를 사용해야 한다.

관련 경로:

- `analysis/rslora_all_attention_validation/protocol.md`
- `analysis/rslora_all_attention_validation/report.md`
- `analysis/rslora_all_attention_validation/run_metrics.csv`
- `analysis/rslora_all_attention_validation/paired_comparisons.csv`
- `analysis/rslora_all_attention_validation/aggregate_effects.csv`
- `analysis/rslora_all_attention_validation/summary.json`
- `analysis/generate_rslora_all_attention_configs.py`
- `analysis/summarize_rslora_all_attention_validation.py`
- `config_runners/run_vgwd_clean_rslora_all_attention_validation.sh`
- `configs/finetune_vgwd_clean_rslora_r16_allattn_seed7.yaml`
- `configs/finetune_vgwd_clean_rslora_r16_allattn_seed123.yaml`
- `configs/finetune_vgwd_clean_rslora_r16_allattn_seed2026.yaml`
- `predictions/vgwd_clean_rslora_r16_seed7_valid_beam10/`, `predictions/vgwd_clean_rslora_r16_seed123_valid_beam10/`, `predictions/vgwd_clean_rslora_r16_seed2026_valid_beam10/`
- `predictions/vgwd_clean_rslora_r16_allattn_seed7_valid_beam10/`
- `predictions/vgwd_clean_rslora_r16_allattn_seed123_valid_beam10/`
- `predictions/vgwd_clean_rslora_r16_allattn_seed2026_valid_beam10/`

### 2026-08-30 — r32 all-attention projection validation-only screen

#### 사전 고정과 실행 범위

r16 screen의 사전 진행 규칙 통과 후, rsLoRA r32/alpha64에서 q/v만 학습하는 기준군과 q/k/v/out을 모두 학습하는 실험군을 같은 seed끼리 비교했다.

- 결과 생성 전에 seeds 7/123/2026, checkpoint 3500, deterministic Beam 10, filtered validation 664개, strict Top-1 primary endpoint를 고정했다.
- 사전 선택 규칙은 all-attention의 평균 Top-1 이득이 양수이고 3개 중 최소 2개 seed에서 Top-1 이득이 양수이면 향후 잠금 holdout 또는 외부 평가 후보로 선택하는 것이었다. Top-10은 secondary endpoint였다.
- rank 32, alpha 64, dropout 0.05 및 나머지 학습 조건은 q/v 기준군과 동일하게 두고 target module만 `q_proj`, `k_proj`, `v_proj`, `out_proj`로 넓혔다.
- smoke test에서 q/k/v/out projection이 각각 36개, 총 144개 LoRA 대상에 연결되고 trainable parameter가 9,437,184개임을 확인했다.
- 기존 q/v와 새 all-attention 모두 같은 seed의 checkpoint 3500을 사용했다. seed 123 학습 프로세스는 checkpoint 3500 저장 후 중단됐지만, 이 지점이 사전 고정된 평가 checkpoint였으므로 추가 step을 결과 선택에 사용하지 않았다. 이후 실행은 seed 2026부터 재개됐다.
- VGWD test prediction은 생성하거나 열람하지 않았다.

#### 개별 validation 결과

| seed | q/v Top-1 | all-attention Top-1 | 이득 | q/v Top-10 | all-attention Top-10 | 이득 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 227/664 (34.1867%) | 261/664 (39.3072%) | +5.1205 pp | 551/664 (82.9819%) | 581/664 (87.5000%) | +4.5181 pp |
| 123 | 213/664 (32.0783%) | 277/664 (41.7169%) | +9.6386 pp | 547/664 (82.3795%) | 581/664 (87.5000%) | +5.1205 pp |
| 2026 | 211/664 (31.7771%) | 250/664 (37.6506%) | +5.8735 pp | 538/664 (81.0241%) | 582/664 (87.6506%) | +6.6265 pp |

#### seed 집계와 paired validation 통계

| 지표 | 평균 이득 | seed SD | seed 범위 | 양수 seed | seed-t 95% CI |
|---|---:|---:|---:|---:|---:|
| strict Top-1 | +6.8775 pp | 2.4206 pp | +5.1205~+9.6386 pp | 3/3 | [+0.8644, +12.8906] pp |
| strict Top-10 | +5.4217 pp | 1.0860 pp | +4.5181~+6.6265 pp | 3/3 | [+2.7239, +8.1195] pp |

동일 canonical structure를 cluster로 함께 재표집한 5,000회 bootstrap과 같은 validation row의 성공/실패를 비교한 exact McNemar 결과는 다음과 같다. 이 값은 각 seed 내부의 validation sample/structure 분석이며 새로운 seed 또는 외부 데이터에 대한 검정이 아니다.

| seed | 지표 | structure-cluster bootstrap 95% CI | exact McNemar p |
|---:|---|---:|---:|
| 7 | Top-1 | [+0.4708, +9.6678] pp | 0.0206353 |
| 123 | Top-1 | [+5.4628, +13.6924] pp | 1.82940e-06 |
| 2026 | Top-1 | [+1.3761, +10.4234] pp | 0.00582338 |
| 7 | Top-10 | [+2.1773, +6.9231] pp | 0.000176327 |
| 123 | Top-10 | [+2.5954, +7.6586] pp | 7.55581e-05 |
| 2026 | Top-10 | [+4.0995, +9.2310] pp | 1.02899e-07 |

#### 결론과 제한

- all-attention은 primary Top-1과 secondary Top-10 모두 3/3 seeds에서 q/v보다 높았고 사전에 정한 선택 규칙을 통과했다. 따라서 r32/alpha64 q/k/v/out은 향후 잠금 holdout 또는 외부 평가에 가져갈 validation-selected 후보이다.
- 이 결과로 기존 VGWD test에서 all-attention이 더 높다고 주장할 수 없다. 같은 validation이 모델 선택에 사용됐으므로 이 split에서의 이득은 선택 편향의 영향을 받을 수 있다.
- seed가 3개뿐이므로 seed-t interval은 탐색적이다. 외부 일반화, 새로운 scaffold, 화생방 현장 데이터에서의 성능 또는 운용 적합성을 확정하지 않는다.
- 다음 강한 확인 단계는 기존 개발용 VGWD test를 다시 사용해 선택하는 것이 아니라, 잠근 holdout 또는 외부 데이터에서 q/v와 all-attention을 한 번 비교하는 것이다. FFN 포함 여부는 아직 실험하지 않았다.

관련 경로:

- `analysis/rslora_r32_all_attention_validation/protocol.md`
- `analysis/rslora_r32_all_attention_validation/report.md`
- `analysis/rslora_r32_all_attention_validation/run_metrics.csv`
- `analysis/rslora_r32_all_attention_validation/paired_comparisons.csv`
- `analysis/rslora_r32_all_attention_validation/aggregate_effects.csv`
- `analysis/rslora_r32_all_attention_validation/summary.json`
- `analysis/generate_rslora_r32_all_attention_configs.py`
- `analysis/summarize_rslora_r32_all_attention_validation.py`
- `config_runners/run_vgwd_clean_rslora_r32_all_attention_validation.sh`
- `configs/finetune_vgwd_clean_rslora_r32_allattn_seed7.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_allattn_seed123.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_allattn_seed2026.yaml`
- `predictions/vgwd_clean_rslora_r32_seed7_valid_beam10/`, `predictions/vgwd_clean_rslora_r32_seed123_valid_beam10/`, `predictions/vgwd_clean_rslora_r32_seed2026_valid_beam10/`
- `predictions/vgwd_clean_rslora_r32_allattn_seed7_valid_beam10/`
- `predictions/vgwd_clean_rslora_r32_allattn_seed123_valid_beam10/`
- `predictions/vgwd_clean_rslora_r32_allattn_seed2026_valid_beam10/`

### 2026-09-01 — r32 all-attention validation 오류·강건성 분석과 후보 동결

#### 분석 범위와 무결성 원칙

- r32 q/v와 all-attention의 기존 validation prediction만 사용한 기술적 분석이다. 새 학습이나 architecture 선택은 수행하지 않았다.
- 결과를 생성하기 전에 분자량, 최대 관측 m/z, peak 수, train scaffold 상태, stereochemistry, Top-10 회수, invalid SMILES, seed 간 공통 난제의 정의를 `analysis/rslora_r32_all_attention_diagnostics/protocol.md`에 고정했다.
- seeds 7/123/2026, checkpoint 3500, Beam 10, filtered validation 664개를 그대로 사용했고 test는 열람하거나 생성하지 않았다.
- 하위그룹 n<20은 CSV에는 보존하되 요약 해석 표에서는 제외했다. 하위그룹 결과는 다중검정에 의한 확증 결과가 아니라 기술 통계이다.

#### 오류와 강건성 결과

- all-attention의 strict Top-1/Top-10은 seed 7에서 39.3072%/87.5000%, seed 123에서 41.7169%/87.5000%, seed 2026에서 37.6506%/87.6506%였다.
- Top-10 내 invalid SMILES 후보는 q/v와 all-attention의 3개 seed 모두 0개였다.
- connectivity-correct이지만 strict stereochemistry Top-1이 틀린 사례는 all-attention에서 seed 7/123/2026 각각 0/1/0개, q/v에서 3/3/0개였다. 이 validation의 strict-connectivity 차이는 작았다.
- 같은 row의 q/v→all-attention strict Top-1 전환에서 새로 맞힌 수/기존 정답을 잃은 수는 seed 7이 119/85, seed 123이 121/57, seed 2026이 115/76이었다. 순이득은 각각 34/64/39개였다. 따라서 all-attention의 개선은 개별 sample에서 단조롭지 않다.
- 두 방법이 3개 seed 모두에서 틀린 공통 난제는 188/664개였다. q/v는 3개 seed 모두 틀렸으나 all-attention은 최소 2개 seed에서 맞힌 일관된 개선 사례는 35개, 반대의 일관된 퇴행 사례는 19개였다.
- 사전 정의하고 n≥20인 모든 분자량·최대 m/z·peak 수·scaffold 상태·stereochemistry 하위그룹에서 all-attention의 평균 Top-1 이득이 양수였고 3/3 seeds에서 양수였다.
- 평균 Top-1 이득은 분자량 `<150`, `150-249`, `250-349`, `>=350` Da에서 각각 +8.6022, +6.2196, +7.0588, +10.4762 pp였다. 각 그룹 n은 31/343/255/35이다.
- 최대 m/z `100-199`, `200-299`, `300-399` 그룹의 평균 Top-1 이득은 각각 +5.4313, +8.5842, +6.0000 pp였고 n은 313/299/50이었다. 작은 `<100` 및 `400-500` 그룹은 요약 해석에서 제외했다.
- acyclic 518개에서는 평균 Top-1 +6.1133 pp, train-seen scaffold 141개에서는 +10.1655 pp였다. train-unseen scaffold는 5개뿐이어서 unseen-scaffold 일반화를 판단할 수 없다.
- 이 분석은 all-attention의 validation 개선이 사전 정의한 큰 하위그룹 하나에만 국한된 현상은 아니라는 기술적 근거를 제공한다. 그러나 같은 validation이 방법 선택에 사용됐으므로 독립적인 일반화 확인은 아니다.

#### 동결 모델

- seed 123 checkpoint 3500을 `models/vgwd_clean_rslora_r32_all_attention_seed123_frozen/`에 검증 기반 운용 후보로 동결했다.
- 선택 이유는 사전 실행한 세 all-attention seed 가운데 primary validation strict Top-1이 가장 높았기 때문이다. test는 선택에 사용하지 않았으며, 이 선택은 validation에 낙관적일 수 있음을 모델 카드와 manifest에 명시했다.
- 동결 패키지는 추론에 필요한 `adapter_model.bin`, adapter 설정, tokenizer metadata만 포함한다. optimizer와 학습 재개 상태는 제외했다.
- adapter SHA-256은 `9471d17dda3ead052360ca0574d7f4f4989864b4ff4a237b7fb1488ed6b17513`이다. 기반 `SpecTUS_pretrained/pytorch_model.bin`의 SHA-256은 `4f8d487a51ab787dd2353253c16a71170c1843c612684ec884a0c9a5c20c13c4`이다.
- `SHA256SUMS` 검증에서 adapter 및 tokenizer 관련 5개 파일이 모두 일치했다.
- 이 동결은 재현 가능한 연구 후보를 보존한다는 뜻이며 외부 검증, 현장 검증 또는 배포 승인을 뜻하지 않는다.

관련 경로:

- `analysis/rslora_r32_all_attention_diagnostics/protocol.md`
- `analysis/rslora_r32_all_attention_diagnostics/report.md`
- `analysis/rslora_r32_all_attention_diagnostics/summary.json`
- `analysis/rslora_r32_all_attention_diagnostics/diagnostic_metrics.csv`
- `analysis/rslora_r32_all_attention_diagnostics/paired_top1_transitions.csv`
- `analysis/rslora_r32_all_attention_diagnostics/subgroup_by_seed.csv`
- `analysis/rslora_r32_all_attention_diagnostics/subgroup_summary.csv`
- `analysis/rslora_r32_all_attention_diagnostics/cross_seed_sample_consistency.csv`
- `analysis/rslora_r32_all_attention_diagnostics/common_hard_cases.csv`
- `analysis/rslora_r32_all_attention_diagnostics/consistent_promotions_regressions.csv`
- `analysis/analyze_rslora_r32_all_attention_diagnostics.py`
- `analysis/freeze_rslora_r32_all_attention.py`
- `models/vgwd_clean_rslora_r32_all_attention_seed123_frozen/MODEL_CARD.md`
- `models/vgwd_clean_rslora_r32_all_attention_seed123_frozen/manifest.json`
- `models/vgwd_clean_rslora_r32_all_attention_seed123_frozen/SHA256SUMS`

개별 validation 분자의 SMILES와 예측이 포함된 row-level CSV는 로컬 연구 감사용으로만 보존하고 GitHub 커밋에서는 제외한다. 공개 커밋에는 재현 스크립트와 집계 통계만 포함한다.

## 4. Validation에서 결정한 사항

현재까지 validation 근거로 정한 주요 사항은 다음과 같다.

1. r32 주 모델의 checkpoint는 validation Morgan 최고값 때문에 step 3500으로 선택했다. test 점수로 checkpoint를 변경하지 않았다.
2. Beam 10은 validation strict Top-1, 메모리 안정성, Top-10 성능을 함께 고려해 선택했다. Beam 20은 후보 회수율은 높지만 Top-1이 낮았고, Beam 30/50은 완료되지 않았다.
3. r32 seed 재현성, r16 matched seed, scaling ablation에서는 checkpoint 3500과 Beam 10을 seed/condition마다 동일하게 고정했다. 5-seed 확장에서도 이 규칙을 그대로 적용했으며 개별 seed의 validation 최고점을 골라 test를 최적화하지 않았다.
4. spectral analog reranker와 learned cross-modal reranker는 structure-grouped OOF에서 robust gain이 없어서 test로 진행하지 않았다.
5. r16 all-attention validation-only screen은 primary Top-1 평균 +7.4799 pp와 3/3 seed 양수로 사전 진행 규칙을 통과했다. 따라서 다음 단계로 r32/alpha64 all-attention을 validation에서 비교할 수 있다. 이 결정에 test는 사용하지 않았다.
6. r32 all-attention validation-only screen은 primary Top-1 평균 +6.8775 pp와 3/3 seed 양수로 사전 선택 규칙을 통과했다. 따라서 r32/alpha64 q/k/v/out을 향후 잠금 holdout 또는 외부 평가 후보로 선택했다. 이 결정에도 test는 사용하지 않았다.

초기 LoRA/DoRA/AdaLoRA/rsLoRA r16 adapter screening은 test 결과도 비교에 사용했다. 따라서 이 초기 방법 선택을 순수 validation-only 선택으로 재서술하면 안 된다.

## 5. Test 결과의 현재 요약

주 clean VGWD test 675개에서 현재 보고할 핵심 결과는 다음과 같다.

- 기존 test-evaluated incumbent r32/alpha64 q/v, seed 42: strict Top-1 37.0370%, Top-10 85.0370%, Morgan Top-1 0.722644, best-10 0.943691.
- primary matched seeds 7/123/314/1729/2026에서 r32/alpha64 strict Top-1 range는 33.6296~37.7778%, Top-10 range는 81.7778~84.8889%였다.
- 같은 seed의 r16/alpha32 대비 r32/alpha64 이득은 Top-1 +3.7037~+9.4815 pp, Top-10 +7.1111~+10.3704 pp였고, 두 지표 모두 5/5 seeds에서 양수였다.
- rank × scaling 통제에서도 r32의 rank 효과는 두 scale 수준의 Top-1·Top-10 모두 5/5 seeds에서 양수였다.

이 test는 초기 adapter screening, r16/r32 비교, seed 반복, scaling ablation 및 오류 분석에서 여러 번 열람되었다. 따라서 앞으로 개발되는 새 방법에 대해 이 split을 “한 번도 보지 않은 최종 test”라고 부르면 안 된다.

## 6. Seed별 재현성과 rank × scaling 결론

- 과거 r32 자체 재현성은 seed 42/123/2026 세 번으로 먼저 확인했다. 이 기록은 historical analysis로 유지한다.
- primary matched 2×2 분석은 사전 고정한 seeds 7/123/314/1729/2026으로 확장했다. r32/alpha64 strict Top-1 range는 33.6296~37.7778%, Top-10 range는 81.7778~84.8889%였다.
- rank 효과는 low/high scaling의 Top-1·Top-10 모두 5/5 seeds에서 양수였고 seed-t 95% CI도 0보다 높았다.
- scaling 효과는 rank 16에서 5/5 seeds 양수였지만, rank 32에서는 Top-1·Top-10 각각 4/5 seeds만 양수였다. rank 32 Top-1 seed-t 95% CI는 0을 포함했다.
- r32/alpha64 target-module 비교에서는 seeds 7/123/2026 모두에서 q/k/v/out이 q/v보다 validation strict Top-1과 Top-10이 높았다. 평균 이득은 각각 +6.8775 pp와 +5.4217 pp였다.
- n=5도 작은 표본이므로 “모든 seed에서 항상 우월하다” 또는 “seed 분산이 정확히 이 값이다”라고 주장하지 않는다.

## 7. 통계 검정과 신뢰구간 해석

- 동일 test sample의 binary 성공/실패 비교에는 exact McNemar 검정을 사용했다.
- 연속 Morgan 차이와 비율 차이에는 paired bootstrap을 사용했다.
- 초기 paper 분석은 sample bootstrap 20,000회를 사용했다.
- split 내부 동일 구조 반복을 확인한 뒤, scaling ablation에는 canonical-structure cluster bootstrap을 사용했다.
- r16 및 r32 all-attention validation screen에도 canonical-structure cluster bootstrap 5,000회와 exact McNemar를 사용했다.
- 여러 endpoint를 함께 본 분석에서는 Benjamini-Hochberg 보정 결과를 기록했다.
- sample-level p-value가 작다는 사실은 새로운 seed, 새로운 scaffold, 새로운 데이터셋에 대한 일반화를 자동으로 보장하지 않는다.
- 초기 seed 2~3개 분석과 확장된 n=5 분석 모두 seed 모집단을 정밀하게 추정하기에는 작다. n=5 t interval도 불확실하며, sample-level 유의성을 seed-level 유의성으로 표현하지 않는다.

## 8. 채택한 방법과 중단한 방법

### validation에서 채택한 다음 평가 후보와 프로토콜

- rsLoRA r32/alpha64
- 향후 잠금 holdout·외부 평가용 validation-selected target module: q_proj/k_proj/v_proj/out_proj
- 동결 운용 후보: all-attention seed 123, checkpoint 3500. 이는 validation 기반 선택이며 외부 검증 완료 모델이 아님
- dropout 0.05
- checkpoint 3500
- deterministic Beam 10 / return 10
- strict exact Top-k와 Morgan 지표 병행 보고
- 반복 구조를 고려한 structure-cluster bootstrap

### 기준선 또는 비교군으로 유지

- pretrained no-adapter
- LoRA r8/alpha16
- LoRA r16/alpha32
- rsLoRA r16/alpha32
- rsLoRA r32/alpha64 q/v: 기존 test 결과가 있는 주 비교 기준이며, all-attention의 외부 확인 전까지 test-evaluated incumbent로 유지
- rsLoRA r16/alpha32 all-attention: validation에서 다음 단계 진행 기준을 통과한 후보이며 아직 최종 채택 모델은 아님
- scaling control r16/alpha45 및 r32/alpha45

### 현재 경로에서 중단

- DoRA r8: clean test에서 rsLoRA r16보다 낮았으며 추가 반복을 진행하지 않음
- AdaLoRA init16→target8: clean test 성능이 낮아 추가 반복을 진행하지 않음
- train-only spectral analog reranker: structure-grouped OOF에서 baseline보다 낮음
- learned cross-modal reranker: +1/664, CI가 0을 포함하고 McNemar p=1.0
- Beam 20: validation Top-1이 Beam 10보다 낮아 주 프로토콜에서 제외
- Beam 30/50: 메모리/CUBLAS 실패로 완전한 비교 불가

### 아직 수행하지 않음

- attention projection 외 FFN module 포함

FFN 포함에는 성능·과적합·공정성에 관한 실험적 결론이 없다. r16과 r32 all-attention은 validation-only로 실행했으며 test 또는 외부 holdout 결론은 없다.

## 9. 데이터 누수와 연구 무결성 점검

확인된 긍정적 근거:

- train-validation, train-test, validation-test 사이 exact molecule overlap 0.
- connectivity overlap 0.
- spectrum hash overlap 0.
- reranker 두 종류는 validation의 structure-grouped OOF로 판정했고 robust gain이 없어 test로 보내지 않았다.
- seed 및 scaling 비교에서는 checkpoint와 beam을 조건별로 고정했다.
- 5-seed 확장의 추가 seeds 7/314/1729와 제외 없음 규칙을 결과 생성 전에 프로토콜에 고정했다.
- r16 all-attention의 seed, primary endpoint, r32 진행 규칙을 결과 전에 고정했고 664-row filtered validation만 사용했다. 이 screen에서는 test prediction을 생성하지 않았다.
- r32 all-attention도 seeds 7/123/2026, checkpoint 3500, Beam 10, primary endpoint와 선택 규칙을 결과 전에 고정하고 같은 664-row filtered validation만 사용했다. 이 screen에서도 test prediction을 생성하지 않았다.

남아 있는 위험과 제한:

- split은 scaffold-disjoint가 아니다. test의 19.7037%가 train과 공유 scaffold에 속한다.
- test 675 rows에는 494 unique exact structures만 있으며 동일 구조의 여러 스펙트럼이 있다.
- 현재 test는 모델·rank·seed·scaling 비교와 오류 분석에 반복 노출되었고 5-seed 확장에서도 다시 사용되었다. 새 아이디어를 이 test에서 반복 선택하면 adaptive test overfitting 위험이 커진다.
- 기존 sample-level 통계는 동일 구조 반복 때문에 지나치게 좁을 수 있다. 가능한 경우 structure-cluster bootstrap 결과를 우선한다.

향후 운영 규칙:

1. 새 architecture, target module, FFN 포함 여부, reranker hyperparameter는 validation 또는 structure-grouped validation CV에서만 선택한다.
2. 현재 VGWD test는 기존 benchmark 재현용으로 고정하고, 새 선택의 판단 근거로 사용하지 않는다.
3. 논문의 강한 최종 확인에는 별도의 잠금 holdout 또는 외부 데이터셋을 사용한다.
4. 가능하면 molecule/scaffold 단위 split 또는 최소한 structure-cluster uncertainty를 함께 보고한다.
5. test를 열람한 모든 분석은 날짜와 목적을 이 문서에 남긴다.

## 10. 관련 코드·설정·결과 파일 경로

### 핵심 학습·추론·평가 코드

- `spectus/train_spectus_lora.py`
- `spectus/train_spectus_adalora.py`
- `spectus/predict_lora.py`
- `spectus/evaluate_predictions.py`
- `analysis/audit_exact_topk.py`

### 핵심 설정 및 실행기

- `configs/finetune_vgwd_clean_lora.yaml`
- `configs/finetune_vgwd_clean_lora_r16.yaml`
- `configs/finetune_vgwd_clean_dora_r8.yaml`
- `configs/finetune_vgwd_clean_adalora.yaml`
- `configs/finetune_vgwd_clean_rslora_r16.yaml`
- `configs/finetune_vgwd_clean_rslora_r32.yaml`
- `config_runners/run_vgwd_clean_rslora_r32.sh`
- `config_runners/run_vgwd_clean_rslora_scaling_ablation.sh`
- `config_runners/run_vgwd_clean_rslora_all_attention_validation.sh`
- `config_runners/run_vgwd_clean_rslora_r32_all_attention_validation.sh`
- `configs/finetune_vgwd_clean_rslora_r16_allattn_seed7.yaml`
- `configs/finetune_vgwd_clean_rslora_r16_allattn_seed123.yaml`
- `configs/finetune_vgwd_clean_rslora_r16_allattn_seed2026.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_allattn_seed7.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_allattn_seed123.yaml`
- `configs/finetune_vgwd_clean_rslora_r32_allattn_seed2026.yaml`

### 핵심 분석 산출물

- `analysis/rslora_r32_experiment_summary.md`
- `analysis/r16_vs_r32_paper/`
- `analysis/r32_seed_reproducibility/`
- `analysis/rslora_rank_matched_seeds/`
- `analysis/rslora_scaling_ablation/`
- `analysis/rslora_scaling_ablation_5seeds/`
- `analysis/generate_rslora_scaling_seed_configs.py`
- `analysis/summarize_rslora_scaling_ablation.py`
- `analysis/rslora_all_attention_validation/`
- `analysis/generate_rslora_all_attention_configs.py`
- `analysis/summarize_rslora_all_attention_validation.py`
- `analysis/rslora_r32_all_attention_validation/`
- `analysis/generate_rslora_r32_all_attention_configs.py`
- `analysis/summarize_rslora_r32_all_attention_validation.py`
- `analysis/rslora_r32_all_attention_diagnostics/`
- `analysis/analyze_rslora_r32_all_attention_diagnostics.py`
- `analysis/freeze_rslora_r32_all_attention.py`
- `analysis/vgwd_split_leakage/`
- `analysis/spectral_reranker/`
- `analysis/learned_candidate_reranker/results/`

### 데이터 및 prediction

- `data/vgwd/`
- `data/vgwd_clean/`
- `data/vgwd_clean/valid_filtered_mz500.jsonl`
- `predictions/vgwd_original_full/`
- `predictions/vgwd_lora_full/`
- `predictions/vgwd_clean_original/`
- `predictions/vgwd_clean_lora/`
- `predictions/vgwd_clean_lora_r16/`
- `predictions/vgwd_clean_dora_r8/`
- `predictions/vgwd_clean_adalora/`
- `predictions/vgwd_clean_rslora_r16/`
- `predictions/vgwd_clean_rslora_r32_test_beam10/`
- `predictions/vgwd_clean_rslora_r32_seed7_valid_beam10/`, `predictions/vgwd_clean_rslora_r32_seed123_valid_beam10/`, `predictions/vgwd_clean_rslora_r32_seed2026_valid_beam10/`
- `predictions/vgwd_clean_rslora_r32_allattn_seed7_valid_beam10/`, `predictions/vgwd_clean_rslora_r32_allattn_seed123_valid_beam10/`, `predictions/vgwd_clean_rslora_r32_allattn_seed2026_valid_beam10/`
- `models/vgwd_clean_rslora_r32_all_attention_seed123_frozen/`
- 새 5-seed 실행의 prediction 경로는 `predictions/vgwd_clean_rslora_r16_seed<seed>_test_beam10/`, `predictions/vgwd_clean_rslora_r16_alpha45_seed<seed>_test_beam10/`, `predictions/vgwd_clean_rslora_r32_alpha45_seed<seed>_test_beam10/`, `predictions/vgwd_clean_rslora_r32_seed<seed>_test_beam10/` 형식이며 `<seed>`는 7, 314, 1729이다.

### 저장소 내 기존 SpecTUS 원 연구 산출물의 범위

저장소에는 `configs/finetune_exp1_*`부터 `exp5` 계열 설정, NIST/library prediction 등 원 SpecTUS 연구의 참고 산출물이 함께 존재한다. 이 문서는 현재 수행한 VGWD adaptation 연구의 실행 로그와 결론을 정리한다. 원 연구 산출물은 이 VGWD 실험에서 새로 재실행하거나 재검증한 결과로 간주하지 않았고, 그 수치를 이 프로젝트의 새 결론으로 옮기지 않았다.

## 11. 논문에서 주장 가능한 결론과 과도한 주장

### 현재 근거로 주장 가능한 내용

- 고정된 VGWD clean split과 현재 평가 프로토콜에서 PEFT는 pretrained no-adapter보다 크게 높은 구조 예측 성능을 보였다.
- rsLoRA r32/alpha64 q/v는 validation으로 선택한 checkpoint 3500 및 Beam 10에서 seed 42 test strict Top-1 37.0370%, Top-10 85.0370%를 기록했다.
- primary matched seeds 7/123/314/1729/2026에서 r32/alpha64는 r16/alpha32보다 strict Top-1과 Top-10이 모두 높았다.
- 2×2 rank × scaling ablation의 5 seeds에서 rank 32의 Top-1·Top-10 이점이 low/high scaling 모두에 남았고 seed-t 95% CI도 0보다 높았다. 따라서 이 데이터와 프로토콜에서 관찰된 r32 향상을 scaling 증가만으로 설명하기 어렵다.
- r16/alpha32 validation-only screen의 seeds 7/123/2026에서 q/k/v/out all-attention은 q/v보다 strict Top-1과 Top-10이 모두 높았고, 사전 설정한 r32 validation 확장 기준을 통과했다.
- r32/alpha64 validation-only screen의 seeds 7/123/2026에서도 q/k/v/out all-attention은 q/v보다 strict Top-1과 Top-10이 모두 높았고, 평균 이득은 각각 +6.8775 pp와 +5.4217 pp였다. 사전 선택 규칙에 따라 잠금 holdout 또는 외부 평가 후보로 선택됐다.
- r32 all-attention의 기술적 오류 분석에서 사전 정의한 n≥20 하위그룹 모두 Top-1 평균 이득이 양수였고, Top-10 내 invalid SMILES는 0개였다. 다만 train-unseen scaffold는 5개뿐이어서 scaffold 일반화 근거는 부족하다.
- exact molecule, connectivity, spectrum hash 기준의 직접적인 split 간 중복은 감사에서 발견되지 않았다.
- 두 reranker는 structure-grouped OOF에서 robust improvement를 보이지 않아 중단했다.

### 제한을 함께 밝혀야 하는 내용

- VGWD split은 scaffold-disjoint가 아니며 train-test scaffold overlap의 영향을 받는 test rows가 19.7037%이다.
- 동일 구조의 반복 spectrum이 있으므로 sample-level CI와 p-value는 molecule-independent inference가 아니다.
- test가 여러 개발 비교에 반복 사용되었으므로 새로운 방법에 대한 완전히 untouched test라고 할 수 없다.
- primary matched rank 및 scaling ablation은 5개 training seeds로 확장했지만 여전히 작은 seed 표본이며, 동일 test와 동일 데이터 분포에 한정된다. seed 42는 이 matched primary set이 아닌 과거 r32 재현성 기록이다.
- r16 all-attention 결과는 3개 training seeds와 한 validation split에 한정된 모델 선택 근거이다. test 개선이나 외부 일반화의 증거가 아니다.
- r32 all-attention 결과도 3개 training seeds와 모델 선택에 사용한 같은 validation split에 한정된다. 기존 VGWD test, 외부 데이터 또는 현장 데이터의 개선 증거가 아니다.
- 동결 seed 123은 세 seed 중 validation Top-1이 가장 높은 실행을 선택했으므로 개별 artifact 성능에는 추가적인 validation 선택 낙관성이 있을 수 있다.
- rank × scaling의 low/high scale은 alpha 정수 제약 때문에 완벽히 동일한 scale pair가 아니다.

### 과도하게 주장하면 안 되는 내용

- 관측한 5개를 넘어 r32가 모든 seed, 데이터셋, 기기 또는 화학 공간에서 항상 r16보다 우월하다는 주장
- 현재 결과가 unseen scaffold나 완전히 새로운 외부 분포로 일반화된다는 주장
- test 결과를 근거로 선택한 초기 adapter screening 전체가 validation-only 또는 preregistered였다는 주장
- sample-level McNemar 유의성을 seed-level 혹은 데이터셋-level 유의성으로 바꾸어 표현하는 것
- Morgan 1.0 또는 connectivity exact를 strict stereochemical exact와 동일하게 표현하는 것
- reranker가 성능을 개선했다고 주장하는 것
- r16 또는 r32 validation 결과만으로 FFN 포함, test 성능, 외부 데이터 또는 화생방 현장 데이터에서도 개선된다고 주장하는 것
- r32 all-attention이 validation에서 선택됐다는 이유만으로 기존 q/v 모델보다 최종적으로 우월하거나 배포 준비가 완료됐다고 주장하는 것
- rank 증가의 효과가 모든 조건에서 보편적인 인과 효과라고 단정하는 것. 5-seed 2×2 ablation이 현재 조건에서 rank 기여를 지지하지만 seed·모듈·데이터 범위는 제한적임

## 다음 실험 추가용 템플릿

아래 블록을 복사해 3절의 날짜순 위치에 추가한다.

```text
### YYYY-MM-DD — 실험명

- 연구 질문:
- 사전에 고정한 조건:
- 변경한 변수:
- 데이터 split 및 샘플 수:
- seed:
- checkpoint 선택 규칙:
- validation 결과와 결정:
- test 사용 여부 및 결과:
- 통계 검정/CI와 분석 단위:
- 실패·중단 사항:
- 결론과 제한:
- 코드/설정/체크포인트/예측/결과 경로:
```
