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

### 2026-09-02 — r32 all-attention 논문용 그림 세트

- 기존 validation 집계 결과만 사용해 논문용 그림 5종을 만들었다. 새 모델 학습, hyperparameter 선택 또는 test 열람은 수행하지 않았다.
- 모든 그림은 q/v를 blue, q/k/v/out을 vermillion으로 통일한 색각 이상 친화 팔레트와 영문 라벨을 사용했다.
- 각 그림은 600-dpi PNG, 벡터 PDF, 편집 가능한 SVG로 저장했다. PNG의 실제 해상도와 600-dpi metadata, PDF 1 page, SVG vector text를 확인했다.
- Figure 1은 seeds 7/123/2026의 strict Top-1과 Top-10 절대 성능을 0에서 시작하는 축으로 표시한다.
- Figure 2는 seed별 paired 이득과 canonical-structure cluster-bootstrap 95% CI, 그리고 3-seed 평균과 seed-t 95% CI를 구분해 표시한다. n=3 seed interval은 탐색적이라는 제한을 캡션에 명시했다.
- Figure 3은 n≥20인 분자량, 최대 관측 m/z, peak 수 하위그룹의 3-seed 평균과 개별 seed 점을 표시한다. n<20 그룹 제외와 기술 통계라는 점을 캡션에 명시했다.
- Figure 4는 1,992개 paired seed-sample outcomes에서 retained correct 433, newly correct 355, lost correct 218, remained wrong 986 및 순이득 +137을 흐름으로 표시한다. 188/664 공통 난제는 sample 단위라는 점을 분리해 표기했다.
- Figure 5는 데이터 감사, 초기 PEFT screen, rank × scaling, r16/r32 all-attention screen, 후보 동결, 중단 방법, FFN 보류와 향후 confirmatory stage를 하나의 결정 흐름으로 정리한다. exact molecule/connectivity/spectrum overlap은 0이지만 scaffold overlap은 남았음을 함께 표시했다.
- `README.md`에 바로 사용할 수 있는 영문 캡션, 추천 파일 형식과 재생성 명령을 기록했다. `figure_manifest.json`에는 입력 집계표와 산출물의 SHA-256을 기록했다.
- 그림·캡션·manifest·재생성 코드·집계표 25개를 `r32_all_attention_paper_figures_2026-09-02.zip`으로 묶었고 무결성 검사에 통과했다. ZIP SHA-256은 `1385c2043538918981c1a5b74aeffcefb1e1c04b57db55097a96b4ce46c7d67e`이다. row-level SMILES와 prediction은 포함하지 않았다.

관련 경로:

- `analysis/plot_r32_all_attention_paper_figures.py`
- `analysis/paper_figures/r32_all_attention/README.md`
- `analysis/paper_figures/r32_all_attention/figure_manifest.json`
- `analysis/paper_figures/r32_all_attention/figure1_seedwise_qv_vs_all_attention.{png,pdf,svg}`
- `analysis/paper_figures/r32_all_attention/figure2_seedwise_gain_with_confidence_intervals.{png,pdf,svg}`
- `analysis/paper_figures/r32_all_attention/figure3_subgroup_top1_performance.{png,pdf,svg}`
- `analysis/paper_figures/r32_all_attention/figure4_top1_outcome_flow.{png,pdf,svg}`
- `analysis/paper_figures/r32_all_attention/figure5_experiment_decision_flow.{png,pdf,svg}`
- `analysis/paper_figures/r32_all_attention_paper_figures_2026-09-02.zip`

### 2026-09-02 — 동결 r32 all-attention 실행 환경 패키징

- 새 학습이나 모델 선택 없이, 2026-09-01에 validation으로 선택·동결한 seed 123 checkpoint 3500 adapter를 다른 Linux 컴퓨터에서 실행할 수 있도록 배포 패키지를 만들었다. test 또는 validation 성능을 새로 계산하거나 선택에 사용하지 않았다.
- 재현 환경은 Python 3.8.16, PyTorch 2.0.0, CUDA runtime 11.8, Transformers 4.31.0, PEFT 0.9.0 등 실제 검증 환경의 핵심 버전으로 고정했다. `conda env create --dry-run` 의존성 해석이 성공했다.
- 기본 SpecTUS 모델은 공식 Hugging Face `MS-ML/SpecTUS_pretrained_only`의 commit `1218bd94411fcc9f35d67e58a30e9fec36fbf0b0`으로 고정했다. 추론에 필요한 세 파일만 내려받으며 `pytorch_model.bin` SHA-256 `4f8d487a51ab787dd2353253c16a71170c1843c612684ec884a0c9a5c20c13c4`를 포함한 파일별 해시를 확인한다.
- `spectus/predict_lora.py`에 base checkpoint, 입력 JSONL, tokenizer, device의 선택적 CLI override를 추가했다. 기존 인수를 그대로 사용할 때의 동작은 유지했다.
- 정답 SMILES가 없는 실제 추론 입력을 허용하도록 inference mode의 구조 label 검사를 분리했다. 별도 입력 검증기는 빈 입력, JSON 오류, 중복 sample ID, mz/intensity 길이 불일치, 300개 초과 peak, m/z 범위 초과, 정렬 오류, 부적합 intensity를 추론 전에 차단한다.
- 합성 형식 예제 1건으로 RTX 3070 Laptop GPU 8GB에서 base model 로딩, adapter 결합, Beam 10 후보 10개 생성, `predictions.jsonl` 저장까지 end-to-end smoke test가 성공했다. 이 예제에는 VGWD row 또는 정답 구조가 포함되지 않으며 smoke test 결과는 성능 근거가 아니다.
- 인터넷 설치용 ZIP은 35,031,890 bytes, 56 files이며 SHA-256은 `18e2ee2afd3f8518d33aae18a139762e7e75361b2b196fc7ba0793af399552c1`이다. adapter와 실행 코드는 포함하고 1.42GB 기본 모델은 설치 중 내려받는다.
- 기본 모델까지 포함한 오프라인 모델 ZIP은 1,350,436,390 bytes, 59 files이며 SHA-256은 `624a059f4746e639685f0b51599380cde2143405244205219b2d31855225ee53`이다. 두 ZIP 모두 CRC 검사를 통과했다. Conda 패키지까지 완전 오프라인으로 설치하려면 기관 내부 미러 또는 별도 캐시가 필요하다.
- 패키징 완료는 소프트웨어 재실행 가능성을 높인 것이며 외부 성능 검증, 현장 검증, 화생방 운용 적합성 또는 배포 승인을 의미하지 않는다. 다른 OS·GPU·CUDA 커널에서 bitwise 동일성도 보장하지 않는다.

관련 경로:

- `deployment/r32_all_attention/README_KO.md`
- `deployment/r32_all_attention/environment.yml`
- `deployment/r32_all_attention/setup.sh`
- `deployment/r32_all_attention/run_inference.sh`
- `deployment/r32_all_attention/download_base_model.py`
- `deployment/r32_all_attention/verify_package.py`
- `deployment/r32_all_attention/validate_input.py`
- `deployment/r32_all_attention/build_release_bundle.py`
- `deployment/r32_all_attention/inference_beam10.yaml`
- `deployment/r32_all_attention/example_input.jsonl`
- `deployment/r32_all_attention/SHA256SUMS`
- `dist/spectus_r32_all_attention_online_2026-09-02.zip`
- `dist/spectus_r32_all_attention_offline_2026-09-02.zip`
- `dist/SHA256SUMS`

### 2026-09-04 — 잠금 내부 재분할 및 주 모델 grouped 5-fold CV 시작

- 연구 질문: validation에서 선택한 rsLoRA r32/alpha64 q/k/v/out all-attention이 q/v 기준 모델보다, SpecTUS 주 모델 자체를 fold마다 다시 학습하는 structure-grouped OOF 평가에서도 나은지 확인한다.
- 사전에 고정한 조건: canonical connectivity SMILES를 그룹 단위로 사용하고 seed 20260904로 20% row-target holdout을 한 번 생성했다. CV는 k=5, 첫 전체 비교 training/data seed는 123, 두 방법 모두 checkpoint 3500 고정, Beam 10, strict canonical-isomeric Top-1을 primary endpoint로 정했다. outer OOF 성능으로 checkpoint를 선택하거나 early stopping하지 않는다.
- 데이터 pool: 기존 `data/vgwd_clean/{train,valid,test}.jsonl` 6,543 rows를 합쳤다. 기존 전처리 한계와 같은 기준으로 6,536 rows가 적격이었고 7 rows가 제외됐다. 적격 canonical connectivity groups는 4,927개였다.
- 잠금 holdout: 1,308 rows, 997 connectivity groups, 전체 적격 rows의 20.0122%이다. 개발 데이터는 5,228 rows, 3,930 groups이다.
- grouped 5-fold: OOF row 수는 fold 0~4 순서로 1,046, 1,046, 1,046, 1,045, 1,045이다. 각 구조 그룹의 모든 spectrum을 동일 partition과 동일 OOF fold에 배치했으며, 개발-holdout 및 OOF fold 간 connectivity group overlap은 0이다.
- 잠금·해시 검증: `LOCK.json`과 `SHA256SUMS` 검사를 통과했고 상태는 `LOCKED_NOT_EVALUATED`이다. runner에는 holdout 평가 명령이 없으며 경로에 `holdout`이 들어간 prediction 요청을 거부한다.
- smoke test: q/v 4,718,592 trainable LoRA parameters와 all-attention 9,437,184 parameters가 의도한 projection에 연결됨을 확인했다. 두 방법 모두 RTX 3070 Laptop GPU에서 2-step 학습 checkpoint 생성과 fold 0 OOF의 첫 2 rows Beam-10 예측까지 통과했다. 이 2-row 출력은 성능 결과로 사용하지 않는다.
- fold 0 완료 상태: q/v와 all-attention 모두 3,500/3,500 steps, epoch 1.0으로 완료됐고 동일한 1,046-row OOF에 Beam-10 prediction 1,046 rows씩 생성했다. q/v prediction wall time은 13분 34초, all-attention prediction wall time은 15분 19초였다. 두 조건 모두 Top-10 내 invalid prediction은 0개였다.
- fold 0 partial OOF 성능: q/v strict Top-1/3/5/10은 각각 32.4092%, 58.9866%, 71.7973%, 80.3059%이고, all-attention은 36.3289%, 65.0096%, 77.8203%, 85.2772%였다. all-attention의 차이는 각각 +3.9197, +6.0229, +6.0229, +4.9713 percentage points였다.
- fold 0 paired Top-1 전이와 통계: q/v wrong→all-attention right 164 rows, q/v right→all-attention wrong 123 rows였다. exact McNemar p=0.0180651이고, canonical-connectivity cluster bootstrap 5,000회의 Top-1 차이 95% CI는 +0.3913~+7.3728 pp였다.
- 해석 제한: 위 값은 5개 중 한 fold의 partial OOF 결과다. fold 0에서 all-attention 방향의 양수 차이를 보였지만, 나머지 fold가 없으므로 grouped 5-fold의 최종 효과나 일반화 우월성으로 결론 내리지 않는다.
- test/holdout 사용 여부: 새 잠금 holdout prediction은 생성하지 않았다. 기존 source test가 재분할 pool에 포함됐지만, 세 source split 모두 이전 실험에서 역할이 있었으므로 이 holdout은 완전히 never-observed인 외부 검증이 아니다. 이후 논문에서는 `prospectively locked internal re-split`로 제한을 명시해야 한다.
- 향후 분석: 5 folds가 모두 끝난 뒤 pooled OOF strict Top-1/3/5/10, paired exact McNemar, canonical-connectivity cluster bootstrap 5,000회, calibration·abstention·risk-coverage를 계산한다. partial fold 결과는 최종 결론으로 사용하지 않는다.
- 재개 안전성: fold 1부터는 500 steps마다 최신 operational checkpoint 하나를 보존하되 최종 평가 endpoint는 3,500 steps로 고정한다. 이 저장 주기는 OOF 성능에 따른 checkpoint 선택이 아니며 중단 시 같은 학습 상태를 재개하기 위한 것이다.
- 후속 실행 상태: 2026-09-04 21:36 KST에 완료된 fold 0 산출물을 재사용하고 fold 1~4를 순차 실행하는 `full` runner를 시작했다. 시작 시 잠금 해시 검증은 다시 통과했고 fold 0의 두 checkpoint와 두 OOF prediction은 재학습 없이 재사용됐다. 이 실행이 완료되기 전까지 fold 0 partial 값만 존재한다.
- 코드·프로토콜: `analysis/main_model_grouped_cv/prepare_locked_split.py`, `analysis/main_model_grouped_cv/generate_configs.py`, `analysis/main_model_grouped_cv/summarize_oof.py`, `analysis/main_model_grouped_cv/protocol.md`.
- 잠금 데이터·manifest: `data/vgwd_main_grouped_cv_locked/`.
- 설정·실행기: `configs/main_grouped_cv/`, `config_runners/run_vgwd_main_grouped_cv.sh`.
- smoke 산출물: `checkpoints/vgwd_main_grouped_cv_smoke_fold0_{qv,allattn}_seed123/`, `predictions/vgwd_main_grouped_cv_smoke/fold0/`.
- full-fold 산출물 경로: `checkpoints/vgwd_main_grouped_cv_fold0_{qv,allattn}_seed123/`, `predictions/vgwd_main_grouped_cv/fold0/`.

### 2026-09-05 — 주 모델 grouped 5-fold CV 진행 갱신

- fold 1 완료 상태: q/v와 all-attention 모두 3,500 steps 학습과 동일한 1,046-row OOF Beam-10 예측을 완료했다. 양쪽 prediction은 각각 1,046 rows이며 Top-10 내 invalid prediction은 0개다.
- fold 1 partial OOF 성능: q/v strict Top-1/3/5/10은 30.4015%, 57.3614%, 68.7380%, 81.6444%이고, all-attention은 35.8509%, 62.5239%, 73.8050%, 84.3212%이다. all-attention 차이는 각각 +5.4493, +5.1625, +5.0669, +2.6769 percentage points다.
- fold 0~1 단순 pooled 중간값: 2,092 rows에서 q/v Top-1은 657/2,092=31.4054%, all-attention은 755/2,092=36.0899%로 차이는 +4.6845 pp다. 이는 2/5 fold의 중간값이며 사전 정의한 최종 grouped OOF 통계가 아니다.
- fold 2 진행 상태(2026-09-05 00:26 KST): q/v 3,500-step 학습과 1,046-row OOF Beam-10 예측은 완료됐다. q/v strict Top-1/3/5/10은 32.1224%, 59.9426%, 70.2677%, 80.2103%이며 Top-10 내 invalid prediction은 0개다. all-attention은 1,960/3,500 steps까지 진행 중이므로 paired fold 2 결과는 아직 없다.
- 실행 무결성: fold 0~1과 fold 2 q/v의 checkpoint 및 prediction은 존재하며, runner는 fold 2 all-attention 이후 fold 3~4를 순차 실행한다. 잠금 holdout은 계속 `LOCKED_NOT_EVALUATED` 상태이며 prediction을 생성하지 않았다.
- 해석 제한: 현재 all-attention의 Top-1 차이는 완료된 fold 0과 fold 1에서 모두 양수지만, 5개 fold 전체 완료·connectivity-cluster bootstrap·paired 검정 전에는 최종 우월성으로 결론 내리지 않는다.
- 추가 산출물 경로: `checkpoints/vgwd_main_grouped_cv_fold1_{qv,allattn}_seed123/`, `checkpoints/vgwd_main_grouped_cv_fold2_{qv,allattn}_seed123/`, `predictions/vgwd_main_grouped_cv/fold1/`, `predictions/vgwd_main_grouped_cv/fold2/qv/`.

### 2026-09-05 — 주 모델 grouped 5-fold CV 전체 완료

- 완료 상태: fold 0~4에서 q/v와 all-attention을 각각 고정된 3,500 steps로 다시 학습하고, 각 fold의 group-disjoint OOF를 Beam-10으로 예측했다. 총 10개 학습 checkpoint와 10개 prediction이 모두 존재한다. OOF row 수는 조건별 5,228개이며 fold별 1,046, 1,046, 1,046, 1,045, 1,045개와 정확히 일치한다.
- pooled strict Top-k: q/v Top-1/3/5/10은 32.1155%, 59.1431%, 70.5241%, 80.6809%이고, all-attention은 38.1790%, 65.1301%, 76.4728%, 85.3481%이다. all-attention 차이는 각각 +6.0635, +5.9870, +5.9487, +4.6672 percentage points다.
- primary Top-1 정답 수: q/v 1,679/5,228, all-attention 1,996/5,228이다.
- fold별 Top-1 이득: fold 0~4 순서로 +3.9197, +5.4493, +8.1262, +5.6459, +7.1770 pp이며 5/5 folds에서 양수다.
- paired Top-1 전이와 통계: q/v wrong→all-attention right 851 rows, q/v right→all-attention wrong 534 rows다. paired exact McNemar p=1.4878575×10^-17이며, canonical-connectivity cluster bootstrap 5,000회의 pooled Top-1 차이 95% CI는 +4.5870~+7.5149 pp다.
- prediction 유효성: 모든 fold·조건에서 Top-10 내 invalid prediction은 0개다.
- 잠금·누수 상태: `prepare_locked_split.py --verify`가 다시 PASS했고 holdout 1,308 rows는 `LOCKED_NOT_EVALUATED` 상태다. holdout prediction은 생성하지 않았다.
- 해석: validation에서 선택한 r32/alpha64 all-attention은 seed 123의 이 prospectively locked internal re-split에서 q/v보다 structure-grouped OOF Top-1 및 Top-k가 높았다. 따라서 기존 한 validation split만의 현상이 아니라 개발 pool의 5개 group-disjoint OOF fold 전반에서 같은 방향이 관찰됐다고 주장할 수 있다.
- 제한: 이는 외부 데이터 또는 잠금 holdout 최종 평가가 아니고, 5 folds는 5개의 독립 training seed가 아니다. source pool에는 과거 실험에서 사용한 기존 train/validation/test가 포함돼 있으며 training/data seed는 123 하나다. 따라서 외부 일반화, 화생방 현장 성능, 모든 seed에서의 보편적 우월성 또는 최종 배포 준비 완료로 과장하지 않는다.
- 총 실행 시간: fold 1~4 연속 runner는 2026-09-04 21:36 KST에 시작해 마지막 fold 4 all-attention OOF 평가가 2026-09-05 03:36 KST에 완료되어 약 6시간이 걸렸다. fold 0은 그 전에 별도로 완료됐다.
- 결과: `analysis/main_model_grouped_cv/results/summary.json`, `analysis/main_model_grouped_cv/results/report.md`, `analysis/main_model_grouped_cv/results/run_metrics.csv`, `analysis/main_model_grouped_cv/results/paired_fold_results.csv`.
- 전체 산출물: `checkpoints/vgwd_main_grouped_cv_fold{0,1,2,3,4}_{qv,allattn}_seed123/`, `predictions/vgwd_main_grouped_cv/fold{0,1,2,3,4}/{qv,allattn}/`.

### 2026-09-05 — OOF calibration·기권·risk–coverage·Top-k 정량 활용성 분석

- 연구 질문: grouped 5-fold OOF prediction의 저장된 sequence score로 strict Top-1 정답 확률을 보정할 수 있는지, 신뢰도가 낮을 때 기권하면 선택 정확도가 어떻게 변하는지, Top-k 후보가 추가로 회수하는 정답과 검토 부담은 얼마인지 평가했다.
- 사전 고정 범위: q/v와 all-attention의 OOF만 사용하고 잠금 holdout은 열지 않았다. all-attention을 primary operational candidate, q/v를 secondary comparator로 두었다. 결과 확인 전 primary 기권 기준을 cross-fitted calibrated P(Top-1 correct)≥0.80으로 고정했고, 0.50/0.60/0.70/0.90은 secondary threshold로 정했다.
- calibration feature: log top-1 sequence score, log top-1/top-2 score ratio, Beam 후보 score-share의 normalized entropy, valid candidate count를 고정했다. sequence score는 native class probability가 아니라 empirical ranking score로 취급했다.
- calibration 설계: 각 OOF fold를 한 번씩 완전히 제외하고 나머지 4 folds에서 standard scaling+L2 logistic regression(C=1)을 적합하는 5-fold cross-fitting을 사용했다. 따라서 각 row와 같은 connectivity group은 그 row의 calibrator 적합에 쓰이지 않았다. cross-fitted 평가 후 같은 고정법을 전체 OOF에 적합한 계수를 향후 holdout 적용용 JSON으로 동결했다.
- 데이터 품질 amendment: all-attention 5,228 rows 중 1 row에서 valid candidate가 0개였다. 이는 오답으로 전체 분모에 유지하되 confidence=0으로 강제 기권하고 calibrator 적합에서는 제외했다. 이 규칙은 holdout 접근 전에 기록했으며 모델·threshold·나머지 feature는 변경하지 않았다.
- all-attention calibration 결과: raw Beam score share의 Brier/ECE-10은 0.275246/0.252551이었고, cross-fitted calibration은 0.203792/0.017805였다. log loss도 0.811533에서 0.594453으로 낮아졌다. q/v의 raw→calibrated Brier는 0.247979→0.191991, ECE는 0.223341→0.011197이었다.
- primary 기권 결과(all-attention, threshold 0.80): 5,228 rows 중 44 rows에 답하고 5,184 rows는 기권했다. coverage 0.8416%, selective accuracy 90.9091%(40/44), risk 9.0909%였다. canonical-connectivity cluster bootstrap 5,000회(seed 20260905)의 95% CI는 coverage 0.5593~1.1597%, selective accuracy 80.9524~98.4615%였다.
- primary 기권 해석: 고정 0.80 기준은 정확도가 높지만 99.1584%를 기권하므로 일반 운용 기준으로는 coverage가 지나치게 작다. 결과를 본 뒤 primary threshold를 바꾸지 않았으며, 매우 제한적인 고신뢰 표시로만 해석한다.
- secondary all-attention threshold: 0.50은 coverage 26.3772%/accuracy 62.9442%, 0.60은 15.1875%/70.2771%, 0.70은 6.2357%/76.9939%, 0.90은 accepted row 0개였다. 이 값은 선택 가능한 운용 trade-off를 보여주는 보조 분석이지 새 primary 기준이 아니다.
- risk–coverage: confidence 상위 약 5%, 10%, 20%, 50%, 100% coverage에서 all-attention selective accuracy는 각각 79.0076%, 73.8050%, 67.5908%, 52.3718%, 38.1790%였다. coverage가 커질수록 risk가 증가하는 예상 방향을 보였다.
- Top-k 정량 활용성(all-attention): Top-1/3/5/10 strict accuracy는 38.1790/65.1301/76.4728/85.3481%, hits는 1,996/3,405/3,998/4,462건, MRR@10은 0.539359였다. Top-1 대비 추가 hits는 Top-3 +1,409, Top-5 +2,002, Top-10 +2,466이었다.
- 후보 검토 부담 proxy: 모든 row에서 고정 K까지 본다고 가정할 때 Top-1 대비 추가 정답 1개당 추가 candidate slots는 Top-3 7.42, Top-5 10.45, Top-10 19.08이었다. 이는 정량적 workload proxy일 뿐 실제 연구자 사용자 연구 또는 현장 유용성 증명이 아니다.
- 잠금 및 동결: 분석 종료 후 protocol, script, summary, frozen calibrator, calibration/abstention/risk-coverage/Top-k 표에 SHA-256 manifest를 생성했고 전 항목 검증이 통과했다. holdout 상태는 계속 `LOCKED_NOT_EVALUATED`이다.
- 결론: 내부 grouped OOF에서 score calibration은 raw Beam share보다 Brier/log loss/ECE를 개선했고, confidence 순위는 유용한 risk–coverage 분리를 제공했다. 그러나 primary 0.80 정책은 coverage가 1% 미만이며, holdout·외부 데이터에서 calibration과 threshold가 재현되는지는 아직 확인하지 않았다.
- 코드·프로토콜: `analysis/main_model_grouped_cv/analyze_decision_support.py`, `analysis/main_model_grouped_cv/decision_support_protocol.md`.
- 결과·그림·동결 artifact: `analysis/main_model_grouped_cv/decision_support/`의 `report.md`, `summary.json`, `sample_scores.csv`, `calibration_metrics.csv`, `reliability_bins.csv`, `abstention_thresholds.csv`, `risk_coverage.csv`, `topk_quantitative_utility.csv`, `frozen_calibrators.json`, `calibration_reliability.{png,pdf}`, `risk_coverage.{png,pdf}`, `topk_quantitative_utility.{png,pdf}`, `SHA256SUMS`.

### 2026-09-05 — 컴퓨터 교체용 연구 마이그레이션 백업 생성

- grouped-CV와 decision-support 분석 이후 새로 생긴 코드·설정·문서·잠금 split·OOF prediction·10개 최종 checkpoint·동결 모델을 하나의 private 연구 ZIP으로 묶었다. 잠금 데이터와 checkpoint를 포함하므로 공개 GitHub가 아닌 LAN 직접 전송 또는 접근 제한 개인 저장소용이다.
- `trainSpectus` Conda 환경을 build string 없이 export했고, 새 컴퓨터에서 사용할 복원 지침과 bundle 생성 코드를 함께 포함했다.
- 연구 ZIP은 846,266,034 bytes, 내부 항목 633개이며 Python ZipFile CRC test가 PASS했다. SHA-256은 `b5a48c6f48c39aeeaad5d549f3eaf622e58c3844e83b00c8e7b3b9d79434d018`이다.
- 기존 기본 모델 포함 오프라인 ZIP 1,350,436,390 bytes와 온라인 ZIP 35,031,890 bytes를 연구 ZIP과 함께 최상위 `MIGRATION_SHA256SUMS`에 연결했고 세 항목 모두 `sha256sum -c` 검증이 통과했다.
- 이 백업은 현재 연구 상태 보존용이며 external validation 또는 잠금 holdout 평가를 추가하지 않았다. holdout은 계속 `LOCKED_NOT_EVALUATED`이다.
- 경로: `migration/README_KO.md`, `migration/trainSpectus_environment_2026-09-05.yml`, `migration/build_research_bundle.py`, `dist/spectus_research_migration_2026-09-05.zip`, `dist/MIGRATION_SHA256SUMS`.

### 2026-09-05 — grouped-CV 및 decision-support 공개 GitHub 백업

- 사용자의 명시적 승인 후 `https://github.com/gwajamukkosip/SpecTUS-VGWD-LoRA.git`의 `main` branch에 grouped-CV 코드·설정·집계 결과·논문용 PNG/PDF·연구 문서·환경 복원 도구를 push했다.
- 최초 공개 커밋은 `5b857a3bc9cfe942c5b8a005a6026be50fa328ac`이며 원격 `refs/heads/main`이 해당 커밋을 가리키는 것을 `git ls-remote`로 확인했다.
- 공개 제외: row-level `analysis/main_model_grouped_cv/decision_support/sample_scores.csv`, 전체 Codex 대화 기록, 잠금 holdout 및 assignments, 원시 OOF predictions, 학습 checkpoints, private migration ZIP. 이 파일들은 로컬과 checksum 검증된 비공개 마이그레이션 ZIP에만 유지한다.
- 공개 집계에는 개별 SMILES·원본 spectrum·잠금 holdout row가 포함되지 않는다. 이번 업로드는 보존·재현 목적이며 새 평가나 모델 선택을 수행하지 않았다.

## 4. Validation에서 결정한 사항

현재까지 validation 근거로 정한 주요 사항은 다음과 같다.

1. r32 주 모델의 checkpoint는 validation Morgan 최고값 때문에 step 3500으로 선택했다. test 점수로 checkpoint를 변경하지 않았다.
2. Beam 10은 validation strict Top-1, 메모리 안정성, Top-10 성능을 함께 고려해 선택했다. Beam 20은 후보 회수율은 높지만 Top-1이 낮았고, Beam 30/50은 완료되지 않았다.
3. r32 seed 재현성, r16 matched seed, scaling ablation에서는 checkpoint 3500과 Beam 10을 seed/condition마다 동일하게 고정했다. 5-seed 확장에서도 이 규칙을 그대로 적용했으며 개별 seed의 validation 최고점을 골라 test를 최적화하지 않았다.
4. spectral analog reranker와 learned cross-modal reranker는 structure-grouped OOF에서 robust gain이 없어서 test로 진행하지 않았다.
5. r16 all-attention validation-only screen은 primary Top-1 평균 +7.4799 pp와 3/3 seed 양수로 사전 진행 규칙을 통과했다. 따라서 다음 단계로 r32/alpha64 all-attention을 validation에서 비교할 수 있다. 이 결정에 test는 사용하지 않았다.
6. r32 all-attention validation-only screen은 primary Top-1 평균 +6.8775 pp와 3/3 seed 양수로 사전 선택 규칙을 통과했다. 따라서 r32/alpha64 q/k/v/out을 향후 잠금 holdout 또는 외부 평가 후보로 선택했다. 이 결정에도 test는 사용하지 않았다.
7. 주 모델 grouped 5-fold OOF 완료 후, holdout 접근 전에 confidence feature, cross-fitted logistic calibration, primary 기권 threshold 0.80, Top-k workload proxy 정의를 동결했다. 0.80 결과의 coverage가 낮더라도 사후에 primary 기준을 교체하지 않았다.

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
- OOF cross-fitted confidence calibration 및 고정 primary 기권 threshold 0.80. 단, 현재 coverage 0.8416%의 매우 제한적인 고신뢰 표시이며 holdout 확인 전 배포 안전 기준은 아님

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

### 동결 모델 실행 환경과 배포 파일

- `models/vgwd_clean_rslora_r32_all_attention_seed123_frozen/`
- `deployment/r32_all_attention/`
- `dist/spectus_r32_all_attention_online_2026-09-02.zip`
- `dist/spectus_r32_all_attention_offline_2026-09-02.zip`
- `dist/SHA256SUMS`

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
- prospectively locked internal re-split의 주 모델 structure-grouped 5-fold OOF(seed 123, 조건별 5,228 rows)에서 r32/alpha64 all-attention Top-1은 38.1790%, q/v는 32.1155%였고, fold별 차이가 5/5 양수였다. pooled 차이 +6.0635 pp의 canonical-connectivity cluster bootstrap 95% CI는 +4.5870~+7.5149 pp였다.
- 같은 grouped OOF에서 fold-cross-fitted logistic calibration은 all-attention raw Beam share 대비 Brier, log loss, ECE-10을 모두 낮췄고 confidence 상위 subset은 전체보다 높은 Top-1 정확도를 보였다. 고정 threshold 0.80은 90.9091% selective accuracy를 보였으나 coverage가 0.8416%뿐이었다.
- all-attention OOF의 Top-3/5/10은 65.1301/76.4728/85.3481%로 Top-1보다 각각 1,409/2,002/2,466개의 정답을 추가 회수했다. 이는 후보 제공의 정량적 잠재력을 지지하지만 실제 연구자 사용자 연구는 아니다.
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
- 주 모델 grouped 5-fold OOF는 training/data seed 123 한 번의 내부 재분할 결과이며 5 folds를 5개의 독립 training seed 반복으로 해석하면 안 된다. 새 잠금 holdout과 외부 데이터 평가는 아직 수행하지 않았다.
- calibration·기권·risk–coverage 수치도 동일한 내부 OOF에 한정된다. 특히 threshold 0.80의 accepted sample은 44건뿐이므로 90.9091%를 일반 운용 정확도나 안전 보장으로 표현하면 안 된다.

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
