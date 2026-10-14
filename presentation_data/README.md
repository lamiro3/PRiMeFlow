# PRiMeFlow 결과 데이터 (발표자료 제작용)

이 폴더는 PRiMeFlow 결과 발표 자료를 만드는 데 필요한 데이터 전부입니다.
모든 수치는 `metric_each_model_graph.py --export` 로 생성했고, 아래 본문의 숫자는 CSV 와 같습니다.

## 1. 배경

- **과제**: 단일세포 perturbation 반응 예측. 학습 때 보지 않은 유전자 perturbation 을 받은 세포의 발현(HVG 2000개)을 예측한다.
- **데이터 / 평가**: Norman19 (K562, CRISPRa 단일·이중 유전자 perturbation), PerturBench 파이프라인, test split (`final_test`), seed 245. 모든 baseline 과 PRiMeFlow 가 같은 split 과 같은 평가 코드를 사용했다.
- **PRiMeFlow**: 유전자 발현 공간에서 직접 동작하는 conditional flow matching 모델.
  - 가우시안 노이즈에서 출발해 perturbation 조건을 보고 실제 발현으로 흘려보내는 속도장을 학습한다.
  - 속도장은 1D U-Net 이고, classifier-free guidance(CFG)를 쓴다.
- **지난 발표 피드백**: "overall 점수가 적절하지 않다."
  - 이전 overall 점수는 지표마다 정규화 방식이 달랐다. 그래서 일부 지표가 사실상 상수처럼 작동해 의도치 않은 가중치가 생겼다.
  - 비교하는 모델 집합이 바뀌면 다른 모델의 점수까지 바뀌었다.
  - 성격이 다른 지표를 숫자 하나로 합쳐 trade-off 를 가렸다.

## 2. 이번에 바꾼 평가 방식 (피드백 반영)

1. **정규화 = 지표별 순위.** 각 지표에서 모델들을 1위(최고)부터 순위를 매긴다. 동점이면 평균 순위를 준다. 스케일이나 이상치와 무관하고, 모든 지표를 같은 무게로 다룬다.
2. **지표 10개를 성격별로 4개 축으로 묶음.** 축 점수는 축 안 지표들의 평균 순위다.
   - **Accuracy**: 예측 평균 발현이 실제와 얼마나 가까운가 (RMSE, cos PCA, cos logFC, R²)
   - **Specificity**: 예측이 다른 perturbation 이 아니라 *자기* perturbation 에 가장 가까운가 (각 지표의 rank 버전 4개)
   - **DEG recall**: 실제 top-k 차등발현 유전자를 맞히는가 (top-k recall)
   - **Distribution**: 단일세포 분포 수준에서 실제와 같은가 (MMD PCA)
3. **overall 점수는 만들지 않음.** 축 간 trade-off 를 그대로 보여준다. 축에 가중치를 줄 근거도 없다.
4. **메인 비교와 ablation 을 분리.** 순위는 비교 집합 안에서 매겨진다. 그래서 PRiMeFlow 변형을 여러 개 넣으면 baseline 순위가 밀려 왜곡된다.
   - 메인 비교: baseline 6개 + 대표 PRiMeFlow 1개 (7개 모델)
   - ablation: PRiMeFlow 변형 4개끼리만 비교

## 3. 핵심 결과

### 메인 비교 (7개 모델, `main_axis_ranks.csv`, 평균 순위, 1 = 최고)

| 모델 | Accuracy | Specificity | DEG recall | Distribution |
|---|---|---|---|---|
| **PRiMeFlow (Unet+gemb, CFG=1.0)** | 3.25 | 7.0 | **1.0** | **1.0** |
| Latent_additive | 2.25 | **1.0** | 5.5 | 6.0 |
| Decoder_only | 2.25 | 2.25 | 2.0 | 7.0 |
| CPA | 2.75 | 3.25 | 3.0 | 5.0 |
| SAMS_VAE | 5.25 | 4.5 | 4.0 | 4.0 |
| Biolord | 6.5 | 4.25 | 5.5 | 2.0 |
| Linear_additive | 5.75 | 5.75 | 7.0 | 3.0 |

- **강점: DEG recall 1위, Distribution 1위.**
  - top-k recall 0.188: 2위 Decoder_only 0.168. CPA 0.050, 나머지는 0.001 이하.
  - MMD PCA 0.867: baseline 은 1.67 ~ 3.22.
  - R² 도 0.220 으로 1위다 (2위 Decoder_only 0.165, baseline 4개는 음수).
- **중간: Accuracy 3.25.** RMSE 0.0659 로, 최고인 Latent_additive 0.0417 보다 나쁘다. cos PCA 0.618 은 상위 3개 모델(0.83 ~ 0.88)보다 낮다.
- **약점: Specificity 꼴찌(7위).**
  - 예를 들어 RMSE rank 는 0.0855 이고, Latent_additive 는 0.0135 다.
  - 즉 예측이 평균적으로는 맞지만 perturbation 마다 충분히 구별되지 않는다.
  - 다음 개선 방향으로 제시할 부분이다.
- **baseline 들의 trade-off.** Latent_additive 와 Decoder_only 는 Accuracy·Specificity 상위지만 Distribution 은 하위권(6·7위)이다. 축마다 1위가 다르다. 이것이 overall 점수 하나로 요약하면 안 되는 근거다.

### Ablation (PRiMeFlow 변형 4개, `ablation_axis_ranks.csv`)

| 변형 | Accuracy | Specificity | DEG recall | Distribution |
|---|---|---|---|---|
| Unet, CFG=1.0 (gene embedding 없음) | 4.0 | 4.0 | 4.0 | 4.0 |
| Unet+gemb, CFG=1.0 | 1.75 | **1.5** | 3.0 | **1.0** |
| Unet+gemb, CFG=3.0 | **1.5** | 1.75 | 2.0 | 2.0 |
| Unet+gemb, CFG=5.0 | 2.75 | 2.75 | **1.0** | 3.0 |

- **gene embedding 추가 효과가 가장 크다.** 모든 지표에서 4위에서 상위로 올라간다.
  - RMSE: 0.548 → 0.066
  - MMD PCA: 14.76 → 0.867
  - cos PCA: 0.135 → 0.618
- **CFG weight 를 올리면 trade-off 가 생긴다.**
  - DEG recall 은 오른다: top-k 0.188 → 0.349 → 0.358.
  - 대신 분포와 특이성은 나빠진다: MMD 0.867 → 3.03 → 6.08, RMSE rank 0.086 → 0.315 → 0.388.
  - 메인 모델로 CFG=1.0 을 고른 이유다.

## 4. 해석할 때 주의할 점 (발표에서 질문 대비)

- **seed 1개 결과다.** 오차 막대나 유의성 검정은 없다.
- **순위는 차이의 크기를 버린다.** 1·2위가 근소한지 큰 차이인지는 `raw_metrics.csv` 원값으로 확인해야 한다 (예: DEG recall 1·2위는 0.188 vs 0.168 로 근소).
- **gene embedding ablation 에는 다른 변경이 섞여 있다.**
  - gemb run 은 4-GPU DDP(유효 batch 512)로 학습했다.
  - gemb 없는 Unet run 은 단일 GPU, batch 128 이다.
  - 따라서 개선을 전부 gene embedding 효과로 단정할 수는 없다.
- **CFG 3.0 / 5.0 은 재학습 없이 평가만 다시 한 것이다.** Unet+gemb 와 같은 checkpoint(epoch 40)를 쓰고 guidance weight 만 바꿨다.
- **rank 지표는 0 에 가까울수록 좋다.** 0 = 예측이 항상 자기 perturbation 에 가장 가까움.

## 5. 파일 안내

| 파일 | 내용 |
|---|---|
| `figures/metric_summary.png` | **메인 슬라이드용.** 4개 축 패널, 축별 평균 순위 dot plot, PRiMeFlow 강조 |
| `figures/metric_ablation.png` | ablation 슬라이드용. 같은 형식으로 PRiMeFlow 변형 4개 비교 |
| `figures/metric_heatmap.png` | 부록용. 지표 10개 원값 표, 색 = 지표별 순위 |
| `raw_metrics.csv` | 전체 10개 run × 10개 지표 원값 (wide) |
| `main_metric_ranks.csv` / `main_axis_ranks.csv` | 메인 비교(7개 모델)의 지표별 순위 / 축별 평균 순위 |
| `ablation_metric_ranks.csv` / `ablation_axis_ranks.csv` | ablation(4개 변형)의 지표별 순위 / 축별 평균 순위 |
| `metrics_long.csv` | 차트용 tidy 형식: comparison, model, metric, axis, direction(+1 높을수록 좋음 / -1 낮을수록 좋음), value, rank, n_models |
| `metrics_info.csv` | 지표 정의: 이름, 축, 방향, 설명 |
| `models_info.csv` | 모델 설명, baseline/PRiMeFlow 구분, 메인·ablation 포함 여부 |

## 6. 제안하는 슬라이드 흐름

1. 지난 피드백과 평가 방식 변경 (2절): 이전 방식의 문제, 순위 + 4개 축 + overall 없음
2. 메인 결과 (`metric_summary.png`): DEG recall·Distribution 1위, Specificity 약점
3. Ablation (`metric_ablation.png`): gene embedding 효과, CFG trade-off
4. 한계와 다음 단계: Specificity 개선, 다중 seed, ablation 교란 요인 정리
5. 부록 (`metric_heatmap.png`): 원값 전체
