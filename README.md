# PRiMeFlow

[PerturBench](https://github.com/altoslabs/perturbench) 프레임워크 위에 **PRiMeFlow**(Conditional Flow Matching 기반 단일세포 섭동 반응 예측 모델)를 구현하고, PerturBench가 제공하는 8개 baseline 모델과 성능을 비교한 프로젝트입니다.

## 모델 개요

- **입력/조건**: 대조군(control) 세포의 유전자 발현, 섭동(perturbation) one-hot, 그리고 covariate(선택)
- **Velocity field**: 1D U-Net (`unet1d.py`, FiLM 방식으로 조건 주입, `channel_mult=(1,2,4,8)`, `num_res_blocks=2`) + sinusoidal time embedding
- **Source distribution**: 표준 가우시안 노이즈에서 출발 (`x0 ~ N(0,1)`) — control이 아닌 noise-source를 쓰는 것이 원 PRiMeFlow 논문 방식
- **학습**: Conditional Flow Matching loss (`L_cfm`, straight-line interpolant `x_t = (1-t)x0 + t*x1`), Classifier-Free Guidance를 위해 학습 중 `p_uncond=0.1` 확률로 조건을 0-vector로 드롭
- **추론**: Euler method로 100-step ODE 적분, `cfg_weight`로 guidance 강도 조절, VRAM overflow 방지를 위해 512개 단위 chunked inference 적용

## 실험

`CFG=1.0 / 3.0 / 5.0` 세 가지 guidance 강도로 PRiMeFlow를 학습하고, PerturBench의 8개 baseline(Linear_additive, Latent_additive, Decoder_only, CPA, SAMS_VAE, Biolord, Flow_Matching-MLP, Flow_Matching-Unet)과 10개 지표(RMSE, cosine similarity(PCA/logFC), R², top-k recall, MMD 등, raw + rank 버전)로 비교했습니다.

결과 시각화는 `metric_each_model_graph.py`로 생성합니다.

```bash
python metric_each_model_graph.py
# -> metric_radar.png   (모델별 성능 프로파일, 레이더 차트)
# -> metric_heatmap.png (지표별 정규화 점수 히트맵 + 종합 점수)
```

## 결과 요약 및 분석

<img width="2021" height="1076" alt="metric_heatmap" src="https://github.com/user-attachments/assets/6d6ac405-27ec-4790-9155-9fe47a95187e" />
<img width="2259" height="3134" alt="metric_radar" src="https://github.com/user-attachments/assets/fb5d2182-9bbd-4afe-a2ef-df0b444e08ed" />


PRiMeFlow는 scale-invariant 지표(R², top-k recall, cosine similarity)에서는 baseline과 비슷하거나 나은 수준이지만, RMSE·MMD처럼 절대 스케일에 민감한 지표에서는 baseline보다 크게 낮은 점수를 보였습니다. 원인으로 다음 두 가지를 확인했습니다.

Noise-source sampling의 구조적 영향: 같은 코드베이스에서 Flow_Matching-Unet baseline도 source를 noise로 바꾸면 동일하게 RMSE/MMD가 급격히 나빠지는 패턴이 재현됩니다 (source=control: RMSE .112/MMD 1.5 → source=noise: RMSE .545/MMD 16.3). 즉 이 벤치마크의 pointwise 지표 자체가 noise-source 샘플링에 구조적으로 불리하게 작동하는 것으로 보입니다.
학습 수렴 부족: CFG 1.0/3.0/5.0 세 variant 모두 22~56 epoch 내에서 val_loss가 약 0.34 근처에서 정체되었고, 이는 학습 목적함수의 하한에 가까운 값이거나 아직 충분히 수렴하지 않은 상태로 판단됩니다.
FiLM 기반 조건 주입 자체(unet1d.py)는 구조적으로 문제가 없는 것으로 확인했습니다. 평가 파이프라인(예: MMD 계산 시 perturbation당 샘플 수 등)은 아직 코드로 직접 재현하지 못해 추가로 검증이 필요한 부분입니다.

## 저장소 구조

```
PRiMeFlow/
├── perturbench/              # PerturBench 프레임워크 + PRiMeFlow 모델 구현
│   └── src/perturbench/
│       ├── modelcore/models/primeflow.py  # PrimeFlow (velocity field, CFM loss, ODE 추론)
│       └── configs/model/primeflow.yaml   # 하이퍼파라미터
├── load_data.py, load_data_2.py   # Hugging Face 데이터셋 로드/전처리 실험
├── inspect_batch.py               # 배치 단위 데이터 확인용 스크립트
├── utils.py                       # 데이터로더 확인 등 유틸 함수
├── metric_each_model_graph.py     # 모델별 평가 지표 시각화 (radar / heatmap)
├── metric_radar.png, metric_heatmap.png
└── logs/                          # 학습 로그 및 체크포인트 (PRiMeFlow_CFG_1/3/5)
```

## 설치

```
conda create -n primeflow python=3.11
conda activate primeflow
cd perturbench
pip install -e .
데이터셋 다운로드 및 기본 사용법은 perturbench/README.md를 참고하세요.
```

## 참고

베이스 프레임워크인 PerturBench를 사용한다면 아래 논문을 인용해주세요.

```
@inproceedings{wu2025perturbench,
  title={PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis},
  author={Yan Wu and Esther Wershof and Sebastian M Schmon and Marcel Nassar and Błażej Osiński and Ridvan Eksi and Zichao Yan and Rory Stark and Kun Zhang and Thore Graepel},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=PPPDuyiZaG},
}
```
