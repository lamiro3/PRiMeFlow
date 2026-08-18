# Perturbench 사용 가이드라인 및 진행 과정

### 1. Load Data: load_data_2.py 참고 바람

### 2. train: 6가지 기존 모델 이용해 metric get -> 이걸 baseline으로 삼아 PrimeFlow와 비교할 예정
- **command**: `train experiment=neurips2025/{사용할 데이터셋 명}/linear_best_params_{사용할 데이터셋 명}`

- train 돌릴 때, **data 파일(.h5ad) 위치**: `/perturbench/notebooks/neurips2025/perturbench_data/`
- train command 돌릴 때, **dir 위치**: `/data1/project/taehyeon/PrimeFlow/perturbench`

### 3. datalodaer
- **파일 위치**: `/data1/project/taehyeon/PrimeFlow/perturbench/src/perturbench/data/modules.py`