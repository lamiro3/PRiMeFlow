from datasets import load_dataset

# [원본 데이터 주소]: https://huggingface.co/datasets/altoslabs/perturbench/tree/main

# 논문에서는 perturbench의 Srivatsen20, Jiang24, Norman19 이 데이터 3개를 이용했음
# 본인(고태현)은 frangieh21, Jiang24, op3 이렇게 3개 이용해보겠음
# 원래 3개의 df 모두 한번에 합칠려고 했으나, 구조가 다르기 때문에 따로따로 가져오는 방식을 택했음

# [Features & 데이터 개수] // 차례대로
# [주의!!] - 셋 다 생김새가 모두 다른데, 이게 정확히 뭘 의미하는 건지 & 알게 되면 전처리도 진행해야 할 듯

# 1. <frangieh21>
# features: ['CELL_1', 'train'],
# num_rows: 218330

# 2. <jiang24>
# features: ['07_48_88_1_1_1_1_1_1_1_1_1', 'train'],
# num_rows: 1628475

# 3. <op3>
# features: ['AAACGAAAGAGCGACT-1_SRTP0006403-0', 'train'],
# num_rows: 298086
# AAACGAAAGAGCGACT: 16개 염기 (A/C/G/T)로 된 10x Genomics cell 바코드 ~ 각 세포에 붙은 고유 바코드
# -1: CellRanger가 붙이는 GEM well 접미사 (샘플 여러개 합칠 때 1, 2, ... n 으로 구분)
# SRTP0006403: 샘플/도너 ID. 이 세포가 어느 검체에서 나왔는지를 알려줌

PERTURBENCH = "altoslabs/perturbench" # Perturbench data source url

FRANGIEH = "frangieh21"
JIANG = "jiang24"
OP = "op3"

TEST_RATE = 0.2 # 전체 데이터 비중: train: 0.8 & test: 0.2

class Df:
    def __init__(self, train, test):
        self.train = train
        self.test = test


df_info = {
    "frangieh21": "frangieh21_split.csv",
    "jiang24": "jiang24_split.csv",
    "op3": "op3_split.csv"
}

feat_name = ["CELL_1", "07_48_88_1_1_1_1_1_1_1_1_1", "AAACGAAAGAGCGACT-1_SRTP0006403-0"]

df = {}

for idx, name, url in enumerate(df_info.items()):
    train = load_dataset(PERTURBENCH, data_files=url)["train"] # 이 train은 split명 얘기하는 거 (Hugginface 사이트에서 보면 뭔 말인지 알거임)
    train =  train.remove_columns("train") # train label 모두 제거
    splits = train.train_test_split(test_size=TEST_RATE, seed=42)
    df[name] = Df(splits["train"], splits["test"])
    

# load 된 데이터 결과
print(df[OP].test[:100])