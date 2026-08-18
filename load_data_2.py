# 굳이 load_data.py 처럼 고생할 필요 없이 이거 하나로 데이터 로드 가능 (심지어 전처리 된 버전임)

from perturbench.data.accessors.norman19 import Norman19
from perturbench.data.accessors.jiang24 import Jiang24
from perturbench.data.accessors.mcfaline23 import McFaline23

from utils import showDataset

# <ds_name>_accessor.get_anndata(): Get the preprocessed anndata object
# <ds_name>_accessor.get_dataset(): Get a PyTorch Dataset

norman19_accessor = Norman19()
jiang24_accessor = Jiang24()
mcfaline23_accessor = McFaline23()

# jiang24_data = jiang24_accessor.get_dataset()
# showDataset(jiang24_data, 10)

norman19_data = norman19_accessor.get_anndata()