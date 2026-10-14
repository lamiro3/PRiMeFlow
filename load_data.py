# 굳이 load_data.py 처럼 고생할 필요 없이 이거 하나로 데이터 로드 가능 (심지어 전처리 된 버전임)

from perturbench.data.accessors.norman19 import Norman19
import pandas as pd
from torch.utils.data import DataLoader

def showDataset(dataset, batch_size):
    
    #데이터 로더 정의
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_batch, labels_batch = next(iter(dataloader))
    
    data_dict = {
        "Feature (X)": data_batch.squeeze().tolist(),
        "Label (y)": labels_batch.tolist()
    }
    
    df = pd.DataFrame(data_dict)
    print(df)

# <ds_name>_accessor.get_anndata(): Get the preprocessed anndata object
# <ds_name>_accessor.get_dataset(): Get a PyTorch Dataset

norman19_accessor = Norman19()

# jiang24_data = jiang24_accessor.get_dataset()
# showDataset(jiang24_data, 10)

norman19_data = norman19_accessor.get_dataset()
showDataset(norman19_data, 10)