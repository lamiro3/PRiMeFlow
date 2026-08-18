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