#Use this library to convert a dataframe to a batched data loader for NN 
import pandas as pd
import numpy as np 
from torch.utils import data
from torch import nn 
import torch.save 

def initDataloader(fpath, nocols): 
    df = pd.read_parquet(fpath)
    df = df[[a for a in data.columns not in nocols]
    tensor = torch.tensor(df[df.columns != 'target'].to_numpy)
    ttensor = torch.tensor(df[df.columns == 'target'].to_numpy)
    save = input("Do you want to save the dataset as a tensor?")
    if (save): 
        torch.save(tensor, "feature-trainparq.pt")
        torch.save(ttensor, "target-trainparq.pt")
        #Each of these saved tensors will be pre-filtered 
    return tensor, ttensor
        

