#Use this library to convert a dataframe to a batched data loader for NN 
import pandas as pd
import numpy as np 
import torch
#from torch.utils import data
#from torch import nn 
#import torch.save 
typed = torch.float32

"""
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
"""        
class dataManager: #Data MUST BE PREFILTERED. 

    """
    Data must be prefiltered to use datamanager. End goal is to return a DataLoader object so pytorch can do much less work. 
    Idea is this: 
    df = pd.read_filetype(path)
    feature_df = df[features]
    traindf = df["target"] #Or also traindf = df[targets] where targets is a lst containing multiple targets. 
    del df 
    feature_set = dataManager(feature_df).main() #Returns type tensor
    train_set = dataManager(traindf).main() #Returns type tensor 
    #Maybe I won't seperate actually... 
    """
    def __init__(self, data = None): 
        self.data = data
    def __len__(self): #Ngl this was totally recommended by GPT, not sure why yet... -- NVM I GET IT NOW. 
        return len(self.dataframe())
    def __getitem__(self, index): 
        row = self.data.iloc[index]
        return row
    def main(self, bsize): 
        label = torch.tensor(self.data["target"].to_numpy(), dtype = typed)
        feats = torch.tensor(self.data[[a for a in self.data.columns if a != "target"]].to_numpy(), dtype = typed)
        dataset = TensorDataset(feats, label)
        return DataLoader(dataset, batch_size = bsize, shuffle = True)

    @classmethod 
    def read_parquet(cls, filepath, fill_na = 0, **kwargs):
        data = pd.read_parquet(filepath)
        data = data.fillna(fill_na)
        return cls(data)

if __name__ == "__main__": 
    dm = dataManager.read_parquet("../v5.0/train.parquet")
    #dm = dataManager(df)
    len = 0
    for batch in dm.main(10): 
        print(batch)
        if (len  % 20 == 0): 
            print("Success")
            exit() 

