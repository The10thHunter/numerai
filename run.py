#DEPRECATED: from data import dataManager
from model import torchMod
import random
import pandas as pd 
import torch 
from data import Dataset
data = pd.read_parquet("../v5.0/train.parquet")
#Get lst of features #Specifying era1 for now.... 
#data = data[data["era"] == '0001']
rand_feats = random.sample(list(a for a in data.columns if "feature" in a), 100)

feats = torch.tensor(data[rand_feats].to_numpy())
labels = torch.tensor(data["target"].to_numpy())

print("Attempting to save...")
torch.save({"feats": feats, "labels" : labels}, "trialset_1.pt") 

print("Saved. Attempting to read....")
loaded = torch.load("trialset_1.pt")
f_loaded = loaded["feats"]
t_loaded = loaded["labels"]

print("Dataset loaded successfully!")
print("Features shape:", f_loaded.shape)
print("Labels shape:", t_loaded.shape)

dataloader = DataLoader(Dataset(f_loaded, t_loaded), batch_size = 128, shuffle = True)
model = torchMod()
