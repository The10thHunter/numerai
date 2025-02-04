import torch 
from model import torchMod
import pandas as pd 
import numpy as np 
import random
from sklearn.model_selection import train_test_split as tts

df = pd.read_parquet("../v5.0/train.parquet")
feats = df[random.sample([f for f in data.columns if "feature" in f], 99)].to_numpy()
targets = df[target].to_numpy()
del df #free the memory


trainx, testx, trainy, testy = tts(feats, targets, test_size = 0.33, random_state = 99)

ftensor = torch.tensor(trainx) 
ttensor = torch.tensor(trainy)
ftest = torch.tensor(testx)
ttest = torch.tensor(testy)

model = torchMod()
y_pred = model.stack(ftensor)
print(model.backwardprop(y_pred, ttensor))
print("Training and backprop 1 Complete...")
loss = model.eval(ftest)
print(model.lossFn(loss, ttest))

if (input("Save mod? (y/n): ") == 'y'): 
    torch.save(model, "../v5.0/modeltest.pt")
