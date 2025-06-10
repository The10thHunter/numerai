#Model file for testing the model as well as the deployable basic functions.
import pandas as pd
import pickle
import numpy as np
import torch 
import time
#from sshout import fileout #sshout.py will be included later... 
#from sklearn import DecisionTreeRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import train_test_split
import random

RAND = 99

#class torchMod() should be trainable on a numpy array. It will not use raw data, perhaps include method using sk to break down certain features or aggregate certain ones? 

class torchMod(torch.nn.Module): #Layerlst is a var describing feature to input transition sizes. #Ex is for 500 inputs and 5 resulting outs [500, 250, 125, 75 , 25, 5]. On that thought, see comment on line line 27
    def testing(testfunc): 
        def wrap(*args, **kwargs): 
            #For testing later on, this way you can call @testing and it will allow you to spit out test results prettier for different error calcs
            start = time.time()
            print("Timer started.")
            result = testfunc(*args, **kwargs)
            end = time.time()
            print(f"Task finished: {(end - start) / 60} minutes.")
            return result
        return wrap

    def __init__(self): #Perhaps implement a loop based on divisible inputs of features for testing purposes. You can do this by literally just iteratting with a for and then trying to square root down.
        super().__init__()
        self.layerlst = [2376, 792, 264, 33, 11, 1]
        #self.ftensor = ftensor #feature tensor
        #self.ttensor = ttensor #target tensor
        self.stack = torch.nn.Sequential(
                torch.nn.Linear(self.layerlst[0], self.layerlst[1]),
                torch.nn.Tanh(),
                torch.nn.Linear(self.layerlst[1], self.layerlst[2]), 
                torch.nn.Tanh(), 
                torch.nn.Linear(self.layerlst[2], self.layerlst[3]), 
                torch.nn.Tanh(),
                torch.nn.Linear(self.layerlst[3], self.layerlst[4]),
                torch.nn.Tanh(), 
                torch.nn.Linear(self.layerlst[4], self.layerlst[5])
            )
        self.loss_fn = torch.nn.MSELoss()
                        
    @testing
    def forwardprop(self, ftensor):
        #You can alter the input here later. But for now we'll manually shove it in. Come back to it later. 
        return self.stack(ftensor)

    def lossFn(self, inpt, target):
        return self.loss_fn(inpt, target)

    @testing
    def backwardprop(self):
        loss = self.loss_fn(prediction, target)
        loss.backward()

    @staticmethod
    def cudaTest():
        if (torch.cuda.is_available()):
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device (#): {torch.cuda.current_device()}")
            print(f"CUDA device (name): {torch.cuda.get_device_name}")
        else: 
            print("Cuda is unavailable... using CPU.")

class NumeraiDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

def training():
    torchMod.cudaTest()

    # === Load and prepare data ===
    filepath = "../v5.0/train.parquet"
    data = pd.read_parquet(filepath)
    target_col = "target"
    #filt = random.sample([f for f in data.columns if "feature" in f], 100)  # Select 100 random features

    # === Prepare dataset and dataloader ===
    dataset = NumeraiDataset(data, data.columns, target_col)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # === Initialize model, move to device, define optimizer ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchMod().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Training loop ===
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (features, targets) in enumerate(dataloader):
            features, targets = features.to(device), targets.to(device)

            # Forward
            predictions = model.forwardprop(features)

            # Loss
            loss = model.lossFn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}")

    print(f"Epoch [{epoch+1}] completed. Average Loss: {running_loss / len(dataloader):.6f}")
    torch.save(model, "../assets/torchMod.pt")

def validation():
    filepath = "../v5.0/validation.parquet"
    data = pd.read_parquet(filepath)
    dataset = NumeraiDataset(pd.read_parquet(filepath), data.columns, target_col) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True) 
    model = torch.load("../assets/torchMod.pt").to(device)

    for batch_idx, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device), targets.to(device)

        pred = model.eval(features)
        loss = model.lossFn(pred, targets)

        running_loss += loss.item()

        print(f"Epoch [{epoch+1}] completed. Average Loss: {running_loss / len(dataloader):.6f}")


if __name__ == '__main__':
    training()
    validation()

"""
def basicGBRTrain(train_df, target):
    #Basic model skeleton 
    #DEFINE CONSTS
    N_EST = 50
    LEARN_RATE = 1.25
    #See outside var, for scope reasons. RAND = 99
    #SETUP: 
    x = train_df[[a for a in features_df.columns if "feature" in a]]#[:, :10000] <- Limiter for compute intensive reasons 
    #If you want, just swap functions 
    mod = GradientBoostingRegressor(random_state = RAND, learning_rate = LEARN_RATE, n_estimators = N_EST)
    if type(target) == list():
        mod.fit(x, train_df[[target]])
    elif type(target) == str():
        mod.fit(x, train_df[target])
    else: 
        print("Failed. Target col is not list or str")
        exit()
    return mod
    #END OF BASIC MODEL


if __name__ == "__main__":
    random.seed(RAND)
    x, y = train_test_split(pd.read_parquet("../v5.0/train.parquet"), test_size = 0.33, random_state = RAND)
    for a in range(500, 2001, 500): 
        mod = basicGBRTrain(x[[random.sample(x.columns,k = a)]].sample(frac = 0.5), "target")
        print(mod.score(y, y["target"]))
"""
