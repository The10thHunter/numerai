#Model file for testing the model as well as the deployable basic functions.
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

class torchMod(nn.Module, layerlst): #Layerlst is a var describing feature to input transition sizes. #Ex is for 500 inputs and 5 resulting outs [500, 250, 125, 75 , 25, 5]. On that thought, see comment on line line 27
    def testing(testfunc): 
        def wrap(): 
            #For testing later on, this way you can call @testing and it will allow you to spit out test results prettier for different error calcs
            start = time.time()
            print("Timer started.")
            testfunc()
            end = time.time()
            print(f"Task finished: {end - start / 60} minutes.")
        return wrap

    def __init__(self): #Perhaps implement a loop based on divisible inputs of features for testing purposes. You can do this by literally just iteratting with a for and then trying to square root down.
        super().__init__()
        self.linear_relu_stack = nn.Sequential( #Sequential is the ordering of the layers. 
                nn.Linear(),
                nn.tanh(), 
                nn.Linear(),
                nn.tanh(),
                nn.Linear(), #Linear(inputs, outputs)
                )
    @testing
    def forwardprop(self, x):
        #You can alter the input here later. But for now we'll manually shove it in. Come back to it later. 
        return self.linear_relu_stack(x)
    def lossfunc(self): 

    def backwards(self):

    def cudaTest():
        if (torch.cuda_is_available()):
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device: {torch.cuda.current_device()}")
        else: 
            print("Cuda is unavailable... using CPU.")


def main(): 
    filepath = "../v5.0/train.parquet"
    data = pd.read_parquet(filepath) 
    filt = random.sample([f for f in data.columns if "feature" in f], 100) #Short for filter

    

if __name__ == '__main__': 
    main()

        


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
