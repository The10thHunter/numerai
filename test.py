from feature_selectors import TreeSelector
import pandas as pd
import time 

data = pd.read_parquet("../v5.0/train.parquet")
subcols = data.columns[0:100]
subcols.append("target")
data = data.dropna()
data = data.sample(frac = 0.075, random_state = 99)
print(data.columns)
print("Data Read Successful! Feeding...")

"""
for f in data.columns: 
    var = data[f].dtypes
    if (var == 'int8' or var == 'float32'): 
        pass
    else: 
        #print(f"Feature: {f} has type: {var}")
        print(data[f].head())
#print(data[["target"]].dtypes)
"""

print("Staring timer...")
start = time.time()
print(TreeSelector(data))
end = time.time()
print("--- %s seconds ---" % (time.time() - start_time))
