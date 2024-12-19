import pandas as pd 
import numpy as np 

data = pd.read_parquet("../v5.0/train.parquet")
targetcols = data[[a for a in data.columns if "target" in a]].columns 
featurecols = data[[a for a in data.columns if "feature" in a]].columns 
#Main target: target
data[featurecols].corrwith(data[targetcols])

