import pandas as pd 
import numpy as np 
data = pd.read_parquet("../v5.0/train.parquet")
#len(data_features) = 2376
data = data[[a for a in data.columns if "feature" in a]]

feats = np.array_split(data.columns, 24)
# DEBUG: print(len(feats[1]), len(feats[0]), len(feats[23]))

for chunk in feats: 
    subtable = data[chunk].corr().melt(id_vars = data.columns, value_name = 'Correlation')
    print(subtable.head())

    
