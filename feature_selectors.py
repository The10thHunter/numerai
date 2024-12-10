import pandas as pd 
import numpy as np 
data = pd.read_parquet("../v5.0/train.parquet")
#len(data_features) = 2376
data = data[[a for a in data.columns if "feature" in a]]

feats = np.array_split(data.columns, 24)
# DEBUG: print(len(feats[1]), len(feats[0]), len(feats[23]))
features_useful = data.columns.tolist()
for chunk in feats: 
    subtable = abs(data[chunk].corr())
    subtable = subtable.reset_index().melt(id_vars = "index", var_name = "feature2",value_name = 'Correlation')
    subtable.sort_values(by = "Correlation", ascending = False)
    for subtable["Correlation"] < 0.90: 
        features_useful.append(subtable["index"])

    print(len(features_useful))
