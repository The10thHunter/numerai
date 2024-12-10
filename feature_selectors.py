import pandas as pd 
import numpy as np 
data = pd.read_parquet("../v5.0/train.parquet")
#len(data_features) = 2376
data = data[[a for a in data.columns if "feature" in a]]
bad_features = []
feats = np.array_split(data.columns, 24)
# DEBUG: print(len(feats[1]), len(feats[0]), len(feats[23]))
for chunk in feats: 
    subtable = abs(data[chunk].corr())
    subtable = subtable.reset_index().melt(id_vars = "index", var_name = "feature2",value_name = 'Correlation')
    subtable = subtable[subtable["index"] != subtable["feature2"]]
    subtable = subtable.sort_values(by = "Correlation", ascending = False)
    subtable = subtable[(subtable["Correlation"] >= 0.80)]
    #["index"].drop_duplicates().tolist()
    bad_features.append(subtable["index"].drop_duplicates())

filename = "bad_features.txt"
outfile = open(filename, 'w')
outfile.write('\n'.join(str(i) for i in bad_features))
