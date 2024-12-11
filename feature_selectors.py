import pandas as pd 
import numpy as np 

data = pd.read_parquet("../v5.0/train.parquet")
data = data[[a for a in data.columns if "feature" in a]]

# bad_features = []
def feature_selector(data):
    #x = len(data.columns) (figure out divisors) 
    for i in range(250, 100, -1): 
        if len(data.columns) % i == 0: 
            return i 
            break
    else:
        return data
    
    feats = np.array_split(data.columns, i)
    relapse_lst = []
    # DEBUG: 
    #print(len(feats[1]), len(feats[0]), len(feats[23]))
    for chunk in feats:
        subtable = abs(data[chunk].corr())
        subtable = subtable.reset_index().melt(id_vars = "index", var_name = "feature2",value_name = 'Correlation')
        subtable = subtable[subtable["index"] != subtable["feature2"]].drop_duplicates()
        subtable = subtable.sort_values(by = "Correlation", ascending = False)
        subtable = subtable.drop(subtable[subtable["Correlation"] > 0.80].index, inplace = True)

        relapse_lst.append(subtable["feature2"].tolist())

    print("Next Round, starting recursion")
    feature_selector(data[[a for a in data.columns if a in relapse_lst]])   
    #subtable = subtable[(subtable["Correlation"] >= 0.80)]
    # bad_features.append(subtable["index"].drop_duplicates())

x = feature_selector(data).columns
print(type(x))
print(x)
with open("good_features.txt", 'w') as outfile: 
    outfile.write('\n'.join(i for i in x))
"""
filename = "bad_features.csv"
outfile = open(filename, 'w')
outfile.write(','.join(i for i in bad_features))
print(i)
"""
