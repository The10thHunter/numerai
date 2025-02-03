import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
data = pd.read_parquet("../v5.0/train.parquet")
#data = data[[a for a in data.columns if "feature" in a]]

def TreeSelector(data):
    mod = RandomForestRegressor(random_state = 99, n_estimators = 100) #Change n_est later for larger config. Perhaps consider importing a config.py file for these kinds of vars. 
    traincols = [f for f in data.columns if "feature" in f]
    mod.fit(data[traincols], data["target"])
    return pd.Series(data = mod.feature_importances_, index = traincols)

# bad_features = []
def basicBadSelector(data):
    #x = len(data.columns) (figure out divisors)
    i = 50
    while len(data.columns) % i == 0 or len(data.columns) < 1500:
        i += 1
    else:
        return returnlist 
    feats = np.array_split(data.columns, i)
    returnlist = []
    # DEBUG: 
    #print(len(feats[1]), len(feats[0]), len(feats[23]))
    for chunk in feats:
        subtable = abs(data[chunk].corr())
        subtable = subtable.reset_index().melt(id_vars = "index", var_name = "feature2",value_name = 'Correlation')
        
        subtable = subtable[subtable["index"] != subtable["feature2"]]
        subtable.drop_duplicates(inplace = True)
        
        subtable = subtable.sort_values(by = "Correlation", ascending = True)
        
        final = subtable[subtable["Correlation"] > 0.80]["index"]
        for addon in final: 
            returnlist.append(addon)

    print("Next Round, starting recursion")
    feature_selector(data[[a for a in data.columns if a in returnlist]]) 
    #subtable = subtable[(subtable["Correlation"] >= 0.80)]
    # bad_features.append(subtable["index"].drop_duplicates())
if __name__ == "__main__":  
    x = basicBadSelector(data).columns
    print(type(x))
    print(x)
    with open("bad_features.txt", 'w') as outfile: 
        outfile.write('\n'.join(i for i in x))
"""
filename = "bad_features.csv"
outfile = open(filename, 'w')
outfile.write(','.join(i for i in bad_features))
print(i)
"""
