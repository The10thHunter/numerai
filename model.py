import pandas as pd
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import train_test_split
"""
from sklearn import 
from sklearn import 
#...
"""
SEED = 0
#data = pd.read_parquet("../v5.0/train.parquet").sample(frac = 0.1, random_state = SEED, replace = False)


def corrTable(dataframe):
    #test = dataframe[[b for b in dataframe.columns if "target" in b]]
    corr_matrix = dataframe.corr(method = "pearson").abs()
    corr_matrix = corr_matrix.unstack().reset_index()

    corr_matrix.columns = ["feature1", "feature2", "correlation"]
    corr_matrix = corr_matrix[corr_matrix["feature1"] != corr_matrix["feature2"]]
    
    return corr_matrix

data = pd.read_parquet("../v5.0/train.parquet").sample(frac = 0.05, random_state = SEED)

data = data.iloc[:,:50]
print("Data Read.")
final = corrTable(data)
final.to_csv("sample_corr.csv")

#print("Length of Training: " + str(len(train.columns)))
#print("Length of Test: " + str(len(test.columns)))

"""
class MicroModel: 
    private: 
        #Private members
    protected: 
        #Protected 
    public:
        #Public members

"""
