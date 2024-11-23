import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
"""
from sklearn import 
from sklearn import 
#...
"""
#SEED = 
data = pd.read_parquet("../v5.0/train.parquet")

train = data[[a for a in data.columns if "feature" in a]]
test = data[[b for b in data.columns if "target" in b]]
data = None
corr_matrix = train.corr(method = "pearson") #Method can be defined using a function.

corr_matrix.to_csv()

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
