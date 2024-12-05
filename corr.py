import pandas as pd 
import time 
def test1(x, data):    
    start = time.time()
    data = data[[a for a in data.columns if "feature" in a]]
    data = data.iloc[:,:x]
    data.corr().to_csv("sample_corr.csv")
    end = time.time()
    return {x: x, y :end-start}

def test2(x, data): 
    start = time.time()
    data = data[[a for a in data.columns if "feature" in a]]

    #result = pd.DataFrame()
    final = []
    for z in range(0,x+1,1): 
        colpop = data.columns[z]
        curr_feature = data.pop(colpop)
        for y in range(z, x+1, 1): 
            corr_value = curr_feature.corr(data.iloc[y])
            result = {"Feature 1" : colpop,
                      "Feature 2" : data.iloc[y],
                      "Corr" : corr_value}
            final.append(result)

    temp = pd.DataFrame(final).to_csv("sample_corr.csv")
    end = time.time()
    print(str(x) + " features in " + str(end-start) + " time (sec)")

data = pd.read_parquet("../v5.0/train.parquet")
for x in range(300, 500, 50):
    test2(x, data)
