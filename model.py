import pickle
import pandas as pd
#from sklearn import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import random

def basicGBRTrain(filtered_df): #n_est exists as a test var, it will be deployed via range() for testing
    #DEFINE CONSTS
    N_EST = 50
    LEARN_RATE = 1.25
    RAND = 99
    #SETUP: 
    x = filtered_df[[a for a in filtered_df.columns if "feature" in a]]
    y = filtered_df["target"] #Main Target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = RAND, shuffle = True)

    #If you want, just swap functions 
    mod = GradientBoostingRegressor(random_state = RAND, learning_rate = LEARN_RATE, n_estimators = N_EST)
    mod.fit(x_train, y_train)
    
    return mod.score(x_test, y_test)
    #END OF BASIC MODEL

if __name__ == "__main__":
    df = pd.read_parquet("../v5.0/train.parquet")
    #Sample feature columns from lst 
    features = df[[a for a in df.columns if "feauture" in a]].columns
    for n in range(500, 2001, 500): 
        random.seed(99)
        random_cols = random.sample(features, n)
        score = basicGBRTrain(df[[a for a in df.columns if a in random_cols or a == "target"]])

        print(f"Scored: {round(score, 4)} where random features = {n}")
