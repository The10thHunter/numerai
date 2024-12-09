import pandas as pd 

corr = pd.read_csv("sample_corr.csv")
corr = corr.drop_duplicates()
print(corr.head())
