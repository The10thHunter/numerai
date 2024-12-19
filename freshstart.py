import pandas as pd 

dataset = pd.read_parquet("../v5.0/train.parquet")
features = dataset[[a for a in dataset.columns if "feature" in a]]
targets = dataset[[a for a in dataset.columns if "target" in a]]
maintarget = dataset["target"] # "target" is main target
def ideal_features(features, targetcol):
    targetcorr = features.corrwith(targetcol).abs()
    return targetcorr.head()

ideal_features(features, maintarget)
