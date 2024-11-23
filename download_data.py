#Download Script for NumerAI
from numerapi import NumerAPI
import pandas as pd 

napi = NumerAPI()
napi.download_dataset("v5.0/train.parquet")
