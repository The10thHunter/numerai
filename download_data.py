#Download Script for NumerAI

from numerapi import NumerAPI
from os import listdir
#from sqlalchemy import create_engine 
import pandas as pd 

napi = NumerAPI()
#engine = create_engine("mysql+pymysql://user:password@host:port/dbname")
#Dataset already installed on local. Edit this out during commit 

def basicInstall(): 
    napi.download_dataset("v5.0/train.parquet")

def allInstall(path):
#If it is desirable, you may install all data formats this way instead:
    for file in [a for a in napi.list_datasets() if a.endswith(".parquet") and a isnot in listdir(path)]: 
        napi.download_dataset(file)

def autoSql(filepath, eng):
#If you manually installed and just want an easy mySQL db conversion: 
    for item in [a for a in listdir(filepath) if a.endswith(".parquet")]:
        sql = pd.read_parquet(item)
        sql.to_sql(item, eng, index = True, index_col = "id")
