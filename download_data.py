#Download Script for NumerAI
from numerapi import NumerAPI
#Only import create_engine, pandas will handle everything else
from sqlalchemy import create_engine 
import pandas as pd 

napi = NumerAPI()
#Dataset already installed on local. Edit this out during commit 
#napi.download_dataset("v5.0/train.parquet")
engine = create_engine("mysql+pymysql://user:password@host:port/dbname")
sql = pd.read_parquet("../v5.0/train.parquet")
#"data" corresponds to table_name
sql.to_sql("data", engine,index = True, index_col = "id")
