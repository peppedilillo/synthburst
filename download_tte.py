
import pandas as pd
from sqlalchemy import  create_engine

import sqlite3

# Connect to the SQLite database file
conn = sqlite3.connect('C:\\profili\\u423831\\Downloads\\GBMdatabase.db')


name_tab = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).values[0, 0]


# Query to fetch the table data
query = f"SELECT * FROM {name_tab}"

# Read the table into a pandas DataFrame
df = pd.read_sql_query(query, conn)

print(df.loc[0, 'trig_det'])

import urllib.request
urllib.request.urlretrieve("https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/2018/bn180113011/current/glg_tte_b0_bn180113011_v00.fit", 'glg_tte_b0_bn180113011_v00.fit')

