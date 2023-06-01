import os
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import urllib.request
import requests

# Define Path to save
PATH_TO_SAVE = "TTE"
isExist = os.path.exists(PATH_TO_SAVE)
if not isExist:
   os.makedirs(PATH_TO_SAVE)

# Connect to the SQLite database file
conn = sqlite3.connect('GBMdatabase.db')
name_tab = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).values[0, 0]

# Query to fetch the table data
query = f"SELECT * FROM {name_tab}"

# Read the table into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Define the NAI detector id number map
dct_map_td = {"NAI_" + i: i[1] for i in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]}
dct_map_td["NAI_10"] = "a"
dct_map_td["NAI_11"] = "b"

# For loop for download TTE events
for idx, row in df.iterrows():
    grb_id = row['id']
    year = grb_id[0:2]
    grb_td = row['trig_det']
    # For loop for the triggered detectors
    for td in dct_map_td.keys():
        if td in grb_td:
            # Get the id of the detectors
            n_td = dct_map_td[td]
            # Build the HTTPS folder link and get the last version of the TTE file
            str_http_folder = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/20{year}/bn{grb_id}/current/"
            try:
                response_txt = (requests.get(str_http_folder)).text
            except:
                print(f"can t get response on {str_http_folder}")
                continue
            idx_txt_version = response_txt.find(f"glg_tte_n{n_td}_bn{grb_id}_v")
            tte_version = response_txt[idx_txt_version+24:idx_txt_version+26]
            # Define the TTE file name founded and the complete HTTPS link path
            str_tte_file = f"glg_tte_n{n_td}_bn{grb_id}_v{tte_version}.fit"
            str_ftp_http = str_http_folder + str_tte_file
            # If the file already exists skip the file
            isExist = os.path.exists(PATH_TO_SAVE + "/" + str_tte_file)
            if isExist:
                continue
            try:
                print("Downloading: ", str_ftp_http)
                urllib.request.urlretrieve(str_ftp_http, PATH_TO_SAVE + "/" + str_tte_file)
            except:
                print(f"can t download event bn{grb_id}")
