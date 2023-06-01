import os
import pandas as pd
import sqlite3
import urllib.request
import requests


def map_det():
    """
    Define a map between detector string in DB and [0, 1, ..., a, b].
    :return: dict of mapping
    """
    # Define the NAI detector id number map
    dct_map_td = {"NAI_" + i: i[1] for i in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]}
    dct_map_td["NAI_10"] = "a"
    dct_map_td["NAI_11"] = "b"
    return dct_map_td


def set_path(PATH_TO_SAVE='TTE'):
    """
    Create a folder to save the TTE files
    :param PATH_TO_SAVE: str, path to save the downloaded files. By default, goes into "TTE" inside the GitHub project.
    :return: None
    """
    # Define Path to save
    isExist = os.path.exists(PATH_TO_SAVE)
    if not isExist:
       os.makedirs(PATH_TO_SAVE)


def get_db(db_path='GBMdatabase.db'):
    """
    Return the table of Burst in the db_path selected.
    :param db_path: str, DB path in which is stored the SQLite DB with events information.
    :return: pd.DataFrame, the table of GRBs information.
    """
    # Connect to the SQLite database file
    conn = sqlite3.connect(db_path)
    name_tab = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).values[0, 0]
    # Query to fetch the table data
    query = f"SELECT * FROM {name_tab}"
    # Read the table into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    return df


def download_all_grb(PATH_TO_SAVE="TTE", db_path='GBMdatabase.db'):
    """
    Download all the GRBs listed in the DB in db_path and save those in the PATH_TO_SAVE folder.
    :param PATH_TO_SAVE: str, path to save the TTE files.
    :param db_path: str, path to find the DB Burst info.
    :return: None

    Example of run: download_all_grb
    """
    # Set path, DB and det variable
    set_path(PATH_TO_SAVE)
    df = get_db(db_path)
    dct_map_td = map_det()
    # For loop for download TTE events
    for idx, row in df.iterrows():
        grb_id = row['id']
        grb_td = row['trig_det']
        _download_grb(grb_id, grb_td, dct_map_td, PATH_TO_SAVE)


def _download_grb(grb_id, grb_td, dct_map_td, PATH_TO_SAVE="TTE"):
    """
    Private method. This method download a single GRB given its name, triggered detectors, detectors mapping and path to
    save those. There are performed https requests to query the last version of the TTE and then download it if not
    already present in PATH_TO_SAVE.
    :param grb_id: str, id NUMBER of the GRB. E.g. "080714086".
    :param grb_td: str, list of triggered detector. E.g. "NAI_01, NAI_11".
    :param dct_map_td: dict, dictionary of mapping detectors name. E.g. NAI_00 -> 0.
    :param PATH_TO_SAVE: str, path where to save TTE files.
    :return: None
    """
    # Get th year of the GRB
    year = grb_id[0:2]

    # For loop for the triggered detectors
    for td in dct_map_td.keys():
        if td in grb_td:
            # Get the id of the detectors
            n_td = dct_map_td[td]
            # Build the HTTPS folder link and get the last version of the TTE file
            str_http_folder = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/20{year}/bn{grb_id}/current/"
            try:
                response = (requests.get(str_http_folder))
                if response.status_code == 404:
                    raise
                else:
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


def download_grb(grb_id, PATH_TO_SAVE="TTE"):
    """
    Download a single GRB given its id name.
    :param grb_id: id of the GRB. Note: specify the "bn" at the beginning. E.g. bn080714086.
    :param PATH_TO_SAVE: Path to save the TTE file.
    :return: None

    Example of run: download_grb("bn080714086")
    """
    df = get_db()
    dct_map_td = map_det()
    grb_td = df.loc[df['id'] == grb_id[2:], 'trig_det'].values[0]
    _download_grb(grb_id[2:], grb_td, dct_map_td, PATH_TO_SAVE=PATH_TO_SAVE)
