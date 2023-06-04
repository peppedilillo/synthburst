import sqlite3
from pathlib import Path

import pandas as pd

import paths


def get_db(dbpath: Path=paths.database):
    """
    Return the table of Burst in the db_path selected.
    :param dbpath: str, DB path in which is stored the SQLite DB with events information.
    :return: pd.DataFrame, the table of GRBs information.
    """
    # Connect to the SQLite database file
    conn = sqlite3.connect(dbpath)
    name_tab = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).values[0, 0]
    # Query to fetch the table data
    query = f"SELECT * FROM {name_tab}"
    # Read the table into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    return df


def get_bkg_day(day_bkg: Path=paths.day_bkg):
    """
    Return the table background prediction in the day_bkg selected.
    :param dbpath: str, bakcground day path in which is stored the prediction of the background count rates
     for each of the 12 detectors (n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, na, nb) in three ranges
      28-50 (r0), 50-300 (r1), 300-500 (r2). Met time in float and datetime in strings are provided.
    :return: pd.DataFrame, the table of background count rates estimated.
    """
    bkg_pred = pd.read_csv(day_bkg)
    bkg_pred['met_0'] = bkg_pred.met - bkg_pred.met.min()
    return bkg_pred
