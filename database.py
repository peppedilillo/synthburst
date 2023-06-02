import sqlite3

import pandas as pd

import paths


def get_db(dbpath=paths.database):
    """
    Return the table of Burst in the db_path selected.
    :param dbpath: str, DB path in which is stored the SQLite DB with events information.
    :return: pd.DataFrame, the table of GRBs information.
    """
    # Connect to the SQLite database file
    conn = sqlite3.connect(dbpath)
    name_tab = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).values[0, 0]
    # Query to fetch the table data
    query = f"SELECT * FROM {name_tab}"
    # Read the table into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    return df
