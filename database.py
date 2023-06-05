import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

import paths


_memory = {}


def get_db(dbpath: Path=paths.database):
    """
    Return the table of Burst in the db_path selected.
    :param dbpath: str, DB path in which is stored the SQLite DB with events information.
    :return: pd.DataFrame, the table of GRBs information.
    """
    if "database" in _memory:
        return _memory["database"]
    conn = sqlite3.connect(dbpath)
    name_tab = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).values[0, 0]
    # Query to fetch the table data
    query = f"SELECT * FROM {name_tab}"
    # Read the table into a pandas DataFrame
    df = pd.read_sql_query(query, conn)
    _memory["database"] = df
    return df


def triggered_detectors(
    grb_id: str,
) -> List[str]:
    df = get_db()
    grb_td_string = df.loc[df["id"] == grb_id, "trig_det"].values[0]
    # TODO: reformat db "trig_det" column to avoid this parsing step
    grb_td = grb_td_string.replace(" ", "").replace("\"", "")[1:-1].split(",")
    return grb_td


if __name__ == "__main__":
    print(triggered_detectors("120707800"))
