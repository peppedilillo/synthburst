from pathlib import Path
from typing import Dict, List

import pandas as pd

import paths
from utils import DETECTOR_MAP_KEYS

_memory = {}


def get_db(dbpath: Path = paths.database):
    """
    Return the table of Burst in the db_path selected.
    :param dbpath: str, DB path in which is stored the SQLite DB with events information.
    :return: pd.DataFrame, the table of GRBs information.
    """
    if "database" in _memory:
        return _memory["database"]
    df = pd.read_csv(dbpath, dtype={"bcat_detector_mask": object})
    _memory["database"] = df
    return df


def get_metadata(grb_id: str) -> Dict:
    """
    Return burst catalog entry relative to grb
    :param grb_id: id NUMBER of the GRB. E.g. "080714086".
    :return: a dictionary of metadata
    """
    df = get_db()
    metadata = df.loc[df["name"] == "GRB" + grb_id].to_dict(orient="records")[0]
    return metadata


def triggered_detectors(grb_id: str) -> List[str]:
    """
    Returns GBM detectors triggered by a grb.
    :param grb_id: id NUMBER of the GRB. E.g. "080714086".
    :return: a list of string
    """
    df = get_db()
    bitmask = df.loc[df["name"] == "GRB" + grb_id, "bcat_detector_mask"].values[0]
    nai_bitmask = bitmask[: len(DETECTOR_MAP_KEYS)]
    output = [DETECTOR_MAP_KEYS[i] for i, bit in enumerate(nai_bitmask) if int(bit)]
    return output


if __name__ == "__main__":
    print("Triggered detectors:")
    print(triggered_detectors("120707800"))
    print("Metadata from burst catalog:")
    print(get_metadata("120707800"))
