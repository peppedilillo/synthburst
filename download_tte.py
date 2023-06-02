import os
import urllib.request
from pathlib import Path
from typing import Dict, List

import requests

import paths
from database import get_db
from errors import TTEDownloadError


def map_det():
    """
    Define a map between detector string in DB and [0, 1, ..., a, b].
    :return: dict of mapping
    """
    # Define the NAI detector id number map
    dct_map_td = {
        "NAI_" + i: i[1]
        for i in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]
    }
    dct_map_td["NAI_10"] = "a"
    dct_map_td["NAI_11"] = "b"
    return dct_map_td


def _url(grb_id: str) -> str:
    year = grb_id[0:2]
    url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/20{year}/bn{grb_id}/current/"
    return url


def _download_grb(
    grb_id: str,
    grb_td: List[str],
    dct_map_td: Dict[str, str],
    folderpath: Path = paths.ttes(),
) -> None:
    """
    Private method. This method download a single GRB given its name, triggered detectors, detectors mapping and path to
    save those. There are performed https requests to query the last version of the TTE and then download it if not
    already present in PATH_TO_SAVE.
    :param grb_id: id NUMBER of the GRB. E.g. "080714086".
    :param grb_td: list of triggered detector. E.g. "NAI_01, NAI_11".
    :param dct_map_td: dictionary of mapping detectors name. E.g. NAI_00 -> 0.
    :param folderpath: path where to save TTE files.
    :returns: None
    """
    for td in dct_map_td.keys():
        if td not in grb_td:
            continue
        # Get the id of the detectors
        n_td = dct_map_td[td]
        # Build the HTTPS folder link and get the last version of the TTE file
        str_http_folder = _url(grb_id)
        response = requests.get(str_http_folder)
        if response.status_code == 404:
            raise TTEDownloadError()
        response_txt = (requests.get(str_http_folder)).text
        idx_txt_version = response_txt.find(f"glg_tte_n{n_td}_bn{grb_id}_v")
        tte_version = response_txt[idx_txt_version + 24 : idx_txt_version + 26]
        # Define the TTE file name founded and the complete HTTPS link path
        str_tte_file = f"glg_tte_n{n_td}_bn{grb_id}_v{tte_version}.fit"
        str_ftp_http = str_http_folder + str_tte_file
        # If the file already exists skip the file
        filepath = Path(folderpath).joinpath(str_tte_file)
        if filepath.is_file():
            continue
        print("Downloading: ", str_ftp_http)
        urllib.request.urlretrieve(str_ftp_http, filepath)


def download_all_grb(
    folderpath: Path = paths.ttes(), dbpath: Path = paths.database
) -> None:
    """
    Download all the GRBs listed in the DB in db_path and save those in the PATH_TO_SAVE folder.
    :param folderpath: str, path to save the TTE files.
    :param dbpath: str, path to find the DB Burst info.
    :return: None

    Example of run: download_all_grb
    """
    df = get_db(dbpath)
    dct_map_td = map_det()
    # For loop for download TTE events
    for idx, row in df.iterrows():
        grb_id = row["id"]
        grb_td = row["trig_det"]
        _download_grb(grb_id, grb_td, dct_map_td, folderpath)


def download_grb(grb_id: str, folderpath: Path = paths.ttes()) -> None:
    """
    Download a single GRB given its id name.
    :param grb_id: id of the GRB. Note: specify the "bn" at the beginning. E.g. bn080714086.
    :param folderpath: Path to save the TTE file.
    :return: None

    Example of run: download_grb("bn080714086")
    """
    df = get_db()
    dct_map_td = map_det()
    grb_td = df.loc[df["id"] == grb_id[2:], "trig_det"].values[0]
    _download_grb(grb_id[2:], grb_td, dct_map_td, folderpath=folderpath)


if __name__ == "__main__":
    download_grb("bn080714086")
