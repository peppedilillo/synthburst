import urllib.request
from pathlib import Path
from typing import Dict, List

import requests
import numpy as np
from astropy.io import fits

import paths
from database import triggered_detectors
from errors import TTEDownloadError


_DETECTOR_MAP = {
    "NAI_00": "0",
    "NAI_01": "1",
    "NAI_02": "2",
    "NAI_03": "3",
    "NAI_04": "4",
    "NAI_05": "5",
    "NAI_06": "6",
    "NAI_07": "7",
    "NAI_08": "8",
    "NAI_09": "9",
    "NAI_10": "a",
    "NAI_11": "b",
}


def _url(grb_id: str) -> str:
    year = grb_id[0:2]
    url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/20{year}/bn{grb_id}/current/"
    return url


def _download_ttes(
    grb_id: str,
    grb_td: List[str],
    folderpath: Path = paths.ttes(),
) -> List[Path]:
    """
    Private method. This method download a single GRB given its name, triggered detectors, detectors mapping and path to
    save those. There are performed https requests to query the last version of the TTE and then download it if not
    already present in PATH_TO_SAVE.
    :param grb_id: id NUMBER of the GRB. E.g. "080714086".
    :param grb_td: list of triggered detector. E.g. "NAI_01, NAI_11".
    :param folderpath: path where to save TTE files.
    :returns: list of filepaths
    """
    filepaths = []
    for td in _DETECTOR_MAP.keys():
        if td not in grb_td:
            continue
        # Get the id of the detectors
        n_td = _DETECTOR_MAP[td]
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
        filepaths.append(filepath)
    return filepaths


def download_ttes(grb_id: str, folderpath: Path = paths.ttes()) -> List[Path]:
    """
    Download a single GRB given its id name.
    :param grb_id: id of the GRB, e.g. "080714086".
    :param folderpath: Path to save the TTE file.
    :returns: list of filepaths
    """
    grb_td = triggered_detectors(grb_id)
    return _download_ttes(grb_id, grb_td, folderpath=folderpath)


def fetch_datafiles(
    grb_id: str,
) -> List[Path]:
    """
    Returns all stored datafiles relative to a grb. Download if not available.
    :param grb_id: id of the GRB, e.g. "080714086".
    :return: a list of paths
    """
    detectors = triggered_detectors(grb_id)
    cached = [f for f in list(paths.ttes().iterdir()) if grb_id in f.name]
    _INV_DETECTOR_MAP = {v: k for k, v in _DETECTOR_MAP.items()}
    cached_detectors = [_INV_DETECTOR_MAP[f.name[9:10]] for f in cached]
    missing_detectors = set(detectors) - set(cached_detectors)
    downloaded = _download_ttes(grb_id, missing_detectors)
    return cached + downloaded


def get_events(grb_id: str) -> np.ndarray:
    """
    A function returning all
    :param grb_id: id of the GRB, e.g. "080714086".
    :return: an array with columns corresponding to time, low energy bin edge,
             hi energy bin edge.
    """
    filepaths = fetch_datafiles(grb_id)
    times = np.array([])
    lo_energy = np.array([])
    hi_energy = np.array([])
    for file in filepaths:
        hdul = fits.open(file)
        _times = hdul[2].data["TIME"]
        channels = hdul[2].data["PHA"]
        _lo_energy = hdul[1].data["E_MIN"][channels]
        _hi_energy = hdul[1].data["E_MAX"][channels]
        ii = np.searchsorted(times, _times)
        times = np.insert(times, ii, _times)
        lo_energy = np.insert(lo_energy, ii, _lo_energy)
        hi_energy = np.insert(hi_energy, ii, _hi_energy)
    output = np.dstack(
        (times, lo_energy, hi_energy),
    )[0]
    return output


if __name__ == "__main__":
    print("Downloading data if missing.")
    print([f.name for f in download_ttes("171030729")])
    print("GRB has been stored here.")
    print([f.name for f in fetch_datafiles("171030729")])
    print("These are the first 5 events:")
    print(get_events("171030729")[:5])