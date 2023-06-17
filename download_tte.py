import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import requests
from astropy.io import fits

import paths
from database import triggered_detectors
from errors import TTEDownloadError
from utils import DETECTOR_MAP, DETECTOR_MAP_INVERTED


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
    Private method. This method download a single GRB given its name,
    triggered detectors, detectors mapping and path to save those.
    There are performed https requests to query the last version of the
    TTE and then download it if notalready present in PATH_TO_SAVE.
    :param grb_id: id NUMBER of the GRB. E.g. "080714086".
    :param grb_td: list of triggered detector. E.g. "NAI_01, NAI_11".
    :param folderpath: path where to save TTE files.
    :returns: list of filepaths to files downloaded.
    """
    # TODO: we shouldn't be waiting if data are cached
    # possible solution:
    # list_tte_dl = os.listdir(Path(folderpath))
    # ...
    # for ...
    #     if len(([1 for i in list_tte_dl if f"glg_tte_n{n_td}_bn{grb_id}_v" in i]) >= 1:
    #         continue
    filepaths = []
    for td in DETECTOR_MAP.keys():
        if td not in grb_td:
            continue
        # Get the id of the detectors
        n_td = DETECTOR_MAP[td]
        # Build the HTTPS folder link and get the last version of the TTE file
        str_http_folder = _url(grb_id)
        response = requests.get(str_http_folder)
        if response.status_code == 404:
            raise TTEDownloadError()
        response_txt = (requests.get(str_http_folder)).text
        # works when version is greater than 0
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
    Returns all stored datafiles relative to a grb.
    Downloads if not cached already.
    :param grb_id: id of the GRB, e.g. "080714086".
    :return: a list of paths
    """
    detectors = triggered_detectors(grb_id)
    cached = [f for f in list(paths.ttes().iterdir()) if grb_id in f.name]
    cached_detectors = [DETECTOR_MAP_INVERTED[f.name[9:10]] for f in cached]
    missing_detectors = set(detectors) - set(cached_detectors)
    downloaded = _download_ttes(grb_id, missing_detectors)
    return cached + downloaded


def get_events(grb_id: str) -> np.ndarray:
    """
    A function returning all grb's tte events from its triggered detectors,
    meshed (sorted by time) together.
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
        channels = hdul[2].data["PHA"]
        _times = hdul[2].data["TIME"]
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
    print([f.name for f in download_ttes("150118409")])
    print("GRB stored here.")
    print([f.name for f in fetch_datafiles("150118409")])
    print("These are the first 5 events from triggered detectors:")
    print(get_events("171030729")[:5])
