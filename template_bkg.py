import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from database import get_bkg_day
from utils import DETECTOR_MAP_VALUES, ENRANGE_VALUES
import logging


def bkg_template_split(min_split: float = 96.5) -> Dict:
    """
    Take a background day estimate and produce a template dictionary (4 dimensions: orbit, detector, energy range,
     values).
    :param min_split: float, Minutes of the background split. By default, it is chosen the Fermi orbit (96.5 minutes).
    :return: Dict, a template which the first dimension is the number of templates,by default are 15 orbits. The next
    dimension is the detector (e.g. n7). Then the range (e.g. r1). Finally, the last dimension can be 'met' that is
    referring the time and 'counts' that is the count rates estimated (e.g. count rates for detector n7,
     in range r1 at the particular met time).
    """
    bkg_pred = get_bkg_day()
    orbit_duration = min_split * 60
    templates = {}
    for norbit, tstart in enumerate(np.arange(0, bkg_pred.time.max() + orbit_duration, orbit_duration)):
        templates[norbit] = {}
        for ndet in DETECTOR_MAP_VALUES:
            templates[norbit][ndet] = {}
            for enrange in ENRANGE_VALUES:
                df_template_tmp = pd.DataFrame()
                mask = (bkg_pred.time >= tstart) & (bkg_pred.time < tstart + orbit_duration)
                df_template_tmp['counts'] = bkg_pred.loc[mask, ndet + '_' + enrange].copy()
                df_template_tmp['time'] = bkg_pred.loc[mask, 'time'] - bkg_pred.loc[mask,'time'].min()
                templates[norbit][ndet][enrange] = df_template_tmp
    return templates


def sample_count(counts: int,  type_random: str = 'poisson', en_range=None, scale_mean_rate=1) -> int:
    """
    Sample a count from the random variable chosen.
    :param counts: int, number of counts.
    :param type_random: str, type of random variable. 'poisson' or 'normal'.
    :param en_range: str, the energy range of the count rates. For poisson is ignored.
    :param scale_mean_rate: Amplify the count rates by scale.
    :return: the count rate sampled.
    """
    if type_random == 'poisson':
        return np.random.poisson(counts*scale_mean_rate)
    elif type_random == 'normal':
        # Error or the prediction for each energy range. The scale factor it to convert MAD to STD.
        if en_range == 'r2':
            std = 2/(2/3)
        elif en_range == 'r1':
            std = 6/(2/3)
        elif en_range == 'r0':
            std = 5/(2/3)
        else:
            std = 5/(2/3)
        return max(int(np.random.normal(counts*scale_mean_rate, std)), 0)
    else:
        logging.error("Type of random variable not specified in sample_count.")
        raise


def sample_energy(counts: int, en_range: str, type_random: str = 'uniform') -> List:
    """
    Sample the energy range for each N photons. N = counts.
    :param counts: int, Number of counts to sample the energy.
    :param en_range: str, type of energy range of the photons.
    :param type_random: str, type of random variable. 'uniform' or 'custom'.
    :return: list of energy sampled.
    """
    if type_random == 'uniform':
        if en_range == 'r0':
            energy_tte = np.random.randint(28, 50, counts)
        elif en_range == 'r1':
            energy_tte = np.random.randint(50, 300, counts)
        elif en_range == 'r2':
            energy_tte = np.random.randint(300, 500, counts)
        else:
            logging.error("Energy range not specified in sample_energy.")
            raise
    elif type_random == 'custom':
        # The probability is supposed to be P(E) = 1/E^k. Then scaled to sum up to one.
        exp_prob = 1 / np.arange(28, 500)**1.6 / (sum(1 / np.arange(28, 500)))
        if en_range == 'r0':
            exp_prob = exp_prob[0:50-28]/(sum(exp_prob[0:50-28]))
            energy_tte = np.random.choice(range(28, 50), size=counts, p=exp_prob)
        elif en_range == 'r1':
            exp_prob = exp_prob[50-28:300-28]/(sum(exp_prob[50-28:300-28]))
            energy_tte = np.random.choice(range(50, 300), size=counts, p=exp_prob)
        elif en_range == 'r2':
            exp_prob = exp_prob[300-28:500-28]/(sum(exp_prob[300-28:500-28]))
            energy_tte = np.random.choice(range(300, 500), size=counts, p=exp_prob)
        else:
            logging.error("Energy range not specified in sample_energy.")
            raise
    else:
        logging.error("Type of random variable not specified in sample_energy.")
        raise
    return energy_tte.tolist()


def template2tte(template: Dict, bin_time: float = 4.096) -> List:
    """
    Get a template with energy range r0, r1 and r0 and convert it to a TTE list.
    :param template: dict, the background template structure (orbit, detector, energy range, values).
    :param bin_time: float, binned time of the background.
    :return: list of tuples. A tuple is composed by the time and the energy of the photon simulated.
    """
    list_tte = []
    # Iterate per energy range
    for rng in ['r0', 'r1', 'r2']:
        for idx, row in template[rng].iterrows():
            # From the estimated count draw a poisson random variable
            counts = sample_count(row['counts'], type_random='normal', en_range=rng)
            # Generate the time arrival of the N photons. N = counts. The time arrival must be in the interval.
            time_tte = np.random.uniform(row['time'], row['time'] + bin_time, counts).tolist()
            # According to the energy range, sample a specific energy for the photons.
            energy_tte = sample_energy(counts, rng, type_random='custom')
            # Add the events list
            list_tte = list_tte + list(zip(time_tte, energy_tte))
    # Use the pandas Structure and sort the time event by the time arrival
    list_tte = pd.DataFrame(list_tte, columns=['time', 'energy'])
    list_tte = list_tte.sort_values('time', ascending=True)
    return list(zip(list_tte['time'], list_tte['energy']))


if __name__ == "__main__":
    # Create the templates
    templates = bkg_template_split()
    # Select the 10th orbit and the detector n7
    list_tte = template2tte(templates[10]['n7'])
    print(list_tte[0:20])
    plt.hist([i[1] for i in list_tte], bins=64, density=True)
    plt.loglog()
    plt.title("Spectral background density simulated")
    plt.figure()
    plt.hist((np.array([i[0] for i in list_tte])[1:] - np.array([i[0] for i in list_tte])[0:-1]), log=True, bins=128)
    plt.title("Distribution time arrival")
    plt.show()
