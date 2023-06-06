""""
light_curves.py
A module for extracting relevant information from GBM data files and, starting from these,
generating synthetic and 'customizable' light-curves.
""" ""

from bisect import insort
from math import ceil, floor, log10
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize

from legacy.gbm_interface import _LightCurveGBM
from legacy.mcarlo import generate_by_invsampl
from legacy.models import (Band, _background_S, _background_X, _instrument_S,
                           _instrument_X)
from legacy.shared import _LightCurve, _pdf, convert_to_fits


def estimate_burst_activity_dur(
    burst_pdf, fraction=0.1
):  # TODO: is this safe with uneven binlength?
    """
    :param burst_pdf:
    :param fraction: like 0.1, 0.5
    :return:
    """
    widths = np.array(burst_pdf.bins[1:]) - np.array(burst_pdf.bins[:-1])
    counts = np.array(burst_pdf.counts)
    a = np.column_stack((widths, counts))
    a = a[a[:, -1].argsort()]  # sort pdf by counts value (last column)
    a[:, 1] = np.cumsum(a[:, -1])  # cumulative sum last column
    delta_t = sum(a[a[:, -1] > fraction][:, 0])  # subtract
    return delta_t


def estimate_burst_counts(burst_pdf, factor):
    """
    eurysthic. it attempt to get the burst most
    active section as bright as background*factor**-1
    :return:
    """
    model_pdf_counts = np.array(burst_pdf.counts)
    model_pdf_bins = np.array(burst_pdf.bins)
    num = (
        sum(
            (model_pdf_counts / max(model_pdf_counts))
            * (model_pdf_bins[1:] - model_pdf_bins[:-1])
            * (_background_X.rate + _background_S.rate)
        )
        / factor
    )
    return round(num)


def ttes(l):
    return list(zip(*l))[0]


def chns(l):
    return list(zip(*l))[1]


def flags(l):
    return list(zip(*l))[2]


def energies(l):
    return [
        (_instrument_X.en_bins[c], _instrument_X.en_bins[c + 1])
        if f == "X"
        else (_instrument_S.en_bins[c], _instrument_S.en_bins[c + 1])
        for t, c, f in l
    ]


def _make_events_list(time_tags, energies):
    """
    :param time_tags: list of tte
    :param energies: dic with keys 'X' and 'S' and values equal to channel numbers
    :return: list of tuples like (tte, chn, instrument_flag)
    """
    tupled_en_list = [
        (value, key) for key in energies.keys() for value in energies[key]
    ]
    shuffle(tupled_en_list)
    events = [(time_tags[i], *tupled_en_list[i]) for i in range(len(time_tags))]
    return events


class _LightCurveSim(_LightCurve):
    """
    Master simulation class.
    """

    def __init__(self, *, duration=300, burst_start_time=20, burst_spectrum=Band()):
        """
        :param duration: the simulation time duration ([s]). default value = 129 s.
        :param burst_start_time: time at which the transient should be placed ([s]). default value = 20 s.
        :param burst_spectrum: a band object (see models.py)
        """
        self.binning = 0.01
        self.duration = duration
        self.burst_start_time = burst_start_time
        self.burst_spectrum = burst_spectrum
        self.bkg_count = 0
        self.burst_count = 0
        self.data = []

    def reset(self):
        """
        reset object counts
        """
        self.data = []
        self.bkg_count = 0
        self.burst_count = 0

    def add_bkg(self, counts="hermes_default", is_flux=False, bkg_model=None):
        """
        generate background events
        :param counts: int, number of events or, if 'is_flux == True'
                        (number of events)/(event duration)
        :param is_flux: bool, the routine will generate a number of events
                        equal to (counts)/(event duration)
        :param bkg_model: model for background time profile
        :return:
        """
        if counts == "hermes_default":
            counts = (_background_X.rate + _background_S.rate) * self.duration
        elif isinstance(counts, int) and is_flux:
            counts *= self.duration
        counts = round(counts)

        timetags = self._simulate_bkg_tte(counts, bkg_model)
        energies = self._sim_bkg_energies(counts)
        events = _make_events_list(timetags, energies)

        if not self.data:
            self.data = sorted(events, key=lambda x: x[0])
            self.bkg_count += len(events)
        else:
            for event in events:
                insort(self.data, event)
                self.bkg_count += 1

    def add_burst(self, counts):
        """
        :param counts: numer of burst counts to simulate
        :return:
        """
        time_tags = [
            time_tags + self.burst_start_time
            for time_tags in self._simulate_burst_tte(counts)
        ]
        energies = self._sim_burst_energies(counts, self.burst_spectrum)
        events = _make_events_list(time_tags, energies)
        for event in events:
            insort(self.data, event)
            self.burst_count += 1

    # methods for simulating ttes.

    def _simulate_burst_tte(self, counts):
        time_tags, _ = generate_by_invsampl(self.burst_pdf, counts)
        return time_tags

    def _simulate_bkg_tte(self, counts, bkg_model=None):
        if bkg_model:
            binning = self.duration / 1000
            bins = np.arange(0, self.duration + binning, binning)
            values = bkg_model(bins[:-1])
            time_tags, _ = generate_by_invsampl(
                _pdf(values / sum(values), bins), counts
            )
        else:
            time_tags = self.duration * np.random.rand(counts)
        return time_tags

    @staticmethod
    def _sim_bkg_energies(N):
        """
        generate energy lists for N background photons
        according to spectra of band object.
        :param N: number of photons
        :return: dictionary with different keys for each instruments
        """

        def compute_photons_num_per_instrument():
            N_X = ceil(
                N * _background_X.rate / (_background_X.rate + _background_S.rate)
            )
            N_S = floor(
                N * _background_S.rate / (_background_X.rate + _background_S.rate)
            )
            return N_X, N_S

        out_dic = {}
        N_X, N_S = compute_photons_num_per_instrument()
        for instrument, background, num, key in list(
            zip(
                *[
                    [_instrument_X, _instrument_S],
                    [_background_X, _background_S],
                    [N_X, N_S],
                    ["X", "S"],
                ]
            )
        ):
            _, channel_nums = generate_by_invsampl(
                _pdf(background.counts / sum(background.counts), instrument.en_bins),
                num,
            )
            out_dic.setdefault(key, channel_nums)
        return out_dic

    @staticmethod
    def _sim_burst_energies(N, band):
        """
        generate energy lists for N source photons
        according to spectra of band object
        :param band: band object
        :param N: number of photons
        :return: dictionary with different keys for each instruments
        """

        def convolvify(en_bins, arf, rmf_mat):
            """
            convolve band function to instrument response
            :param en_bins:
            :param arf: arf function
            :param rmf:
            :return:
            """

            # compute middle points of energy bins
            mid_points = [
                (en_bins[i] + en_bins[i - 1]) / 2 for i in range(1, len(en_bins))
            ]
            # compute band*eff_area at middle points
            arfdotband = [arf(p) * band(p) for p in mid_points]

            counts = []
            for i in range(len(mid_points)):
                # compute counts multiplying rmf columns by arfdotband
                counts.append(sum(rmf_mat[:, i] * np.array(arfdotband)))
            # return as normalized array
            return np.array(counts) / sum(counts), mid_points

        def compute_photons_num_per_instrument():
            C = [0, 0]
            for i, instrument in enumerate([_instrument_X, _instrument_S]):
                mid_points = [
                    (instrument.en_bins[i] + instrument.en_bins[i - 1]) / 2
                    for i in range(1, len(instrument.en_bins))
                ]
                binwidths = [
                    (instrument.en_bins[i] - instrument.en_bins[i - 1])
                    for i in range(1, len(instrument.en_bins))
                ]
                arfdotband = [
                    instrument.arf(p) * band(p) * bw
                    for p, bw in list(zip(mid_points, binwidths))
                ]
                C[i] = sum(arfdotband)
            return ceil(N * C[0] / (C[0] + C[1])), floor(N * C[1] / (C[0] + C[1]))

        out_dic = {}
        N_X, N_S = compute_photons_num_per_instrument()
        for instrument, num, key in list(
            zip(*[[_instrument_X, _instrument_S], [N_X, N_S], ["X", "S"]])
        ):
            counts_arr, mid_points = convolvify(
                instrument.en_bins, instrument.arf, instrument.rmf_mat
            )
            _, channel_nums = generate_by_invsampl(
                _pdf(counts_arr, instrument.en_bins), num
            )
            out_dic.setdefault(key, channel_nums)
        return out_dic

    def get_data(self, energy=None, time=None, flag=None):
        out = self.data
        if energy:
            low, hi = energy
            condition = (
                lambda d: low <= _instrument_S.en_bins[d[1]] < hi
                if d[2] == "S"
                else low <= _instrument_X.en_bins[d[1]] < hi
            )
            out = list(filter(condition, out))
        if time:
            first, last = time
            condition = lambda d: first <= d[0] < last
            out = list(filter(condition, out))
        if flag:
            condition = lambda d: d[2] == flag
            out = list(filter(condition, out))
        return out

    def get_burst_times(self):
        try:
            start = (
                self.burst_start_time
                + floor(self.burst_pdf.bins[0] / self.binning) * self.binning
            )
            end = (
                self.burst_start_time
                + ceil(self.burst_pdf.bins[-1] / self.binning) * self.binning
            )
            return start, end
        except AttributeError:
            raise AttributeError("please call simulate() a burst first!")

    def plot(self, figsize=None, xlims=None, ylims=None, binning=1.0, **kwargs):
        """
        :param figsize:
        :param xlims:
        :param binning:
        :param kwargs: routed to get_data. use example: x.plot(**{'energy' : (50,300)})
        :return:
        """
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

        data = ttes(self.get_data(**kwargs))
        hist_counts, hist_bin, _ = plt.hist(
            data,
            bins=np.arange(data[0], data[-1] + binning, binning),
            color="#607c8e",
            histtype="step",
        )
        plt.xlabel("Times [s]")
        if xlims:
            plt.xlim(*xlims)
        if ylims:
            plt.ylim(*ylims)
        plt.ylabel("Counts/{:.2f} s bin".format(binning))

        return fig

    def plot_spectra(self, binnum=64, figsize=None, xlims=None):
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

        bins = np.logspace(
            log10(_instrument_X.en_bins[0]), log10(_instrument_S.en_bins[-1]), binnum
        )
        mid_energies = [sum(e) / 2 for e in energies(self.data)]
        counts, *_ = np.histogram(mid_energies, bins=bins)

        plt.bar(
            x=bins[:-1],
            height=counts,
            width=bins[1:] - bins[:-1],
            facecolor="white",
            edgecolor="orange",
        )
        plt.loglog()
        plt.ylabel("N$_E$")
        plt.xlabel("E")
        if xlims:
            plt.xlim(*xlims)
        return fig

    def to_dataframe(self, **kwargs):
        """
        :param kwargs: routed to get_data. use example: x.plot(**{'energy' : (50,300)})
        :return: dataframe
        """
        data = self.get_data(**kwargs)
        out = pd.DataFrame({"tte": ttes(data), "pha": chns(data), "flag": flags(data)})
        return out

    def export(self, filename, **kwargs):
        extension = filename[-3:]
        if extension == "csv":
            self.to_dataframe(**kwargs).to_csv(filename, index=False)
            print("exported to {}.".format(filename))
        elif extension == "pickle":
            self.to_dataframe(**kwargs).to_pickle(filename)
            print("exported to {}.".format(filename))
        elif extension == "fits":
            hdul = convert_to_fits(self.to_dataframe(**kwargs))
            hdul.writeto(filename)
            print("exported to {}.".format(filename))
        else:
            raise ValueError("i don't know this export mode")


class LCSFromModel(_LightCurveSim):
    def simulate(self, id, *, bkg_scenario=None, orbit_fraction=0, duration_adjust=0.0):
        """
        :param id:              str, gbm model id
        :param bkg_scenario:    optional str, 'worst' or 'gentle'. default is None
        :param orbit_fraction:  float between -1 and 1.
                                -1 for SAA exit, +1 for SAA entering. 0 for mid-orbit.
                                default is 0.
        :param duration_adjust: change model start and end points.
        :return:
        """
        self.model = _LightCurveGBM(id)
        self.bkg_scenario = bkg_scenario
        self.bkg_orbit_fraction = orbit_fraction
        if type(duration_adjust) != tuple:
            self.duration_adjust = (-duration_adjust, duration_adjust)
        else:
            self.duration_adjust = duration_adjust
        self.fit_model_background()
        self.make_burst_pdf()

    def fit_model_background(self):
        def model_residual(pars, t, data=None):
            t0 = t[0]
            v = pars.valuesdict()
            model = (
                v["c0"] + v["c1"] * (t - t0) + v["c2"] * (t - t0) ** 2
            )  # + v['c3'] * (t - t0) ** 3
            if data is None:
                return model
            return data - model

        lo_bot, lo_top, hi_bot, hi_top = *self.model.lobckint, *self.model.hibckint

        bkg = [
            x
            for x in self.model.data
            if ((x > lo_bot and x < lo_top) or (x > hi_bot and x < hi_top))
        ]
        lo, hi = np.arange(lo_bot, lo_top, step=self.binning), np.arange(
            hi_bot, hi_top, step=self.binning
        )
        bkg_counts, bkg_bins = np.histogram(bkg, bins=np.concatenate([lo, hi]))

        fit_params = Parameters()
        fit_params.add("c0", value=1000 * self.binning, min=0)
        fit_params.add("c1", value=0)
        fit_params.add("c2", value=0)
        fit_params.add("c3", value=0)

        out = minimize(
            model_residual,
            fit_params,
            args=((bkg_bins[1:] + bkg_bins[:-1]) / 2,),
            kws={"data": bkg_counts},
        )
        out.model = lambda t: model_residual(out.params, t, data=None)
        self.bkg_fit = out

    def make_burst_pdf(self):
        lo_top, hi_bot = (
            self.model.tTrigger + self.duration_adjust[0],
            self.model.tTrigger + self.model.t90 + self.duration_adjust[1],
        )

        signal = [x for x in self.model.data if (x > lo_top and x < hi_bot)]
        burst_counts, burst_bins = np.histogram(
            signal, np.arange(signal[0], signal[-1], self.binning)
        )
        try:
            bkg_est = self.bkg_fit.model((burst_bins[1:] + burst_bins[:-1]) / 2)
        except IndexError:
            raise IndexError(
                "I'm failing at background subtraction. Binning parameter possibly exceeding model's t90."
            )
        burst_counts_minus_bkg = (burst_counts - bkg_est).clip(min=0)

        self.burst_pdf = _pdf(
            list(burst_counts_minus_bkg / np.sum(burst_counts_minus_bkg)),
            list(burst_bins - self.model.tTrigger),
        )

    def plot_model_diagnostic(self, binning=None, *args, **kwargs):
        lo_bot, lo_top, hi_bot, hi_top = *self.model.lobckint, *self.model.hibckint

        if binning is None:
            binning = self.binning

        fig = self.model.plot(binning, *args, **kwargs)
        plt.axvspan(
            lo_bot - self.model.tTrigger,
            lo_top - self.model.tTrigger,
            alpha=0.2,
            color="red",
        )
        plt.axvspan(
            hi_bot - self.model.tTrigger,
            hi_top - self.model.tTrigger,
            alpha=0.2,
            color="red",
        )
        plt.axvline(
            self.duration_adjust[0], color="red", linewidth=1.5, label="model_start"
        )
        plt.axvline(
            self.model.t90 + self.duration_adjust[1],
            color="cornflowerblue",
            linewidth=1.5,
            label="model_end",
        )
        x = np.linspace(lo_bot, hi_top, 100)
        # we compute a correction to the background which account for the fact that
        # the background is computed on satellite time while plot routines
        # works on trigger time (seconds since and from model trigger)
        trig_time_shift_corr = (
            -(
                self.bkg_fit.model(np.array([0.0]))
                - self.bkg_fit.model(np.array([self.model.tTrigger]))
            )
            * binning
            / self.binning
        )
        plt.plot(
            x - self.model.tTrigger,
            self.bkg_fit.model(x) * binning / self.binning - trig_time_shift_corr,
            c="k",
        )
        plt.legend()
        return fig


class LCSCalibration(_LightCurveSim):
    """
    A class for calibration signals.
    Recipe:
    1.  Build a calibration object e.g. x = LCSCalibration()
    2.  Simulate specifying duration e.g. x.simulate(2) will generate
        the triangular burst pdf with duration 2 seconds.
    3.  Add background and burst events via add_bkg and add_bursts methods
    """

    def simulate(
        self,
        burst_duration,
        *,
        burst_shape="rectangular",
        bkg_scenario=None,
        orbit_fraction=0
    ):
        """
        :param burst_duration:  float
        :param burst_shape:     str, 'triangular' or 'rectangular'. default is 'triangular'
        :param bkg_scenario:    optional str, 'worst' or 'gentle'. default is None
        :param orbit_fraction:  float between -1 and 1.
                                -1 for SAA exit, +1 for SAA entering. 0 for mid-orbit.
                                default is 0.
        :return:
        """
        self.burst_shape = burst_shape
        self.bkg_scenario = bkg_scenario
        self.bkg_orbit_fraction = orbit_fraction
        self.burst_duration = burst_duration
        self.make_burst_pdf()

    def make_burst_pdf(self):
        if self.burst_shape == "triangular":
            distr_data = np.random.triangular(
                0, self.burst_duration / 2, self.burst_duration, size=10**6
            )  # TODO: this pdf are kind of bad.
        elif (
            self.burst_shape == "rectangular"
        ):  #       No random generator should be involved here.
            distr_data = np.random.uniform(0, self.burst_duration, size=10**6)

        burst_counts, burst_bins = np.histogram(
            distr_data, bins=np.arange(0, self.burst_duration, self.binning)
        )
        self.burst_pdf = _pdf(
            list(burst_counts / np.sum(burst_counts)), list(burst_bins)
        )


if __name__ == "__main__":
    model_id = "120707800"
    lc = LCSFromModel(duration=120, burst_start_time=30)
    lc.simulate(model_id, duration_adjust=(-0.5, 1))

    lc.add_bkg(350, is_flux=True)
    lc.add_burst(8000)
    print("i've generated a total of {} photons.".format(len(lc.get_data())))
