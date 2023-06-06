from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from database import get_metadata
from download_tte import get_events


class Lightcurve:
    def __init__(self, grb_id: str):
        # TODO: add an option for selecting detectors
        self.grb_id = grb_id
        self.data = get_events(grb_id)
        self.metadata = get_metadata(grb_id)
        trigger_time_met = self.metadata["trigger_time_met"]
        self._tmin = self.data[0, 0] - trigger_time_met
        self._tmax = self.data[-1, 0] - trigger_time_met
        self._energies = None
        self.background_interval = self._background_interval_from_t90()

    def get_times(self) -> np.ndarray:
        """
        :return: array containing time tagged events time in s (from trigger).
        """
        return self.data[:, 0] - self.metadata["trigger_time_met"]

    def get_emin(self) -> np.ndarray:
        """
        :return: array containing time tagged events low energy value ("E_MIN"in fits).
        """
        return self.data[:, 1]

    def get_emax(self) -> np.ndarray:
        """
        :return: array containing time tagged events high energy value ("E_MAX" in fits).
        """
        return self.data[:, 2]

    def get_energies(self):
        return (self.get_emin() + self.get_emin()) / 2

    def _background_interval_from_catalog(self) -> Tuple[float, float, float, float]:
        """
        Gets background interval boundaries from burst catalog
        :return: a 4-tuple representing
        (back_low_start, back_low_stop, back_hi_start, back_hi_stop)
        """
        lo_bot = self.metadata["back_interval_low_start"]
        lo_top = self.metadata["back_interval_low_stop"]
        hi_bot = self.metadata["back_interval_high_start"]
        hi_top = self.metadata["back_interval_high_stop"]
        return lo_bot, lo_top, hi_bot, hi_top

    def _background_interval_from_t90(self) -> Tuple[float, float, float, float]:
        """
        Especially for very long GRBs the bkg intervals from
        metadata refers to times not represented in TTE data lists
        and hence are no good. This routine builds original
        bkg intevals based on the actual available data,
        t90 and trigger time.
        :return: a 4-tuple representing
        (back_low_start, back_low_stop, back_hi_start, back_hi_stop)
        """
        trigger_time = self.metadata["trigger_time_met"]
        t90 = self.metadata["t90"]

        margin_lo = -self._tmin / 3
        margin_hi = (self._tmax - t90) / 5
        lo_bot, lo_top = self._tmin + 1, -margin_lo
        hi_bot, hi_top = t90 + margin_hi, self._tmax - 1
        return lo_bot, lo_top, hi_bot, hi_top

    def fit_background(self, binning, deg=2) -> Callable:
        """
        Fits binned data according to boundaries attribute `background_interval`
        and returns a polynomial function.
        :param binning: histogram binlength
        :param deg: polynomial degree
        :return: a function, call over array of times to get background estimates.
        """
        lo_bot, lo_top, hi_bot, hi_top = self.background_interval
        times = self.get_times()
        times_lo = times[(times > lo_bot) & (times < lo_top)]
        times_hi = times[(times > hi_bot) & (times < hi_top)]
        bins_lo = np.arange(lo_bot, lo_top + binning, binning)
        counts_lo, _ = np.histogram(times_lo, bins=bins_lo)
        bins_hi = np.arange(hi_bot, hi_top + binning, binning)
        counts_hi, _ = np.histogram(times_hi, bins=bins_hi)
        midpoints_lo = (bins_lo[1:] + bins_lo[:-1]) / 2
        midpoints_hi = (bins_hi[1:] + bins_hi[:-1]) / 2
        midpoints = np.concatenate((midpoints_lo, midpoints_hi))
        counts = np.concatenate((counts_lo, counts_hi))
        z = np.polyfit(midpoints, counts, deg=deg)
        p = np.poly1d(z)
        return p

    def plot(
        self,
        binning: float = 1.0,
        energy_range: Tuple[float, float] = (2, 2000),
        xlims: Tuple[float, float] | None = None,
        ylims: Tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.axis]:
        """
        Plots count histogram for data from triggered detectors.
        :param binning: histogram bin length
        :param energy_range: energy range
        :param xlims: x-axis plot limits (in units of s from trigger)
        :param ylims: y-axis plot limits (in units of s from trigger)
        :param kwargs: optional arguments passed to plt.subplots().
        :return:
        """
        lo_en, hi_en = energy_range
        data = self.data[(self.get_emin() > lo_en) & (self.get_emax() < hi_en)]
        times = data[:, 0] - self.metadata["trigger_time_met"]
        lo_bot, lo_top, hi_bot, hi_top = self.background_interval
        bins = np.arange(self._tmin, self._tmax, binning)
        counts, bins = np.histogram(times, bins=bins)
        midpoints = (bins[:-1] + bins[1:]) / 2
        p = self.fit_background(binning)
        background = p(midpoints)

        # fmt: off
        fig, ax = plt.subplots(**kwargs)
        ax.step(
            bins[:-1],
            counts,
            color="k",
        )
        ax.plot(
            midpoints,
            background,
            color="red",
            label="Best background fit",
        )
        ax.axvspan(
            lo_bot,
            lo_top,
            color="red",
            alpha=0.1,
            label="Background region",
        )
        ax.axvspan(
            hi_bot,
            hi_top,
            color="red",
            alpha=0.1,
        )
        ax.axvline(
            0,
            linestyle="dotted",
            c="orange",
            linewidth=1,
            label="Trigger time",
        )

        xlims_ = (self._tmin, self._tmax) if xlims is None else xlims
        ax.set_xlim(*xlims_)
        if ylims: ax.set_ylim(*ylims)
        ax.set_title(
            "GRB{}. t90: {:.2f}. {:.1f} - {:.1f} keV".format(
                self.grb_id, self.metadata["t90"], lo_en, hi_en
            )
        )
        ax.set_xlabel("Time (from trigger) [s]")
        ax.set_ylabel("Counts/{:.3f} s bin".format(binning))
        plt.legend()
        # fmt: on
        return fig, ax


if __name__ == "__main__":
    fig, ax = Lightcurve("200219317").plot(dpi=150, figsize=(11, 4), binning=0.1)
    plt.show()
