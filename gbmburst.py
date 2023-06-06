from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from database import get_metadata
from download_tte import get_events


class Lightcurve:
    def __init__(self, grb_id: str):
        self.data = get_events(grb_id)
        self._tmin = self.data[0, 0]
        self._tmax = self.data[0, -1]
        self._energies = None
        self.grb_id = grb_id
        self.metadata = get_metadata(grb_id)
        self.background_interval = self._background_interval()

    def get_times(self) -> np.ndarray:
        """
        :return: array containing time tagged events time in s ("TIME" in fits, MET).
        """
        return self.data[:, 0]

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

        margin_lo = (trigger_time - self._tmin) / 3
        margin_hi = (self._tmax - (trigger_time + t90)) / 5
        lo_bot, lo_top = self._tmin - trigger_time + 1, -margin_lo
        hi_bot, hi_top = t90 + margin_hi, self._tmax - trigger_time - 1
        return lo_bot, lo_top, hi_bot, hi_top

    def _background_interval(self) -> Tuple[float, float, float, float]:
        """
        returns background interval boundaries based on catalog (if sane)
        or burst's t90.
        :return: a 4-tuple representing
        (back_low_start, back_low_stop, back_hi_start, back_hi_stop)
        """
        lo_bot, lo_top, hi_bot, hi_top = self._background_interval_from_catalog()
        assert lo_bot < lo_top
        assert hi_bot < hi_top

        if (lo_bot < self._tmin) & (hi_top > self._tmax):
            lo_bot, lo_top, hi_bot, hi_top = self._background_interval_from_t90()
        elif lo_bot < self._tmin:
            lo_bot, lo_top, *_ = self._background_interval_from_t90()
        elif hi_top < self._tmax:
            *_, hi_bot, hi_top = self._background_interval_from_t90()
        return lo_bot, lo_top, hi_bot, hi_top

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
        data = self.data[(self.data[:, 2] > lo_en) & (self.data[:, 1] < hi_en)]
        times = data[:, 0] - self.metadata["trigger_time_met"]
        lo_bot, lo_top, hi_bot, hi_top = self.background_interval
        bins = np.arange(times[0], times[-1], binning)
        counts, bins = np.histogram(times, bins=bins)

        # fmt: off
        fig, ax = plt.subplots(**kwargs)
        ax.step(
            bins[:-1],
            counts,
            color="k",
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

        if xlims: ax.set_xlim(*xlims)
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
    fig, ax = Lightcurve("120707800").plot(dpi=150, figsize=(11, 4), binning=0.1)
    plt.show()
