from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from database import get_metadata
from download_tte import get_events


class Lightcurve:
    def __init__(self, grb_id: str):
        self.grb_id = grb_id
        self.metadata = get_metadata(grb_id)
        self.data = get_events(grb_id)
        self.background_interval = self._background_interval()

    def plot(
        self,
        binning: float = 1.0,
        xlims: Tuple[float, float] | None = None,
        ylims: Tuple[float, float] | None = None,
        energy_range: Tuple[float, float] | None = (50, 300),
        **kwargs: Any,
    ) -> Tuple[plt.Figure, plt.axis]:
        if energy_range:
            lo, hi = energy_range
            data = self.data[(self.data[:, 2] > lo) & (self.data[:, 1] < hi)]
        times = data[:, 0] - self.metadata["trigger_time_met"]
        lo_bot, lo_top, hi_bot, hi_top = self.background_interval
        counts, bins = np.histogram(
            times,
            bins=np.arange(times[0], times[-1], binning),
        )

        fig, ax = plt.subplots(**kwargs)
        ax.step(bins[:-1], counts, color="k")
        ax.axvspan(lo_bot, lo_top, color="red", alpha=0.1, label="background region")
        ax.axvspan(
            hi_bot,
            hi_top,
            color="red",
            alpha=0.1,
        )
        ax.axvline(0, linestyle="dotted", c="orange", linewidth=1, label="trigger")
        ax.set_title("GRB{}. t90: {:.2f}".format(self.grb_id, self.metadata["t90"]))
        ax.set_xlabel("Time (from trigger) [s]")
        if xlims:
            plt.xlim(*xlims)
        if ylims:
            plt.ylim(*ylims)
        plt.ylabel("Counts/{:.3f} s bin".format(binning))
        return fig, ax

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
        times = self.data[:, 0]
        mintime, maxtime = times[0], times[-1]
        trigger_time = self.metadata["trigger_time_met"]
        t90 = self.metadata["t90"]

        margin_lo = (trigger_time - mintime) / 3
        margin_hi = (maxtime - (trigger_time + t90)) / 5
        lo_bot, lo_top = mintime - trigger_time + 1, -margin_lo
        hi_bot, hi_top = t90 + margin_hi, maxtime - trigger_time - 1
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
        mintime, maxtime = self.data[0, 0], self.data[0, -1]

        if (lo_bot < mintime) & (hi_top > maxtime):
            lo_bot, lo_top, hi_bot, hi_top = self._background_interval_from_t90()
        elif lo_bot < mintime:
            lo_bot, lo_top, *_ = self._background_interval_from_t90()
        elif hi_top < maxtime:
            *_, hi_bot, hi_top = self._background_interval_from_t90()
        return lo_bot, lo_top, hi_bot, hi_top


if __name__ == "__main__":
    fig, ax = Lightcurve("140603476").plot(dpi=150, figsize=(11, 4), binning=0.1)
    plt.show()
