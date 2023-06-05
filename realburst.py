from download_tte import get_events
from database import get_metadata
import matplotlib.pyplot as plt
import numpy as np


class Lightcurve:
    def __init__(self, grb_id):
        self.grb_id = grb_id
        self.metadata = get_metadata(grb_id)
        self.data = get_events(grb_id)

    def get_metadata(self):
        get_metadata(self.grb_id)

    def plot(self, binning=1.0, xlims=None, ylims=None, **kwargs):
        fig = plt.figure(**kwargs)
        data = self.data[:, 0] - self.metadata["trigger_time"]
        hist_counts, hist_bin, _ = plt.hist(
            data,
            bins=np.arange(data[0], data[-1] + binning, binning),
            color="#607c8e",
            histtype="step",
        )
        plt.axvline(0, linestyle="dotted", c="orange", linewidth=1, label="tTrigger")
        # plt.axvline(self.tTrigger, linestyle = 'dotted', c = 'lightblue', linewidth = 1, label = 't90')
        plt.title("GRB{}. t90: {:.2f}".format(self.id, self.t90))
        plt.xlabel("Time since trigger [s]")
        if xlims:
            plt.xlim(*xlims)
        if ylims:
            plt.ylim(*ylims)
        plt.ylabel("Counts/{:.3f} s bin".format(binning))
        return fig


if __name__ == "__main__":
    Lightcurve("171030729").plot()
    print("1")