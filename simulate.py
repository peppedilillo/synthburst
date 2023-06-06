import numpy as np

from gbmburst import Lightcurve as GRBModel
import matplotlib.pyplot as plt


def inversion_sampling(
        size: int,
        bins: np.ndarray,
        probability: np.ndarray,
) -> np.ndarray:
    """
    Fast inversion sampling. Does not require bins to be equispaced. ~peppe
    :param size: number of samples to draw
    :param bins: bin edges (len n + 1)
    :param probability: unnormalized probability values (len n).
    :return: !!unsorted!! samples
    """
    probability_cdf = np.cumsum(probability)
    ys = np.random.rand(size) * np.sum(probability)
    ii = np.searchsorted(probability_cdf, ys)
    top_edges = bins[ii + 1]
    samples = top_edges - np.random.rand(size) * (top_edges - bins[ii])
    return samples


class ConstantBackground:
    def __init__(self, rate):
        self.rate = rate

    def generate_times(self, t_min, t_max):
        size = int(self.rate * (t_max - t_min))
        sample = np.random.rand(size)
        times = (t_max - t_min) * sample + t_min
        return times


class Burst:
    def __init__(self, model):
        self.model = GRBModel(model)

    def generate_times(self, size):
        return inversion_sampling(size, *self.pdf())

    def pdf(self):
        _, lo_top, hi_bot, _ = self.model.background_interval
        times = np.unique(self.model.get_times()) - self.model.metadata["trigger_time_met"]
        bins = times[(times > lo_top) & (times <= hi_bot)]
        probability = 1 / np.diff(bins)
        return bins, probability


class Lightcuve:
    def __init__(self, duration):
        pass

    def add_background(self, background):
        pass

    def add_burst(self, burst):
        pass


if __name__ == "__main__":
    print("loading model")
    x = Burst(model="120707800")
    print("generating burst events")
    data = x.generate_times(1_000_000)
    print("plotting")
    bins = np.arange(np.min(data), np.max(data), 0.1)
    counts, _ = np.histogram(data, bins=bins)
    plt.step(bins[:-1], counts)
    plt.show()