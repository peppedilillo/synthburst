import matplotlib.pyplot as plt
import numpy as np

from gbmburst import Lightcurve as GRBModel
from template_bkg import bkg_template_split, sample_count


def timer(fn):
    from time import perf_counter

    def inner(*args, **kwargs):
        start_time = perf_counter()
        to_execute = fn(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        print('{0} took {1:.8f}s to execute'.format(fn.__name__, execution_time))
        return to_execute

    return inner


def inversion_sampling(
    size: int,
    bins: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    """
    Fast inversion sampling from discrete distribution.
    Does not require bins to be equispaced.
    :param size: number of samples to draw
    :param bins: bin edges (len n + 1)
    :param probability: unnormalized probability values (len n).
    :return: !!unsorted!! samples
    """
    probability_cdf = np.cumsum(probability)
    ys = np.random.rand(size) * probability_cdf[-1]
    ii = np.searchsorted(probability_cdf, ys)
    top_edges = bins[ii + 1]
    samples = top_edges - np.random.rand(size) * (top_edges - bins[ii])
    return samples


class ConstantBackground:
    def __init__(self, rate):
        self.rate = rate

    @timer
    def generate_times(self, t_min, t_max):
        size = int(self.rate * (t_max - t_min))
        sample = np.random.rand(size)
        times = (t_max - t_min) * sample + t_min
        return times


class TemplateBackground:
    def __init__(self, id_t):
        """
        Init the background template class.
        :param id_t: Tuple identifier of the template (orbit, detector, energy range). E.g. (3, 'n7', 'r1').
        """
        print("Load template")
        self.template = bkg_template_split()[id_t[0]][id_t[1]][id_t[2]].reset_index(drop=True)
        # Bin time of the template. Usually is 4.096.
        self.bin_time = 4.096
    @timer
    def generate_times(self, mean_rate, tmin, tmax, distribution='inversion_sampling'):
        """
        Given a template defined by id_t a TTE list is created to emulate the background dynamics.
        :param mean_rate: Average number of events expected in 1 s.
        :param tmin: Minimum time to select the background template (usually is 0).
        :param tmax: Maximum time to select the background template.
        :param distribution: str, define the type of sampling. E.g. 'inversion_sampling', 'normal', 'poisson'.
        :return: list, the ordered event list.
        """
        template = self.template
        template = template.loc[(template.met >= tmin) & (template.met <= tmax), :]
        # Define list TTE
        list_tte = np.array([])
        print("Begin bkg generation")
        if distribution in ['normal', 'poisson']:
            scale_mean_rate = mean_rate/template['counts'].mean()
            for i in template.index:
                # From the estimated count draw a poisson random variable
                counts = sample_count(template.loc[i, 'counts'], type_random=distribution, en_range=self.id_t[2],
                                      scale_mean_rate=scale_mean_rate)
                # Generate the time arrival of the N photons. N = counts. The time arrival must be in the interval.
                time_tte = np.sort(np.random.uniform(template.loc[i, 'met'], template.loc[i, 'met'] +
                                                     self.bin_time, counts))
                # equispaced event generation
                # time_tte = np.arange(template.loc[i, 'met'], template.loc[i, 'met'] + bin_time, 1/counts)
                list_tte = np.append(list_tte, time_tte)
            print(f"INFO: parameter 'size' wasn't used. Number of events generated: {len(list_tte)}")
        else:
            size = mean_rate * (tmax - tmin)
            bins = template.loc[:, 'met'].values
            probabilities = template.loc[:, 'counts'].values[:-1]
            list_tte = inversion_sampling(size, bins, probabilities)

        print("finish generation")
        return list_tte - tmin


class Burst:
    def __init__(self, model):
        self.model = GRBModel(model)

    @timer
    def generate_times(self, size, rm_bkg=1.024, q=0.5):
        """
        Generate list of TTE event for the GRB chosen.
        :param size: Number of events in the TTE.
        :param rm_bkg: float, this parameter is intended for removing a constant background. The GRB is binned with
        rem_bkg seconds, then a quantile is computed and in each bin it is removed that quantity.
        selected for removing a constant background. Per  the quantile count rates values per each bin
        :param q: Quantile threshold to decide how many events remove.
        :return: list of TTE event.
        """
        data = inversion_sampling(size, *self.pdf())
        # Remove event for constant background subtraction
        if isinstance(rm_bkg, float):
            bins = np.arange(np.min(data), np.max(data), rm_bkg)
            counts, _ = np.histogram(data, bins=bins)
            sample_to_rm = int(np.quantile(counts, q))
            data_resampled = np.array([])
            for i in range(0, len(bins)-1):
                data_tmp = data[(data >= bins[i]) & (data < bins[i + 1])]
                data_tmp = np.random.choice(data_tmp, max(len(data_tmp) - sample_to_rm, 0), replace=False)
                data_resampled = np.append(data_resampled, data_tmp)
            data = data_resampled
        return data

    def pdf(self):
        _, lo_top, hi_bot, _ = self.model.background_interval
        times = np.unique(self.model.get_times())
        bins = times[(times > lo_top) & (times <= hi_bot)]
        probability = 1 / np.diff(bins)
        return bins, probability


class Lightcurve:
    def __init__(self):
        # TODO add duration?
        self.background = None
        self.burst = None

    def add_background(self, background):
        self.background = background

    def add_burst(self, burst):
        self.burst = burst

    def generate_times(self, toffset):
        """
        Generate list of TTE event for the GRB chosen compounded with che background template.
        :param toffset: Which seconds let the GRB start respect the bakground TTE list.
        :return: TTE list of the complete lightcurve
        """
        return np.append(self.burst + toffset, self.background)


def plot_lightcurve(data, bin_time=4):
    bins = np.arange(np.min(data), np.max(data), bin_time)
    counts, _ = np.histogram(data, bins=bins)
    plt.figure()
    plt.step(bins[:-1], counts)
    plt.show()


if __name__ == "__main__":

    print("loading bkg model")
    tb = TemplateBackground(id_t=(3, 'n7', 'r1'))
    bkg_data = tb.generate_times(mean_rate=500, distribution='inversion_sampling', tmin=0, tmax=1500)
    print("plotting bkg")
    plot_lightcurve(bkg_data)

    print("loading GRB model")
    x = Burst(model="120707800")
    print("generating burst events")
    grb_data = x.generate_times(3200)
    print("plotting GRB")
    plot_lightcurve(grb_data)

    print("Define lightcurve model")
    lc = Lightcurve()
    lc.add_burst(grb_data)
    lc.add_background(bkg_data)
    lc_data = lc.generate_times(toffset=500)
    print("plotting lightcurve")
    plot_lightcurve(lc_data)

    pass
