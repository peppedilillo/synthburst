import numpy as np
from bisect import bisect_left
from math import exp
import scipy.integrate as integrate
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from legacy._paths import filepath_instruments, filepath_background_sim

# a few band parametization. note that these are median values.
grb990123 =  {'alpha' : -0.6, 'beta' : -3.1, 'e_peak' : 720}
nava2011 = {'alpha': -1, 'beta': -2.33, 'e_peak': 174}
goldstein2012 = {'alpha' : -1.05, 'beta' : -2.25, 'e_peak': 205}
gruberbest2014 = {'alpha' : -1.08, 'beta' : -2.14, 'e_peak': 196}


class Band:
    """
    Band function power-law emission
    """
    def __init__(self, alpha=-1, beta=-2.33, e_peak=174):
        assert alpha != -2
        self.alpha = alpha
        self.beta = beta
        self.e_peak = e_peak
        self.e_0 = self.e_peak / (2 + self.alpha)
        self.norm = self.compute_norm_constant()

    def _band_f(self, en, constant=1):
        '''
        energy in keV
        '''
        if en < (self.alpha - self.beta) * self.e_0:
            return ((en / 100) ** self.alpha) * exp(- en / self.e_0) / constant
        else:
            return ( ((self.alpha - self.beta) * self.e_0 / 100) ** (self.alpha - self.beta) )\
            * exp(self.beta - self.alpha) * (( en / 100) ** self.beta) / constant

    def compute_norm_constant(self):
        # note we normalize over the interval where detector efficiency is defined
        return integrate.quad(lambda x: self._band_f(x), 1.0, 4116.0)[0]

    def __call__(self, en):
        return self._band_f(en, self.norm)

    def cdf(self, en):
        return integrate.quad(lambda x: self.__call__(x), 1, en)[0]


class BackgroundSpectra():
    """
    background simulation wrapper class
    """
    def __init__(self,in_mode):
        channels, counts = self.read_bkg_matrix(in_mode)
        self.channels = channels
        self.counts = counts
        self.rate = sum(counts)/1000

    def read_bkg_matrix(self,in_mode):
        '''
        :param in_mode: 'X' or 'S'
        :return:
        '''
        bkg_file = filepath_background_sim(in_mode)
        data = fits.getdata(bkg_file, 'SPECTRUM', header=False)
        channels, counts = data['CHANNEL'], data['COUNTS']
        return channels, counts


class Instrument:
    """
    instruments redistribution (RMF) and effective area (ARF) wrapper class.
    """
    def __init__(self,in_mode):
        specresp, en_bins = self.read_arf_data(in_mode)
        self.rmf_mat = self.read_rmf_matrix(in_mode)
        self.specresp = specresp
        self.en_bins = en_bins

    def read_rmf_matrix(self,in_mode):
        '''
        Read a rmf fits built according to OGIP standard  (cf. CAL/GEN/92-002, George, I.M. 1992)
        and returns a numpy array of the redistribution matrix.

        par rmf_file: str, .rmf fits file name
        return m: numpy array
        '''
        rmf_file = filepath_instruments('rmf', in_mode)
        (data, header) = fits.getdata(rmf_file, 'MATRIX', header=True)
        CH_NUM = header['DETCHANS']
        m = np.zeros((CH_NUM, CH_NUM))
        for row in range(CH_NUM):
            N_GRP, MAT = data[row][2], data[row][-1]
            # mapping tuple of arrays to tuples of arrays clipping unnecessary information
            F_CHAN, N_CHAN = tuple(map(lambda x: x[:N_GRP], data[row][3:-1]))
            for i, (f_chan, n_chan) in enumerate(zip(F_CHAN, N_CHAN)):
                m[row, f_chan:f_chan + n_chan] = MAT[:n_chan]
                # no np.pop() in numpy. we peel the matrix elements
                MAT = np.delete(MAT, np.s_[:n_chan])
        return m

    def read_arf_data(self,in_mode):
        '''
        open an arf file and returns bins and effective areas
        :param arf_file:
        :return:
        '''
        arf_file = filepath_instruments('arf', in_mode)
        data = fits.getdata(arf_file, 'SPECRESP', header=False)
        low_en, hi_en, specresp = list(zip(*data))
        en_bins = list(low_en) + [hi_en[-1]]
        return specresp, en_bins

    def arf(self,en):
        def find_lt(a, x):
            '''
            Find first leftmost index lesser than x
            '''
            i = bisect_left(a, x)
            if i:
                # e.g. a, x = [10,11,12,13], 11.5
                # bisect_left(a,x) > 2
                return i - 1
            raise ValueError
        index = find_lt(self.en_bins,en)
        return self.specresp[index]

    def plot_rmf(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.rmf_mat, norm=LogNorm(vmin=0.00001, vmax=1))
        plt.gca().xaxis.tick_bottom()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Full energy ($20$ - $4116$ keV) and channel scale ($0$-$1023$)', pad=20)
        plt.xlabel('Channel')
        plt.ylabel('Energy bin')
        plt.show()


_instrument_X, _background_X = Instrument(in_mode='X'), BackgroundSpectra(in_mode='X')
_instrument_S, _background_S = Instrument(in_mode='S'), BackgroundSpectra(in_mode='S')


# background time profiles


def cosine(period = 60*90, var = 0.5, phase = 0):
    assert var <= 1

    def f(bins):
        from math import pi
        ys = var*np.cos(bins*(2*pi/period) + phase)  + 1
        return ys
    return f


def linear(var = 0.5):
    assert abs(var) <= 1

    def f(bins):
        assert var <= 1
        ys = 2*var/bins[-1]*bins + (1-var)
        return ys
    return f