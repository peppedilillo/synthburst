import numpy as np
import astropy.io.fits as fits
from synthburst._paths import filepath_instruments


class _GenericDisplay:
    def __str__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.gather_attrs())

    def gather_attrs(self):
        attrs = '\n'
        for key in self.__dict__:
            if (lambda obj: isinstance(obj, (list, np.ndarray)))(self.__dict__[key]) and (len(self.__dict__[key]) > 3):
                attrs += '\t{} = {}..\n'.format(key,self.__dict__[key][:3])
            else:
                attrs += '\t{} = {}\n'.format(key,self.__dict__[key])
        return attrs


class _LightCurve(_GenericDisplay):
    '''
    dunno why i still i have this but i guess someday could come handy.
    '''
    pass


class _pdf(_GenericDisplay):
    '''
    A class serving as a factory for the distributions from
    which we compute random numbers via inversion sampling.
    NOTE: these are supposed to be normalized in a density
    sense i.e. the sum of counts should be 1. This in general
    makes the integral of counts over bins different from 1!
    '''
    def __init__(self, counts, bins):
        self.counts = counts
        self.bins = bins

    def plot(self, binning = None):
        printf("not devel yet")
        pass

    def filter(self):
        for i, count in enumerate(self.counts[1:-1]):
            if self.counts[i+1] == 0 and self.counts[i-1] == 0:
                self.counts[i] = 0
        self.counts /= np.sum(self.counts)


def convert_to_fits(file, rmf_x=filepath_instruments('rmf', 'X'), rmf_s=filepath_instruments('rmf', 'S')):
    """Converts the given data frame into a .fits file that can
       be exported. Since we want to distinguish between the energies of
       Scintillator and SDD, their responses are needed so the paths have
       to be supplied.
       Usage example:
        1) my_fits=convert_to_fits(my_data_frame, rmf_x=my_rmf_x.rmf, rmf_s=my_rmf_s.rmf)"""

    # assign the names explicitly for better reading
    # also make sure they are numpy arrays
    arrival_time = np.array(file['tte'].values)
    pha = np.array(file['pha'].values)
    flag = np.array(file['flag'].values)

    # now to create the fits file/columns
    # empty primary as is customary
    primary_hdu = fits.PrimaryHDU(None)

    # columns:
    # I decided to keep the phontons from different detectors mixed in the
    # events column but add another header with the detector flag and then
    # create two ebounds extensions for X and S energy channels

    # events column:
    tte = fits.Column(name='time', format='D', unit='s', array=arrival_time)
    detchan = fits.Column(name='PHA', format='I', array=pha)
    dettype = fits.Column(name='FLAG', format='20A', array=flag)

    event_cols = fits.ColDefs([tte, detchan, dettype])
    events = fits.BinTableHDU.from_columns(event_cols, name='EVENTS')

    # now ebounds:
    # first for the SDD
    temp = fits.open(rmf_x)
    ebounds_x = temp['EBOUNDS']
    ebounds_x.name = 'EBOUNDSX'

    # now the scintillator
    temp = fits.open(rmf_s)
    ebounds_s = temp['EBOUNDS']
    ebounds_s.name = 'EBOUNDSS'

    # finally create the HDU list
    hdul = fits.HDUList([primary_hdu, events, ebounds_x, ebounds_s])

    return hdul
