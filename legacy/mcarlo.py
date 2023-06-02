from bisect import bisect_left
import numpy as np


def generate_by_invsampl(distribution, num):
    ys = np.random.rand(num)
    ys_indices = [bisect_left(np.cumsum(distribution.counts), y) for y in ys]
    bin_length = (lambda index: distribution.bins[index + 1] - distribution.bins[index])
    events = [distribution.bins[y_index] + bin_length(y_index)*np.random.rand() for y_index in ys_indices]
    return events, ys_indices

