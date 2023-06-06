import numpy as np


class ConstantBackground:
    def __init__(self, rate):
        self.rate = rate

    def generate_times(self, t_min, t_max):
        size = int(self.rate * (t_max - t_min))
        sample = np.random.rand(size)
        times = np.cumsum((t_max - t_min) * sample + t_min)
        return times


class Burst:
    def __init__(self, model):
        pass

    def generate_burst(self, size):
        pass


class Lightcuve:
    def __init__(self, duration):
        pass

    def add_background(self, background):
        pass

    def add_burst(self, burst):
        pass


if __name__ == "__main__":
    pass
