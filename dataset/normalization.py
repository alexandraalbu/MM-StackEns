import numpy as np


def normalize_min_max(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def normalize_std(data, mean, std):
    return (data - mean) / std


def normalize_pair(scaler, pair_data):
    return np.stack(
        [scaler.transform(pair_data[:, 0]), scaler.transform(pair_data[:, 1])]
        , axis=1
    )


class MinMaxNormalization:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):

        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def normalize(self, data):
        return normalize_min_max(data, self.min, self.max)


class StddevNormalization:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def normalize(self, data):
        return normalize_std(data, self.mean, self.std)


class MinMaxStdNormalization:
    def __init__(self):
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

        std_scaled = normalize_std(data, self.mean, self.std)

        self.min = np.min(std_scaled)
        self.max = np.max(std_scaled)

    def transform(self, data):
        return normalize_min_max(normalize_std(data, self.mean, self.std),
                                 self.min,
                                 self.max)
