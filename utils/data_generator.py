import numpy as np

class DataGenerator:
    def __init__(self, filename, n_features):
        # load data here
        data = np.load(filename)
        self.feature = data[:, 1:n_features+1]
        self.target1 = data[:, 0]
        self.target2 = -data[:, (n_features+1):]
