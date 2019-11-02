import numpy as np
import distances as dst

EPS = 1e-5


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
    
    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def find_kneighbors(self, X, return_distance):
        all_distances = dst.euclidean_distance(X, self.train_X)
        k_distances = []
        k_indexes = []
        for i in range(len(X)):
            el_distances = [(index, value) for index, value in enumerate(all_distances[i])]
            el_distances.sort(key=lambda x: x[1])
            k_distances.append(map(lambda x: x[1], el_distances[:self.k]))
            k_indexes.append(map(lambda x: x[0], el_distances[:self.k]))
        if return_distance:
            return (np.array(k_distances), np.array(k_indexes))
        else:
            return np.array(k_indexes)

    def predict(self, X):
        #####################
        # your code is here #
        #####################
        pass

