import numpy as np
import distances as dst
from sklearn.neighbors import NearestNeighbors

EPS = 1e-5


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy is not "my_own":
            self.neigh = NearestNeighbors(k, algorithm=strategy, metric=metric)

    def fit(self, X, y):
        if self.strategy is not "my_own":
            self.neigh.fit(X, y)
        else:
            self.X = X
            self.y = y

    def find_kneighbors(self, X, return_distance=False):
        if self.metric is "euclidean":
            distances = dst.euclidean_distance(X, self.X)
        else:
            distances = dst.cosine_distance(X, self.X)

        k_indices = np.argpartition(distances, -self.k, axis=1)[:, -self.k :]
        k_distances = np.take_along_axis(distances, k_indices, axis=1)

        if return_distance:
            return (k_distances, k_indices)
        else:
            return k_indices

    def predict(self, X):
        if self.strategy is not "my_own":
            neighbors = self.neigh.kneighbors(X, return_distance=self.weights)
        else:
            neighbors = self.find_kneighbors(X, return_distance=self.weights)

        distances, indices = (
            neighbors if type(neighbors) is "tuple" else (None, neighbors)
        )

        k_nearest_classes = self.y[indices]
        counter = self.__bincount(k_nearest_classes, distances)
        return np.argmax(counter, axis=1)

    def __bincount(self, A, distances=None):
        N = A.max() + 1
        arr = (A + (N * np.arange(A.shape[0]))[:, None]).ravel()
        weights = distances.ravel()
        min_len = N * A.shape[0]
        return np.bincount(arr, weights, min_len).reshape(-1, N)
