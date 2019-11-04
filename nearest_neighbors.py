import numpy as np

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

        self.X = X
        self.y = y

    def find_kneighbors(self, X, return_distance=False):
        if self.metric is "euclidean":
            distances = euclidean_distance(X, self.X)
        else:
            distances = cosine_distance(X, self.X)

        k_indices = np.argpartition(-distances, -self.k, axis=1)[:, -self.k :]
        k_distances = distances[np.arange(k_indices.shape[0])[:, None], k_indices]

        if return_distance:
            return (k_distances, k_indices)
        else:
            return k_indices

    def predict(self, X):
        if self.strategy is not "my_own":
            neighbors = self.neigh.kneighbors(X, return_distance=self.weights)
        else:
            neighbors = self.find_kneighbors(X, return_distance=self.weights)

        distances, indices = neighbors if self.weights else (None, neighbors)

        k_nearest_classes = self.y[indices]
        counter = self.__bincount(k_nearest_classes, distances)
        return np.argmax(counter, axis=1)

    def __bincount(self, A, distances=None):
        N = A.max() + 1
        arr = (A + (N * np.arange(A.shape[0]))[:, None]).ravel()
        min_len = N * A.shape[0]
        if distances is not None:
            return np.bincount(arr, distances.ravel(), min_len).reshape(-1, N)
        else:
            return np.bincount(arr, minlength=min_len).reshape(-1, N)


def euclidean_distance(x, y):
    return (
        -2 * np.dot(x, y.T)
        + np.sum(y ** 2, axis=1)
        + np.sum(x ** 2, axis=1)[:, np.newaxis]
    ) ** 0.5


def cosine_distance(x, y):
    c = np.sum(y ** 2, axis=1) ** 0.5 * np.sum(x ** 2, axis=1)[:, np.newaxis] ** 0.5
    return 1 - np.dot(x, y.T) / c

