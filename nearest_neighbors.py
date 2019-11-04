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

        if self.weights:
            return self.__weighted_classifier(*neighbors)
        else:
            return self.__plain_classifier(neighbors)

    def __plain_classifier(self, indices):
        k_nearest_classes = self.y[indices]
        (classes, counts) = np.unique(k_nearest_classes, return_counts=True)
        i = np.argmax(counts)
        return classes[i]

    def __weighted_classifier(self, distances, indices):
        return None
        # if self.strategy is not "my_own":
        #     if self.weights:
        #         distances, indices = self.neigh.kneighbors(X, return_distance=True)
        #     else:
        #         indices = self.neigh.kneighbors(X, return_distance=False)
        #         distances = [1]*len(indices)
        # else:
        #     if self.weights:
        #         distances, indices = self.find_kneighbors(X, return_distance=True)
        #     else:
        #         indices = self.find_kneighbors(X, return_distance=False)
        #         distances = [1]*len(indices)

        # y = []
        # for i in range(len(indices)):
        #     classes, weights = [
        #         (self.train_y[k], 1 / (distances[i][j] + EPS))
        #         for j, k in enumerate(indices[i])
        #     ]

        #     class_counter = {}
        #     for c_w in zip(classes, weights):
        #         if class_counter[c_w]:
        #             class_counter[c_w][0] += 1
        #             class_counter[c_w][1] += c_w[1]
        #         else:
        #             class_counter[c_w] = (1, c_w[1])

        #     frequent_class = max(class_counter.items(), key=lambda x: x[1][0])[0]
        #     heavy_class = max(class_counter.items(), key=lambda x: x[1][1])[0]

        #     if self.weights:
        #         y.append(heavy_class)
        #     else:
        #         y.append(frequent_class)
        # return y

