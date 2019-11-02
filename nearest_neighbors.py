import numpy as np
import distances as dst
from statistics import mode

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

    def find_kneighbors(self, X, return_distance=False):
        all_distances = dst.euclidean_distance(X, self.train_X)
        k_distances = []
        k_indices = []
        for i in range(len(X)):
            el_distances = [
                (index, value) for index, value in enumerate(all_distances[i])
            ]
            el_distances.sort(key=lambda x: x[1])
            k_distances.append(map(lambda x: x[1], el_distances[: self.k]))
            k_indices.append(map(lambda x: x[0], el_distances[: self.k]))
        if return_distance:
            return (np.array(k_distances), np.array(k_indices))
        else:
            return np.array(k_indices)

    def predict(self, X):
        distances, indices = self.find_kneighbors(X)

        y = []
        for i in range(len(indices)):
            classes, weights = [
                (self.train_y[k], 1 / (distances[i][j] + EPS))
                for j, k in enumerate(indices[i])
            ]

            class_counter = {}
            for c_w in zip(classes, weights):
                if class_counter[c_w]:
                    class_counter[c_w][0] += 1
                    class_counter[c_w][1] += c_w[1]
                else:
                    class_counter[c_w] = (1, c_w[1])

            frequent_class = max(class_counter.items(), key=lambda x: x[1][0])[0]
            heavy_class = max(class_counter.items(), key=lambda x: x[1][1])[0]

            if self.weights:
                y.append(heavy_class)
            else:
                y.append(frequent_class)
        return y

        # kneighbors = self.find_kneighbors(X)
        # y = []
        # for i in range(len(kneighbors)):
        #     weights = kneighbors
        #     classes = [self.train_y[j] for j in kneighbors[i]]
        #     y.append(mode(classes))
        # return y

