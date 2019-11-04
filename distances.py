import numpy as np


def euclidean_distance(x, y):
    return (
        -2 * np.dot(x, y.T)
        + np.sum(y ** 2, axis=1)
        + np.sum(x ** 2, axis=1)[:, np.newaxis]
    ) ** 0.5


def cosine_distance(x, y):
    c = np.sum(y ** 2, axis=1) ** 0.5 * np.sum(x ** 2, axis=1)[:, np.newaxis] ** 0.5
    return 1 - np.dot(x, y.T) / c

