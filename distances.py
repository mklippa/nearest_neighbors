import numpy as np


def euclidean_distance(x, y):
    N = x.shape[0]
    M = y.shape[0]
    dists = np.zeros((N, M))
    for i in range(M):
        dists[:, i] = np.sum((x - y[i, :]) ** 2, axis=1) ** 0.5
    return dists


def cosine_distance(x, y):
    N = x.shape[0]
    M = y.shape[0]
    dists = np.zeros((N, M))
    for i in range(M):
        numerator = np.sum(x * y[i, :], axis=1)
        denominator = np.sum(x ** 2, axis=1) ** 0.5 * np.sum(y[i, :] ** 2) ** 0.5
        dists[:, i] = 1 - numerator / denominator
    return dists
