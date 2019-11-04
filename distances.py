import numpy as np


def euclidean_distance(x, y):
    return np.sum((x[:, None] - y) ** 2, axis=2) ** 0.5


def cosine_distance(x, y):
    a = x[:, None]
    b = y
    numerator = np.sum(a * b, axis=2)
    denominator = np.sum(a ** 2, axis=2) ** 0.5 * np.sum(b ** 2, axis=1) ** 0.5
    return 1 - numerator / denominator
