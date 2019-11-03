import numpy as np
# from scipy.spatial.distance import pdist


def euclidean_distance(x, y):
    dist = []
    for i in range(len(x)):
        dist.append([])
        for j in range(len(y)):
            s = 0
            for k in range(len(x[i])):
                s += (x[i][k] - y[j][k]) ** 2
            dist[i].append(s ** (1 / 2))
            # print(f"debug: {pdist(np.array([x[i], y[j]]))}")
            # print(f"my: {dist[i][-1]}")
    return dist

def cosine_distance(x, y):
    dist = []
    for i in range(len(x)):
        dist.append([])
        for j in range(len(y)):
            numerator = 0
            a = 0
            b = 0
            for k in range(len(x[i])):
                numerator += x[i][k] * y[j][k]
                a += x[i][k] ** 2
                b += y[j][k] ** 2
            denominator = a ** (1 / 2) * b ** (1 / 2)
            dist[i].append(1-numerator / denominator)
            # print(f"debug: {pdist(np.array([x[i], y[j]]), 'cosine')}")
            # print(f"my: {dist[i][-1]}")
    return dist
