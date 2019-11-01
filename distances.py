import numpy as np
from scipy.spatial.distance import pdist 


def euclidean_distance(x, y):
    dist = []
    for i in range(len(x)):
        dist.append([])
        for j in range(len(y)):
            s = 0
            for k in range(len(x[i])):
                s += (x[i][k] - y[j][k]) ** 2
            dist[i].append(s ** (1 / 2))
    return dist


# x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
# y = [[8, 7, 6, 5], [4, 3, 2, 1]]

x = [[1, 2]]
y = [[5, 6]]
print(euclidean_distance(x,y))
# print(pdist(x))

def cosine_distance(x, y):
    #####################
    # your code is here #
    #####################
    pass
