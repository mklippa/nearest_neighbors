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


x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
# y = [[8, 7, 6, 5], [4, 3, 2, 1]]

# print(euclidean_distance(x, y))


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

# print(cosine_distance(x, y))


def foo(x):
    dst = euclidean_distance(x,x)
    print(dst)
    a = {}
    for i in range(len(x)):
        for j in range(len(x)):
            if i < j:
                a[(i,j)] = dst[i][j]
    
    sorted_a = sorted(a.items(), key=lambda kv: kv[1])
    for item in sorted_a:
        print(item)

foo(x)
