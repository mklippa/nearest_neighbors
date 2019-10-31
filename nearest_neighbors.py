import numpy as np

EPS = 1e-5


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
    
    def fit(self, X, y):
        #####################
        # your code is here #
        #####################

    def find_kneighbors(self, X, return_distance):
        #####################
        # your code is here #
        #####################
    
    def predict(self, X):
        #####################
        # your code is here #
        #####################

