import sys

import numpy as np
from sklearn import metrics
from collections import Counter
import utilities
import matplotlib.pyplot as plt


class ENNSmoothFilter:

    def __init__(self, k=5, dist_metric=utilities.euclidean):
        self.data = None
        self.removed = []
        self.resultSet = []
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, data):
        self.data = np.array(data)

    def metrics(self):
        print(" ----------------ENN----------------")
        print("New data length: ", len(self.resultSet))
        print("Old data length: ", len(self.data))
        print("Removed points: ", len(self.removed))
        print(" ----------------ENN----------------")

    def evaluate(self):
        self.resultSet = []
        self.removed = []
        for idx, dataPoint in enumerate(self.data):
            mostFrequentClassLabel = utilities.retrieveKPoints(dataPoint, self.data, self.dist_metric, self.k)
            if dataPoint[-1] == mostFrequentClassLabel:
                self.resultSet.append(dataPoint)
            else:
                self.removed.append(dataPoint)
