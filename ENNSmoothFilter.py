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

    def metrics(self, debug=True, display=True):
        if debug:
            print(" ----------------ENN----------------")
            print("New data length: ", len(self.resultSet))
            print("Old data length: ", len(self.data))
            print("Removed points: ", len(self.removed))
            print(" ----------------ENN----------------")
        if display:
            print("----------------")

    def evaluate(self):
        self.resultSet = []
        self.removed = []
        justIndexes = []

        for idx, dataPoint in enumerate(self.data):
            # Removed point from data
            justIndexes.append(idx)
            print(justIndexes)
            dataWithoutPoint = np.delete(self.data, justIndexes, 0)
            mostFrequentClassLabel = utilities.getMajorClass(dataPoint, dataWithoutPoint, self.dist_metric, self.k)
            if dataPoint[-1] == mostFrequentClassLabel:
                justIndexes.pop()
                self.resultSet.append(dataPoint)
            else:
                self.removed.append({"point": dataPoint, "index": idx})
