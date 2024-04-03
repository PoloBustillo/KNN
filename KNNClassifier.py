import sys

import numpy as np
from sklearn import metrics
import utilities
import matplotlib.pyplot as plt


class KNNClassifier:

    def __init__(self, k=5, dist_metric=utilities.euclidean):
        self.train = None
        self.test = None
        self.correct = 0
        self.incorrect = 0
        self.accuracy = 0
        self.actual = []
        self.plot = []
        self.test_indexes = []
        self.predicted = []
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, train, test, test_indexes):
        self.train = np.array(train)
        self.test = np.array(test)
        self.test_indexes = test_indexes

    def metrics(self, debug=True, display=True, headers=[]):
        print("Correctly Classified Instances:", self.correct, " / ", str("%.5f" % (self.accuracy * 100)) + "%")
        print("Incorrectly Classified Instances :", self.incorrect, " / ",
              str("%.5f" % ((1 - self.accuracy) * 100)) + "%")
        if display:
            confusion_matrix = metrics.confusion_matrix(self.actual, self.predicted)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            cm_display.plot()
            plt.show()

    def evaluate(self, keepData=True):
        if not keepData:
            self.actual = []
            self.predicted = []
            self.correct = 0
            self.incorrect = 0
        for idx, testPoint in enumerate(self.test):
            mostFrequentClassLabel = utilities.getMajorClass(testPoint, self.train, self.dist_metric, self.k)
            # Compare the class label with the most frequent class label
            self.plot.append([self.test_indexes[idx], mostFrequentClassLabel == testPoint[-1]])
            if testPoint[-1] == mostFrequentClassLabel:
                self.correct = self.correct + 1
                self.actual.append(testPoint[-1])
                self.predicted.append(testPoint[-1])
            else:
                self.incorrect = self.incorrect + 1
                self.actual.append(testPoint[-1])
                self.predicted.append(mostFrequentClassLabel)

            self.accuracy = self.correct / (self.correct + self.incorrect)
