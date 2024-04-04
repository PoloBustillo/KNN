import sys

import numpy as np
import pandas as pd
from sklearn import metrics
import utilities
import matplotlib.pyplot as plt


class KNNClassifier:

    def __init__(self, k=5, dist_metric=utilities.euclidean):
        self.train = None
        self.test = None
        self.correct = []
        self.incorrect = []
        self.knn_evaluation = pd.DataFrame([])
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
        print("Correctly Classified Instances:", len(self.correct), " / ", str("%.5f" % (self.accuracy * 100)) + "%")
        print("Incorrectly Classified Instances :", len(self.incorrect), " / ",
              str("%.5f" % ((1 - self.accuracy) * 100)) + "%")
        if display:
            confusion_matrix = metrics.confusion_matrix(np.array(self.actual)[:, -1], np.array(self.predicted)[:, -1])
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            cm_display.plot()
            plt.show()

    def evaluate(self, keepData=True):
        if not keepData:
            self.actual = []
            self.predicted = []
            self.correct = []
            self.incorrect = []
        for idx, testPoint in enumerate(self.test):
            mostFrequentClassLabel = utilities.getMajorClass(testPoint, self.train, self.dist_metric, self.k)
            # Compare the class label with the most frequent class label
            self.plot.append([self.test_indexes[idx], mostFrequentClassLabel == testPoint[-1]])
            if testPoint[-1] == mostFrequentClassLabel:
                self.correct.append(testPoint)
                self.actual.append(testPoint)
                self.predicted.append(testPoint)
            else:
                self.actual.append(testPoint)
                testPoint[-1] = mostFrequentClassLabel
                self.predicted.append(testPoint)
                self.incorrect.append(testPoint)

        self.knn_evaluation = pd.DataFrame(self.incorrect)
        self.accuracy = len(self.correct) / (len(self.correct) + len(self.incorrect))
        if len(self.incorrect) > 0:
            self.knn_evaluation.columns = [*self.knn_evaluation.columns[:-1], 'Evaluation']
            self.knn_evaluation['Status'] = 'Failure'
        correctDf = pd.DataFrame(self.correct)
        correctDf.columns = [*correctDf.columns[:-1], 'Evaluation']
        correctDf['Status'] = 'Success'
        self.knn_evaluation = pd.concat([self.knn_evaluation, correctDf])
