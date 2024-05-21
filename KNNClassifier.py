import time

import numpy as np
import pandas as pd
from sklearn import metrics
import utilities
import matplotlib.pyplot as plt
import plotly.express as px


class KNNClassifier:

    def __init__(self, k=5, dist_metric=utilities.euclidean):
        self.runningTime = 0
        self.k = k
        self.dist_metric = dist_metric
        self.train = None
        self.test = None
        self.knn_evaluation = pd.DataFrame([])

        self.correct = []  # only points classified correctly
        self.incorrect = []  # only points classified incorrectly
        self.predicted = []  # all points classified correctly and incorrectly with found class
        self.allPoints = []  # keep track of all points with correct class

        self.accuracy = 0  # percentage of accuracy

        # keep track of indexes for DataFrame
        # and if classification is correct or incorrect
        # self.plot = []
        self.test_indexes = []  # save indexes for test data

    def fit(self, data, train_indexes, test_indexes):
        # From indexes create train data and test data
        trainData = [data.iloc[index, :] for index in train_indexes]
        testData = [data.iloc[index, :] for index in test_indexes]
        self.train = np.array(trainData)
        self.test = np.array(testData)
        self.test_indexes = test_indexes

    def metrics(self, debug=True, display=True, xAxis=0, yAxis=1, k_ENN=None):

        confusion_matrix = metrics.confusion_matrix(np.array(self.allPoints)[:, -1],
                                                    np.array(self.predicted)[:, -1])
        FP = (confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)).sum()
        TP = np.diag(confusion_matrix).sum()

        print("Correctly Classified Instances:", TP, " / ", self.accuracy)
        print("Incorrectly Classified Instances :", FP, " / ", (1 - self.accuracy))
        report = metrics.classification_report(np.array(self.allPoints)[:, -1],
                                               np.array(self.predicted)[:, -1], digits=6)
        print(report)
        if display:
            # precision, recall, _ = metrics.precision_recall_curve(np.array(self.allPoints)[:, -1],
            #                                                       np.array(self.predicted)[:, -1])
            # disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
            # disp.plot()

            confusion_matrix = metrics.confusion_matrix(np.array(self.allPoints)[:, -1],
                                                        np.array(self.predicted)[:, -1],
                                                        )
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            cm_display.plot()
            cm_display.ax_.set_title("KNN: " + str(self.k) + " k_ENN: " + str(k_ENN) + "/ " + str(self.accuracy))
            plt.show()

            fig1 = px.scatter(self.knn_evaluation, x=xAxis, y=yAxis,
                              color=self.knn_evaluation['Status'],
                              symbol=self.knn_evaluation['Evaluation'],
                              color_continuous_scale=["orange", "green", "blue", "purple"]).update_traces(
                                                                                      opacity=0.8,
                                                                                      marker_size=10,
                                                                                      marker_line_width=1)
            fig1.update_layout(
                title="KNN " + str(self.k) + " k_ENN: " + str(k_ENN) + " / " + str(self.accuracy),
                title_font=dict(size=20,
                                color='green',
                                family='Arial'),
                coloraxis_colorbar=dict(yanchor="top",
                                        y=1, x=0,
                                        ticks="outside"),

            )
            fig1.show()
        return {"accuracy": self.accuracy, "confusionMatrix": confusion_matrix}

    def evaluate(self, keepData=True):
        if not keepData:
            self.allPoints = []
            self.predicted = []
            self.correct = []
            self.incorrect = []
            self.runningTime = 0
        start_time = time.time()
        for idx, testPoint in enumerate(self.test):
            mostFrequentClassLabel = utilities.getMajorClass(testPoint, self.train, self.dist_metric, self.k)
            # Compare the class label with the most frequent class label
            # self.plot.append([self.test_indexes[idx], mostFrequentClassLabel == testPoint[-1]])
            if testPoint[-1] == mostFrequentClassLabel:
                self.correct.append(testPoint)
                self.allPoints.append(testPoint)
                self.predicted.append(testPoint)
            else:
                self.allPoints.append(testPoint.copy())
                testPoint[-1] = mostFrequentClassLabel
                self.predicted.append(testPoint)
                self.incorrect.append(testPoint)
        self.runningTime = self.runningTime + (time.time() - start_time)
        print("--- %s seconds ---" % self.runningTime)
        self.knn_evaluation = pd.DataFrame(self.incorrect)
        # self.accuracy = len(self.correct) / (len(self.correct) + len(self.incorrect))
        if len(self.incorrect) > 0:
            self.knn_evaluation.columns = [*self.knn_evaluation.columns[:-1], 'Evaluation']
            self.knn_evaluation['Status'] = 'Failure'
        correctDf = pd.DataFrame(self.correct)
        correctDf.columns = [*correctDf.columns[:-1], 'Evaluation']
        correctDf['Status'] = 'Success'
        self.knn_evaluation = pd.concat([self.knn_evaluation, correctDf])
        self.accuracy = metrics.accuracy_score(np.array(self.allPoints)[:, -1],
                                               np.array(self.predicted)[:, -1])
