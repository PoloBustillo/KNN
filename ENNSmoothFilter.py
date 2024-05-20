import numpy as np
import pandas as pd
import time
import utilities
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


class ENNSmoothFilter:

    def __init__(self, k=5, dist_metric=utilities.euclidean):
        self.data = pd.DataFrame()
        self.removed = []
        self.resultSet = pd.DataFrame()
        self.k = k
        self.runningTime = 0
        self.dist_metric = dist_metric

    def fit(self, data):
        self.data = data

    def metrics(self, x, y, debug=True, display=True):
        if debug:
            print(" ----------------ENN----------------")
            print("New data length: ", len(self.resultSet))
            print("Old data length: ", len(self.data))
            print("Removed points: ", len(self.removed))
            print(" ----------------ENN----------------")
        if display:
            allData = pd.concat([self.removed, self.resultSet])
            classes = allData.iloc[:, -1]
            fig1 = px.scatter_matrix(allData, dimensions=[str(x), str(y)],
                                     symbol=classes, color=classes,
                                     color_discrete_sequence=px.colors.qualitative.G10)
            fig1.update_layout(
                title="ENN: K= " + str(self.k) + " <sub>" + str(self.runningTime) + "seg</sub>" +
                      "<br><sup>New data length: " + str(len(self.resultSet)) + " - " +
                      "Old data length: " + str(len(self.data)) + " - " +
                      "Removed points: " + str(len(self.removed)) + "</sup>",
                title_font=dict(size=20,
                                color='black',
                                family='Arial')

            )
            fig1.show()
        return len(self.resultSet), len(self.data), len(self.removed)

    def evaluate(self):
        self.resultSet = []
        values = np.array(self.data)
        start_time = time.time()
        for idx, dataPoint in enumerate(values):
            # Removed point from data
            dataWithoutPoint = np.delete(values, idx, 0)
            mostFrequentClassLabel = utilities.getMajorClass(dataPoint, dataWithoutPoint, self.dist_metric, self.k)
            if dataPoint[-1] != mostFrequentClassLabel:
                self.removed.append(idx)
        self.resultSet = self.data.drop(index=self.removed)
        self.removed = pd.DataFrame(self.data.iloc[self.removed, :])
        self.removed['Classes'] = 'Removed'
        self.runningTime = time.time() - start_time
        print("--- %s seconds ---" % self.runningTime)
