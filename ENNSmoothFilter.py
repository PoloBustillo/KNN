import numpy as np
import pandas as pd
import time
import utilities
import plotly.express as px


class ENNSmoothFilter:

    def __init__(self, k=5, dist_metric=utilities.euclidean):
        self.data = pd.DataFrame()
        self.removed = []
        self.resultSet = pd.DataFrame()
        self.k = k
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
            removedDataFrame = pd.DataFrame(self.data.iloc[self.removed, :])
            removedDataFrame['Classes'] = 'Removed'
            allData = pd.concat([removedDataFrame, self.resultSet])
            classes = self.resultSet.iloc[:, -1]
            classesWithEnn = allData.iloc[:, -1]
            classesRemoved = self.data.iloc[self.removed, -1]
            classesRemoved = pd.concat([classesRemoved, classes])
            fig1 = px.scatter(allData, x=str(x), y=str(y), symbol=classesRemoved, color=classesWithEnn)
            fig1.update_layout(
                title="ENN: K= " + str(self.k),
                title_font=dict(size=20,
                                color='green',
                                family='Arial')
            )
            fig1.show()

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
        print("--- %s seconds ---" % (time.time() - start_time))
        self.resultSet = self.data.drop(index=self.removed)
