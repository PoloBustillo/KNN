import math

import numpy as np
from sklearn.model_selection import StratifiedKFold
from ENNSmoothFilter import ENNSmoothFilter
from KNNClassifier import KNNClassifier
import random
import pandas as pd
import utilities
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

path = "./Dt1.txt"
# path = "./iris.data.txt"
# path = "./sintetico.txt"
debug = True
display = False
kFolds = 10
corrT = 0.3
findDataFlag = True,
skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)
configs = [
    {"k_KNN": 3, "k_ENN": 3, "PCA_enabled": False, "ENN_enabled": True,
     "distance_metric": utilities.euclidean},
    {"k_KNN": 3, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True,
     "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 7, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean}
]

if __name__ == '__main__':
    # By default, selected attributes to plot are 0 and 1
    xAttr = 0
    yAttr = 1
    # Get data from file
    dataAndClasses, classes, dataWithoutClasses = utilities.read_file(path)
    data = dataAndClasses
    # Just display data to get an idea of it
    if debug:
        utilities.printStart('DATA INFO:')
        print(dataAndClasses)
    if data.shape[1] != 2:
        matrixCorrelation = dataWithoutClasses.corr()

    ennPlot = []
    ennData = []
    if findDataFlag:
        # Calculate two attributes for plotting using data
        xAttr, yAttr = utilities.calculateTwoSignificantAttributes(data, matrixCorrelation, corrT)
        # Display data without classification
        utilities.plotData(display, data, xAttr, yAttr, classes)

    for config in configs:
        print(config)
        # Configure initial setup for KNN and ENN
        knn = KNNClassifier(config.get('k_KNN'), config.get('distance_metric'))
        enn = ENNSmoothFilter(config.get('k_ENN'), config.get('distance_metric'))
        # Print current configuration
        if config.get("ENN_enabled"):
            utilities.printStart('KNN:' + str(config.get('k_KNN')) + ' - ENN:' + str(config.get('k_ENN')))
        else:
            utilities.printStart('KNN:' + str(config.get('k_KNN')))
        if config.get('ENN_enabled'):
            # Recalculate new data and classes
            enn.fit(dataAndClasses)
            enn.evaluate()
            ennPlot.append(pd.concat([enn.removed, enn.resultSet]))
            data = pd.DataFrame(enn.resultSet)
            classes = data.iloc[:, -1]

        # KFold execution
        i = 0
        for train_indexes, test_indexes in skf.split(data, classes):
            if debug:
                i = i + 1
                print('### KFold: ', i, " ####")
            knn.fit(data, train_indexes, test_indexes)
            knn.evaluate()
        if config.get('ENN_enabled'):
            ennData.append(enn.metrics(xAttr, yAttr, debug, display))
        knn.metrics(debug, display, xAttr, yAttr)

    fig = make_subplots(rows=math.ceil(len(configs) / 2), cols=2,
                        subplot_titles=[
                            "KNN:" + str(configs[idx].get('k_KNN')) + " - ENN:" + str(configs[idx].get('k_ENN')) +
                            " - Removed:" + str(ennData[idx][2]) +
                            " - Old Data:" + str(ennData[idx][1])
                            for idx in range(len(configs))
                        ])
    symbols = random.choices(SymbolValidator().values, k=20)
    for idx in range(len(ennPlot)):
        by_class = ennPlot[idx].groupby('Classes')
        indexCol = (idx % 2) + 1
        indexRow = (idx // 2) + 1
        index = 0
        for groups, data in by_class:
            fig.add_trace(
                go.Scatter(
                    name=groups,
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    marker=dict(
                        symbol=symbols[index]
                    ),
                    marker_line_width=1,
                    marker_size=5,
                    mode="markers",
                ),
                row=indexRow, col=indexCol
            )
            index += 1

    fig.update_layout(title_text="Archivo: " + path)
    fig.show()
