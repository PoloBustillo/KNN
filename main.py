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
#path = "./iris.data.txt"
# path = "./sintetico.txt"
debug = True
display = True
kFolds = 10
corrT = 0.3
findDataFlag = True,
skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)
configs = [
    {"k_KNN": 3, "k_ENN": 3, "PCA_enabled": False, "ENN_enabled": False,
     "distance_metric": utilities.euclidean},
    {"k_KNN": 5, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True,
     "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 7, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": False,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": False,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True,
    #  "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": False,
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
    knnPlot = []
    ennData = []
    knnData = []
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
            data = dataAndClasses
            classes = data.iloc[:, -1]
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
        knnData.append(knn.metrics(debug, display, xAttr, yAttr, config.get('k_ENN') if config.get('ENN_enabled') else " "))
        knnPlot.append(knn.knn_evaluation)

    # TODO: MOVE TO FUNCTION
    # PLOT KNN ENN SECTION
    symbols = random.choices(SymbolValidator().values, k=20)

    figKNN = make_subplots(rows=math.ceil(len(configs) / 2), cols=2,
                           subplot_titles=[

                               ("KNN " + str(configs[idx].get('k_KNN'))
                                + ' ENN: ' + str(configs[idx].get('k_ENN'))
                                + " / " + str(knnData[idx]['accuracy']))
                               if configs[idx].get('ENN_enabled') == True
                               else
                               ("KNN " + str(configs[idx].get('k_KNN'))
                                + " / " + str(knnData[idx]['accuracy']))
                               for idx in range(len(configs))
                           ])
    for idx in range(len(knnPlot)):
        by_class = knnPlot[idx].groupby('Status')
        indexCol = (idx % 2) + 1
        indexRow = (idx // 2) + 1
        index = 0
        for groups, data in by_class:
            figKNN.add_trace(
                go.Scatter(
                    name=groups,
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    marker=dict(
                        symbol=symbols[index]
                    ),
                    marker_line_width=1,
                    marker_size= 10 if len(knnPlot)<=4 else 7,
                    mode="markers",
                ),
                row=indexRow, col=indexCol
            )
            index += 1
    figKNN.update_layout(title_text="Resultados KNN con Archivo: " + path)
    figKNN.show()

    res_list = [i for i in range(len(configs)) if configs[i].get('ENN_enabled') == True]
    figENN = make_subplots(rows=math.ceil(len(ennPlot) / 2), cols=2,
                           subplot_titles=[
                               "KNN:" + str(configs[res_list[idx]].get('k_KNN')) + " - ENN:" + str(
                                   configs[res_list[idx]].get('k_ENN')) +
                               " - Removed:" + str(ennData[idx][2]) +
                               " - Data:" + str(ennData[idx][0]) +
                               " - Accuracy:" + str(knnData[res_list[idx]]['accuracy'])
                               for idx in range(len(ennPlot))
                           ])

    for idx in range(len(ennPlot)):
        by_class = ennPlot[idx].groupby('Classes')
        indexCol = (idx % 2) + 1
        indexRow = (idx // 2) + 1
        index = 0
        for groups, data in by_class:
            figENN.add_trace(
                go.Scatter(
                    name=groups,
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    marker=dict(
                        symbol=symbols[index]
                    ),
                    marker_line_width=1,
                    marker_size=10 if len(ennPlot) <= 4 else 6,
                    mode="markers+text",
                ),
                row=indexRow, col=indexCol
            )
            index += 1

    figENN.update_layout(title_text="Resultados ENN con Archivo: " + path + " Data: " + str(ennData[0][1]))
    figENN.show()
