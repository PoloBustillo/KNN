import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from ENNSmoothFilter import ENNSmoothFilter
from KNNClassifier import KNNClassifier
import utilities
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

path = "./sintetico.txt"
debug = True
display = True
kFolds = 10
corrT = 0.3
skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)
configs = [
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": False, "distance_metric": utilities.euclidean},
    {"k_KNN": 5, "k_ENN": 11, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 3, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean}
]

if __name__ == '__main__':

    dataAndClasses, classes, data = utilities.read_file(path)
    # Apply PCA only to visualize data if PCA flag is enabled then PCA components are used as data
    dataPCA = utilities.apply_PCA(dataAndClasses)
    # Get correlation matrix (pearson) to get value closer to threshold
    matrixCorrelation = dataAndClasses.corr().dropna(how='all', axis=1).dropna(how='all')

    if data.shape[1] == 2:
        fig = px.scatter(dataAndClasses, x="Data_0", y="Data_1", symbol=classes, color=classes)
        fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
        fig.show()
    else:
        print("select data")

    if debug:
        utilities.printStart('DATA INFO:')
        print(dataAndClasses)

    if display:
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrixCorrelation)

    gen = utilities.getIndicesFromCorrelation(matrixCorrelation.abs().to_numpy())

    # for i in gen:
    #     print(i)

    # ij_min = np.unravel_index(matrix.argmin(), matrix.shape)
    # print(matrix[ij_min])
    # print(ij_min)
    # utilities.displayTwoAttributes(ij_min[0], ij_min[1], dataAndClasses, classes)

    for config in configs:
        # Configure initial setup for KNN and ENN
        knn = KNNClassifier(config.get('k_KNN'), config.get('distance_metric'))
        enn = ENNSmoothFilter(config.get('k_ENN'), config.get('distance_metric'))

        utilities.printStart('KNN:' + str(config.get('k_KNN')))
        print(config)
        if config.get('PCA_enabled'):
            dataPCA['Classes'] = dataAndClasses.values[:, -1].T
            data = dataPCA
            if debug:
                print(dataPCA)
        else:
            data = dataAndClasses

        if config.get('ENN_enabled'):
            enn.fit(data)
            enn.evaluate()
            enn.metrics(debug, display)

            removeArray = list(set(enn.removed))
            data = dataAndClasses.drop(index=removeArray)
            classes = data.iloc[:, -1]

            ennDataFrame = pd.DataFrame(dataAndClasses.iloc[removeArray, :])
            ennDataFrame['Classes'] = 'Removed'
            ennPlusData = pd.concat([ennDataFrame, data])

            classesWithEnn = ennPlusData.iloc[:, -1]
            classesOld = dataAndClasses.iloc[removeArray, -1]
            classesOld = pd.concat([classesOld, classes])

            fig1 = px.scatter(ennPlusData, x="Data_0", y="Data_1", symbol=classesOld, color=classesWithEnn)
            fig1.show()

        # KFold execution
        i = 0
        for train_indexes, test_indexes in skf.split(data, classes):
            if debug:
                i = i + 1
                print('### KFold: ', i, " ####")
            # From indexes create train data and test data
            trainData = [data.iloc[index, :] for index in train_indexes]
            testData = [data.iloc[index, :] for index in test_indexes]
            knn.fit(trainData, testData)
            knn.evaluate()
        knn.metrics(debug, display)
