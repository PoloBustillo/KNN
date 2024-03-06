from array import array

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from ENNSmoothFilter import ENNSmoothFilter
from KNNClassifier import KNNClassifier
import utilities

path = "./Dt1.txt"
debug = False
display = False
# KFold stratified
kFolds = 10
skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)

configs = [
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": False, "distance_metric": utilities.euclidean},
    {"k_KNN": 1, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 3, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean}
]

if __name__ == '__main__':
    # Read file and convert to numeric
    dataAndClasses = pd.read_csv(path, sep=",", header=None)
    # Drop duplicate rows
    dataAndClasses = dataAndClasses.drop_duplicates()
    # TODO: ask for this step
    # Drop attribute with same value for all elements / constants
    dataAndClasses = dataAndClasses[[i for i in dataAndClasses if len(set(dataAndClasses[i])) > 1]]
    # Get classes
    classes = dataAndClasses.values[:, -1].T

    # Apply PCA only to visualize data if PCA flag is enabled then PCA components are used
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(dataAndClasses.values[:, :-1])
    dataPCA = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    # Get correlation matrix (pearson) to get value closer to 0
    matrixCorrelation = dataAndClasses.corr().dropna(how='all', axis=1).dropna(how='all')
    # matrixCorrelation = dataAndClasses.corr()
    # print(matrixCorrelation.values[64, 384])
    matrix = matrixCorrelation.abs().to_numpy()
    ij_min = np.unravel_index(matrix.argmin(), matrix.shape)
    print(matrix[ij_min])
    print(ij_min)
    utilities.displayTwoAttributes(ij_min[0], ij_min[1], dataAndClasses, classes)

    if debug:
        # Create header for data only for display
        headers = utilities.createHeaders('Data_', len(dataAndClasses.values[0]) - 1)
        headers = np.append(headers, 'Classes')
        dataAndClasses.columns = headers
        print(dataAndClasses)
        print(matrixCorrelation)
    if display:
        # HEATMAP correlation
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrixCorrelation)
        # PCA
        fig = px.scatter(dataPCA, x="PC1", y="PC2", symbol=classes, color=classes)
        fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
        fig.show()

    for config in configs:
        print(config)
        if config.get('PCA_enabled'):
            dataPCA['Classes'] = dataAndClasses.values[:, -1].T
            data = dataPCA.values
            if debug:
                print(dataPCA)
        else:
            data = dataAndClasses.values

        # Configure initial setup for KNN and ENN
        knn = KNNClassifier(config.get('k_KNN'), config.get('distance_metric'))
        enn = ENNSmoothFilter(config.get('k_ENN'), config.get('distance_metric'))

        # KFold execution
        i = 0
        for train_indexes, test_indexes in skf.split(data, classes):
            if debug:
                i = i + 1
                print('### KFold: ', i, " ####")
            # From indexes create train data and test data
            trainData = [data[index, :] for index in train_indexes]
            testData = [data[index, :] for index in test_indexes]

            # ENN use train data only to clean up outliers and smooth frontiers
            if config.get('ENN_enabled'):
                enn.fit(trainData)
                enn.evaluate()
                enn.metrics(debug, display)
                trainData = enn.resultSet
            knn.fit(trainData, testData)
            knn.evaluate()
        knn.metrics(debug, display)
