import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

from ENNSmoothFilter import ENNSmoothFilter
from KNNClassifier import KNNClassifier
import utilities


path = "./Dt1.txt"
kFolds = 10

if __name__ == '__main__':

    # Read file and convert to float array
    dataAndClasses = pd.read_csv(path, sep=",", header=None)
    # dataAndClasses = np.array(utilities.read_file(path), dtype=float)

    # Create column tag only for display
    columnsArray = np.arange(0, len(dataAndClasses.values[0]) - 1, 1, dtype=int)
    f = np.vectorize(lambda t: 'X' + str(t))
    columnsTags = f(columnsArray)
    columnsTags = np.append(columnsTags, 'Classes')
    dataAndClasses.columns = columnsTags
    print(dataAndClasses)

    data = dataAndClasses.values[:, :-1]
    classes = dataAndClasses.values[:, -1].T

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(dataAndClasses.values[:, :-1])
    data = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    print(data)

    fig = px.scatter(data, x="PC1", y="PC2", symbol=classes, color=classes)
    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
    fig.show()
    # KFold stratified
    skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)
    # Configure initial setup for KNN
    knn = KNNClassifier(1, utilities.euclidean)
    enn = ENNSmoothFilter(1, utilities.euclidean)
    i = 0
    for train_indexes, test_indexes in skf.split(data, classes):
        i = i + 1
        print('### KFold: ', i, " ####")

        # Create trainData
        trainData = [dataAndClasses.values[index] for index in train_indexes]
        testData = [dataAndClasses.values[index] for index in test_indexes]
        enn.fit(trainData)
        enn.evaluate()
        enn.metrics()
        knn.fit(enn.resultSet, testData)
        knn.evaluate()
    knn.metrics()
