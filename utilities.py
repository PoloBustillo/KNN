import sys
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
from art import text2art
from sklearn.decomposition import PCA


def euclidean(point, data):
    """Euclidean distance between a point & data"""
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def calculateTwoSignificantAttributes(data, matrixCorrelation, corrT):
    """Display two attributes of data"""
    xAttr = "0"
    yAttr = "1"
    if data.shape[1] != 2:
        values = getIndicesFromCorrelation(matrixCorrelation.abs().to_numpy(), corrT)
        dataFrequencies = []
        for value in values:
            dataFrequencies.append(
                len(set(data.iloc[:, value["i"]])) + len(
                    set(data.iloc[:, value["j"]])))
        xAttr = values[dataFrequencies.index(np.max(dataFrequencies))]["j"]
        yAttr = values[dataFrequencies.index(np.max(dataFrequencies))]["i"]
    return xAttr, yAttr


def getMajorClass(dataPoint, data, dist_metric=euclidean, k=3):
    """Retrieve from data the most k common distance measures"""
    # Calculate the distance without the class label
    distances = dist_metric(dataPoint[:-1], data[:, :-1])
    # Add class label, sort and retrieve first k elements
    # distancesAndClasses = sorted(zip(distances, self.train[:, -1]))[:self.k]
    kClosedClasses = data[np.argsort(distances)[:k], -1]
    # Count the number of occurrences of each class label
    classCounts = Counter(kClosedClasses)
    # Find the most frequent class label
    return classCounts.most_common(1)[0][0]


def printStart(title):
    """Print the data"""
    Art = text2art(title)
    print(Art)


def createHeaders(size=100):
    """Create the data headers to be displayed"""
    columnsArray = np.arange(0, size, 1, dtype=int)
    f = np.vectorize(lambda t: str(t))
    return f(columnsArray)


def getIndicesFromCorrelation(matrix, corrT):
    """Get the indices of the minimum values in a correlation matrix"""
    valuesInT = []
    tri = np.tril_indices_from(matrix)
    for i in range(len(tri[0])):
        if matrix[tri[0][i], tri[1][i]] < corrT:
            valuesInT.append({"i": tri[0][i], "j": tri[1][i]})
    return valuesInT


def apply_PCA(dataAndClasses, display=True):
    dataPCA = dataAndClasses
    classes = dataAndClasses.iloc[:, -1]
    if dataAndClasses.shape[1] > 3 and display:
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(dataAndClasses.values[:, :-1])
        dataPCA = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
        fig = px.scatter(dataPCA, x="PC1", y="PC2", symbol=classes.to_numpy().T, color=classes.to_numpy().T)
        fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
        fig.show()
    return dataPCA


def read_file(path="./Dt1.txt"):
    """Read file and retrieve data
    \n-Drop duplicate rows
    \n-Drop attribute with same value for all elements / constants

    :param path: path to file
    :type path: str
    :returns: dataAndClasses:data with classes , classes: classes for all elements, data: data without classes
    :rtype dataAndClasses: pd.DataFrame
    """
    try:
        dataAndClasses = pd.read_csv(path, sep=",", header=None)
        # dataAndClasses = dataAndClasses.drop_duplicates()
        # Create header for data only for display
        headers = createHeaders(len(dataAndClasses.values[0]) - 1)
        headers = np.append(headers, 'Classes')
        dataAndClasses.columns = headers
        # dataAndClasses = dataAndClasses[[i for i in dataAndClasses if len(set(dataAndClasses[i])) > 1]]
        classes = dataAndClasses.iloc[:, -1]
        data = dataAndClasses.iloc[:, :-1]

        return dataAndClasses, classes, data

    except Exception as error:
        print(f"File {path} does not exist!", file=sys.stderr)
        return
