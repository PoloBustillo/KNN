import sys
from collections import Counter

import numpy as np
import plotly.express as px


def euclidean(point, data):
    """Euclidean distance between a point & data"""
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def displayTwoAttributes(firstColumn, secondColumn, data, classes):
    """Display two attributes of data"""
    # print(data[:, firstColumn])
    # print(data[:, secondColumn])
    # two_vars = [data[:, firstColumn], data[:, secondColumn]]
    # print(two_vars)
    # print(np.shape(two_vars))
    fig = px.scatter(x=data.values[:, firstColumn], y=data.values[:, secondColumn], color=classes)
    # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
    fig.show()


def getMajorClass(dataPoint, data, dist_metric=euclidean, k=3):
    """Retrieve from data the the most k common distance measures"""
    # Calculate the distance without the class label
    distances = dist_metric(dataPoint[:-1], data[:, :-1])
    # Add class label, sort and retrieve first k elements
    # distancesAndClasses = sorted(zip(distances, self.train[:, -1]))[:self.k]
    kClosedClasses = data[np.argsort(distances)[:k], -1]
    # Count the number of occurrences of each class label
    classCounts = Counter(kClosedClasses)
    # Find the most frequent class label
    return classCounts.most_common(1)[0][0]


def createHeaders(prefix='X', size=100):
    """Create the data headers to be displayed"""
    columnsArray = np.arange(0, size, 1, dtype=int)
    f = np.vectorize(lambda t: prefix + str(t))
    return f(columnsArray)


def read_file(path="./Dt1.txt"):
    """Read file and retrieve data"""
    data = list()
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                xi = line.rstrip().split(",")
                data.append(xi)
        return data
    except FileNotFoundError as e:
        print(f"File {path} does not exist!", file=sys.stderr)
        return
