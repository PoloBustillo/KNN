import sys
from collections import Counter
from math import sqrt

import numpy as np


def euclidean(point, data):
    """Euclidean distance between a point  & data"""
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def retrieveKPoints(dataPoint, data, dist_metric=euclidean, k=3):
    # Calculate the distance without the class label
    distances = dist_metric(dataPoint[:-1], data[:, :-1])
    # Add class label, sort and retrieve first k elements
    # distancesAndClasses = sorted(zip(distances, self.train[:, -1]))[:self.k]
    kClosedClasses = data[np.argsort(distances)[:k], -1]
    # Count the number of occurrences of each class label
    classCounts = Counter(kClosedClasses)
    # Find the most frequent class label
    return classCounts.most_common(1)[0][0]


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
