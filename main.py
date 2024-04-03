import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ENNSmoothFilter import ENNSmoothFilter
from KNNClassifier import KNNClassifier
import utilities
import plotly.express as px
import numpy as np

path = "./Dt1.txt"
debug = True
display = True
kFolds = 10
corrT = 0.3
skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)
configs = [
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": False, "distance_metric": utilities.euclidean},
    {"k_KNN": 5, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 3, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 5, "PCA_enabled": False, "ENN_enabled": True, "distance_metric": utilities.euclidean}
]

if __name__ == '__main__':

    # Get data from file
    # TODO: indices when remove constant attribute
    dataAndClasses, classes, dataWithoutClasses = utilities.read_file(path)
    data = dataAndClasses
    matrixCorrelation = dataWithoutClasses.corr()
    # By default, selected attributes to display are 0 and 1
    xAttr = "0"
    yAttr = "1"

    # Just display data to get an idea of it
    if debug:
        utilities.printStart('DATA INFO:')
        print(dataAndClasses)

    # Display heatmap for pearson correlation
    # if display:
    #     sns.heatmap(matrixCorrelation)

    for config in configs:
        # Configure initial setup for KNN and ENN
        knn = KNNClassifier(config.get('k_KNN'), config.get('distance_metric'))
        enn = ENNSmoothFilter(config.get('k_ENN'), config.get('distance_metric'))
        # Print current configuration
        utilities.printStart('KNN:' + str(config.get('k_KNN')))
        print(config)

        if config.get('ENN_enabled'):
            enn.fit(data)
            enn.evaluate()
            # Recalculate new data and classes
            data = pd.DataFrame(enn.resultSet)
            classes = data.iloc[:, -1]
            # Calculate two attributes for plotting using enn_data
            xAttr, yAttr = utilities.calculateTwoSignificantAttributes(data, matrixCorrelation, corrT)
            enn.metrics(xAttr, yAttr, debug, display)
        else:
            # Calculate two attributes for plotting using whole data
            xAttr, yAttr = utilities.calculateTwoSignificantAttributes(data, matrixCorrelation, corrT)

        # Display data without classification
        if display:
            fig = px.scatter(data, x=str(xAttr), y=str(yAttr), symbol=classes, color=classes)
            fig.update_layout(
                title="Data without classification",
                title_font=dict(size=20,
                                color='green',
                                family='Arial'),
                coloraxis_colorbar=dict(yanchor="top",
                                        y=1, x=0,
                                        ticks="outside")
            )
            fig.show()
            plt.figure(figsize=(12, 8))

        # KFold execution
        i = 0
        for train_indexes, test_indexes in skf.split(data, classes):
            if debug:
                i = i + 1
                print('### KFold: ', i, " ####")
            # From indexes create train data and test data
            trainData = [data.iloc[index, :] for index in train_indexes]
            testData = [data.iloc[index, :] for index in test_indexes]
            knn.fit(trainData, testData, test_indexes)
            knn.evaluate()

        # TODO: fix indices failures
        knn.metrics(debug, display)
        print(np.array(knn.plot)[:, 0])
        df = pd.DataFrame(data, index=np.array(knn.plot)[:, 0])
        fig1 = px.scatter(df, x=str(xAttr), y=str(yAttr), color=np.array(knn.plot)[:, 1], symbol=df.iloc[:, -1],
                          color_continuous_scale=["orange", "red", "green", "blue", "purple"])
        fig1.update_layout(
            title="KNN Classifier",
            title_font=dict(size=20,
                            color='green',
                            family='Arial'),
            coloraxis_colorbar=dict(yanchor="top",
                                    y=1, x=0,
                                    ticks="outside"),

        )
        fig1.show()
