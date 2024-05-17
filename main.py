from sklearn.model_selection import StratifiedKFold
from ENNSmoothFilter import ENNSmoothFilter
from KNNClassifier import KNNClassifier
import pandas as pd
import utilities

# path = "./Dt1.txt"
# path = "./iris.data.txt"
path = "./sintetico.txt"
debug = True
display = True
kFolds = 10
corrT = 0.3
skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=1)
configs = [
    {"k_KNN": 3, "k_ENN": 3, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True,
     "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean},
    # {"k_KNN": 3, "k_ENN": 1, "PCA_enabled": False, "ENN_enabled": True, "FindData_enabled": True, "distance_metric": utilities.euclidean}
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

        if config.get('FindData_enabled'):
            # Calculate two attributes for plotting using data
            xAttr, yAttr = utilities.calculateTwoSignificantAttributes(data, matrixCorrelation, corrT)
            # Display data without classification
            utilities.plotData(display, data, xAttr, yAttr, classes)
        if config.get('ENN_enabled'):
            # Recalculate new data and classes
            enn.fit(dataAndClasses)
            enn.evaluate()
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
            enn.metrics(xAttr, yAttr, debug, display)
        knn.metrics(debug, display, xAttr, yAttr)
