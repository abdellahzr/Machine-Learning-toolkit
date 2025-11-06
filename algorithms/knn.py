import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, mode='classification'):
        self.k = k  # Number of neighbors
        self.mode = mode  # 'classification' or 'regression'

    def fit(self, X_train, y_train):
        """
        Store the training data.
        :param X_train: Features of the training dataset (2D array)
        :param y_train: Labels of the training dataset (1D array)
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict the labels for the test dataset.
        :param X_test: Features of the test dataset (2D array)
        :return: Predicted labels (1D array)
        """
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predict the label for a single test sample.
        :param x: Test sample (1D array)
        :return: Predicted label
        """
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # Sort the distances and get the indices of the k closest points
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Classification: return the most common label among the k nearest neighbors
        if self.mode == 'classification':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        
        # Regression: return the mean of the k nearest labels
        elif self.mode == 'regression':
            return np.mean(k_nearest_labels)

