import numpy as np
from optimizers import GD

def add_polynomial_features(X, degree=2):
    """
    Adds polynomial features to the input matrix X up to the given degree.
    For example, if degree=2, it will add X^2 terms to the features.
    """
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.concatenate((X_poly, np.power(X, d)), axis=1)
    return X_poly

import numpy as np

class NonLinearRegression:
    def __init__(self, alpha=0.001, epochs=5000, optimizer=None):
        self.alpha = alpha
        self.epochs = epochs
        self.optimizer = optimizer
        self.theta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Create a feature transformation to include X^2 (quadratic term)
        X_transformed = np.concatenate((X, X ** 2), axis=1)  # Add X^2 as a new feature
        self.theta = np.random.randn(X_transformed.shape[1] + 1, 1)  # Initialize theta
        
        # Add a column of ones for the bias term
        X_transformed = np.concatenate((np.ones((n_samples, 1)), X_transformed), axis=1)
        
        # Use the selected optimizer
        if self.optimizer is not None:
            self.theta = self.optimizer(X_transformed, y, self.theta, self.alpha, self.epochs)
        else:
            raise ValueError("Optimizer not specified")

    def predict(self, X):
        n_samples = X.shape[0]
        # Create the same transformation as during training (X^2)
        X_transformed = np.concatenate((X, X ** 2), axis=1)
        
        # Add a column of ones for the bias term
        X_transformed = np.concatenate((np.ones((n_samples, 1)), X_transformed), axis=1)
        
        # Predict using the learned model
        return X_transformed.dot(self.theta)
