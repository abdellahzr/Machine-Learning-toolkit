import numpy as np
class LinearRegression:
    def __init__(self, alpha=0.001, epochs=5000, optimizer=None):
        self.alpha = alpha
        self.epochs = epochs
        self.optimizer = optimizer  # Store the selected optimizer
        self.theta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.random.rand(n_features + 1, 1)  # Initialize theta
        X_new = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        
         # Use the selected optimizer
        if self.optimizer is not None:
            self.theta = self.optimizer(X_new, y, self.theta, self.alpha, self.epochs)
        else:
            raise ValueError("Optimizer not specified")

    def predict(self, X):
        n_samples = X.shape[0]
        X_new = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
        return X_new.dot(self.theta)
