import numpy as np

def GD(X, y, theta, alpha, epochs):
    """
    Performs Gradient Descent optimization to fit the parameters (theta)
    to the data using a linear model.
    """
    n_samples = X.shape[0]

    for epoch in range(epochs):
        y_hat = X.dot(theta)  # Model prediction
        error = y_hat - y
        gradient = (1 / n_samples) * X.T.dot(error)  # Compute gradient
        theta = theta - alpha * gradient  # Update theta
        
    return theta