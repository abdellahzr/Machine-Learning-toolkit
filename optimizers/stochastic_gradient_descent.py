import numpy as np

def SGD(X, y, theta, alpha, epochs):
    n_samples = X.shape[0]
    
    for epoch in range(epochs):
        for i in range(n_samples):
            xi = X[i:i+1]  # Select one sample
            yi = y[i:i+1]  # Corresponding target value
            
            # Compute prediction (y_hat)
            y_hat = xi.dot(theta)
            
            # Calculate error
            error = y_hat - yi
            
            # Compute the gradient (derivative of the cost function)
            gradient = (xi.T.dot(error))  # Gradient for one data point
            
            # Update theta based on the gradient and learning rate
            theta = theta - alpha * gradient  # Gradient descent update step
    
    return theta
