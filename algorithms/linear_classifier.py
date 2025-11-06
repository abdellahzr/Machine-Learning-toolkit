import numpy as np

class LinearClassifier:
    def __init__(self, alpha=0.01, epochs=10000):
        self.alpha = alpha
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        # Reshape y to be a 1D array (if it's not already)
        y = y.flatten()  # Shape: (50,)

        # Check the shape of X and y
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        # Add bias term (column of ones) to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Shape: (50, 2)
        
        # Initialize theta (weights + bias) to zeros
        self.theta = np.zeros(X_bias.shape[1])  # Shape: (2,)
        
        for epoch in range(self.epochs):
            # Compute predictions (linear output)
            predictions = X_bias.dot(self.theta)  # Shape: (50,)
            
            # Compute the error
            error = predictions - y  # Shape: (50,)
            
            # Compute the gradient
            gradient = X_bias.T.dot(error) / X.shape[0]  # Shape: (2,)
            
            # Update the parameters (theta)
            self.theta -= self.alpha * gradient  # Shape: (2,)

            # Optionally, print the loss or other metrics for monitoring
            if epoch % 1000 == 0:
                loss = np.mean(error**2)  # Mean squared error
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        predictions = X_bias.dot(self.theta)  # Shape: (50,)
        return (predictions >= 0).astype(int)  # Return 0 or 1 for classification based on threshold at 0

