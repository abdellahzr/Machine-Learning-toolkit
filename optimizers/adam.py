import numpy as np  # Make sure this is included

def adam(X, y, theta, alpha, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam Optimizer implementation.
    
    Parameters:
    - X: Feature matrix (n_samples x n_features)
    - y: Target values (n_samples x 1)
    - theta: Model parameters (n_features x 1)
    - alpha: Learning rate
    - epochs: Number of iterations (epochs)
    - beta1: Exponential decay rate for first moment estimate
    - beta2: Exponential decay rate for second moment estimate
    - epsilon: Small constant to avoid division by zero
    
    Returns:
    - theta: Updated model parameters (n_features x 1)
    """
    n_samples = len(y)
    
    # Initialize moment vectors
    V = np.zeros_like(theta)  # First moment (mean of gradients)
    S = np.zeros_like(theta)  # Second moment (variance of gradients)
    
    t = 0  # Time step
    
    for epoch in range(epochs):
        t += 1
        y_hat = X.dot(theta)  # Prediction
        
        # Calculate error
        error = y_hat - y
        MSE = (error ** 2).mean()
        
        # Gradient calculation
        gradient = (1 / n_samples) * X.T.dot(error)

        # Update biased first moment estimate
        V = beta1 * V + (1 - beta1) * gradient
        
        # Update biased second moment estimate
        S = beta2 * S + (1 - beta2) * (gradient ** 2)
        
        # Correct bias in first and second moment estimates
        V_hat = V / (1 - beta1 ** t)
        S_hat = S / (1 - beta2 ** t)
        
        # Update parameters (theta)
        theta = theta - alpha * V_hat / (np.sqrt(S_hat) + epsilon)
    
    return theta
