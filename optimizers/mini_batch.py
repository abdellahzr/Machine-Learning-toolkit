def mini_batch_GD(X, y, theta, alpha, epochs, batch_size):
    """
    Mini-Batch Gradient Descent (MBGD) implementation.
    
    Parameters:
    - X: Feature matrix (n_samples x n_features)
    - y: Target values (n_samples x 1)
    - theta: Model parameters (n_features x 1)
    - alpha: Learning rate
    - epochs: Number of iterations (epochs)
    - batch_size: Size of the mini-batch
    
    Returns:
    - theta: Updated model parameters (n_features x 1)
    """
    n_samples = len(y)
    
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            xi = X[i:i + batch_size, :]  # Mini-batch of features
            yi = y[i:i + batch_size, :]  # Corresponding target values

            # Make prediction
            y_hat = xi.dot(theta)

            # Calculate error
            error = y_hat - yi

            # Gradient calculation
            gradient = (1 / batch_size) * xi.T.dot(error)

            # Update parameters (theta)
            theta = theta - alpha * gradient
        
        # Calculate Mean Squared Error (MSE) at the end of each epoch
        y_hat_batch = X.dot(theta)
        error = y_hat_batch - y
        MSE = (error ** 2).mean()

    return theta
