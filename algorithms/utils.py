import numpy as np

def generate_data_regression(n_samples=100, noise_level=2, seed=42):
    """
    Generates non-linear data with a quadratic relationship and added noise.
    
    Parameters:
    - n_samples: The number of data points to generate.
    - noise_level: The standard deviation of the noise to add to the target values.
    - seed: Random seed for reproducibility.
    
    Returns:
    - X: Feature matrix of shape (n_samples, 1).
    - y: Target vector of shape (n_samples, 1).
    """
    np.random.seed(seed)  # For reproducibility
    X = 2 * np.random.rand(n_samples, 1)  # Random values for X in range [0, 2]
    y = 5 * (X ** 2) + np.random.randn(n_samples, 1) * noise_level  # Quadratic relationship + noise
    return X, y


def generate_data_classification(n_samples=100, noise_level=0.5):
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1) - 1  # Random data in range [-1, 1]
    y = (X > 0).astype(int)  # Binary classification (0 or 1)
    return X, y

def generate_data_Classification(n_samples=100, noise_level=0.5):
    """
    Generates a simple 1D binary classification dataset with a noisy threshold.
    
    Args:
        n_samples (int): The number of data points to generate.
        noise_level (float): The amount of noise to add. Higher values introduce more noise to the class boundaries.
    
    Returns:
        X (numpy array): Feature matrix of shape (n_samples, 1).
        y (numpy array): Target vector of shape (n_samples,).
    """
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1) - 1  # Random data in range [-1, 1]

    # Add noise to the decision boundary at 0
    noise = np.random.randn(n_samples, 1) * noise_level

    # Assign class 0 or class 1 based on the noisy threshold
    y = (X + noise > 0).astype(int)
    
    return X, y