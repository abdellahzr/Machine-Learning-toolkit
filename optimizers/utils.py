import numpy as np

def generate_data(n_samples=50, n_features=1):
    x = np.random.rand(n_samples, n_features)
    y = 3 * x + 4 + np.random.randn(n_samples, 1) * 0.5
    return x, y
