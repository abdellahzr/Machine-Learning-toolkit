
import numpy as np
class SVM:
    def __init__(self, learning_rate=0.0001, epochs=10000, lambda_param=0.0001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent to optimize the SVM objective
        for epoch in range(self.epochs):
            # Compute the margin (decision boundary) output
            margin_output = np.dot(X, self.weights) + self.bias
            # Compute the hinge loss for each sample
            hinge_loss = np.maximum(0, 1 - y * margin_output)  # Hinge loss per sample
            # Total loss is the hinge loss + regularization term
            loss = 0.5 * np.dot(self.weights, self.weights) + self.lambda_param * np.sum(hinge_loss)

            # Debugging: Print loss every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # Compute the gradients
            dw = np.zeros_like(self.weights)
            db = 0

            # Loop through each sample to calculate gradients
            for i in range(num_samples):
                if hinge_loss[i] > 0:  # Misclassified sample
                    dw += -y[i] * X[i]  # Update weights
                    db += -y[i]          # Update bias

            # Regularization gradient
            dw += self.lambda_param * self.weights

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw / num_samples
            self.bias -= self.learning_rate * db / num_samples

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

