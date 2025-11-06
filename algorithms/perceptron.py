import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights with Xavier initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = self.relu(self.hidden_layer_input)  # Using ReLU activation
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output_layer_output = self.sigmoid(self.output_layer_input)  # Sigmoid for output layer
        return self.output_layer_output

    def backward(self, X, y, output):
        # Backward pass (Gradient Descent)
        error = y - output
        output_gradient = self.sigmoid_derivative(output)
        hidden_layer_error = error.dot(self.weights_hidden_output.T)
        hidden_layer_gradient = self.relu_derivative(self.hidden_layer_output)

        # Update weights using the gradients
        self.weights_input_hidden += X.T.dot(hidden_layer_error * hidden_layer_gradient) * self.learning_rate
        self.weights_hidden_output += self.hidden_layer_output.T.dot(error * output_gradient) * self.learning_rate

    def fit(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)