import numpy as np
import matplotlib.pyplot as plt
from algorithms.linear_regression import LinearRegression
from algorithms.non_linear_regression import NonLinearRegression
from algorithms.utils import generate_data_regression
from algorithms.utils import generate_data_Classification
from algorithms.linear_classifier import LinearClassifier
from algorithms.non_linear_classifier import NonLinearClassifier
from algorithms.decision_tree import DecisionTree  
from algorithms.perceptron import MLP  
from algorithms.knn import KNN
from algorithms.svm import SVM 
from optimizers import GD, mini_batch_GD, adam, SGD


def main():
    
    print("Generating data...")
    x, y = generate_data_regression(n_samples=100)

    
    model_choice = input("Choose the model (1: Linear Regression, 2: Non-Linear Regression, 3: Linear Classifier, 4: Non-Linear Classifier, 5: Decision Tree, 6: MLP, 7: KNN, 8: SVM): ")
    
    optimizer_choice = input("Choose optimizer (1: GD, 2: MiniBatch, 3: Adam): ")
    if optimizer_choice == '1':
        optimizer = GD
    elif optimizer_choice == '2':
        optimizer = mini_batch_GD
    elif optimizer_choice == '3':
        optimizer = adam
    else:
        print("Invalid optimizer choice! Defaulting to GD.")
        optimizer = GD

    
    if model_choice == '1':
        print(f"Training Linear Regression model with {optimizer.__name__} optimizer...")
        model = LinearRegression(alpha=0.00001, epochs=5000, optimizer=optimizer)
        model.fit(x, y)

        print("Making predictions...")
        y_pred = model.predict(x)

        
        print("Plotting results...")
        plt.scatter(x, y, color='blue', label='Original Data')
        plt.plot(x, y_pred, color='red', label='Fitted Line')
        plt.legend()
        plt.title(f'Linear Regression Fit with {optimizer.__name__}')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()

    elif model_choice == '2':
        print(f"Training Non-Linear Regression model with {optimizer.__name__} optimizer...")
        model = NonLinearRegression(alpha=0.001, epochs=10000, optimizer=optimizer)
        model.fit(x, y)

        print("Making predictions...")
        y_pred = model.predict(x)
        x_sorted, y_pred_sorted = zip(*sorted(zip(x.flatten(), y_pred.flatten())))

        
        print("Plotting results...")
        plt.scatter(x, y, color='blue', label='Original Data')
        plt.plot(x_sorted, y_pred_sorted, color='red', label='Fitted Curve')
        plt.legend()
        plt.title(f'Non-Linear Regression Fit with {optimizer.__name__}')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()

    elif model_choice == '3':
        print(f"Training Linear Classifier with {optimizer.__name__} optimizer...")
        classifier = LinearClassifier(alpha=0.01, epochs=10000)
        classifier.fit(x, y)

        print("Making predictions...")
        y_pred = classifier.predict(x)

        
        plt.scatter(x, y, color='blue', label='Original Data')

        
        plt.scatter(x, y_pred, color='red', label='Predicted Data')

        #
        x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        X_bias = np.c_[np.ones((x_range.shape[0], 1)), x_range]
        
        
        decision_boundary = X_bias.dot(classifier.theta)

        
        plt.plot(x_range, decision_boundary, color='green', label='Decision Boundary')

        plt.legend()
        plt.title('Linear Classifier')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()

    elif model_choice == '4':
        print(f"Training NonLinear Classifier (Logistic Regression) with {optimizer.__name__} optimizer...")
        classifier = NonLinearClassifier(alpha=0.01, epochs=10000)
        classifier.fit(x, y)

        print("Making predictions...")
        y_pred = classifier.predict(x)

        
        plt.scatter(x, y, color='blue', label='Original Data')

        
        plt.scatter(x, y_pred, color='red', label='Predicted Data')

        
        x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        X_bias = np.c_[np.ones((x_range.shape[0], 1)), x_range]
        
        
        decision_boundary = classifier.sigmoid(X_bias.dot(classifier.theta))
        
        
        plt.plot(x_range, decision_boundary, color='green', label='Decision Boundary')

        plt.legend()
        plt.title('NonLinear Classifier with Logistic Regression')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()

    elif model_choice == '5':
        print(f"Training Decision Tree with {optimizer.__name__} optimizer...")
        tree = DecisionTree(max_depth=50)
        tree.fit(x, y)

        print("Making predictions...")
        y_pred = tree.predict(x)

        
        plt.scatter(x, y, color='blue', label='Original Data')
        plt.scatter(x, y_pred, color='red', label='Predicted Data')
        plt.legend()
        plt.title('Decision Tree Classifier/Regressor')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()

    elif model_choice == '6':
        print(f"Training MLP (Multi-Layer Perceptron) with {optimizer.__name__} optimizer...")
        mlp = MLP(input_size=1, hidden_size=5, output_size=1, learning_rate=0.01, epochs=10000)
        mlp.fit(x, y)

        print("Making predictions...")
        y_pred = mlp.predict(x)

        
        plt.scatter(x, y, color='blue', label='Original Data')
        plt.scatter(x, y_pred, color='red', label='Predicted Data')
        plt.legend()
        plt.title('Multi-Layer Perceptron')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()
    elif model_choice == '7':
        print(f"Training KNN with {optimizer.__name__} optimizer...")
        knn = KNN(k=3, mode='regression')  # Change mode to 'classification' for classification task
        knn.fit(x, y)

        print("Making predictions...")
        y_pred = knn.predict(x)

        plt.scatter(x, y, color='blue', label='Original Data')
        plt.scatter(x, y_pred, color='red', label='Predicted Data')
        plt.legend()
        plt.title('KNN Regressor')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()
    elif model_choice == '8':
        print(f"Training SVM with {optimizer.__name__} optimizer...")
        svm = SVM(learning_rate=0.001, epochs=10000)
        svm.fit(x, y)

        print("Making predictions...")
        y_pred = svm.predict(x)

        plt.scatter(x, y, color='blue', label='Original Data')
        plt.scatter(x, y_pred, color='red', label='Predicted Data')
        plt.legend()
        plt.title('Support Vector Machine (SVM)')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.show()    
    else:
        print("Invalid model choice!")

if __name__ == "__main__":
    main()
