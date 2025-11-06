import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini(self, y):
        # Gini impurity calculation
        class_probs = [np.sum(y == c) / len(y) for c in np.unique(y)]
        return 1 - sum(p**2 for p in class_probs)
    
    def best_split(self, X, y):
        # Find the best feature and threshold to split
        best_gini = float('inf')
        best_split = None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                
                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)
                gini_split = (len(left_y) * gini_left + len(right_y) * gini_right) / len(y)
                
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = (feature_index, threshold)
        
        return best_split

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return {'label': np.unique(y)[0]}
        
        if depth == self.max_depth or len(y) < self.min_samples_split:
            return {'label': np.argmax(np.bincount(y))}

        feature_index, threshold = self.best_split(X, y)
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': feature_index, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        def traverse_tree(x, node):
            if 'label' in node:
                return node['label']
            if x[node['feature']] <= node['threshold']:
                return traverse_tree(x, node['left'])
            else:
                return traverse_tree(x, node['right'])

        return np.array([traverse_tree(x, self.tree) for x in X])

