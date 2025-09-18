import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier


class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini  # Gini's impurity of the node
        self.num_samples = num_samples  # Number of samples in the node
        self.num_samples_per_class = num_samples_per_class  # Sample counts per class
        self.predicted_class = predicted_class  # Predicted class of the node
        self.feature_index = 0  # Index of the feature to split on
        self.threshold = 0  # Threshold for splitting
        self.left = None  # Left child
        self.right = None  # Right child


class DecisionTreeClassifier:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    # Building the decision tree
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    # Recursive function to grow the decision tree
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(10)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Stop if max depth is reached, all samples are of one class, or we have very few samples
        if depth >= self.max_depth or node.gini == 0 or y.size < 5:
            return node

        idx, thr = self._best_split(X, y)
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]

            # Avoid splits that don't divide the data
            if len(y_left) == 0 or len(y_right) == 0:
                return node

            node.feature_index = idx
            node.threshold = thr
            node.left = self._grow_tree(X_left, y_left, depth + 1)
            node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    # Calculate Gini's impurity for labels y
    def _gini(self, y):
        m = y.size
        if m == 0:
            return 0
        counts = np.bincount(y)
        return 1.0 - np.sum((counts / m) ** 2)

    # Find the best split for a node
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        # Initial Gini impurity before split
        best_gini = self._gini(y)
        best_idx, best_thr = None, None

        num_classes = 10  # Assuming CIFAR-10 with 10 classes
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = np.zeros(num_classes, dtype=int)
            num_right = np.bincount(classes, minlength=num_classes)
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # Skip splits with empty left or right nodes
                if i < 1 or i >= m - 1:
                    continue

                gini_left = 1.0 - np.sum((num_left / i) ** 2)
                gini_right = 1.0 - np.sum((num_right / (m - i)) ** 2)
                gini = (i * gini_left + (m - i) * gini_right) / m

                # Only update best split if gini improves
                if thresholds[i] != thresholds[i - 1] and gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    # Predicting classes for samples in X
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    # Predict class for a single sample input
    def _predict(self, inputs):
        node = self.root
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


# Decision Tree Classifier using sklearn DecisionTreeClassifier
def train_sklearn_decision_tree(training_features, training_labels, max_depth=50):
    model = SklearnDecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=42)
    model.fit(training_features, training_labels)
    return model


# Use the trained decision tree classifier to predict class labels
def predict_sklearn_decision_tree(model, test_features):
    return model.predict(test_features)
