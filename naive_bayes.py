import numpy as np


# Gaussian Naive Bayes Algorithm using basic Numpy
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    # Fit the model to the training data X and labels y
    def fit(self, X, y):
        # Identify unique classes and initialize parameters
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # Initialize mean, variance, and priors dictionaries
        self.mean = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.var = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)

        # Calculate mean, variance, and priors for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  # Filter samples belonging to class c
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]  # Prior is the fraction of samples in this class

    # Compute the gaussian probability density function for each feature
    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    # Predict class labels for samples in X
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    # Predict the class for a single sample
    def _predict(self, x):
        posteriors = []

        # Calculate posterior for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])  # Log prior probability of the class
            conditional = np.sum(np.log(self.gaussian_density(idx, x)))  # Log likelihood of the features
            posterior = prior + conditional  # Combine prior and likelihood
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self.classes[np.argmax(posteriors)]


# Naive Bayes using Scikit’s Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB


def train_sklearn_naive_bayes(training_features, training_labels):
    model = GaussianNB()  # Build a Gaussian Classifier
    model.fit(training_features, training_labels)  # Train the model
    return model


def predict_sklearn_naive_bayes(model, test_features):
    return model.predict(test_features)  # Predict the output
