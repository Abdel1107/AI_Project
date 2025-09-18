from sklearn.decomposition import PCA
import torch


# Apply PCA to reduce the dimensionality of the feature vectors
def apply_pca(training_features, test_features, n_components=50):
    # Convert PyTorch tensors to NumPy arrays for compatibility with PCA
    train_features_np = training_features.numpy()
    test_features_np = test_features.numpy()

    # Initialize PCA and fit it on the training features
    pca = PCA(n_components=n_components)
    train_features_pca = pca.fit_transform(train_features_np)

    # Transform the test features using the same PCA model
    test_features_pca = pca.transform(test_features_np)

    # Convert the reduced features back to PyTorch tensors
    train_features_pca = torch.tensor(train_features_pca)
    test_features_pca = torch.tensor(test_features_pca)

    return train_features_pca, test_features_pca
