from data_loading import load_and_limit_cifar10
from feature_extraction import load_resnet, extract_feature
from dimensionality_reduction import apply_pca

from naive_bayes import GaussianNaiveBayes, train_sklearn_naive_bayes, predict_sklearn_naive_bayes
from decision_tree import DecisionTreeClassifier, train_sklearn_decision_tree, predict_sklearn_decision_tree
from multi_layer_perceptron import MLP, train_mlp, evaluate_mlp
from convolutional_neural_network import VGG11, train_vgg11, evaluate_vgg11

from evaluation import evaluate_model, print_evaluation_metrics
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


def main():
    # Step 1: Load and limit the CIFAR-10 dataset
    print("Loading and limiting CIFAR-10 dataset...")
    train_loader, test_loader = load_and_limit_cifar10()
    print("Data loading and limiting complete.")

    # Step 2: Load ResNet-18 model as a feature extractor
    print("Loading ResNet-18 model for feature extraction...")
    resnet18, device = load_resnet()

    # Step 3: Extract features from the dataset
    print("Extracting features from training and test sets...")
    train_features, train_labels = extract_feature(train_loader, resnet18, device)
    test_features, test_labels = extract_feature(test_loader, resnet18, device)
    print("Feature extraction complete.")
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Step 4: Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    train_features_pca, test_features_pca = apply_pca(train_features, test_features)
    print("Dimensionality reduction with PCA complete.")
    print(f"Reduced training features shape: {train_features_pca.shape}")
    print(f"Reduced test features shape: {test_features_pca.shape}")
    # Convert PyTorch tensors to NumPy arrays for compatibility with Naive Bayes
    train_features_np = train_features_pca.numpy()
    train_labels_np = train_labels.numpy()
    test_features_np = test_features_pca.numpy()
    test_labels_np = test_labels.numpy()

    # Step 5: Train and evaluate Custom Naive Bayes Classifier Implementation
    print("Training and evaluating Custom Naive Bayes model...")
    custom_gnb = GaussianNaiveBayes()
    custom_gnb.fit(train_features_np, train_labels_np)
    custom_predictions = custom_gnb.predict(test_features_np)
    custom_metrics = evaluate_model(test_labels_np, custom_predictions)
    print_evaluation_metrics(custom_metrics, model_name="Custom Naive Bayes")

    # Step 6: Train and evaluate Scikit-learn Naive Bayes Classifier Implementation
    print("Training and evaluating Scikit-learn Naive Bayes model...")
    sklearn_gnb = train_sklearn_naive_bayes(train_features_np, train_labels_np)
    sklearn_predictions = predict_sklearn_naive_bayes(sklearn_gnb, test_features_np)
    sklearn_metrics = evaluate_model(test_labels_np, sklearn_predictions)
    print_evaluation_metrics(sklearn_metrics, model_name="Scikit-learn Naive Bayes")

    # Step 7: Train and evaluate Custom Decision Tree Classifier Implementation
    print("Training and evaluating Custom Decision Tree model...")
    custom_tree = DecisionTreeClassifier(max_depth=50)
    custom_tree.fit(train_features_np, train_labels_np)
    custom_tree_predictions = custom_tree.predict(test_features_np)
    custom_tree_metrics = evaluate_model(test_labels_np, custom_tree_predictions)
    print_evaluation_metrics(custom_tree_metrics, model_name="Custom Decision Tree")

    # Step 8: Train and evaluate Scikit-learn Decision Tree Classifier Implementation
    print("Training and evaluating Scikit-learn Decision Tree model...")
    sklearn_tree = train_sklearn_decision_tree(train_features_np, train_labels_np)
    sklearn_tree_predictions = predict_sklearn_decision_tree(sklearn_tree, test_features_np)
    sklearn_tree_metrics = evaluate_model(test_labels_np, sklearn_tree_predictions)
    print_evaluation_metrics(sklearn_tree_metrics, model_name="Scikit-learn Decision Tree")

    # Step 9: Train and evaluate multi-layer Perceptron
    # Convert the features and labels to PyTorch tensors and create DataLoaders
    train_data = TensorDataset(torch.tensor(train_features_pca.numpy()).float(), torch.tensor(train_labels.numpy()))
    test_data = TensorDataset(torch.tensor(test_features_pca.numpy()).float(), torch.tensor(test_labels.numpy()))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=50, hidden_size=512, output_size=10).to(device1)
    criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD with momentum
    print("Training MLP model...")
    train_mlp(model, train_loader, criterion, optimizer, device1, num_epochs=10)
    print("Evaluating MLP model...")
    y_true, y_pred = evaluate_mlp(model, test_loader, device)
    mlp_metrics = evaluate_model(y_true, y_pred)
    print_evaluation_metrics(mlp_metrics, model_name="MLP")

    # Step 10: Training and evaluating the CNN VGG11 model
    # Load CIFAR-10 dataset with batch_size of 64
    print("Loading and limiting CIFAR-10 dataset for CNN")
    train_loader, test_loader = load_and_limit_cifar10(batch_size=64)
    # Initialize model, optimizer, and loss function
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11(num_classes=10).to(device2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("Training VGG11 model...")
    train_vgg11(model, train_loader, optimizer, criterion, device2, num_epochs=10)
    print("Evaluating VGG11 model...")
    y_true, y_pred = evaluate_vgg11(model, test_loader, device)
    vgg11_metrics = evaluate_model(y_true, y_pred)
    print_evaluation_metrics(vgg11_metrics, model_name="VGG11")


if __name__ == "__main__":
    main()
