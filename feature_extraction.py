import torch
import torchvision.models as models


# Load pre-trained ResNet-18 model and remove the final fully connected layer to use it as a feature extractor
def load_resnet():
    resnet18 = models.resnet18(pretrained=True)  # Load pre-trained ResNet-18 model
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  # Remove the last fully connected layer
    resnet18.eval()  # Set model to evaluation mode (for feature extraction)

    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)

    return resnet18, device


# Extracts 512-Dimensional feature vectors for each image in the data_loader using the given model
def extract_feature(data_loader, model, device):
    features = []
    labels = []

    # Disable gradient computation for faster extraction
    with torch.no_grad():
        for images, label in data_loader:
            images = images.to(device)
            output = model(images)  # Forward pass through ResNet-18 to get features

            # Flatten the output to create a 512-dimensional feature vector per image
            output = output.view(output.size(0), -1)
            features.append(output.cpu())
            labels.append(label)

    # Concatenate all features and labels into single tensors
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels
