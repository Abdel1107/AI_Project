import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# transformation for resizing and normalizing the images
def get_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# load the CIFAR-10 dataset with specific transformations and return the full training and test datasets
def load_cifar10_dataset():
    transform = get_transform()
    training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return training_dataset, test_dataset


# limit the dataset to 500 training and 100 test images per class
def limit_dataset(dataset, num_per_class):
    class_counts = {i: 0 for i in range(10)}
    limited_data = []

    for img, label in dataset:
        if class_counts[label] < num_per_class:
            limited_data.append((img, label))
            class_counts[label] += 1
        if all(count >= num_per_class for count in class_counts.values()):
            break;  # Stop if we have enough images for each class
    return limited_data


def load_and_limit_cifar10(num_training_data_per_class=500, num_test_data_per_class=100, batch_size=64):
    training_dataset, test_dataset = load_cifar10_dataset()  # load full datasets

    # Limit dataset
    limited_training_data = limit_dataset(training_dataset, num_training_data_per_class)
    limited_test_data = limit_dataset(test_dataset, num_test_data_per_class)

    # Convert to DataLoaders
    training_loader = torch.utils.data.DataLoader(limited_training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(limited_test_data, batch_size=batch_size, shuffle=False)

    return training_loader, test_loader
