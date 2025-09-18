import torch
import torch.nn as nn
import torch.optim as optim


# Initialize the MLP model with the specified three-layer architecture
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.layer2 = nn.Sequential(
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_size, hidden_size),  # Second fully connected layer
            nn.BatchNorm1d(hidden_size),  # Batch normalization
            nn.ReLU()  # ReLU activation
        )
        self.output_layer = nn.Linear(hidden_size, output_size)  # Output layer

    # Forward pass through the network
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


# Train the MLP model
def train_mlp(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()


# Evaluates the MLP model on the test set
def evaluate_mlp(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred
