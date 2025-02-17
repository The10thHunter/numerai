import torch
from torch.utils.data import DataLoader
from model import torchMod
from data import Dataset
import torch.optim as optim
import pandas as pd

# Load the training dataset
train_data = torch.load("trialset_1.pt")
train_features, train_labels = train_data["feats"], train_data["labels"]

# Load the validation dataset directly from Parquet
val_data = pd.read_parquet("../v5.0/validation.parquet")

# Select the same feature subset used for training
selected_features = train_data["feats"].shape[1]  # Ensure feature count matches
val_features = torch.tensor(val_data.iloc[:, :selected_features].to_numpy(), dtype=torch.float32)
val_labels = torch.tensor(val_data["target"].to_numpy(), dtype=torch.float32)

# Create Dataset objects
train_dataset = Dataset(train_features, train_labels)
val_dataset = Dataset(val_features, val_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchMod().to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, optimizer, epochs=10, device="cpu"):
    """
    Train function for the model.

    Args:
        model (torchMod): The PyTorch model.
        train_loader (DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        epochs (int): Number of training epochs.
        device (str): "cpu" or "cuda" for training hardware.

    Returns:
        None
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            predictions = model.forwardprop(batch_features).squeeze()

            # Compute loss
            loss = model.lossFn(predictions, batch_labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

def test(model, test_loader, device="cpu"):
    """
    Evaluate the model on a test dataset.

    Args:
        model (torchMod): The trained model.
        test_loader (DataLoader): DataLoader for the validation set.
        device (str): "cpu" or "cuda" for evaluation.

    Returns:
        float: Mean Squared Error on the test set.
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            predictions = model.forwardprop(batch_features).squeeze()

            # Compute loss
            loss = model.lossFn(predictions, batch_labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Validation MSE Loss: {avg_loss:.6f}")
    return avg_loss

# Train the model
train(model, train_loader, optimizer, epochs=10, device=device)

# Evaluate the model on validation set
test(model, val_loader, device=device)
