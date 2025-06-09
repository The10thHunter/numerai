# numerai_vector_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# --- Config ---
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_VAL_SPLIT = 0.8
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Dataset ---
class NumeraiVectorDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_cols: list, target_col: str):
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(dataframe[feature_cols])
        self.targets = dataframe[target_col].values.astype(np.float32)

        # Normalize target to [0, pi] for angle representation
        self.angles = np.clip(self.targets, 0, 1) * np.pi
        self.unit_vectors = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.unit_vectors[idx], dtype=torch.float32)
        )

# --- Model ---
class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Output is 2D vector
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)  # Normalize for angle comparison

# --- Training and Evaluation ---
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_x)
        #loss = criterion(outputs, batch_y)
        target = torch.ones(batch_x.size(0), device=DEVICE)
        loss = criterion(outputs, batch_y, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            #loss = criterion(outputs, batch_y)
            target = torch.ones(batch_x.size(0), device=DEVICE)
            loss = criterion(outputs, batch_y, target) 
            total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(dataloader.dataset)

# --- Main Pipeline ---
def main():
    parquet_path = "/Users/prashant/numerai/v5.0/train.parquet"
    df = pd.read_parquet(parquet_path)
    df = df.sample(n=5000, random_state=SEED)  # or df.head(5000)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    target_col = "target"

    dataset = NumeraiVectorDataset(df, feature_cols, target_col)
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VectorEncoder(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CosineEmbeddingLoss()

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "vector_encoder.pt")
    print("Model saved to vector_encoder.pt")

if __name__ == "__main__":
    main()
