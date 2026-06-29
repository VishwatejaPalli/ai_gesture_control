"""
ai/train_model.py

Implements the training pipeline for the gesture recognition network.
Loads a gesture landmark CSV dataset, maps labels, splits the dataset, 
manages model optimization via Adam and cross-entropy loss, and computes 
evaluation metrics.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Dynamic Path Resolution: Ensures the local 'model.py' can be imported easily
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import the model architecture
from model import GestureNet


class GestureDataset(Dataset):
    """
    Custom PyTorch Dataset class to parse and load hand landmark inputs.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features (np.ndarray): Shape (N, 63) array of normalized coordinates.
            labels (np.ndarray): Shape (N,) array of integer category labels.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes classification performance metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        tuple: (accuracy, precision, recall, f1, confusion_matrix)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, cm


def train(model, dataloader, criterion, optimizer, device) -> tuple:
    """
    Runs a single training epoch.

    Returns:
        tuple: (mean epoch loss, mean epoch accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics accumulation
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device) -> tuple:
    """
    Runs evaluation on the validation set.

    Returns:
        tuple: (loss, accuracy, precision, recall, f1_score, confusion_matrix)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute evaluation metrics
    acc, prec, rec, f1, cm = calculate_metrics(all_labels, all_preds)
    return epoch_loss, acc, prec, rec, f1, cm


def save_best_model(model, path: str) -> None:
    """
    Saves the best state configuration weights of the network.
    """
    model.save_model(path)


def generate_synthetic_csv(path: str, samples: int = 400) -> None:
    """
    Utility method to generate structural mock data if the user runs the 
    training script before completing gesture collection.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    classes = ["fist", "palm", "peace", "pointing", "thumbs_up", "pinch"]
    
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ["label"] + [f"{coord}{i}" for i in range(21) for coord in ("x", "y", "z")]
        writer.writerow(headers)
        
        # Write random coordinates matching normalization format
        for _ in range(samples):
            label = np.random.choice(classes)
            # Create scale normalized vectors
            coords = np.random.randn(21, 3) * 0.4
            coords[0] = 0.0  # Wrist translated to origin
            writer.writerow([label] + coords.flatten().tolist())
    print(f"Generated mock dataset with {samples} entries at {path}")


def main():
    # Parameters
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    dataset_path = "dataset/gesture_data.csv"
    best_model_path = "models/best_model.pth"

    # Handle missing dataset gracefully for testing
    if not os.path.exists(dataset_path):
        generate_synthetic_csv(dataset_path)

    # 1. Load CSV and encode labels
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, 1:].values      # Landmark coordinates shape (N, 63)
    y_raw = df.iloc[:, 0].values   # Label text values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    print(f"\nTraining on {num_classes} categories: {list(encoder.classes_)}")

    # 2. Train-Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Create datasets and data loaders
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Device and Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureNet(num_classes=num_classes).to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track metrics for plotting
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0

    print("\n--- Initiating Training Loop ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = validate(model, val_loader, criterion, device)

        # Log epoch values
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2%}")

        # Checkpoint evaluation & persistence
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_model(model, best_model_path)

    print(f"\nTraining complete. Highest validation accuracy: {best_val_acc:.2%}")

    # Load best weights to compute final detailed validation metrics
    model.load_model(best_model_path)
    _, final_acc, final_prec, final_rec, final_f1, final_cm = validate(model, val_loader, criterion, device)

    print("\n--- Final Best Model Validation Assessment ---")
    print(f"Accuracy:  {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall:    {final_rec:.4f}")
    print(f"F1 Score:  {final_f1:.4f}")
    print("\nConfusion Matrix:")
    print(final_cm)

    # 5. Display and save training curves
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss")
    plt.title("Epoch Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="Train Acc")
    plt.plot(range(1, epochs + 1), history["val_acc"], label="Val Acc")
    plt.title("Epoch Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curves_path = "training_curves.png"
    plt.savefig(curves_path)
    print(f"\nTraining curve figures saved to '{curves_path}'")
    plt.close()


if __name__ == "__main__":
    main()