import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from model import EnergyFraudClassifier, load_and_preprocess_data, calibrate_energy_threshold

DATASET_PATH = r"D:\PROJECTS\ML 2028\data set\creditcard.csv"
MODEL_SAVE_PATH = "model_weights.pt"

print("=" * 70)
print("  Energy-Based Fraud Model DP-SGD Training Script")
print("=" * 70)

if os.path.exists(DATASET_PATH):
    print(f"Loading credit card dataset from: {DATASET_PATH}...")
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATASET_PATH)
        print(f"Dataset successfully loaded. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Falling back to mock data...")
        X_train = torch.randn(1000, 29)
        y_train = torch.randint(0, 2, (1000,))
else:
    print(f"Dataset path '{DATASET_PATH}' not found. Falling back to mock data...")
    X_train = torch.randn(1000, 29)
    y_train = torch.randint(0, 2, (1000,))

dataset = TensorDataset(X_train, y_train)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = EnergyFraudClassifier(input_dim=29, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
print("Privacy Engine attached. Training with Differential Privacy!")

def train_private_model(model, data_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
            
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Privacy Spent (e, delta=1e-5): {epsilon:.2f}")

train_private_model(model, data_loader, optimizer, criterion, epochs=3)

unwrap_model = model._module if hasattr(model, "_module") else model
torch.save(unwrap_model.state_dict(), MODEL_SAVE_PATH)
print(f"Model weights saved successfully to: {MODEL_SAVE_PATH}")

unwrap_model.eval()
calibrated_threshold = calibrate_energy_threshold(unwrap_model, X_train, percentile=95.0)
print(f"Calibrated OOD Energy Threshold (95th percentile): {calibrated_threshold:.4f}")
print("Training completed successfully!")