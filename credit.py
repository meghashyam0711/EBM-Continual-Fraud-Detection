from importlib import metadata
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path=r"D:\PROJECTS\ML 2028\data set\creditcard.csv"):
    df = pd.read_csv(csv_path)
    
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time'], axis=1)
    
    X = df.drop(['Class'], axis=1).values
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

class EnergyFraudClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(EnergyFraudClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def get_energy_score(self, x, temperature=1.0):
        with torch.no_grad():
            logits = self.forward(x)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy

if __name__ == "__main__":
    print("Initializing Energy-Based Model...")
    
    input_features = 29 
    model = EnergyFraudClassifier(input_dim=input_features)
    
    dummy_in_distribution = torch.randn(5, input_features) 
    dummy_ood_attack = torch.randn(5, input_features) * 15.0
    
    normal_energy = model.get_energy_score(dummy_in_distribution)
    ood_energy = model.get_energy_score(dummy_ood_attack)
    
    print("\n--- OOD Detection Layer Smoke Test ---")
    print(f"Normal Transactions Energy Scores: {normal_energy.numpy()}")
    print(f"Adversarial OOD Transactions Energy Scores: {ood_energy.numpy()}")
    print("\n[Notice how the unusual/adversarial data returns massive differences in energy levels compared to standard profiles!]")