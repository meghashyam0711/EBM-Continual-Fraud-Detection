"""
Module 1: Core AI Engine
=========================
Energy-Based Out-of-Distribution (OOD) Detection for High-Frequency Financial Fraud.

Components:
    - EnergyFraudClassifier: Deep Neural Network with Log-Sum-Exp energy scoring
    - EWCRegularizer: Elastic Weight Consolidation for continual learning
    - load_and_preprocess_data: Dual-format data loader (creditcard.csv / PS_*.csv)
    - calibrate_energy_threshold: Auto-calibration from training distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional
import copy
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EnergyFraudClassifier(nn.Module):
    """
    Deep Neural Network that outputs unnormalized logits for binary
    classification (0: Legitimate, 1: Fraudulent).

    Anomaly rejection is performed via the energy score, NOT softmax.
    The energy function is defined as:

        E(x; f) = -T · log Σ exp(f_i(x) / T)

    High energy values indicate Out-of-Distribution (OOD) samples —
    black-swan anomalies that deviate from the learned data manifold.
    """

    def __init__(self, input_dim: int, num_classes: int = 2, dropout_rate: float = 0.3):
        super(EnergyFraudClassifier, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, 128)
        self.gn1 = nn.GroupNorm(num_groups=1, num_channels=128)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.gn2 = nn.GroupNorm(num_groups=1, num_channels=64)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 32)
        self.gn3 = nn.GroupNorm(num_groups=1, num_channels=32)
        self.drop3 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits."""
        x = self.fc1(x)
        x = self.gn1(x.unsqueeze(-1)).squeeze(-1)
        x = self.drop1(F.relu(x))

        x = self.fc2(x)
        x = self.gn2(x.unsqueeze(-1)).squeeze(-1)
        x = self.drop2(F.relu(x))

        x = self.fc3(x)
        x = self.gn3(x.unsqueeze(-1)).squeeze(-1)
        x = self.drop3(F.relu(x))

        logits = self.fc_out(x)
        return logits

    def get_penultimate_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract activations from the penultimate layer (fc3 output, 32-dim).
        Used by SpectralPoisonFilter to detect clean-label poisoning.
        """
        with torch.no_grad():
            x = self.fc1(x)
            x = self.gn1(x.unsqueeze(-1)).squeeze(-1)
            x = self.drop1(F.relu(x))

            x = self.fc2(x)
            x = self.gn2(x.unsqueeze(-1)).squeeze(-1)
            x = self.drop2(F.relu(x))

            x = self.fc3(x)
            x = self.gn3(x.unsqueeze(-1)).squeeze(-1)
            x = F.relu(x)
        return x

    def get_energy_score(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute the Helmholtz Free Energy for input samples.

        Mathematical formulation (Log-Sum-Exp):
            E(x; f) = -T · log( Σ_i exp( f_i(x) / T ) )

        Where:
            - f_i(x) are the unnormalized logits for class i
            - T is the temperature scaling parameter
            - The sum runs over all K output classes

        Returns:
            Tensor of shape (batch_size,) with energy scores.
            Higher energy ⟹ more likely OOD (anomalous / black-swan).
        """
        with torch.no_grad():
            logits = self.forward(x)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy

    def predict_with_ood(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        energy_threshold: float = -5.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Full prediction pipeline: class label + energy score + OOD flag.

        Args:
            x: Input features tensor of shape (batch, input_dim)
            temperature: Temperature scaling for energy computation
            energy_threshold: Threshold above which a sample is flagged OOD.
                              Energy is negative; higher (less negative) = more OOD.

        Returns:
            Dictionary with:
                - "predicted_class": int tensor (0=Legitimate, 1=Fraudulent)
                - "energy_score": float tensor
                - "is_ood": bool tensor (True if energy > threshold)
                - "confidence": float tensor (softmax probability of predicted class)
        """
        with torch.no_grad():
            logits = self.forward(x)

            predicted_class = torch.argmax(logits, dim=1)

            probabilities = F.softmax(logits, dim=1)
            confidence = probabilities.gather(1, predicted_class.unsqueeze(1)).squeeze(1)

            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)

            is_ood = energy > energy_threshold

        return {
            "predicted_class": predicted_class,
            "energy_score": energy,
            "is_ood": is_ood,
            "confidence": confidence,
        }


class EWCRegularizer:
    """
    Elastic Weight Consolidation prevents catastrophic forgetting when
    fine-tuning from Task A (historical data) to Task B (live stream).

    After training on Task A, we compute the diagonal of the Fisher
    Information Matrix (FIM) and snapshot the optimal parameters θ*_A.

    During Task B training, an EWC penalty is added to the loss:

        L_EWC = (λ / 2) · Σ_i F_i · (θ_i − θ*_{A,i})²

    Where:
        - F_i = diagonal Fisher information for parameter i
        - θ*_{A,i} = optimal parameter value after Task A
        - λ = regularization strength (higher = more weight preservation)

    This penalizes changes to parameters that are important for Task A,
    allowing the model to learn new patterns while preserving old ones.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        """
        Args:
            model: The trained model (after Task A completion)
            ewc_lambda: Regularization strength — controls the forgetting/plasticity trade-off
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_diag: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self._is_consolidated = False

    def compute_fisher_information(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Compute the diagonal Fisher Information Matrix after Task A training.

        The FIM diagonal approximates the second derivative of the loss
        w.r.t. each parameter, measuring how sensitive the loss is to
        parameter changes. We compute it as the expectation of the
        squared gradients:

            F_i = E[ (∂L/∂θ_i)² ]

        This is estimated empirically over the Task A dataset.

        Args:
            model: The model trained on Task A
            dataloader: Task A data loader
            criterion: Loss function used during Task A training
            num_samples: Optional cap on samples used for FIM estimation
        """
        logger.info("Computing Fisher Information Matrix for EWC consolidation...")
        model.eval()

        fisher_accum: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param.data)

        total_samples = 0
        for batch_x, batch_y in dataloader:
            if num_samples is not None and total_samples >= num_samples:
                break

            model.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data.clone().pow(2) * batch_x.size(0)

            total_samples += batch_x.size(0)

        for name in fisher_accum:
            fisher_accum[name] /= max(total_samples, 1)

        self.fisher_diag = fisher_accum

        self.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self._is_consolidated = True
        logger.info(
            f"EWC consolidation complete. "
            f"Tracked {len(self.fisher_diag)} parameter groups over {total_samples} samples."
        )

    def ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularization penalty for the current model state.

            L_EWC = (λ / 2) · Σ_i F_i · (θ_i − θ*_{A,i})²

        Returns:
            Scalar tensor representing the EWC penalty to add to the loss.
        """
        if not self._is_consolidated:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self.fisher_diag:
                diff = param - self.optimal_params[name]
                penalty += (self.fisher_diag[name] * diff.pow(2)).sum()

        return (self.ewc_lambda / 2.0) * penalty

    @property
    def is_consolidated(self) -> bool:
        """Whether Fisher information has been computed."""
        return self._is_consolidated


def load_and_preprocess_data(
    csv_path: str = r"D:\PROJECTS\ML 2028\data set\creditcard.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Load and preprocess credit card fraud data from CSV.

    Supports two dataset formats:
        1. Kaggle creditcard.csv — columns: Time, V1-V28, Amount, Class
        2. PaySim PS_*.csv — columns: step, type, amount, nameOrig, oldbalanceOrg,
           newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

    Args:
        csv_path: Path to the CSV file
        test_size: Fraction of data used for testing
        random_state: Random seed for reproducibility

    Returns:
        (X_train, X_test, y_train, y_test, scaler)
        - X tensors: float32, shape (N, num_features)
        - y tensors: long, shape (N,)
        - scaler: fitted StandardScaler for inference-time reuse
    """
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")

    if "Class" in df.columns:
        logger.info("Detected Kaggle creditcard.csv format")
        scaler = StandardScaler()
        df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
        df = df.drop(["Time"], axis=1)
        X = df.drop(["Class"], axis=1).values
        y = df["Class"].values

    elif "isFraud" in df.columns:
        logger.info("Detected PaySim PS_*.csv format")
        df["type_encoded"] = pd.Categorical(df["type"]).codes
        feature_cols = [
            "step", "type_encoded", "amount",
            "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
        ]
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols].values)
        y = df["isFraud"].values
    else:
        raise ValueError(
            f"Unrecognized CSV format. Expected 'Class' or 'isFraud' column. "
            f"Found columns: {list(df.columns)}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Split: Train={X_train.shape[0]}, Test={X_test.shape[0]} | "
        f"Fraud rate: {y.mean() * 100:.4f}%"
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        scaler,
    )


def calibrate_energy_threshold(
    model: EnergyFraudClassifier,
    X_train: torch.Tensor,
    percentile: float = 95.0,
    temperature: float = 1.0,
    batch_size: int = 512,
) -> float:
    """
    Auto-calibrate the OOD energy threshold from the training distribution.

    Computes energy scores for all training samples and sets the threshold
    at the specified percentile. Samples with energy above this threshold
    are flagged as OOD during inference.

    Args:
        model: Trained EnergyFraudClassifier
        X_train: Training features tensor
        percentile: Percentile cutoff (e.g., 95.0 means 5% false-positive rate)
        temperature: Temperature for energy computation
        batch_size: Batch size for energy computation

    Returns:
        Calibrated energy threshold (float)
    """
    model.eval()
    all_energies = []

    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for (batch_x,) in loader:
        energy = model.get_energy_score(batch_x, temperature=temperature)
        all_energies.append(energy)

    all_energies = torch.cat(all_energies)
    threshold = float(np.percentile(all_energies.numpy(), percentile))

    logger.info(
        f"Energy threshold calibrated at {percentile}th percentile: {threshold:.4f} | "
        f"Energy range: [{all_energies.min():.4f}, {all_energies.max():.4f}] | "
        f"Mean: {all_energies.mean():.4f}, Std: {all_energies.std():.4f}"
    )
    return threshold


if __name__ == "__main__":
    print("=" * 70)
    print("  Energy-Based OOD Detection Framework — Module 1 Smoke Test")
    print("=" * 70)

    INPUT_DIM = 29
    model = EnergyFraudClassifier(input_dim=INPUT_DIM)
    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n--- OOD Detection Layer Test ---")
    dummy_normal = torch.randn(5, INPUT_DIM)
    dummy_ood = torch.randn(5, INPUT_DIM) * 15.0

    model.eval()
    normal_energy = model.get_energy_score(dummy_normal)
    ood_energy = model.get_energy_score(dummy_ood)

    print(f"Normal transaction energies : {normal_energy.numpy()}")
    print(f"OOD attack vector energies  : {ood_energy.numpy()}")

    print("\n--- Full Prediction Pipeline Test ---")
    result = model.predict_with_ood(dummy_ood, energy_threshold=-5.0)
    for k, v in result.items():
        print(f"  {k}: {v.numpy()}")

    print("\n--- EWC Regularizer Test ---")
    X_mock = torch.randn(200, INPUT_DIM)
    y_mock = torch.randint(0, 2, (200,))
    mock_loader = DataLoader(TensorDataset(X_mock, y_mock), batch_size=32)

    model.train()
    ewc = EWCRegularizer(model, ewc_lambda=1000.0)
    ewc.compute_fisher_information(model, mock_loader, nn.CrossEntropyLoss())

    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    penalty = ewc.ewc_penalty(model)
    print(f"EWC penalty after parameter drift: {penalty.item():.4f}")

    print("\n--- Energy Threshold Auto-Calibration ---")
    model.eval()
    threshold = calibrate_energy_threshold(model, X_mock, percentile=95.0)
    print(f"Calibrated threshold (95th pct): {threshold:.4f}")

    print("\n[PASS] Module 1 smoke test passed.")
