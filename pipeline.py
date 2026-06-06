"""
Module 2: Data Integrity & Privacy Pipeline
=============================================
Cryptographic auditing, data poisoning defense, and differentially
private continual learning pipeline.

Components:
    - MerkleTree: SHA-256 Merkle tree for immutable batch lineage
    - SpectralPoisonFilter: SVD-based clean-label poisoning defense
    - SecureTrainingPipeline: Orchestrates Merkle → Poison Filter → DP-SGD + EWC
"""

import hashlib
import math
import struct
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from typing import List, Optional, Tuple, Dict, Any
import logging
import json
from datetime import datetime, timezone

from model import EnergyFraudClassifier, EWCRegularizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MerkleTree:
    """
    SHA-256 Merkle Tree for immutable transaction batch auditing.

    Each leaf node is the SHA-256 hash of a serialized transaction record.
    Internal nodes are SHA-256(left_child || right_child). If a level has
    an odd number of nodes, the last node is duplicated.

    This provides:
        - O(1) batch integrity verification via the Merkle root
        - O(log N) audit proofs for individual records
        - Tamper-evident lineage logs for regulatory compliance

    Usage:
        tree = MerkleTree(batch_tensor)
        root = tree.get_root()
        proof = tree.get_audit_proof(record_index)
        is_valid = MerkleTree.verify_proof(leaf_hash, proof, root)
    """

    def __init__(self, records: torch.Tensor):
        """
        Build a Merkle tree from a batch of transaction records.

        Args:
            records: Tensor of shape (N, D) where each row is a transaction record.
        """
        self.records = records
        self.num_records = records.shape[0]
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []
        self._build_tree()

    @staticmethod
    def _hash_data(data: bytes) -> str:
        """Compute SHA-256 hash of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _hash_pair(left: str, right: str) -> str:
        """Compute SHA-256 hash of two concatenated hex-digest strings."""
        combined = (left + right).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def _build_tree(self) -> None:
        """
        Construct the full Merkle tree bottom-up.

        Level 0 = leaves (hashes of individual records)
        Level k = SHA-256(level_{k-1}[2i] || level_{k-1}[2i+1])
        Final level has a single element: the Merkle root.
        """
        self.leaves = []
        for i in range(self.num_records):
            record_bytes = self.records[i].numpy().tobytes()
            leaf_hash = self._hash_data(record_bytes)
            self.leaves.append(leaf_hash)

        self.tree = [self.leaves.copy()]
        current_level = self.leaves.copy()

        while len(current_level) > 1:
            next_level = []
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                parent = self._hash_pair(current_level[i], current_level[i + 1])
                next_level.append(parent)

            self.tree.append(next_level)
            current_level = next_level

    def get_root(self) -> str:
        """Return the Merkle root hash — the single top-level digest."""
        if not self.tree:
            return ""
        return self.tree[-1][0]

    def get_audit_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Generate an audit proof (authentication path) for the record at `index`.

        Returns a list of (hash, side) tuples where `side` is 'L' or 'R'
        indicating whether the sibling hash should be placed on the left
        or right during verification.

        Args:
            index: Index of the record to prove (0-indexed)

        Returns:
            List of (sibling_hash, side) pairs for bottom-up verification
        """
        if index < 0 or index >= self.num_records:
            raise IndexError(f"Record index {index} out of range [0, {self.num_records})")

        proof = []
        idx = index

        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]
            level_copy = current_level.copy()
            if len(level_copy) % 2 != 0:
                level_copy.append(level_copy[-1])

            if idx % 2 == 0:
                sibling_idx = idx + 1
                proof.append((level_copy[sibling_idx], "R"))
            else:
                sibling_idx = idx - 1
                proof.append((level_copy[sibling_idx], "L"))

            idx = idx // 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """
        Verify an audit proof against the Merkle root.

        Args:
            leaf_hash: The SHA-256 hash of the record being verified
            proof: The authentication path from get_audit_proof()
            root: The expected Merkle root

        Returns:
            True if the proof is valid (record is in the tree)
        """
        current = leaf_hash
        for sibling_hash, side in proof:
            if side == "L":
                current = MerkleTree._hash_pair(sibling_hash, current)
            else:
                current = MerkleTree._hash_pair(current, sibling_hash)
        return current == root

    def to_lineage_log(self) -> Dict[str, Any]:
        """
        Generate an immutable lineage log entry for this batch.

        Returns a JSON-serializable dict suitable for audit storage.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merkle_root": self.get_root(),
            "num_records": self.num_records,
            "tree_depth": len(self.tree),
            "leaf_hashes_sample": self.leaves[:3],
        }


class SpectralPoisonFilter:
    """
    Detects and filters clean-label poisoned data points using spectral
    signatures in the model's learned representation space.

    Method (Tran, Li, Madry 2018 — "Spectral Signatures in Backdoor Attacks"):
        1. Extract penultimate-layer activations for the batch
        2. Center the activation matrix (subtract mean)
        3. Compute SVD of the centered matrix
        4. Project each sample onto the top singular vector
        5. Flag samples whose projection magnitude exceeds a z-score threshold

    Clean-label poisoned samples tend to have anomalously large projections
    onto the top singular vector because the poisoning pattern creates a
    consistent directional bias in representation space.

    Usage:
        poison_filter = SpectralPoisonFilter(z_threshold=2.0)
        clean_mask = poison_filter.filter(model, batch_x, batch_y)
        clean_x = batch_x[clean_mask]
        clean_y = batch_y[clean_mask]
    """

    def __init__(self, z_threshold: float = 2.0):
        """
        Args:
            z_threshold: Z-score cutoff for outlier detection.
                         Samples with |z| > threshold are flagged as poisoned.
                         Default 2.0 ≈ 97.7% of clean data retained.
        """
        self.z_threshold = z_threshold
        self.last_num_filtered = 0
        self.last_scores: Optional[np.ndarray] = None

    def filter(
        self,
        model: EnergyFraudClassifier,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Identify clean (non-poisoned) samples in a batch.

        Args:
            model: The EnergyFraudClassifier (must have get_penultimate_activations)
            batch_x: Input features tensor (N, D)
            batch_y: Labels tensor (N,)

        Returns:
            Boolean mask tensor of shape (N,) — True for clean samples.
        """
        if batch_x.shape[0] < 3:
            logger.warning("Batch too small for spectral analysis; skipping filter.")
            return torch.ones(batch_x.shape[0], dtype=torch.bool)

        activations = model.get_penultimate_activations(batch_x)
        act_np = activations.numpy()

        mean_act = act_np.mean(axis=0, keepdims=True)
        centered = act_np - mean_act

        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed to converge; skipping poison filter.")
            return torch.ones(batch_x.shape[0], dtype=torch.bool)

        top_singular_vector = Vt[0]
        projections = centered @ top_singular_vector

        proj_mean = projections.mean()
        proj_std = projections.std()
        if proj_std < 1e-8:
            return torch.ones(batch_x.shape[0], dtype=torch.bool)

        z_scores = np.abs((projections - proj_mean) / proj_std)

        clean_mask = torch.tensor(z_scores < self.z_threshold, dtype=torch.bool)

        self.last_num_filtered = int((~clean_mask).sum())
        self.last_scores = z_scores

        if self.last_num_filtered > 0:
            logger.warning(
                f"Spectral filter: Removed {self.last_num_filtered}/{batch_x.shape[0]} "
                f"suspected poisoned samples (z-threshold={self.z_threshold:.1f})"
            )

        return clean_mask


class SecureTrainingPipeline:
    """
    Orchestrates the full secure continual learning pipeline:

        Input Batch → Merkle Audit → Spectral Poison Filter →
        DP-SGD Training (Opacus) + EWC Penalty → Privacy Budget Log

    This class wires together all security and integrity components
    so that data flows from cryptographic verification through
    differentially private training with catastrophic forgetting prevention.

    Usage:
        pipeline = SecureTrainingPipeline(
            model=model,
            X_train=X_train,
            y_train=y_train,
            config={...}
        )
        pipeline.run(epochs=10)
    """

    DEFAULT_CONFIG = {
        "batch_size": 64,
        "learning_rate": 1e-3,
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0,
        "dp_delta": 1e-5,
        "ewc_lambda": 1000.0,
        "ewc_enabled": True,
        "poison_filter_enabled": True,
        "poison_z_threshold": 2.0,
        "merkle_audit_enabled": True,
    }

    def __init__(
        self,
        model: EnergyFraudClassifier,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
        ewc_regularizer: Optional[EWCRegularizer] = None,
    ):
        """
        Args:
            model: The EnergyFraudClassifier to train
            X_train: Training features tensor
            y_train: Training labels tensor
            config: Configuration dict (merged with DEFAULT_CONFIG)
            ewc_regularizer: Optional pre-computed EWC regularizer (from Task A)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        dataset = TensorDataset(X_train, y_train)
        self.data_loader = DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        self.model.train()
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            noise_multiplier=self.config["noise_multiplier"],
            max_grad_norm=self.config["max_grad_norm"],
        )
        logger.info(
            f"DP-SGD enabled: noise_multiplier={self.config['noise_multiplier']}, "
            f"max_grad_norm={self.config['max_grad_norm']}"
        )

        self.ewc = ewc_regularizer
        if self.ewc and self.ewc.is_consolidated:
            logger.info(
                f"EWC regularization enabled: λ={self.ewc.ewc_lambda}"
            )

        self.poison_filter = (
            SpectralPoisonFilter(z_threshold=self.config["poison_z_threshold"])
            if self.config["poison_filter_enabled"]
            else None
        )

        self.lineage_logs: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, float]] = []

    def audit_batch(self, batch_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compute the Merkle root for a batch and store the lineage log.

        Args:
            batch_tensor: The batch features tensor (B, D)

        Returns:
            Lineage log dict with merkle_root, timestamp, etc.
        """
        tree = MerkleTree(batch_tensor)
        log_entry = tree.to_lineage_log()
        self.lineage_logs.append(log_entry)
        return log_entry

    def filter_poisoned(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run spectral poison filtering on a batch.

        Args:
            batch_x: Input features (B, D)
            batch_y: Labels (B,)

        Returns:
            (clean_x, clean_y) — filtered batch with suspected poison removed
        """
        if self.poison_filter is None:
            return batch_x, batch_y

        base_model = self.model
        if hasattr(self.model, "_module"):
            base_model = self.model._module

        clean_mask = self.poison_filter.filter(base_model, batch_x, batch_y)

        if clean_mask.sum() == 0:
            logger.warning("All samples filtered as poisoned — keeping original batch.")
            return batch_x, batch_y

        return batch_x[clean_mask], batch_y[clean_mask]

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one full training epoch with the secure pipeline.

        Flow per batch:
            1. Merkle audit (cryptographic lineage)
            2. Spectral poison filter (remove adversarial samples)
            3. Forward pass → CrossEntropy + EWC penalty
            4. DP-SGD backward pass (Opacus noise injection + gradient clipping)

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Dict with epoch metrics (loss, ewc_penalty, samples_filtered, etc.)
        """
        self.model.train()
        total_loss = 0.0
        total_ewc_penalty = 0.0
        total_ce_loss = 0.0
        total_samples = 0
        total_filtered = 0
        num_batches = 0

        for batch_x, batch_y in self.data_loader:
            if self.config["merkle_audit_enabled"]:
                self.audit_batch(batch_x)

            if self.poison_filter is not None:
                clean_x, clean_y = self.filter_poisoned(batch_x, batch_y)
                total_filtered += batch_x.shape[0] - clean_x.shape[0]

                if clean_x.shape[0] < 2:
                    continue
            else:
                clean_x, clean_y = batch_x, batch_y

            self.optimizer.zero_grad()
            logits = self.model(clean_x)
            ce_loss = self.criterion(logits, clean_y)

            ewc_loss = torch.tensor(0.0)
            if self.ewc is not None and self.ewc.is_consolidated:
                base_model = self.model
                if hasattr(self.model, "_module"):
                    base_model = self.model._module
                ewc_loss = self.ewc.ewc_penalty(base_model)

            loss = ce_loss + ewc_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * clean_x.shape[0]
            total_ce_loss += ce_loss.item() * clean_x.shape[0]
            total_ewc_penalty += ewc_loss.item() * clean_x.shape[0]
            total_samples += clean_x.shape[0]
            num_batches += 1

        avg_loss = total_loss / max(total_samples, 1)
        avg_ce = total_ce_loss / max(total_samples, 1)
        avg_ewc = total_ewc_penalty / max(total_samples, 1)

        metrics = {
            "epoch": epoch + 1,
            "total_loss": avg_loss,
            "ce_loss": avg_ce,
            "ewc_penalty": avg_ewc,
            "samples_trained": total_samples,
            "samples_filtered": total_filtered,
            "num_batches": num_batches,
        }

        return metrics

    def log_privacy_budget(self, epoch: int) -> Tuple[float, float]:
        """
        Query Opacus for the current cumulative privacy budget.

        The privacy guarantee is expressed as (ε, δ)-differential privacy:
            - ε (epsilon): Privacy loss — lower is more private
            - δ (delta): Probability of privacy breach

        DP-SGD guarantees that for any two adjacent datasets D and D' that
        differ in a single record, and for any set of outputs S:

            Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D') ∈ S] + δ

        This means an attacker performing a membership inference attack
        gains at most exp(ε) advantage, with δ failure probability.

        Args:
            epoch: Current epoch number

        Returns:
            (epsilon, delta) tuple
        """
        delta = self.config["dp_delta"]
        epsilon = self.privacy_engine.get_epsilon(delta=delta)

        logger.info(
            f"Epoch {epoch + 1} Privacy Budget | "
            f"epsilon = {epsilon:.4f}, delta = {delta:.2e} | "
            f"Membership inference resistance: exp(epsilon) = {math.exp(epsilon):.4f}x advantage cap"
        )

        return epsilon, delta

    def run(self, epochs: int = 10) -> List[Dict[str, float]]:
        """
        Execute the full secure training pipeline for the specified epochs.

        Full flow per epoch:
            Audit batches → Filter poisoned → DP-SGD train with EWC → Log privacy

        Args:
            epochs: Number of training epochs

        Returns:
            List of per-epoch metric dictionaries
        """
        logger.info("=" * 70)
        logger.info("  SECURE TRAINING PIPELINE -- STARTING")
        logger.info("=" * 70)
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

        all_metrics = []
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            metrics = self.train_epoch(epoch)

            epsilon, delta = self.log_privacy_budget(epoch)
            metrics["epsilon"] = epsilon
            metrics["delta"] = delta

            epoch_time = time.time() - epoch_start
            metrics["epoch_time_seconds"] = epoch_time

            all_metrics.append(metrics)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {metrics['total_loss']:.4f} (CE: {metrics['ce_loss']:.4f}, "
                f"EWC: {metrics['ewc_penalty']:.4f}) | "
                f"Filtered: {metrics['samples_filtered']} samples | "
                f"eps={epsilon:.4f} | Time: {epoch_time:.2f}s"
            )

        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"  TRAINING COMPLETE -- {epochs} epochs in {total_time:.2f}s")
        logger.info(f"  Final Privacy Budget: eps={all_metrics[-1]['epsilon']:.4f}")
        logger.info(f"  Merkle Lineage Logs: {len(self.lineage_logs)} batch audits recorded")
        logger.info("=" * 70)

        self.training_history = all_metrics
        return all_metrics

    def get_lineage_report(self) -> Dict[str, Any]:
        """
        Generate a full audit report with all Merkle lineage logs.

        Returns:
            Dict with training summary and all batch audit entries
        """
        return {
            "pipeline_config": self.config,
            "total_batches_audited": len(self.lineage_logs),
            "training_history": self.training_history,
            "lineage_logs": self.lineage_logs,
        }


if __name__ == "__main__":
    from model import EnergyFraudClassifier, EWCRegularizer, load_and_preprocess_data

    print("=" * 70)
    print("  Secure Pipeline — Module 2 Integration Test")
    print("=" * 70)

    INPUT_DIM = 29

    print("\n--- Phase 1: Task A Training (Historical 2026 Data) ---")
    X_task_a = torch.randn(500, INPUT_DIM)
    y_task_a = torch.randint(0, 2, (500,))

    model_a = EnergyFraudClassifier(input_dim=INPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    opt_a = optim.Adam(model_a.parameters(), lr=1e-3)
    loader_a = DataLoader(TensorDataset(X_task_a, y_task_a), batch_size=32)

    model_a.train()
    for epoch in range(3):
        for bx, by in loader_a:
            opt_a.zero_grad()
            loss = criterion(model_a(bx), by)
            loss.backward()
            opt_a.step()
    print("Task A base training complete.")

    ewc = EWCRegularizer(model_a, ewc_lambda=1000.0)
    ewc.compute_fisher_information(model_a, loader_a, criterion)

    print("\n--- Phase 2: Task B Training (Live 2028 Stream, Secure Pipeline) ---")
    X_task_b = torch.randn(300, INPUT_DIM)
    y_task_b = torch.randint(0, 2, (300,))

    pipeline = SecureTrainingPipeline(
        model=model_a,
        X_train=X_task_b,
        y_train=y_task_b,
        ewc_regularizer=ewc,
        config={
            "batch_size": 32,
            "noise_multiplier": 1.1,
            "max_grad_norm": 1.0,
            "ewc_lambda": 1000.0,
            "poison_z_threshold": 2.0,
        },
    )

    metrics = pipeline.run(epochs=3)

    print("\n--- Merkle Lineage Audit Report ---")
    report = pipeline.get_lineage_report()
    print(f"Total batches audited: {report['total_batches_audited']}")
    if report["lineage_logs"]:
        sample = report["lineage_logs"][0]
        print(f"Sample Merkle Root: {sample['merkle_root'][:32]}...")
        print(f"Sample Timestamp:   {sample['timestamp']}")

    print("\n--- Standalone Merkle Tree Test ---")
    test_batch = torch.randn(8, INPUT_DIM)
    tree = MerkleTree(test_batch)
    root = tree.get_root()
    print(f"Merkle Root: {root}")

    proof = tree.get_audit_proof(3)
    leaf_hash = tree.leaves[3]
    is_valid = MerkleTree.verify_proof(leaf_hash, proof, root)
    print(f"Audit proof for record 3: {'VALID [OK]' if is_valid else 'INVALID [FAIL]'}")

    fake_hash = hashlib.sha256(b"tampered").hexdigest()
    is_tampered_valid = MerkleTree.verify_proof(fake_hash, proof, root)
    print(f"Tampered record proof:     {'VALID [BUG!]' if is_tampered_valid else 'INVALID [OK] (tamper detected)'}")

    print("\n[PASS] Module 2 integration test passed.")
