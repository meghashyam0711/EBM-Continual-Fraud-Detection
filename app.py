"""
Module 3: Production Backend & Interaction Layer
==================================================
Asynchronous FastAPI service with Redis caching and Prometheus monitoring
for real-time energy-based OOD fraud detection.

Endpoints:
    POST /api/v1/predict   — Run OOD energy detection on transaction features
    GET  /health           — Liveness probe
    GET  /ready            — Readiness probe (model + Redis)
    GET  /metrics          — Prometheus metrics endpoint
"""

import os
import sys
import json
import time
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from model import EnergyFraudClassifier, load_and_preprocess_data, calibrate_energy_threshold
from pipeline import SecureTrainingPipeline, MerkleTree
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import threading
import joblib

try:
    from tasks import run_background_training as celery_train
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration loaded from environment variables."""

    MODEL_PATH: str = os.getenv("MODEL_PATH", "model_weights.pt")
    INPUT_DIM: int = int(os.getenv("INPUT_DIM", "29"))
    NUM_CLASSES: int = int(os.getenv("NUM_CLASSES", "2"))
    ENERGY_TEMPERATURE: float = float(os.getenv("ENERGY_TEMPERATURE", "1.0"))
    OOD_ENERGY_THRESHOLD: float = float(os.getenv("OOD_ENERGY_THRESHOLD", "-5.0"))

    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "300"))

    EMA_ALPHA: float = float(os.getenv("EMA_ALPHA", "0.05"))


config = AppConfig()


PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions",
    ["prediction", "ood"],
)

ENERGY_SCORE_HISTOGRAM = Histogram(
    "energy_score_histogram",
    "Distribution of computed energy scores",
    buckets=[-20, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 2, 5, 10, 20],
)

ENERGY_SCORE_RUNNING_AVG = Gauge(
    "energy_score_running_avg",
    "Exponential moving average of input energy scores (spike = anomaly warning)",
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Inference request latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

OOD_WARNINGS_TOTAL = Counter(
    "ood_anomaly_warnings_total",
    "Total number of OOD anomaly warnings triggered",
)


class AppState:
    """Mutable application state managed during lifespan."""

    model: Optional[EnergyFraudClassifier] = None
    redis_client: Optional[Any] = None
    scaler: Optional[Any] = None
    energy_ema: float = -5.9963
    is_ready: bool = False


    training_status: Dict[str, Any] = {
        "status": "idle",
        "progress": 0,
        "epochs": 3,
        "current_epoch": 0,
        "loss": 0.0,
        "epsilon": 0.0,
        "error": None
    }
    lineage_logs: List[Dict[str, Any]] = []


state = AppState()


class SinglePredictionRequest(BaseModel):
    """Single transaction prediction request."""

    features: List[float] = Field(
        ...,
        description="Transaction feature vector",
        min_length=1,
        examples=[[0.1, -0.5, 1.2, 0.3, -0.8] + [0.0] * 24],
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature scaling for energy computation (default: server config)",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Custom OOD energy threshold (default: server config)",
    )


class BatchPredictionRequest(BaseModel):
    """Batch transaction prediction request."""

    batch: List[List[float]] = Field(
        ...,
        description="List of transaction feature vectors",
        min_length=1,
    )
    temperature: Optional[float] = Field(default=None)
    threshold: Optional[float] = Field(default=None)


class PredictionResult(BaseModel):
    """Single prediction result."""

    prediction: str = Field(description="LEGITIMATE or FRAUDULENT")
    energy_score: float = Field(description="Energy score (higher = more anomalous)")
    is_ood: bool = Field(description="Whether the sample is flagged as OOD")
    flag: Optional[str] = Field(description="SUSPICIOUS_OOD if OOD, else null")
    confidence: float = Field(description="Softmax confidence of predicted class")
    cached: bool = Field(default=False, description="Whether result was served from cache")


class PredictionResponse(BaseModel):
    """API response for prediction requests."""

    results: List[PredictionResult]
    batch_size: int
    avg_energy: float
    ood_count: int
    processing_time_ms: float


def _feature_cache_key(features: List[float]) -> str:
    """
    Generate a deterministic Redis key from a feature vector.
    Uses SHA-256 hash of the serialized features for fixed-length keys.
    """
    raw = json.dumps(features, sort_keys=True).encode("utf-8")
    return f"energy:{hashlib.sha256(raw).hexdigest()}"


def _cache_get(key: str) -> Optional[Dict]:
    """Retrieve cached result from Redis. Returns None on miss or error."""
    if state.redis_client is None:
        return None
    try:
        cached = state.redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis GET error: {e}")
    return None


def _cache_set(key: str, value: Dict) -> None:
    """Store result in Redis with configured TTL. Fails silently."""
    if state.redis_client is None:
        return
    try:
        state.redis_client.setex(key, config.REDIS_TTL, json.dumps(value))
    except Exception as e:
        logger.warning(f"Redis SET error: {e}")


def run_inference(
    features_batch: List[List[float]],
    temperature: Optional[float] = None,
    threshold: Optional[float] = None,
) -> List[PredictionResult]:
    """
    Run energy-based OOD inference on a batch of feature vectors.

    For each sample:
        1. Check Redis cache
        2. If miss: run through EnergyFraudClassifier.predict_with_ood
        3. Cache the result
        4. Update Prometheus metrics

    Args:
        features_batch: List of feature vectors
        temperature: Temperature override (default: config)
        threshold: Energy threshold override (default: config)

    Returns:
        List of PredictionResult objects
    """
    temp = temperature or config.ENERGY_TEMPERATURE
    thresh = threshold or config.OOD_ENERGY_THRESHOLD

    if len(features_batch) > 0:
        try:
            batch_tensor = torch.tensor(features_batch, dtype=torch.float32)
            tree = MerkleTree(batch_tensor)
            log_entry = tree.to_lineage_log()
            log_entry["records"] = features_batch
            log_entry["leaves"] = tree.leaves
            state.lineage_logs.insert(0, log_entry)
            if len(state.lineage_logs) > 50:
                state.lineage_logs.pop()
        except Exception as e:
            logger.warning(f"Merkle audit logging failed: {e}")

    results = []
    uncached_indices = []
    uncached_features = []

    for i, features in enumerate(features_batch):
        cache_key = _feature_cache_key(features)
        cached = _cache_get(cache_key)

        if cached is not None:
            results.append(PredictionResult(**cached, cached=True))
        else:
            results.append(None)
            uncached_indices.append(i)
            uncached_features.append(features)

    if uncached_features:
        input_tensor = torch.tensor(uncached_features, dtype=torch.float32)

        state.model.eval()
        prediction = state.model.predict_with_ood(
            input_tensor, temperature=temp, energy_threshold=thresh
        )

        for j, idx in enumerate(uncached_indices):
            pred_class = int(prediction["predicted_class"][j].item())
            energy = float(prediction["energy_score"][j].item())
            is_ood = bool(prediction["is_ood"][j].item())
            confidence = float(prediction["confidence"][j].item())

            label = "FRAUDULENT" if pred_class == 1 else "LEGITIMATE"
            flag = "SUSPICIOUS_OOD" if is_ood else None

            result = PredictionResult(
                prediction=label,
                energy_score=round(energy, 6),
                is_ood=is_ood,
                flag=flag,
                confidence=round(confidence, 6),
                cached=False,
            )
            results[idx] = result

            cache_payload = {
                "prediction": label,
                "energy_score": round(energy, 6),
                "is_ood": is_ood,
                "flag": flag,
                "confidence": round(confidence, 6),
            }
            _cache_set(_feature_cache_key(features_batch[idx]), cache_payload)

    for r in results:
        ood_label = "true" if r.is_ood else "false"
        PREDICTIONS_TOTAL.labels(prediction=r.prediction, ood=ood_label).inc()
        ENERGY_SCORE_HISTOGRAM.observe(r.energy_score)

        state.energy_ema = (
            config.EMA_ALPHA * r.energy_score + (1 - config.EMA_ALPHA) * state.energy_ema
        )
        ENERGY_SCORE_RUNNING_AVG.set(state.energy_ema)

        if r.is_ood:
            OOD_WARNINGS_TOTAL.inc()
            logger.warning(
                f"[!] OOD ANOMALY DETECTED | Energy: {r.energy_score:.4f} | "
                f"Threshold: {thresh} | Flag: {r.flag}"
            )

    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes model and Redis on startup, cleans up on shutdown.
    """
    logger.info("=" * 60)
    logger.info("  Energy-Based OOD Fraud Detection API — Starting")
    logger.info("=" * 60)

    state.model = EnergyFraudClassifier(
        input_dim=config.INPUT_DIM,
        num_classes=config.NUM_CLASSES,
    )

    if os.path.exists(config.MODEL_PATH):
        logger.info(f"Loading model weights from: {config.MODEL_PATH}")
        weights = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
        state.model.load_state_dict(weights)
    else:
        logger.warning(
            f"No weights found at {config.MODEL_PATH} — using random initialization. "
            f"Run the training pipeline first for production use."
        )

    state.model.eval()
    if os.path.exists('scaler.joblib'):
        try:
            state.scaler = joblib.load('scaler.joblib')
            logger.info('Scaler loaded from scaler.joblib')
        except Exception as e:
            logger.warning(f'Scaler load failed: {e}')
            state.scaler = None
    else:
        state.scaler = None
    logger.info(f"OOD threshold: {config.OOD_ENERGY_THRESHOLD}, Temperature: {config.ENERGY_TEMPERATURE}")

    if REDIS_AVAILABLE:
        import threading

        def _try_redis_connect():
            """Attempt Redis connection in a separate thread."""
            try:
                client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    db=config.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                client.ping()
                state.redis_client = client
                logger.info(f"Redis connected: {config.REDIS_HOST}:{config.REDIS_PORT}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}) -- running without cache.")
                state.redis_client = None

        redis_thread = threading.Thread(target=_try_redis_connect, daemon=True)
        redis_thread.start()
        redis_thread.join(timeout=3)

        if redis_thread.is_alive():
            logger.warning("Redis connection timed out -- running without cache.")
            state.redis_client = None
    else:
        logger.warning("redis-py not installed -- running without cache.")

    state.is_ready = True
    logger.info("API ready to serve requests.")

    yield

    logger.info("Shutting down API...")
    if state.redis_client:
        try:
            state.redis_client.close()
        except Exception:
            pass
    state.is_ready = False


app = FastAPI(
    title="Energy-Based OOD Fraud Detection API",
    description=(
        "Production-grade financial fraud detection using energy-based "
        "Out-of-Distribution scoring with differential privacy, "
        "continual learning (EWC), and cryptographic data auditing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: Request):
    """
    Run energy-based OOD fraud detection on a transaction payload.

    Accepts either:
        - Single: {"features": [f1, f2, ...]}
        - Batch:  {"batch": [[f1, f2, ...], [f1, f2, ...], ...]}

    Returns prediction class, energy score, OOD flag, and confidence
    for each transaction in the payload.
    """
    start_time = time.time()

    if not state.is_ready or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service not ready.")

    body = await request.json()

    if "batch" in body:
        features_batch = body["batch"]
    elif "features" in body:
        features_batch = [body["features"]]
    else:
        raise HTTPException(
            status_code=422,
            detail="Request must contain 'features' (single) or 'batch' (multiple) field.",
        )

    for i, features in enumerate(features_batch):
        if len(features) != config.INPUT_DIM:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Feature vector {i} has {len(features)} dimensions, "
                    f"expected {config.INPUT_DIM}."
                ),
            )

    temperature = body.get("temperature")
    threshold = body.get("threshold")

    results = run_inference(features_batch, temperature=temperature, threshold=threshold)

    elapsed_ms = (time.time() - start_time) * 1000
    REQUEST_LATENCY.observe(elapsed_ms / 1000)

    energies = [r.energy_score for r in results]
    ood_count = sum(1 for r in results if r.is_ood)

    return PredictionResponse(
        results=results,
        batch_size=len(results),
        avg_energy=round(sum(energies) / len(energies), 6),
        ood_count=ood_count,
        processing_time_ms=round(elapsed_ms, 3),
    )


@app.get("/health")
async def health():
    """Liveness probe — returns 200 if the service process is running."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/ready")
async def ready():
    """
    Readiness probe — checks model loaded + Redis connectivity.
    Returns 503 if not ready.
    """
    checks = {
        "model_loaded": state.model is not None,
        "redis_connected": False,
    }

    if state.redis_client:
        try:
            state.redis_client.ping()
            checks["redis_connected"] = True
        except Exception:
            checks["redis_connected"] = False

    is_ready = checks["model_loaded"]

    if not is_ready:
        raise HTTPException(status_code=503, detail=checks)

    return {"status": "ready", "checks": checks}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for scraping."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


class AuditProofRequest(BaseModel):
    batch_index: int
    record_index: int


class VerifyProofRequest(BaseModel):
    leaf_hash: str
    proof: List[List[str]]
    merkle_root: str


@app.get("/api/v1/audit-logs")
async def get_audit_logs():
    """Get list of Merkle audit lineage logs (excluding raw large fields for transmission efficiency)."""
    summary_logs = []
    for i, log in enumerate(state.lineage_logs):
        summary_logs.append({
            "index": i,
            "timestamp": log["timestamp"],
            "merkle_root": log["merkle_root"],
            "num_records": log["num_records"],
            "tree_depth": log["tree_depth"],
        })
    return summary_logs


@app.post("/api/v1/audit-proof")
async def get_audit_proof(req: AuditProofRequest):
    """Retrieve Merkle audit proof and record info for verification."""
    if req.batch_index < 0 or req.batch_index >= len(state.lineage_logs):
        raise HTTPException(status_code=400, detail="Invalid batch index.")
    
    log_entry = state.lineage_logs[req.batch_index]
    records = log_entry["records"]
    
    if req.record_index < 0 or req.record_index >= len(records):
        raise HTTPException(status_code=400, detail="Invalid record index.")
    
    try:
        batch_tensor = torch.tensor(records, dtype=torch.float32)
        tree = MerkleTree(batch_tensor)
        proof = tree.get_audit_proof(req.record_index)
        leaf_hash = tree.leaves[req.record_index]
        return {
            "leaf_hash": leaf_hash,
            "proof": proof,
            "merkle_root": log_entry["merkle_root"],
            "record": records[req.record_index]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/verify-proof")
async def verify_proof(req: VerifyProofRequest):
    """Verify a cryptographic Merkle proof against a Merkle root."""
    try:
        reconstructed_proof = [(item[0], item[1]) for item in req.proof]
        is_valid = MerkleTree.verify_proof(req.leaf_hash, reconstructed_proof, req.merkle_root)
        return {"is_valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _update_local_redis_status():
    """Sync local training status to Redis if available."""
    if REDIS_AVAILABLE and state.redis_client is not None:
        try:
            state.redis_client.set("training_status", json.dumps(state.training_status))
        except Exception as e:
            logger.warning(f"Failed to sync local training status to Redis: {e}")


def run_background_training():
    """Background worker to run training pipeline asynchronously."""
    state.training_status = {
        "status": "running",
        "progress": 0,
        "epochs": 3,
        "current_epoch": 0,
        "loss": 0.0,
        "epsilon": 0.0,
        "error": None
    }
    _update_local_redis_status()
    
    try:
        DATASET_PATH = r"D:\PROJECTS\ML 2028\data set\creditcard.csv"
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
            
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATASET_PATH)
        joblib.dump(scaler, 'scaler.joblib')
        
        train_model = EnergyFraudClassifier(input_dim=29, num_classes=2)
        
        pipeline = SecureTrainingPipeline(
            model=train_model,
            X_train=X_train,
            y_train=y_train,
            config={
                "batch_size": 256,
                "noise_multiplier": 1.1,
                "max_grad_norm": 1.0,
                "ewc_enabled": False,
                "poison_filter_enabled": True,
                "poison_z_threshold": 2.0,
                "merkle_audit_enabled": False
            }
        )
        
        epochs = 3
        for epoch in range(epochs):
            metrics = pipeline.train_epoch(epoch)
            eps, delta = pipeline.log_privacy_budget(epoch)
            
            state.training_status["current_epoch"] = epoch + 1
            state.training_status["loss"] = round(metrics["total_loss"], 4)
            state.training_status["epsilon"] = round(eps, 2)
            state.training_status["progress"] = int(((epoch + 1) / epochs) * 100)
            _update_local_redis_status()
            
        unwrap_model = pipeline.model._module if hasattr(pipeline.model, "_module") else pipeline.model
        torch.save(unwrap_model.state_dict(), config.MODEL_PATH)
        
        state.model.load_state_dict(unwrap_model.state_dict())
        
        state.model.eval()
        calibrated_threshold = calibrate_energy_threshold(state.model, X_train, percentile=95.0)
        config.OOD_ENERGY_THRESHOLD = calibrated_threshold
        
        state.training_status["status"] = "completed"
        state.training_status["progress"] = 100
        state.training_status["calibrated_threshold"] = round(calibrated_threshold, 4)
        _update_local_redis_status()
        logger.info("Background training completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during background training: {e}")
        state.training_status["status"] = "error"
        state.training_status["error"] = str(e)
        _update_local_redis_status()


@app.post("/api/v1/train")
async def start_training(background_tasks: BackgroundTasks):
    """Trigger the differentially private training pipeline via Celery task or BackgroundTasks fallback."""
    status = await get_training_status()
    if status.get("status") == "running":
        return {"message": "Training is already in progress.", "status": status}

    if CELERY_AVAILABLE and REDIS_AVAILABLE and state.redis_client is not None:
        try:
            result = celery_train.delay()
            state.training_status = {
                "status": "running",
                "progress": 0,
                "epochs": 3,
                "current_epoch": 0,
                "loss": 0.0,
                "epsilon": 0.0,
                "error": None,
                "celery_task_id": result.id
            }
            state.redis_client.set("training_status", json.dumps(state.training_status))
            return {"message": "Training started via Celery task.", "status": state.training_status}
        except Exception as e:
            logger.warning(f"Failed to submit Celery task: {e}. Falling back to background thread.")

    background_tasks.add_task(run_background_training)
    state.training_status = {
        "status": "running",
        "progress": 0,
        "epochs": 3,
        "current_epoch": 0,
        "loss": 0.0,
        "epsilon": 0.0,
        "error": None
    }
    return {"message": "Training started in background.", "status": state.training_status}


@app.get("/api/v1/train/status")
async def get_training_status():
    """Get status of the training pipeline (synced via Redis for Celery)."""
    if REDIS_AVAILABLE and state.redis_client is not None:
        try:
            cached_status = state.redis_client.get("training_status")
            if cached_status:
                parsed = json.loads(cached_status)
                state.training_status = parsed
                return parsed
        except Exception as e:
            logger.warning(f"Failed to fetch status from Redis: {e}")
    return state.training_status


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main dashboard UI."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Dashboard frontend under construction. Make sure static files are built. Error: {e}</h3>",
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
