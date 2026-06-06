import os
import sys
import logging
import torch
import joblib
import json
from celery import Celery
from celery.signals import task_failure, task_success

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import EnergyFraudClassifier, load_and_preprocess_data, calibrate_energy_threshold
from pipeline import SecureTrainingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"]
)

def _update_redis_status(state_dict):
    """Helper to update training status in Redis."""
    try:
        import redis
        client = redis.Redis(host=REDIS_HOST, port=int(REDIS_PORT), db=int(REDIS_DB))
        client.set("training_status", json.dumps(state_dict))
    except Exception as e:
        logger.warning(f"Failed to update status in Redis: {e}")

@celery_app.task(bind=True)
def run_background_training(self):
    """Background Celery worker to run training pipeline asynchronously."""
    status = {
        "status": "running",
        "progress": 0,
        "epochs": 3,
        "current_epoch": 0,
        "loss": 0.0,
        "epsilon": 0.0,
        "error": None
    }
    _update_redis_status(status)
    
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
            
            status["current_epoch"] = epoch + 1
            status["loss"] = round(metrics["total_loss"], 4)
            status["epsilon"] = round(eps, 2)
            status["progress"] = int(((epoch + 1) / epochs) * 100)
            _update_redis_status(status)
            
        unwrap_model = pipeline.model._module if hasattr(pipeline.model, "_module") else pipeline.model
        MODEL_PATH = os.getenv("MODEL_PATH", "model_weights.pt")
        torch.save(unwrap_model.state_dict(), MODEL_PATH)
        
        calibrated_threshold = calibrate_energy_threshold(unwrap_model, X_train, percentile=95.0)
        
        status["status"] = "completed"
        status["progress"] = 100
        status["calibrated_threshold"] = round(calibrated_threshold, 4)
        _update_redis_status(status)
        logger.info("Background training completed successfully.")
        return status
        
    except Exception as e:
        logger.error(f"Error during background training: {e}")
        status["status"] = "error"
        status["error"] = str(e)
        _update_redis_status(status)
        raise e
