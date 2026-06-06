# ENERGY-SHIELD // Real-Time DP Fraud & Auditing Dashboard

An unassailable, production-grade financial fraud detection framework combining **Energy-Based Out-of-Distribution (OOD) Scoring**, **Differential Privacy (DP-SGD)**, **Continual Learning (EWC)**, and **Cryptographic Merkle-based Auditing**.

---

##  Key Features

1. **Energy-Based OOD Detection**
   - Combines traditional classification with OOD energy scoring to flag novel, unseen fraud patterns.
   - Automatically calibrates threshold values at startup.

2. **Differentially Private & Continual Training**
   - Minimizes privacy leakage using Opacus (DP-SGD) with customizable privacy budgets.
   - Utilizes Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
   - Spectral Poison Filtering detects and isolates clean-label poisoned samples before training.

3. **Cryptographic Merkle Auditing**
   - Automatically builds a Merkle Tree for incoming transaction batches.
   - Exposes audit endpoints for full cryptographic lineage proof verification.
   - Includes a dynamic record selector in the UI to verify individual transaction authenticity.

4. **Robust Async Architecture**
   - Integrates Celery + Redis for asynchronous, decoupled training execution.
   - Falls back gracefully to FastAPI's asynchronous `BackgroundTasks` when running in single-machine setups without Redis/Celery.
   - Saves preprocess `StandardScaler` state to ensure consistent feature normalization across training and real-time inference.

---

## 🛠 Tech Stack

- **Core ML**: PyTorch, Scikit-Learn, NumPy
- **Differential Privacy**: Opacus
- **API & Serving**: FastAPI, Uvicorn, Pydantic
- **Task Orchestration**: Celery (Optional, falls back to `BackgroundTasks`)
- **Caching & Brokers**: Redis (Optional)
- **Monitoring**: Prometheus
- **Frontend**: Custom HTML5/CSS3 dashboard with high-end glassmorphic theme and micro-animations

---

## Directory Structure

```text
├── app.py             # FastAPI Inference & API Server
├── tasks.py           # Celery Task Worker Configuration & Logic
├── model.py           # PyTorch Energy Classifier, Data Loader, and Calibration
├── pipeline.py        # Secure DP-SGD Training, Spectral Filter, and Merkle Auditing
├── train.py           # Independent local CLI training pipeline
├── credit.py          # Synthetic dataset simulation & helper
├── requirements.txt   # Core Python dependencies
├── static/
│   ├── index.html     # Real-time Web Dashboard UI
│   ├── app.js         # Reactive API Integration & Merkle visualizer
│   └── styles.css     # CSS Styling & Animations
```

---

##  Quick Start

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Locally (Standard Mode)
Run the application server using Uvicorn:
```bash
python app.py
```
By default, the server runs on `http://localhost:8000`.

### 3. Run with Celery & Redis (Distributed Mode)
If you have a Redis broker installed:
1. Start Redis on port `6379`.
2. Install Celery:
   ```bash
   pip install celery
   ```
3. Start the Celery worker process:
   ```bash
   celery -A tasks worker --loglevel=info
   ```
4. Start `app.py` as normal.

---

##  Verification & Auditing

1. Open the dashboard in a browser: `http://localhost:8000`.
2. Generate inferences using the presets or random feature vectors.
3. Observe new batches listed in the **Immutable Audit Logs**.
4. Select a batch to view its Merkle root and dynamic transaction list.
5. Click **Verify Cryptographic Root** to assert the integrity of the record against the Merkle tree, or test tampering with **Verify Tampered Hash**.
