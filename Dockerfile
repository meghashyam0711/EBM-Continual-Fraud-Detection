# =====================================================================
# Energy-Based OOD Fraud Detection Framework — Production Dockerfile
# =====================================================================
# Multi-stage build for minimal image size and non-root execution.
#
# Build:  docker build -t fraud-api .
# Run:    docker run -p 8000:8000 --env-file .env fraud-api
# =====================================================================

# ----- Stage 1: Builder -----
FROM python:3.13-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies (gcc required for some C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into the virtual env
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ----- Stage 2: Runtime -----
FROM python:3.13-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application code
COPY model.py .
COPY pipeline.py .
COPY app.py .

# Optional: copy pre-trained model weights if available
# COPY model_weights.pt .

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default environment variables (overridable at runtime)
ENV MODEL_PATH="model_weights.pt" \
    INPUT_DIM="29" \
    NUM_CLASSES="2" \
    OOD_ENERGY_THRESHOLD="-5.0" \
    ENERGY_TEMPERATURE="1.0" \
    REDIS_HOST="redis" \
    REDIS_PORT="6379" \
    REDIS_TTL="300"

# Run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
