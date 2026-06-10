"""
serving/app.py — FastAPI inference server.

Loads the trained model on startup and exposes:
  POST /predict  — run inference on a feature dict
  GET  /metrics  — Prometheus scrape endpoint
  GET  /health   — lightweight liveness / readiness check

Designed to run inside Docker on EC2:
  uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import sys
import time
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# ── Ensure project root is on sys.path so shared modules resolve ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exporter.metrics import model_accuracy, response_delay_seconds  # noqa: E402

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("serving")

# ──────────────────────────────────────────────
# Paths (relative to project root — the Docker WORKDIR)
# ──────────────────────────────────────────────
MODEL_PATH = Path("model/current_model.pkl")
METADATA_PATH = Path("model/model_metadata.json")

# ──────────────────────────────────────────────
# Load model & metadata at import time
# ──────────────────────────────────────────────
# FastAPI creates the app object at import, so loading here means the
# model is ready before the first request arrives.

if not MODEL_PATH.exists():
    logger.critical("Model file not found: %s — cannot start server.", MODEL_PATH)
    sys.exit(1)

if not METADATA_PATH.exists():
    logger.critical("Metadata file not found: %s — cannot start server.", METADATA_PATH)
    sys.exit(1)

# Load the sklearn model once; it stays in memory for the process lifetime.
_model = joblib.load(MODEL_PATH)
logger.info("Loaded model from %s", MODEL_PATH)

with open(METADATA_PATH, "r") as _f:
    _metadata: dict = json.load(_f)
logger.info(
    "Loaded metadata — version: v%s, accuracy: %.4f, features: %s",
    _metadata["version"],
    _metadata["accuracy"],
    _metadata.get("feature_names", []),
)

# The trained feature order — inference must align to this exact order.
_feature_names: list[str] = _metadata["feature_names"]

# Expose the trained accuracy on the Prometheus gauge immediately so
# /metrics returns a meaningful value even before any retraining occurs.
model_accuracy.set(_metadata["accuracy"])

# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="MLOps Inference Server",
    version=str(_metadata["version"]),
    description="Serves predictions from the latest trained model.",
)


# ──────────────────────────────────────────────
# Request / response schemas
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Incoming prediction request.

    `features` is a dict mapping feature names to numeric values.
    Extra features are silently dropped; missing features default to 0.
    """
    features: dict[str, float]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/predict")
def predict(body: PredictRequest):
    """
    Run model inference on the provided features.

    Steps:
      1. Align incoming feature dict to the trained feature order.
      2. Fill missing features with 0 (safe default for tree-based models).
      3. Drop extra features that the model was not trained on.
      4. Return prediction, confidence, and model version.
    """
    start = time.monotonic()

    # Build the feature vector in the exact order the model expects.
    # Missing features get 0; extra features are simply ignored.
    row = [body.features.get(f, 0.0) for f in _feature_names]

    prediction = _model.predict([row])[0]
    probabilities = _model.predict_proba([row])[0]

    # Confidence = probability of the predicted class.
    confidence = float(max(probabilities))

    elapsed = time.monotonic() - start
    response_delay_seconds.observe(elapsed)

    # Convert numpy types to Python builtins for JSON serialization.
    return {
        "prediction": int(prediction) if hasattr(prediction, "item") else prediction,
        "confidence": confidence,
        "model_version": _metadata["version"],
    }


@app.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint.

    Returns all registered metrics in the Prometheus text exposition format.
    No need for start_http_server — we serve metrics directly through FastAPI.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health():
    """
    Lightweight health / readiness probe.

    Always returns HTTP 200 with the current model version and accuracy.
    Used by the deploy script to confirm the container is alive.
    """
    return {
        "status": "ok",
        "model_version": _metadata["version"],
        "accuracy": _metadata["accuracy"],
    }
