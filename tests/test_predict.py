"""
tests/test_predict.py — Unit test for the /predict endpoint.

Uses FastAPI's TestClient with mocked model and metadata so the test
runs without a real .pkl file or trained model on disk.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_predict_endpoint():
    """
    POST /predict with valid features.

    Mocks:
      - joblib.load  → returns a fake model with .predict() and .predict_proba()
      - metadata file → returns a known version/accuracy/feature_names dict
      - MODEL_PATH.exists() and METADATA_PATH.exists() → True

    Asserts:
      - HTTP 200
      - Response has "prediction" key
      - "confidence" is a float
    """
    # ── Build the fake model ──
    fake_model = MagicMock()
    fake_model.predict.return_value = [1]
    fake_model.predict_proba.return_value = [[0.1, 0.9]]

    fake_metadata = {
        "version": 1,
        "accuracy": 0.91,
        "feature_names": ["a", "b"],
    }
    fake_metadata_json = json.dumps(fake_metadata)

    # We need to patch *before* importing serving.app because the module
    # loads the model at import time (top-level code).
    # The patches must cover:
    #   1. Path.exists() → always True (so the guard doesn't sys.exit)
    #   2. joblib.load   → our fake model
    #   3. builtins.open → our fake metadata JSON

    with patch("pathlib.Path.exists", return_value=True), \
         patch("joblib.load", return_value=fake_model), \
         patch("builtins.open", mock_open(read_data=fake_metadata_json)):

        # Remove any previously cached import so patches take effect.
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("serving"):
                del sys.modules[mod_name]

        # Now import — the patched top-level code will run.
        from serving.app import app  # noqa: E402

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.post(
            "/predict",
            json={"features": {"a": 1.0, "b": 2.0}},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    body = response.json()
    assert "prediction" in body, "Response must contain 'prediction' key"
    assert "confidence" in body, "Response must contain 'confidence' key"
    assert isinstance(body["confidence"], float), (
        f"'confidence' should be float, got {type(body['confidence'])}"
    )
