"""
model/train.py — Model training pipeline.

Loads ingested CSV data, trains a RandomForestClassifier with iterative
n_estimators escalation until a target accuracy is met (or max iterations
are exhausted), then serializes the model as a versioned .pkl artifact.

Can be run standalone:  python -m model.train
"""

import glob
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Ensure project root is importable when invoked as a script ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exporter.metrics import model_accuracy  # noqa: E402
from utils.slack import send_slack_alert      # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Minimum rows required to attempt training (avoids overfitting on tiny data).
MIN_ROWS: int = 50

# Target validation accuracy — we keep trying with more estimators until we
# reach this threshold or exhaust MAX_RETRAIN_ITERS attempts.
TARGET_ACCURACY: float = 0.80

# Maximum extra training iterations with escalating n_estimators.
MAX_RETRAIN_ITERS: int = int(os.getenv("MAX_RETRAIN_ITERS", "5"))

# ── Paths (all relative to project root so they work from any cwd) ──
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "records.csv"
SCHEMA_SNAPSHOT_PATH = DATA_DIR / "schema_snapshot.json"
MODEL_DIR = Path("model")
METADATA_PATH = MODEL_DIR / "model_metadata.json"
CURRENT_MODEL_PATH = MODEL_DIR / "current_model.pkl"


# ──────────────────────────────────────────────
# Versioning helpers
# ──────────────────────────────────────────────

def _next_model_version() -> int:
    """
    Scan model/model_v*.pkl and return max(N) + 1.
    Returns 1 if no versioned models exist yet.
    """
    pattern = str(MODEL_DIR / "model_v*.pkl")
    existing = glob.glob(pattern)
    if not existing:
        return 1

    versions = []
    for path in existing:
        match = re.search(r"model_v(\d+)\.pkl$", path)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1


# ──────────────────────────────────────────────
# Core training logic
# ──────────────────────────────────────────────

def run_training(reason: str = "new_data") -> float | None:
    """
    Full training pipeline: load → split → train → save → alert.

    Parameters
    ----------
    reason : str
        Human-readable context for why training was triggered
        (e.g. "drift_detected", "accuracy_drop", "schema_change", "new_data").

    Returns
    -------
    float or None
        Validation accuracy of the saved model, or None if training
        was skipped (insufficient data / missing CSV).
    """
    logger.info("[TRAIN] Starting training pipeline (reason: %s)", reason)

    # ── Guard: CSV must exist ──
    if not CSV_PATH.exists():
        logger.warning(
            "[TRAIN] %s does not exist. Ingestion has not run yet — skipping.",
            CSV_PATH,
        )
        return None

    # ── Load data ──
    df = pd.read_csv(CSV_PATH)
    df.dropna(inplace=True)

    if len(df) < MIN_ROWS:
        logger.warning(
            "[TRAIN] Only %d rows available (minimum %d). Skipping training.",
            len(df),
            MIN_ROWS,
        )
        return None

    # Last column is the target label; everything else is a feature.
    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]

    X = df[feature_cols].values
    y = df[target_col].values

    logger.info(
        "[TRAIN] Loaded %d rows — %d features, target='%s'.",
        len(df),
        len(feature_cols),
        target_col,
    )

    # ── Stratified train/validation split ──
    # Stratify ensures class proportions are preserved in both sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    # ── Iterative training with escalating n_estimators ──
    best_acc = 0.0
    best_model = None
    n_estimators = 100  # starting point (sklearn default-ish)

    for iteration in range(1, MAX_RETRAIN_ITERS + 1):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        logger.info(
            "[TRAIN] Iteration %d/%d — n_estimators=%d, val_accuracy=%.4f",
            iteration,
            MAX_RETRAIN_ITERS,
            n_estimators,
            acc,
        )

        if acc > best_acc:
            best_acc = acc
            best_model = clf

        if acc >= TARGET_ACCURACY:
            logger.info("[TRAIN] Target accuracy %.2f reached.", TARGET_ACCURACY)
            break

        # Escalate complexity for the next attempt.
        n_estimators += 50
    else:
        # Loop completed without hitting target — proceed with best.
        logger.warning(
            "[TRAIN] Target accuracy not reached. Best: %.4f",
            best_acc,
        )

    # ── Serialize the model ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    version = _next_model_version()
    versioned_path = MODEL_DIR / f"model_v{version}.pkl"
    joblib.dump(best_model, versioned_path)
    logger.info("[TRAIN] Saved %s", versioned_path)

    # Copy (not symlink) to current_model.pkl so it works cross-platform
    # and avoids issues with relative symlinks in different working dirs.
    shutil.copy2(versioned_path, CURRENT_MODEL_PATH)
    logger.info("[TRAIN] Updated %s → v%d", CURRENT_MODEL_PATH, version)

    # ── Save metadata ──
    metadata = {
        "version": version,
        "accuracy": round(best_acc, 6),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "rows_used": len(df),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("[TRAIN] Metadata written to %s", METADATA_PATH)

    # ── Update Prometheus gauge ──
    model_accuracy.set(best_acc)

    # ── Slack notification ──
    send_slack_alert(
        message=f"Model retrained (reason: {reason}). Version: v{version}. Accuracy: {best_acc:.4f}.",
        severity="success",
        title="✅ Model Retrained Successfully"
    )

    return best_acc


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    acc = run_training(reason="manual")
    if acc is not None:
        print(f"Training complete. Accuracy: {acc:.4f}")
    else:
        print("Training skipped (see logs above).")
