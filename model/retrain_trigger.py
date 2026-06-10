"""
model/retrain_trigger.py — Decides when to retrain and dispatches training.

Provides two entry points:
  • check_and_retrain()  — evaluates the current model on recent data and
                            retrains if accuracy has dropped.
  • retrain_on_event()   — unconditionally retrains (called by the ingestion
                            loop on drift, schema changes, or data thresholds).

Can be run standalone:  python -m model.retrain_trigger
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

# ── Ensure project root is importable ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exporter.metrics import retrain_count_total  # noqa: E402
from model.train import run_training              # noqa: E402
from utils.slack import send_slack_alert           # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
CSV_PATH = Path("data/records.csv")
METADATA_PATH = Path("model/model_metadata.json")
CURRENT_MODEL_PATH = Path("model/current_model.pkl")

# Accuracy threshold — below this we retrain automatically.
ACCURACY_FLOOR: float = 0.80


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def check_and_retrain() -> float | None:
    """
    Evaluate the current model on a held-out slice of the CSV.

    If no metadata/model exists → run initial training.
    If live accuracy < ACCURACY_FLOOR → retrain with reason "accuracy_drop".

    Returns
    -------
    float or None
        The current (or new) accuracy, or None if evaluation was not possible.
    """
    # ── No metadata yet → first training ever ──
    if not METADATA_PATH.exists():
        logger.info("[RETRAIN] No model metadata found. Running initial training.")
        return run_training(reason="initial")

    # ── Guard: need both the model and data ──
    if not CURRENT_MODEL_PATH.exists():
        logger.warning("[RETRAIN] current_model.pkl missing. Running fresh training.")
        return run_training(reason="initial")

    if not CSV_PATH.exists():
        logger.warning("[RETRAIN] records.csv missing — cannot evaluate model.")
        return None

    # ── Load model and data ──
    model = joblib.load(CURRENT_MODEL_PATH)

    df = pd.read_csv(CSV_PATH)
    df.dropna(inplace=True)

    if len(df) < 10:
        logger.warning("[RETRAIN] Not enough data to evaluate (%d rows).", len(df))
        return None

    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]

    # Held-out validation slice: last 20 % of rows by index.
    # We use index-based slicing (not random) so the split is deterministic
    # and always uses the most-recent data, which is the best proxy for
    # current distribution.
    split_idx = int(len(df) * 0.80)
    X_val = df.iloc[split_idx:][feature_cols].values
    y_val = df.iloc[split_idx:][target_col].values

    # ── Predict and measure ──
    try:
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
    except Exception as exc:
        # Model may be incompatible with new schema (different feature count).
        logger.error(
            "[RETRAIN] Inference failed (%s). Triggering retraining.", exc
        )
        return retrain_on_event(reason="inference_error")

    logger.info("[RETRAIN] Live accuracy on last 20%%: %.4f", acc)

    if acc < ACCURACY_FLOOR:
        logger.warning(
            "[RETRAIN] Accuracy dropped to %.4f. Triggering retraining.", acc
        )
        retrain_count_total.inc()
        new_acc = run_training(reason="accuracy_drop")
        send_slack_alert(
            message=f"Model retrained due to accuracy drop. Old: {acc:.4f} → New: {new_acc:.4f}.",
            severity="critical",
            title="🔴 Low Model Accuracy"
        )
        return new_acc

    return acc


def retrain_on_event(reason: str) -> float | None:
    """
    Unconditionally retrain the model for a given reason.

    Called by the ingestion loop when:
      • enough new data has accumulated ("new_data")
      • distribution drift is detected ("drift_detected")
      • the schema changes ("schema_change")

    Returns
    -------
    float or None
        New accuracy after retraining, or None if training was skipped.
    """
    logger.info("[RETRAIN] Event-triggered retraining — reason: %s", reason)
    retrain_count_total.inc()
    new_acc = run_training(reason=reason)
    if new_acc is not None:
        send_slack_alert(
            message=f"Model retrained (event: {reason}). Accuracy: {new_acc:.4f}.",
            severity="success",
            title="✅ Model Retrained Successfully"
        )
    return new_acc


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    result = check_and_retrain()
    if result is not None:
        print(f"Current accuracy: {result:.4f}")
    else:
        print("Could not evaluate or train (see logs above).")
