"""
ingestion/drift_detector.py — Statistical drift detection.

Compares each numeric feature's mean against a stored baseline.
If the mean shifts by more than DRIFT_THRESHOLD standard deviations,
drift is reported for that feature.

Baseline stats are persisted to data/baseline_stats.json so the
detector survives process restarts.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # honour .env if present

logger = logging.getLogger(__name__)

# Number of standard deviations a feature mean must shift before we flag drift.
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "2.0"))

# Where the per-feature baseline (mean, std) is persisted.
BASELINE_PATH: Path = Path("data/baseline_stats.json")


def _load_baseline() -> dict | None:
    """Return the saved baseline dict, or None if no baseline exists yet."""
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def _save_baseline(baseline: dict) -> None:
    """Persist the baseline dict to disk so it survives restarts."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)


def _compute_stats(records: list[list], schema: list[str]) -> dict:
    """
    Compute mean and std for every *numeric* feature in the batch.

    Non-numeric columns (strings, None, etc.) are silently skipped because
    statistical drift only makes sense for continuous values.
    """
    stats: dict = {}
    for col_idx, feature_name in enumerate(schema):
        # Extract the column; guard against ragged rows.
        values = []
        for row in records:
            if col_idx < len(row):
                try:
                    values.append(float(row[col_idx]))
                except (TypeError, ValueError):
                    # Skip non-numeric values (e.g. categorical strings).
                    continue

        if len(values) < 2:
            # Not enough data points for a meaningful std; skip.
            continue

        arr = np.array(values)
        stats[feature_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),  # sample std (ddof=1)
        }
    return stats


def detect_drift(records: list[list], schema: list[str]) -> bool:
    """
    Compare the current batch's statistics to the saved baseline.

    Parameters
    ----------
    records : list[list]
        A list of rows, each row being a list of raw values.
    schema : list[str]
        Column names corresponding to the positions in each row.

    Returns
    -------
    bool
        True if any numeric feature's mean has shifted by more than
        DRIFT_THRESHOLD standard deviations from the baseline.
    """
    if not records:
        logger.warning("[DRIFT] Empty batch received — skipping drift check.")
        return False

    current_stats = _compute_stats(records, schema)
    baseline = _load_baseline()

    # ── First call: no baseline yet → save and return False ──
    if baseline is None:
        logger.info("[DRIFT] No baseline found. Saving current batch as baseline.")
        _save_baseline(current_stats)
        return False

    # ── Subsequent calls: compare to baseline ──
    drift_found = False
    for feature, cur in current_stats.items():
        # Handle gracefully if the baseline doesn't have this feature
        # (schema may have changed since the baseline was saved).
        if feature not in baseline:
            logger.info(
                "[DRIFT] Feature '%s' not in baseline (new feature?) — skipping.",
                feature,
            )
            continue

        base = baseline[feature]
        base_std = base["std"]

        # If baseline std is 0 (constant feature), any change counts as drift.
        if base_std == 0:
            if cur["mean"] != base["mean"]:
                logger.warning(
                    "[DRIFT] Feature '%s' was constant (std=0) but mean changed "
                    "from %.4f to %.4f.",
                    feature,
                    base["mean"],
                    cur["mean"],
                )
                drift_found = True
            continue

        # Z-score of the batch mean relative to the baseline distribution.
        z_score = abs(cur["mean"] - base["mean"]) / base_std
        if z_score > DRIFT_THRESHOLD:
            logger.warning(
                "[DRIFT] Feature '%s' drifted: z=%.2f (threshold=%.1f). "
                "Baseline mean=%.4f, current mean=%.4f.",
                feature,
                z_score,
                DRIFT_THRESHOLD,
                base["mean"],
                cur["mean"],
            )
            drift_found = True

    return drift_found
