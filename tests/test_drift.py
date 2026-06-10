"""
tests/test_drift.py — Unit tests for distribution drift detection.

Uses the tmp_path pytest fixture to isolate each test's baseline_stats.json
so tests don't pollute each other or the real data/ directory.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import drift_detector  # noqa: E402


def test_no_drift(tmp_path):
    """
    Feeding the same distribution twice should report NO drift.

    We create a baseline with mean≈5, std≈1, then pass a batch drawn
    from the same distribution. The z-score should stay well below
    the default threshold of 2.0.
    """
    baseline_path = tmp_path / "baseline_stats.json"

    # Pre-seed a baseline: mean=5.0, std=1.0 for a single feature.
    baseline = {
        "feat_0": {"mean": 5.0, "std": 1.0},
    }
    baseline_path.write_text(json.dumps(baseline))

    # Batch with values tightly centred around 5.0 (same distribution).
    records = [[5.1], [4.9], [5.0], [5.05], [4.95]]
    schema = ["feat_0"]

    # Monkeypatch BASELINE_PATH so the detector uses our isolated file.
    with patch.object(drift_detector, "BASELINE_PATH", baseline_path):
        result = drift_detector.detect_drift(records, schema)

    assert result is False, "Expected no drift for same-distribution data"


def test_drift_detected(tmp_path):
    """
    When the batch mean shifts far from the baseline, drift must be flagged.

    Baseline: mean≈0, std≈1. Batch: values around 10.
    The z-score ≈ 10.0, which is way beyond any reasonable threshold.
    """
    baseline_path = tmp_path / "baseline_stats.json"

    baseline = {
        "feat_0": {"mean": 0.0, "std": 1.0},
    }
    baseline_path.write_text(json.dumps(baseline))

    # Every value is ~10 — a massive shift from mean=0.
    records = [[10.0], [10.1], [9.9], [10.2], [9.8]]
    schema = ["feat_0"]

    with patch.object(drift_detector, "BASELINE_PATH", baseline_path):
        result = drift_detector.detect_drift(records, schema)

    assert result is True, "Expected drift when mean shifts by ~10 std devs"
