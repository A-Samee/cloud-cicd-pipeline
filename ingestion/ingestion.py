"""
ingestion/ingestion.py — Main data ingestion loop.

Polls a live HTTP API every POLL_INTERVAL seconds, appends records to a
local CSV, monitors schema changes, triggers drift detection, and exposes
Prometheus metrics.  Slack alerts are sent for significant events when a
webhook URL is configured.
"""

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Ensure the project root is on sys.path so "exporter" is importable
# when this script is executed directly (python ingestion/ingestion.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exporter.metrics import (  # noqa: E402
    datalake_unavailable_total,
    distribution_drift_detected,
    feature_added_total,
    feature_removed_total,
    records_processed_total,
    response_delay_seconds,
)
from ingestion.drift_detector import detect_drift  # noqa: E402
from model.retrain_trigger import retrain_on_event  # noqa: E402
from utils.slack import send_slack_alert             # noqa: E402

# ── Load .env if present (keeps secrets out of source control) ──
load_dotenv()

# ──────────────────────────────────────────────
# Configuration (all tuneable via environment)
# ──────────────────────────────────────────────
API_URL: str = os.getenv("API_URL", "http://149.40.228.124:6500/records")
POLL_INTERVAL: int = int(os.getenv("POLL_INTERVAL", "30"))
RETRAINING_DATA_THRESHOLD: int = int(os.getenv("RETRAINING_DATA_THRESHOLD", "200"))
# SLACK_WEBHOOK_URL is now read in utils/slack.py (single source of truth).

# ── Paths ──
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "records.csv"
SCHEMA_SNAPSHOT_PATH = DATA_DIR / "schema_snapshot.json"

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("ingestion")

# ── In-memory state ──
# Tracks total rows written since process start to decide when to retrain.
_rows_since_last_retrain: int = 0


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
# send_slack_alert  → imported from utils.slack (shared module)
# retrain_on_event  → imported from model.retrain_trigger


# ──────────────────────────────────────────────
# Schema tracking
# ──────────────────────────────────────────────

def _load_schema_snapshot() -> list[str] | None:
    """Load the last-seen schema from disk (returns None on first run)."""
    if not SCHEMA_SNAPSHOT_PATH.exists():
        return None
    with open(SCHEMA_SNAPSHOT_PATH, "r") as f:
        return json.load(f)


def _save_schema_snapshot(schema: list[str]) -> None:
    """Persist the current schema so we can detect changes across restarts."""
    SCHEMA_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_SNAPSHOT_PATH, "w") as f:
        json.dump(schema, f, indent=2)


def compare_schemas(old_schema: list[str], new_schema: list[str]) -> dict:
    """
    Return a dict with 'added' and 'removed' feature lists.

    This is also the function exercised by the unit tests in
    tests/test_schema.py.
    """
    old_set = set(old_schema)
    new_set = set(new_schema)
    return {
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
    }


def _handle_schema_change(new_schema: list[str]) -> None:
    """
    Compare the incoming schema against the saved snapshot.

    Logs, increments Prometheus counters, and fires Slack alerts for
    every added or removed feature.  Saves the new schema afterwards.
    """
    old_schema = _load_schema_snapshot()

    if old_schema is None:
        # First run — nothing to compare against yet.
        logger.info("[SCHEMA] Initial schema snapshot saved (%d features).", len(new_schema))
        _save_schema_snapshot(new_schema)
        return

    diff = compare_schemas(old_schema, new_schema)

    for name in diff["added"]:
        logger.warning("[SCHEMA] Feature added: %s", name)
        feature_added_total.inc()
        send_slack_alert(
            message="New feature detected in schema. Retraining may be required.",
            severity="warning",
            title="🟡 Feature Added to Schema"
        )

    for name in diff["removed"]:
        logger.warning("[SCHEMA] Feature removed: %s", name)
        feature_removed_total.inc()
        send_slack_alert(
            message="Feature dropped from schema. Verify pipeline compatibility.",
            severity="warning",
            title="🟡 Feature Removed from Schema"
        )

    # Persist the updated schema regardless of whether it changed,
    # so the snapshot always reflects the latest successful poll.
    if diff["added"] or diff["removed"]:
        _save_schema_snapshot(new_schema)
        # Schema changed → retrain so the model adapts to new/removed features.
        retrain_on_event(reason="schema_change")


# ──────────────────────────────────────────────
# CSV persistence
# ──────────────────────────────────────────────

def _append_to_csv(schema: list[str], records: list[list]) -> None:
    """
    Append rows to data/records.csv.

    Creates the file (with a header row) if it does not already exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(schema)  # header on first write
        writer.writerows(records)


# ──────────────────────────────────────────────
# Main polling loop
# ──────────────────────────────────────────────

def poll_once() -> None:
    """
    Execute a single poll-parse-store-detect cycle.

    Separated from the infinite loop so it can be unit-tested or
    invoked manually.
    """
    global _rows_since_last_retrain

    try:
        # Time the request so we can feed the histogram.
        start = time.monotonic()
        resp = requests.get(API_URL, timeout=15)
        elapsed = time.monotonic() - start
        response_delay_seconds.observe(elapsed)

    except requests.RequestException as exc:
        logger.error("[HTTP] Request failed: %s", exc)
        return

    # ── Handle 503 (simulated downtime) ──
    if resp.status_code == 503:
        logger.warning("[503] Data lake unavailable")
        datalake_unavailable_total.inc()
        send_slack_alert(
            message="Data source returned 503. Check API availability.",
            severity="critical",
            title="🔴 Data Lake Unavailable"
        )
        return

    if resp.status_code != 200:
        logger.error("[HTTP] Unexpected status %s", resp.status_code)
        return

    # ── Parse JSON payload ──
    # The API may return one of two formats:
    #   1. Documented: {"schema": ["f1", ...], "records": [[v1, ...], ...]}
    #   2. Actual (live): [{"features": [v1, v2], "label": 0}, ...]
    # We normalise both into a (schema, records) pair.
    try:
        payload = resp.json()

        if isinstance(payload, dict) and "schema" in payload:
            # Format 1 — documented shape
            schema: list[str] = payload["schema"]
            records: list[list] = payload["records"]
        elif isinstance(payload, list) and len(payload) > 0:
            # Format 2 — list of dicts from the live API.
            # Build schema by expanding "features" into feat_0 … feat_N
            # and keeping any other top-level keys (e.g. "label").
            first = payload[0]
            feature_count = len(first.get("features", []))
            # Generate feature names: feat_0, feat_1, …
            schema = [f"feat_{i}" for i in range(feature_count)]
            # Append any extra keys (like "label") in a stable order.
            extra_keys = sorted(k for k in first if k != "features")
            schema.extend(extra_keys)

            records = []
            for item in payload:
                row = list(item.get("features", []))
                for k in extra_keys:
                    row.append(item.get(k))
                records.append(row)
        else:
            raise ValueError(f"Unexpected payload type: {type(payload)}")
    except (ValueError, KeyError, TypeError) as exc:
        logger.error("[PARSE] Bad response payload: %s", exc)
        return

    batch_size = len(records)
    logger.info("[POLL] Received %d records with %d features.", batch_size, len(schema))

    # ── Schema monitoring ──
    _handle_schema_change(schema)

    # ── Persist to CSV ──
    _append_to_csv(schema, records)
    records_processed_total.inc(batch_size)

    # ── Drift detection ──
    drifted = detect_drift(records, schema)
    if drifted:
        distribution_drift_detected.set(1)
        send_slack_alert(
            message="Data distribution drift detected. Model may be stale.",
            severity="warning",
            title="🟡 Distribution Drift Detected"
        )
        # Distribution has shifted → retrain to keep the model relevant.
        retrain_on_event(reason="drift_detected")
    else:
        distribution_drift_detected.set(0)

    # ── Retraining trigger (enough new rows accumulated) ──
    _rows_since_last_retrain += batch_size
    if _rows_since_last_retrain >= RETRAINING_DATA_THRESHOLD:
        retrain_on_event(reason="new_data")
        _rows_since_last_retrain = 0  # reset after triggering


def run() -> None:
    """Infinite polling loop.  Press Ctrl-C to stop."""
    logger.info(
        "Starting ingestion loop — polling %s every %ds", API_URL, POLL_INTERVAL
    )
    while True:
        poll_once()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
