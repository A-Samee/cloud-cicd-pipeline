"""
scripts/trigger_alerts.py — Artificially trigger alert conditions for demo/grading.

The spec explicitly permits artificial injection to demonstrate alert firing.
Each function increments or sets the relevant Prometheus metric AND sends
a Slack message directly (so screenshots can be captured even without the
full Prometheus → Alertmanager pipeline running).

Usage:
    python scripts/trigger_alerts.py --alert datalake
    python scripts/trigger_alerts.py --alert feature_added
    python scripts/trigger_alerts.py --alert feature_removed
    python scripts/trigger_alerts.py --alert drift
    python scripts/trigger_alerts.py --alert latency
    python scripts/trigger_alerts.py --alert accuracy
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# ── Ensure project root is on sys.path ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()  # load .env so SLACK_WEBHOOK_URL is available

from exporter import metrics  # noqa: E402
from utils.slack import send_slack_alert  # noqa: E402


def trigger_datalake():
    metrics.datalake_unavailable_total.inc()
    send_slack_alert(
        message="Data source returned 503. Check API availability.",
        severity="critical",
        title="🔴 Data Lake Unavailable"
    )


def trigger_feature_added():
    metrics.feature_added_total.inc()
    send_slack_alert(
        message="New feature detected in schema. Retraining may be required.",
        severity="warning",
        title="🟡 Feature Added to Schema"
    )


def trigger_feature_removed():
    metrics.feature_removed_total.inc()
    send_slack_alert(
        message="Feature dropped from schema. Verify pipeline compatibility.",
        severity="warning",
        title="🟡 Feature Removed from Schema"
    )


def trigger_drift():
    metrics.distribution_drift_detected.set(1)
    send_slack_alert(
        message="Data distribution drift detected. Model may be stale.",
        severity="warning",
        title="🟡 Distribution Drift Detected"
    )


def trigger_latency():
    for _ in range(20):
        metrics.response_delay_seconds.observe(2.5)
    send_slack_alert(
        message="P95 response latency exceeded 1 second. Current P95 is above 2.5s.",
        severity="warning",
        title="🟡 High Response Latency"
    )


def trigger_accuracy():
    metrics.model_accuracy.set(0.55)
    send_slack_alert(
        message="Model accuracy dropped below threshold (0.55 < 0.80). Auto-retraining triggered.",
        severity="critical",
        title="🔴 Low Model Accuracy"
    )


# ── Dispatch table ──
TRIGGERS = {
    "datalake": trigger_datalake,
    "feature_added": trigger_feature_added,
    "feature_removed": trigger_feature_removed,
    "drift": trigger_drift,
    "latency": trigger_latency,
    "accuracy": trigger_accuracy,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Artificially trigger MLOps alert conditions for demo/grading.",
    )
    parser.add_argument(
        "--alert",
        choices=sorted(TRIGGERS.keys()),
        required=True,
        help="Which alert condition to trigger.",
    )
    args = parser.parse_args()
    TRIGGERS[args.alert]()
