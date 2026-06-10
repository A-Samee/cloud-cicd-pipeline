#!/usr/bin/env bash
# ──────────────────────────────────────────────
# scripts/trigger_all_alerts.sh
# ──────────────────────────────────────────────
# Fires all 6 alert conditions in sequence.
# Each one sends a Slack message directly (no Prometheus needed).
#
# Usage:  bash scripts/trigger_all_alerts.sh
# ──────────────────────────────────────────────

set -e

# Load .env so SLACK_WEBHOOK_URL is available to Python.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "══════════════════════════════════════════"
echo "  Triggering all 6 alert conditions..."
echo "══════════════════════════════════════════"
echo ""

if [ -d "venv" ]; then
  source venv/bin/activate
fi

python3 scripts/trigger_alerts.py --alert datalake
python3 scripts/trigger_alerts.py --alert feature_added
python3 scripts/trigger_alerts.py --alert feature_removed
python3 scripts/trigger_alerts.py --alert drift
python3 scripts/trigger_alerts.py --alert latency
python3 scripts/trigger_alerts.py --alert accuracy

echo ""
echo "══════════════════════════════════════════"
echo "  Done. Check your Slack channel."
echo "══════════════════════════════════════════"
