#!/usr/bin/env bash
# ──────────────────────────────────────────────
# scripts/render_configs.sh
# ──────────────────────────────────────────────
# Substitutes environment variables from .env into config templates
# so that secrets (webhook URLs, IPs) are never committed to git.
#
# Usage:  bash scripts/render_configs.sh
# ──────────────────────────────────────────────

set -e

# Load .env variables into the current shell for envsubst.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "⚠️  .env not found — make sure environment variables are exported."
fi

# ── Render alertmanager config ──
envsubst < alertmanager/alertmanager.yml.template > alertmanager/alertmanager.yml
echo "[OK] alertmanager/alertmanager.yml rendered"

# ── Inject EC2_HOST into prometheus.yml if set ──
if [ -n "${EC2_HOST}" ]; then
  sed -i "s|<EC2_PUBLIC_IP>|${EC2_HOST}|g" prometheus/prometheus.yml
  echo "[OK] prometheus/prometheus.yml updated with EC2_HOST=${EC2_HOST}"
else
  echo "⚠️  EC2_HOST not set — prometheus/prometheus.yml still has placeholder."
fi
