#!/usr/bin/env bash
# ──────────────────────────────────────────────
# deploy/deploy.sh — Build, push, and deploy the ML service to EC2
# ──────────────────────────────────────────────
#
# Usage:
#   export EC2_HOST=<your-ec2-ip>
#   export DOCKER_USERNAME=<your-dockerhub-user>
#   bash deploy/deploy.sh
#
# All sensitive values come from environment variables or .env.
# Nothing is hardcoded.
# ──────────────────────────────────────────────

set -e  # Abort immediately on any command failure.

# ── Load .env if present (local dev convenience) ──
if [ -f .env ]; then
  echo "📄 Loading .env file..."
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ──────────────────────────────────────────────
# Configuration — all from environment
# ──────────────────────────────────────────────
EC2_HOST="${EC2_HOST:?❌ EC2_HOST is not set. Export it or add to .env.}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_SSH_KEY="${EC2_SSH_KEY_PATH:-~/.ssh/mlops-key.pem}"
DOCKER_USERNAME="${DOCKER_USERNAME:?❌ DOCKER_USERNAME is not set.}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

IMAGE_NAME="${DOCKER_USERNAME}/mlops-app"
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")

echo "══════════════════════════════════════════"
echo "  MLOps Deploy"
echo "  EC2:   ${EC2_USER}@${EC2_HOST}"
echo "  Image: ${IMAGE_NAME}:latest (${GIT_SHA})"
echo "══════════════════════════════════════════"

# ──────────────────────────────────────────────
# 1. Build the Docker image from the project root
# ──────────────────────────────────────────────
echo ""
echo "🔨 Building Docker image..."
docker build \
  -f serving/Dockerfile \
  -t "${IMAGE_NAME}:latest" \
  -t "${IMAGE_NAME}:${GIT_SHA}" \
  .

# ──────────────────────────────────────────────
# 2. Push both tags to DockerHub
# ──────────────────────────────────────────────
echo ""
echo "🚀 Pushing to DockerHub..."
docker push "${IMAGE_NAME}:latest"
docker push "${IMAGE_NAME}:${GIT_SHA}"

# ──────────────────────────────────────────────
# 3. SSH into EC2 — pull & restart the container
# ──────────────────────────────────────────────
echo ""
echo "🖥  Deploying to EC2..."
ssh -o StrictHostKeyChecking=no -i "${EC2_SSH_KEY}" "${EC2_USER}@${EC2_HOST}" bash -s <<REMOTE_SCRIPT
  set -e
  echo "Pulling ${IMAGE_NAME}:latest..."
  docker pull ${IMAGE_NAME}:latest

  echo "Stopping old container..."
  docker stop mlops-app || true
  docker rm mlops-app   || true

  echo "Starting new container..."
  docker run -d --name mlops-app \
    -p 8000:8000 \
    -e SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
    --restart unless-stopped \
    ${IMAGE_NAME}:latest

  echo "Container started."
REMOTE_SCRIPT

# ──────────────────────────────────────────────
# 4. Health-check polling (up to 10 attempts, 3 s apart)
# ──────────────────────────────────────────────
echo ""
echo "🩺 Waiting for /health to return 200..."

HEALTH_URL="http://${EC2_HOST}:8000/health"
HEALTHY=false

for i in $(seq 1 10); do
  HTTP_CODE=$(curl -s -o /tmp/health_response.json -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null || echo "000")
  if [ "${HTTP_CODE}" = "200" ]; then
    HEALTHY=true
    echo "   ✅ Health check passed on attempt ${i}."
    break
  fi
  echo "   ⏳ Attempt ${i}/10 — HTTP ${HTTP_CODE}. Retrying in 3 s..."
  sleep 3
done

if [ "${HEALTHY}" != "true" ]; then
  echo "❌ Health check failed after 10 attempts. Deployment may have issues."
  exit 1
fi

# ──────────────────────────────────────────────
# 5. Slack notification on success
# ──────────────────────────────────────────────
# Extract model version from the health response.
MODEL_VERSION=$(cat /tmp/health_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_version','unknown'))" 2>/dev/null || echo "unknown")

echo ""
echo "🎉 Deployment successful! Model version: v${MODEL_VERSION}"

if [ -n "${SLACK_WEBHOOK_URL}" ]; then
  curl -s -X POST "${SLACK_WEBHOOK_URL}" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Deployment successful. EC2: ${EC2_HOST}. Model version: v${MODEL_VERSION}.\"}" \
    > /dev/null
  echo "📨 Slack notification sent."
else
  echo "ℹ️  SLACK_WEBHOOK_URL not set — skipping Slack notification."
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Done. Service: http://${EC2_HOST}:8000"
echo "══════════════════════════════════════════"
