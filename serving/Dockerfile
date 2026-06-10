# ──────────────────────────────────────────────
# MLOps Inference Server — Dockerfile
# ──────────────────────────────────────────────
# Build from the PROJECT ROOT (not from serving/):
#   docker build -f serving/Dockerfile -t mlops-app .
#
# This copies the entire project so all relative imports
# (exporter/, utils/, model/, data/) resolve correctly.
# ──────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching — deps change less often than code).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project tree so shared modules are available.
COPY . .

EXPOSE 8000

# Start the FastAPI server via uvicorn.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
