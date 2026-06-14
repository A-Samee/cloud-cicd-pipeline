[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NZbV6Puz)

# MLOps Course Project

End-to-end MLOps pipeline with data ingestion, schema monitoring, drift detection, automated model retraining, inference serving, and full observability.

## Team

- **Name:** Abdul Samee
- **Roll Number:** 22i-1051



## Video Demo

📹 [Watch the full system demo]
(https://drive.google.com/file/d/1zZoFWHLLqBOdrvgOuFfufiObvlc6htK0/view?usp=sharing)

## Configuring Slack Webhook

1. Go to https://api.slack.com/apps → Create App → Incoming Webhooks
2. Activate incoming webhooks → Add to Workspace → select your channel
3. Copy the webhook URL
4. Add to `.env` as `SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...`
5. Run `bash scripts/render_configs.sh` to inject it into alertmanager config

## Live AWS EC2 Endpoints

The inference server is deployed at `http://3.110.56.107:8000`

Test /health:
```bash
curl http://3.110.56.107:8000/health
```

Test /predict:
```bash
curl -X POST http://3.110.56.107:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 1.5, "feature_1": -0.3, "feature_2": 2.1}}'
```

Test /metrics:
```bash
curl http://3.110.56.107:8000/metrics
```

## Project Structure

```
├── .github/workflows/      # Phase 7 — CI/CD
│   └── mlops-ci.yml        # Lint → build → deploy pipeline
├── ingestion/              # Phase 1 — data ingestion & schema monitoring
│   ├── ingestion.py        # Polling loop, CSV storage, Slack alerts
│   └── drift_detector.py   # Statistical drift detection (z-score)
├── model/                  # Phase 2 — training & auto-retraining
│   ├── train.py            # RandomForest training pipeline
│   └── retrain_trigger.py  # Accuracy-based & event-based retraining
├── serving/                # Phase 3 — FastAPI inference server
│   ├── app.py              # /predict, /metrics, /health endpoints
│   └── Dockerfile          # Container image (build from project root)
├── deploy/                 # Phase 3 — EC2 deployment
│   └── deploy.sh           # Build → push → SSH deploy → health check
├── exporter/               # Phase 4 — Prometheus metrics
│   └── metrics.py          # All 8 metric definitions
├── prometheus/             # Phase 5 — Prometheus config
│   ├── prometheus.yml      # Scrape config targeting EC2
│   └── alert_rules.yml     # 7 alert rules
├── grafana/                # Phase 5 — Grafana provisioning
│   ├── datasources/        # Prometheus datasource
│   └── dashboards/         # Dashboard provisioner + JSON dashboard
├── alertmanager/           # Phase 5 — Alertmanager
│   └── alertmanager.yml.template  # Slack webhook template
├── scripts/                # Phase 6 — alert triggers & config rendering
│   ├── render_configs.sh   # Inject .env secrets into config templates
│   ├── trigger_alerts.py   # Fire individual alert conditions
│   └── trigger_all_alerts.sh  # Fire all 6 alerts at once
├── utils/
│   └── slack.py            # Shared Slack alert helper
├── tests/                  # pytest test suite
├── docker-compose.yml      # Observability stack (Prometheus + Grafana + Alertmanager)
├── requirements.txt
└── .env.example            # Environment variable template
```

## Quick Start

```bash
# 0. Clone the repo
git clone https://github.com/NUCES-ISB/course-project-Samee212.git
cd course-project-Samee212

# 1. Create virtual environment and install deps
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Copy and fill in your environment config
cp .env.example .env
# Edit .env — set SLACK_WEBHOOK_URL, EC2_HOST, DOCKER_USERNAME

# 3. Run ingestion (polls the live API every 30s)
python -m ingestion.ingestion

# 4. Train a model (needs ≥50 rows in data/records.csv)
python -m model.train

# 5. Start the inference server locally
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# 6. Run tests
pytest tests/ -v
```

## Running the Observability Stack

```bash
# Inject SLACK_WEBHOOK_URL and EC2_HOST into config templates
bash scripts/render_configs.sh

# Start Prometheus, Grafana, and Alertmanager
# Use whichever command your Docker version supports:
docker-compose up -d        # Docker Compose v1
# docker compose up -d      # Docker Compose v2 (plugin)

# Prometheus UI: http://localhost:9090
# Grafana UI:    http://localhost:3000  (admin/admin)
# Alertmanager:  http://localhost:9093
```

## Deploying to EC2

```bash
# Set required env vars (or add to .env)
export EC2_HOST=<your-ec2-ip>
export DOCKER_USERNAME=<your-dockerhub-user>

# Build, push, and deploy
bash deploy/deploy.sh
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run inference: `{"features": {"feat_0": 0.5, "feat_1": -0.3}}` |
| `/health` | GET | Liveness probe: `{"status": "ok", "model_version": N, "accuracy": float}` |
| `/metrics` | GET | Prometheus metrics (text format) |

## Triggering Alerts for Demonstration

The spec explicitly permits artificial injection to demonstrate alert firing.

### Fire all alerts at once (sends Slack messages directly):

```bash
bash scripts/trigger_all_alerts.sh
```

### Fire a specific alert:

```bash
python scripts/trigger_alerts.py --alert datalake
python scripts/trigger_alerts.py --alert feature_added
python scripts/trigger_alerts.py --alert feature_removed
python scripts/trigger_alerts.py --alert drift
python scripts/trigger_alerts.py --alert latency
python scripts/trigger_alerts.py --alert accuracy
```

### Alert Rules Reference

| # | Alert Name | Expression | Fires After |
|---|---|---|---|
| 1 | DataLakeUnavailable | `increase(datalake_unavailable_total[1m]) > 0` | 0m |
| 2 | FeatureAdded | `increase(feature_added_total[1m]) > 0` | 0m |
| 3 | FeatureRemoved | `increase(feature_removed_total[1m]) > 0` | 0m |
| 4 | DistributionDrift | `distribution_drift_detected == 1` | 0m |
| 5 | FeatureDriftDetected | `distribution_drift_detected > 0` | 0m |
| 6 | HighResponseLatency | `histogram_quantile(0.95, rate(response_delay_seconds_bucket[5m])) > 1.0` | 2m |
| 7 | LowModelAccuracy | `model_accuracy < 0.8` | 0m |

> **Note:** To trigger alerts via Prometheus (requires stack running + EC2 live),
> the alerts fire automatically when conditions are met during real operation.
> For grading screenshots, the trigger scripts above send Slack messages directly.

## GitHub Actions CI/CD

The pipeline runs automatically on every push to `main`.

### Required GitHub repository secrets

Go to Settings → Secrets and Variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| DOCKER_USERNAME | Your DockerHub username |
| DOCKER_PASSWORD | DockerHub access token (not your password) |
| EC2_HOST | `<your-ec2-public-ip>` |
| EC2_USER | ubuntu |
| EC2_SSH_KEY | Full contents of mlops-key.pem |
| SLACK_WEBHOOK_URL | Your Slack webhook URL |

### Jobs

1. **Lint and test** — flake8 + pytest (5 tests)
2. **Build and push** — builds Docker image, pushes `:latest` and `:<git-sha>` to DockerHub
3. **Deploy to EC2** — SSHs in, pulls latest image, restarts container, verifies `/health`

## Screenshots

All evidence screenshots are in the `screenshots/` directory:

| File | Shows |
|---|---|
| ec2_instance_running.png | AWS EC2 instance in Running state |
| ec2_health.png | GET /health returning 200 with model version and accuracy |
| ec2_predict.png | POST /predict returning prediction and confidence |
| ec2_metrics_raw.png | GET /metrics returning Prometheus text format |
| prometheus_target_up.png | Prometheus target health showing EC2 as UP |
| grafana_dashboard.png | Grafana dashboard with all 6 panels showing live data |
| slack_alert_datalake.png | Slack alert: Data Lake Unavailable (CRITICAL) |
| slack_alert_feature_added.png | Slack alert: Feature Added to Schema (WARNING) |
| slack_alert_feature_removed.png | Slack alert: Feature Removed from Schema (WARNING) |
| slack_alert_drift.png | Slack alert: Distribution Drift Detected (WARNING) |
| slack_alert_latency.png | Slack alert: High Response Latency (WARNING) |
| slack_alert_accuracy.png | Slack alert: Low Model Accuracy (CRITICAL) |
| github_actions_green.png | GitHub Actions: all 3 jobs passing (green) |

## Prometheus Metrics (Phase 4)

All 8 metrics exposed at `GET /metrics`:

| Metric | Type | Description |
|---|---|---|
| `model_accuracy` | Gauge | Current validation accuracy (0.0–1.0) |
| `records_processed_total` | Counter | Total records ingested since startup |
| `retrain_count_total` | Counter | Total model retrains triggered |
| `distribution_drift_detected` | Gauge | 1 if drift detected, 0 otherwise |
| `feature_added_total` | Counter | Features added to schema since startup |
| `feature_removed_total` | Counter | Features removed from schema since startup |
| `datalake_unavailable_total` | Counter | Times /records returned 503 |
| `response_delay_seconds` | Histogram | /predict endpoint latency in seconds |
