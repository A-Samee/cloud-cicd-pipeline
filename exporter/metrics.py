"""
exporter/metrics.py — Centralized Prometheus metric definitions.

All 8 metrics are defined here so every module (ingestion, drift detector,
training, serving) imports from a single source of truth.  This avoids
duplicate-metric-registration errors that prometheus_client raises when
the same metric name is instantiated more than once.

No scrape server is started here — metrics are served via the FastAPI
/metrics route in serving/app.py.
"""

from prometheus_client import Counter, Gauge, Histogram

# ──────────────────────────────────────────────
# 1. Model performance
# ──────────────────────────────────────────────

model_accuracy = Gauge(
    "model_accuracy",
    "Current validation accuracy of the deployed model (0.0-1.0)",
)

# ──────────────────────────────────────────────
# 2. Ingestion volume
# ──────────────────────────────────────────────

records_processed_total = Counter(
    "records_processed_total",
    "Total records ingested from API since startup",
)

# ──────────────────────────────────────────────
# 3. Retraining
# ──────────────────────────────────────────────

retrain_count_total = Counter(
    "retrain_count_total",
    "Total number of model retrains",
)

# ──────────────────────────────────────────────
# 4. Drift detection
# ──────────────────────────────────────────────

distribution_drift_detected = Gauge(
    "distribution_drift_detected",
    "Set to 1 when drift detected in current batch, 0 otherwise",
)

# ──────────────────────────────────────────────
# 5–6. Schema changes
# ──────────────────────────────────────────────

feature_added_total = Counter(
    "feature_added_total",
    "Number of features added to schema since startup",
)

feature_removed_total = Counter(
    "feature_removed_total",
    "Number of features removed from schema since startup",
)

# ──────────────────────────────────────────────
# 7. Availability
# ──────────────────────────────────────────────

datalake_unavailable_total = Counter(
    "datalake_unavailable_total",
    "Number of times /records returned 503",
)

# ──────────────────────────────────────────────
# 8. Latency
# ──────────────────────────────────────────────

response_delay_seconds = Histogram(
    "response_delay_seconds",
    "Latency of each /predict API call in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
