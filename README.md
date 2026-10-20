# Iris Continuous Training System

End-to-end MLOps pipeline on the Iris dataset — drift detection, automated
retraining, evaluation gates, and MLflow tracking throughout.

---

## Files at a Glance

| File | Purpose |
|------|---------|
| `config.yaml` | All thresholds, paths, SLOs, MLflow settings — edit here, not in code |
| `src/data_generator.py` | Creates `reference.csv`, `production.csv` (with drift), `golden.csv` |
| `src/train.py` | Trains RandomForest → logs to MLflow → saves candidate model |
| `src/evaluate.py` | Runs 5 gates → promotes candidate to MLflow Model Registry |
| `src/monitor.py` | Computes PSI / KS / JSD drift + EvidentlyAI report → MLflow run |
| `src/retrain_trigger.py` | Evaluates all 4 trigger types → logs decision to MLflow |
| `src/pipeline.py` | **One command to run everything** — monitor → trigger → train → evaluate → promote |
| `src/monitoring_service.py` | Runs the scheduled pipeline and exposes Prometheus metrics |
| `Dockerfile` / `compose.yaml` | Builds and runs the local monitoring demo |
| `monitoring/` | Prometheus configuration and provisioned Grafana dashboard |
| `requirements.txt` | All Python dependencies |
| `src/tests/test_pipeline.py` | 17 unit tests |
| `.github/workflows/continuous_training.yml` | GitHub Actions CI/CD |

---

## Execution Order

### Step 1 — First-time setup

```bash
pip install -r requirements.txt
python src/data_generator.py        # creates data/reference.csv, production.csv, golden.csv
```

### Step 2 — Initial training and deployment

```bash
python src/train.py                 # trains model, logs to MLflow, saves models/candidate_model.pkl
python src/evaluate.py --promote    # runs 5 gates, promotes to MLflow Registry as Production
```

At this point your model is live in the MLflow Model Registry (stage = Production).

### Step 3 — Check drift and monitoring

```bash
python src/monitor.py               # runs PSI / KS / JSD on all features, saves drift_report.html
```

Open `reports/drift_report.html` in a browser for the interactive EvidentlyAI report.

### Step 4 — Check whether retraining is needed

```bash
python src/retrain_trigger.py       # evaluates time / performance / drift / active-learning triggers
```

Prints `DECISION: RETRAIN 🚨` or `DECISION: NO ACTION ✅`.

### Step 5 — Retrain (if triggered)

```bash
python src/train.py --combine       # retrains on reference + production data combined
python src/evaluate.py --promote    # runs gates again, promotes new version if it passes
```

---

## One-Command Automated Pipeline

Instead of running steps 3–5 manually, run the full pipeline:

```bash
python src/pipeline.py              # monitor → trigger → retrain (if needed) → evaluate → promote
python src/pipeline.py --retrain    # skip trigger check, force a retrain
python src/pipeline.py --dry-run    # run everything but do NOT promote
python src/pipeline.py --monitor-only  # check drift and stop, no retrain
```

This is what the GitHub Actions workflow runs on every push and weekly schedule.

The Actions workflow keeps model building separate from inference monitoring:
it generates data, trains/evaluates and caches an initial Production model if
one does not yet exist, then runs drift/performance monitoring against that
model. Retraining runs only when the inference trigger fires (or you force it
from `workflow_dispatch`), followed by all five evaluation gates and promotion.
Model files and MLflow tracking/registry state are transferred between jobs and
cached for subsequent workflow runs.

---

## View Results in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Every training run, monitoring cycle, trigger decision, and pipeline execution
is logged here with full metrics, artefacts, and tags.

---

## What Gets Logged to MLflow

| Script | Experiment | Logged |
|--------|-----------|--------|
| `src/train.py` | `iris_ct` | Hyperparams, CV scores, accuracy/F1/recall per class, feature importances, confusion matrix PNG, model → Registry |
| `src/evaluate.py` | `iris_ct` (same run) | Gate pass/fail, Registry transition None→Staging→Production |
| `src/monitor.py` | `iris_ct_monitoring` | PSI/KS/JSD per feature, class shift, EvidentlyAI HTML report |
| `src/retrain_trigger.py` | `iris_ct_monitoring` | Per-trigger metrics, final retrain=true/false tag |
| `src/pipeline.py` | `iris_ct` | Parent run with all child runs nested inside, pipeline duration, outcome |

Load the Production model at any time:
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/iris_rf_model/Production")
```

---

## The Four Retraining Triggers

| # | Trigger | Fires when | Config |
|---|---------|-----------|--------|
| 1 | **Time-based** | Last retrain > 7 days ago | `triggers.time_based.interval_days` |
| 2 | **Performance** | Accuracy < 0.85 or F1 < 0.82 on labelled production data | `slo.accuracy.critical` |
| 3 | **Drift-based** | PSI > 0.20 or KS p-value < 0.05 on any feature | `drift.psi.trigger` |
| 4 | **Active learning** | > 20% of predictions below 70% confidence | `src/retrain_trigger.py` |

## The Five Evaluation Gates

All must pass before a candidate is promoted to Production in the Registry:

| # | Gate | What it checks |
|---|------|---------------|
| 1 | Baseline comparison | New model ≥ current Production model accuracy (−0.5% margin) |
| 2 | Minimum thresholds | Accuracy ≥ 0.85, F1 ≥ 0.82 — hard floors |
| 3 | Golden dataset | 100% correct on 15 curated representative examples |
| 4 | Per-class recall | Every class ≥ 0.80 recall — no hidden class failures |
| 5 | Distribution check | No missing values, class balance ≥ 5%, at least 30 samples |

## Run Tests

```bash
pytest src/tests/ -v    # 17 tests
```

---

## Local Docker Demo: Prometheus + Grafana

This demo runs three local containers: the MLOps pipeline/metrics exporter,
Prometheus, and Grafana. GitHub Actions builds the application image and
publishes it to GitHub Container Registry (GHCR) after the lint and unit-test
job passes. The build context, Dockerfile, and platform are configured in
the **Build & Publish Docker Image** job in
`.github/workflows/continuous_training.yml`.

### Pull the image built by GitHub Actions

After a workflow run completes, open **Actions → Iris — Continuous Training
Pipeline → the run**. The **Build & Publish Docker Image** job and the workflow
run summary include a link to the GHCR package and a copyable command similar
to `docker pull ghcr.io/galandeshridhar0-a11y/mlops-continuous-training-retraining-automation:run-123456789`.
The unique `run-<run-id>` tag identifies that exact build. The commit tag
(`sha-<commit>`) is also pushed, and successful builds from `main` update
`:latest`.

For a private GHCR package, authenticate locally with a GitHub personal access
token that has `read:packages`:

```powershell
$env:GHCR_TOKEN = Read-Host "GitHub token with read:packages"
$env:GHCR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
Remove-Item Env:GHCR_TOKEN
```

Copy the exact `docker pull` command from the Actions summary and run it. Then
use the downloaded image with the repository's Compose configuration:

```powershell
$env:MLOPS_IMAGE = "ghcr.io/OWNER/REPOSITORY:run-RUN_ID"
docker compose -f compose.yaml -f compose.image.yaml up -d
docker compose -f compose.yaml -f compose.image.yaml ps
```

Replace `OWNER/REPOSITORY:run-RUN_ID` with the exact image reference from the
workflow. `compose.image.yaml` overrides the local build configuration to run
the pulled image. Compose still uses this repository for Prometheus and Grafana
configuration and automatically pulls their pinned upstream images.

If you are running a fork, Actions publishes to that fork's GHCR package; use
the package/image URL shown in that workflow run rather than the example above.

### Start the demo (Windows PowerShell)

Run these commands from the repository root:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f mlops-demo
```

The first startup creates demo data, trains and evaluates a baseline model,
then immediately runs the monitoring pipeline. Allow a few minutes for initial
training and model evaluation. Leave the logs command running to watch each
pipeline stage; press `Ctrl+C` to stop following logs (the containers continue
running).

To run from source instead of using the GitHub Actions image:

```powershell
docker pull prom/prometheus:v2.54.1
docker pull grafana/grafana:11.2.0
docker build -t iris-mlops-demo:local .
docker compose up --no-build -d
```

### See the retraining trigger

The default Compose configuration enables `DEMO_MODE`. It writes a reproducible
labelled production batch of 120 rows with a deliberately shifted feature; this
exceeds the configured drift sample minimum and makes the drift trigger fire.
The service starts a pipeline run immediately and checks again every five
minutes. To see the trigger without waiting for the next cycle, follow the
first run's logs:

```powershell
docker compose logs -f mlops-demo
```

Wait until the logs say `Demo record endpoint available at
http://0.0.0.0:8001/demo/records` before posting records; the app trains its
baseline model before enabling the ingestion endpoint.

Look for the `MONITOR — Drift & Performance` section reporting a drift trigger,
then `TRIGGER — Should we retrain?` with `RETRAIN`, followed by the `TRAIN —
Retrain on combined data` and `EVALUATE — All 5 gates` sections. A trigger means
the pipeline attempted retraining; the evaluation gates can still reject the
candidate and prevent model promotion.

### Open Grafana and Prometheus

| Service | URL | Login |
|---------|-----|-------|
| Grafana dashboard | <http://localhost:3000/d/iris-mlops-demo/iris-mlops-retraining-demo> | `admin` / `admin` |
| Prometheus | <http://localhost:9090> | — |
| Application metrics | <http://localhost:8000/metrics> | — |
| Demo record ingestion API | <http://localhost:8001/demo/records> | — |

Grafana's provisioned **Iris MLOps Retraining Demo** dashboard displays feature
PSI, the monitoring drift signal, the retraining decision, individual trigger
reasons, model accuracy/F1, and pipeline duration. Metrics appear after the
first pipeline cycle completes. In Prometheus, open **Graph**, enter
`mlops_monitor_drift_trigger`, `mlops_retrain_trigger`, or
`mlops_retrain_decisions_total`, then select **Execute**. A value of `1` for
either trigger gauge means that the most recent completed cycle detected drift
or decided to retrain, respectively; the counter increases each time retraining
is triggered.

### Prometheus alert rules

Prometheus loads [monitoring/alert_rules.yml](./monitoring/alert_rules.yml)
on startup. After updating an already-running stack, apply the rules with
`docker compose restart prometheus`. To view current and pending alerts, open
<http://localhost:9090/alerts>. The rules cover a retraining decision, PSI above
the `0.20` drift threshold, pipeline failure, an unreachable metrics exporter,
and no successful pipeline run for 15 minutes. Rules evaluate after the
configured `for` duration; for example, a retraining alert should appear
shortly after the triggered run finishes.

These rules make alerts visible in Prometheus; they do not send email, Slack, or
push notifications. A notification receiver (typically Alertmanager) must be
configured separately if you need alerts delivered outside the Prometheus UI.

### Push test records to trigger retraining now

You can submit a few labelled Iris records to the local demo API. It validates
the records, repeats them to meet the configured minimum drift sample count,
adds a controlled `sepal_width` shift, saves the batch as production data, and
runs the monitoring/retraining pipeline immediately. The request waits until
training and evaluation finish, so allow several minutes on the first request.
This endpoint is enabled only when `DEMO_MODE=true` (the Compose default).

In a second Windows PowerShell window, run:

```powershell
$payload = @{
  records = @(
    @{ sepal_length = 5.1; sepal_width = 3.5; petal_length = 1.4; petal_width = 0.2; species = 0 },
    @{ sepal_length = 6.4; sepal_width = 3.2; petal_length = 4.5; petal_width = 1.5; species = 1 },
    @{ sepal_length = 6.3; sepal_width = 3.3; petal_length = 6.0; petal_width = 2.5; species = 2 }
  )
} | ConvertTo-Json -Depth 5

$result = Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8001/demo/records" `
  -ContentType "application/json" `
  -TimeoutSec 600 `
  -Body $payload

$result | Format-List
```

The response should report `records_received: 3`, at least 50
`production_rows`, and `monitor_trigger: True`, `retrain: True`, with `drift`
in `triggered_by`. The pipeline logs should then show the `TRIGGER`, `TRAIN`,
and `EVALUATE` stages. In Grafana, refresh **Iris MLOps Retraining Demo** and
look for `mlops_retrain_trigger = 1` and the drift trigger reason. The API
returns the pipeline `outcome`; retraining may be triggered even if the
candidate later fails evaluation gates and is not promoted.

To verify Prometheus directly after the request:

```powershell
(Invoke-RestMethod "http://localhost:9090/api/v1/query?query=mlops_retrain_trigger").data.result
```

For a faster repeat demo, set a shorter interval before starting Compose:

```powershell
$env:PIPELINE_INTERVAL_SECONDS = "60"
docker compose up --build -d
```

This interval controls how often another complete monitoring/retraining
decision runs; the first run is still immediate. Clear the PowerShell override
after stopping the stack with `Remove-Item Env:PIPELINE_INTERVAL_SECONDS`.

View recent logs, then stop the stack:

```powershell
docker compose logs --tail 100 mlops-demo
docker compose down
```

Data, models, MLflow state, and monitoring history persist in Docker volumes.
To start over with a clean baseline, remove those volumes:

```bash
docker compose down -v
```

To build the application image without starting the full stack:

```bash
docker build -t iris-mlops-demo:local .
```
