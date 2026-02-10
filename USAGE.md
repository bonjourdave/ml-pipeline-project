# Usage Guide

Detailed instructions for running the pipeline, managing the Feast registry, configuring Airflow, and troubleshooting.

---

## Table of Contents

1. [Running the Pipeline](#running-the-pipeline)
2. [Running src/ Modules Locally](#running-src-modules-locally)
3. [Feast Registry Management](#feast-registry-management)
4. [Airflow Configuration](#airflow-configuration)
5. [Troubleshooting](#troubleshooting)
6. [Production Next Steps](#production-next-steps)

---

## Running the Pipeline

The DAG is triggered manually (`schedule=None`). Each run:

1. **preprocess_data** — Reads `athletes.csv`, drops irrelevant columns, removes outliers, creates `total_lift` target, saves `athletes_clean.parquet`.
2. **engineer_features_v1 / v2** — Produce parquet files consumed by training tasks and registered in the Feast feature store.
3. **train_v*_trees*_depth*** — Each task:
   - Pulls feature paths from Feast registry
   - Trains a `RandomForestRegressor`
   - Logs params, metrics (RMSE, MAE, R²), plots, and emissions to MLFlow
   - Writes CodeCarbon output to `/tmp/emissions/` (temporary, also uploaded as MLFlow artifact)

---

## Running src/ Modules Locally

> **Important:** The `include/src/` modules use `/usr/local/airflow/` paths — the Airflow container mount — not `/workspace/`. Running them directly in the dev container **will fail** for anything that reads or writes processed data or the Feast registry.

### Options

**Option A — Run inside the Airflow scheduler container** (recommended, paths always correct):

```bash
docker exec -it $(docker ps --filter name=scheduler -q) bash

# Then inside the container:
python /usr/local/airflow/include/src/data/preprocess.py
python /usr/local/airflow/include/src/features/engineer.py
python /usr/local/airflow/include/src/models/train.py
```

**Option B — Run from the dev container using overridden paths:**

The `__main__` blocks in `preprocess.py` and `engineer.py` still use `/workspace/` paths and will work from the dev container as long as the MLFlow server is reachable (the dev container must be on `ml-network`).

`train.py` uses `/usr/local/airflow/...` for the Feast repo path in its `__main__` block, which won't resolve from the dev container. You would need to temporarily change it to `/workspace/...` for ad-hoc local runs.

**Option C — Update `devcontainer.json` network name:**

The dev container `runArgs` currently references `services_ml-network` (the old auto-prefixed name). Update it to `ml-network` to ensure connectivity:

```json
"runArgs": ["--network=ml-network"]
```

Then rebuild the container (Ctrl+Shift+P → "Rebuild Container") to access `mlflow-server:5000` directly from the dev container.

---

## Feast Registry Management

Feast stores feature view definitions (including data source paths) in a binary registry file at:

```
include/feast/repo/feature_repo/data/registry.db
```

**The registry is not auto-updated.** Any change to `feature_definitions.py` requires re-running:

```bash
docker exec -it $(docker ps --filter name=scheduler -q) bash -c \
  "cd /usr/local/airflow/include/feast/repo/feature_repo && feast apply"
```

**When must you re-run `feast apply`?**
- After changing `FileSource` paths in `feature_definitions.py`
- After adding or removing feature views or entities
- After first-time setup (the included `registry.db` may have stale paths from a different environment)

**To fully reset Feast state:**

```bash
docker exec -it $(docker ps --filter name=scheduler -q) bash -c \
  "rm -f /usr/local/airflow/include/feast/repo/feature_repo/data/registry.db \
         /usr/local/airflow/include/feast/repo/feature_repo/data/online_store.db && \
   cd /usr/local/airflow/include/feast/repo/feature_repo && feast apply"
```

---

## Airflow Configuration

Configuration lives in `airflow_settings.yaml` (local dev only — imported once at `astro dev start`):

| Type | Key | Value |
|---|---|---|
| Connection | `mlflow_tracking` | `http://mlflow-server:5000` |
| Variable | `FEAST_REPO_PATH` | `/usr/local/airflow/include/feast/repo/feature_repo` |
| Variable | `MLFLOW_TRACKING_URI` | `http://mlflow-server:5000` |

> **Note:** `airflow_settings.yaml` is only processed on startup. After editing it, run `astro dev restart` for changes to take effect.
>
> **Note:** Do not use `BaseHook.get_connection(...).host` to get a full URL — Airflow's `host` field holds only the hostname. Use an Airflow Variable storing the complete URI instead (as done here with `MLFLOW_TRACKING_URI`).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MLFlow is using SQLite instead of Postgres` | `mlflow.set_tracking_uri()` received a partial hostname instead of a full URL | Ensure `MLFLOW_TRACKING_URI` variable is set to `http://mlflow-server:5000` |
| `UnsupportedModelRegistryStoreURIException: generic://` | `Connection.get_uri()` returns `conn_type` as scheme; Airflow stores HTTP connections with `conn_type=generic` in some versions | Use an Airflow Variable for the URI instead of parsing the connection object |
| `Failed to resolve 'mlflow-server'` | Airflow scheduler not on `ml-network` | Verify `docker-compose.override.yml` is present and `astro dev restart` was run after adding it |
| `FileNotFoundError: /workspace/include/...` | Feast registry built outside the container; paths stored as `/workspace/` not `/usr/local/airflow/` | Reset and re-run `feast apply` from inside the scheduler container |
| `PermissionError: emissions/emissions.csv` | `include/emissions/` created with root ownership | CodeCarbon output now goes to `/tmp/emissions/` (always writable) |
| `No such file or directory: .../feast/data` | Feast `data/` directory never created | Run the `feast apply` command (it creates the directory) |

---

## Production Next Steps

This setup is intentionally simple for local experimentation. Moving it toward production would involve:

### Infrastructure

- **Managed Airflow**: Replace Astro CLI local with [Astro Cloud](https://www.astronomer.io/) or [MWAA](https://aws.amazon.com/managed-workflows-for-apache-airflow/) — eliminates local Docker management and adds auto-scaling.
- **Managed MLFlow**: Host on a cloud VM or use [Databricks MLFlow](https://docs.databricks.com/mlflow/index.html). Replace the local Postgres + volume setup with a managed database (RDS, Cloud SQL) and object storage (S3, GCS) for artifacts.
- **Managed Feast**: Swap the local SQLite stores for a production registry (e.g., Feast with a Postgres registry) and an online store (Redis, DynamoDB) for low-latency serving.

### Security

- **Secrets**: Move credentials out of `airflow_settings.yaml` into a secrets backend (HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager). The `.env` file at the project root is the immediate first step.
- **Network isolation**: Remove the `ports` exposure from the MLFlow service; route access through an internal load balancer or API gateway.
- **Least-privilege containers**: Airflow containers should not run as root. Use the `astro` user (Astro Runtime default) and pre-create directories with correct ownership rather than relying on runtime `mkdir`.

### CI/CD & Data Quality

- **Automated DAG tests**: The `.astro/test_dag_integrity_default.py` test suite validates DAG parse — extend it with task-level unit tests for `preprocess.py`, `engineer.py`, and `train.py`.
- **Data validation**: Add a [Great Expectations](https://greatexpectations.io/) or [Soda](https://www.soda.io/) step after preprocessing to assert schema and value constraints before feature engineering runs.
- **Model promotion**: Uncomment the `mlflow.sklearn.log_model()` call in `train.py` and add a post-training task that compares runs and promotes the best model to the MLFlow Model Registry with a `Staging` or `Production` alias.

### Feature Store

- **Decouple Feast apply from manual steps**: Add a `feast_apply` task at the start of the DAG (or a dedicated maintenance DAG) so the registry is always in sync without manual intervention.
- **Proper entity keys**: Replace the `range(len(features))` athlete_id with stable, real athlete identifiers to enable point-in-time correct feature retrieval in production.
