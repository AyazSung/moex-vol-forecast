# moex-vol-forecast

[![CI](https://github.com/AyazSung/moex-vol-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/AyazSung/moex-vol-forecast/actions/workflows/ci.yml)

Daily volatility forecast for MOEX stocks — end-to-end ML pipeline.

## Goal
Predict next-day realized volatility for MOEX index constituents.

## Data
MOEX ISS API (free, daily candles).

## Planned stack
Python, Docker, MLflow, FastAPI, Airflow, PostgreSQL, Prometheus/Grafana, GitHub Actions.

## Roadmap
- [x] Stage 0: project structure, configs, logging, tests
- [x] Stage 1: Docker
- [x] Stage 2: MLflow tracking and model registry
- [x] Stage 3: batch scoring + FastAPI service
- [x] Stage 4: Airflow orchestration
- [x] Stage 5: CI/CD
- [x] Stage 6: monitoring (forecast quality, drift, Grafana dashboards, Telegram alerts)

## CI

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) checks every push to `main` and every pull request on a clean Ubuntu machine: `uv sync --locked`, `ruff check`, `ruff format --check` and `pytest`, including the database tests against a PostgreSQL 18 service container.

When the checks pass, the `image` job builds the Docker image for `linux/amd64` and `linux/arm64`. Builds from `main` are pushed to GitHub Container Registry as `ghcr.io/ayazsung/moex-vol` with the tags `main` (the latest `main`) and `sha-<commit>`; pull requests only build the image.

## Releases

1. Bump the version (`uv version --bump patch` or `minor`) and the image tag in `compose.yaml` in a pull request, merge it.
2. Tag the merge commit and push the tag:

   ```bash
   git tag -a v0.6.2 -m "Release 0.6.2"
   git push origin v0.6.2
   ```

   CI checks that the tag matches `pyproject.toml`, publishes `ghcr.io/ayazsung/moex-vol:0.6.2` (also `0.6` and `latest`) and creates a GitHub Release with the merged pull requests.
3. Deploy: `docker compose pull mlflow api && docker compose up -d mlflow api`, then check that `curl http://localhost:8000/openapi.json` reports the new version.

## Development

`main` is protected: every change goes through a pull request, which can be merged only when the `checks` and `image` CI jobs pass.

```bash
uv sync                        # environment with the dev tools
uv run pre-commit install      # run ruff and file checks before every commit
git switch -c my-change        # work in a branch, push it and open a pull request
```

Dependabot ([`.github/dependabot.yml`](.github/dependabot.yml)) opens weekly pull requests with updates of Python packages, GitHub Actions and pre-commit hooks; releases younger than 14 days are skipped.

## Run locally

```bash
uv sync
uv run --env-file .env fetch-candles
```

Settings come from environment variables: `TICKER` (default `SBER`), `FROM_DATE` (default `2026-10-01`), `DATA_DIR` (default `./data`), `LOG_LEVEL` (default `INFO`).

## Docker

One image, `ghcr.io/ayazsung/moex-vol:<version>`, runs every job and service described in `compose.yaml`. CI builds it for `linux/amd64` and `linux/arm64` and publishes it to GitHub Container Registry; `compose.yaml` pulls it instead of building it. The services listen on localhost only.

```bash
docker compose pull mlflow api              # download the image version set in compose.yaml
docker compose up -d db mlflow api          # start or update the long-running services
docker compose up -d prometheus grafana     # monitoring: Prometheus and Grafana dashboards
docker compose run --rm loader              # one-off job: fetch candles (settings from .env)
docker build -t moex-vol:dev .              # build locally to try a Dockerfile change
```

## Database (docker compose)

```bash
docker compose up -d db                          # PostgreSQL in the background
docker compose run --rm loader                   # fetch candles from MOEX into CSV and DB
docker compose run --rm loader import-candles    # load existing CSV files into DB (no internet)
docker compose exec db psql -U moex -d moex      # SQL console
docker compose down                              # stop; data stays in the pgdata volume
```

## Model training (MLflow)

```bash
uv run train-model --model har               # train a model and log the run to MLflow
uv run train-model --model gbm --register    # also add the model to the registry
uv run mlflow ui --port 5001                 # MLflow UI at http://localhost:5001
uv run predict-vol                           # next-day forecast with the champion model
```

## Model serving (MLflow server, batch scoring, API)

```bash
docker compose exec db createdb -U moex mlflow     # once: a database for MLflow
docker compose up -d db mlflow                     # MLflow UI at http://localhost:5001
export MLFLOW_TRACKING_URI=http://localhost:5001   # train and register into the server
uv run train-model --model har --register
docker compose run --rm predictor                  # batch forecasts into the forecasts table
docker compose up -d api                           # API at http://localhost:8000/docs
```

## Orchestration (Airflow)

Airflow 3 runs on the Mac in its own environment (`~/airflow/.venv`). The DAGs live in `dags/` and start the project's Docker jobs.

```bash
./scripts/airflow.sh                                 # Airflow UI at http://localhost:8080, stop with Ctrl+C
./scripts/airflow.sh dags list-runs moex_vol_daily   # runs of a DAG from the terminal
```

| DAG | Schedule (Moscow time) | Steps |
|---|---|---|
| `moex_vol_daily` | 00:15, Tue–Sat | fetch the last week of candles for 21 tickers (3 at a time, 2 retries) → batch forecast → check that all 21 tickers got a fresh forecast → compare past forecasts with what happened (fails if the serving model is worse than naive over 20 days); in parallel with the forecast, check the champion's input features for drift (fails on drift above the historical limit or on values outside the training range) |
| `moex_vol_retrain` | 02:00, Sat | train a candidate on all data except the last 13 weeks → compare it with the champion on those 13 weeks → move the `champion` alias if the candidate is at least 1% better → restart the API if it serves an old version → check the API → backfill the new champion's forecasts over those 13 weeks |
| `moex_alert_test` | manual | a task that always fails: checks that failure alerts reach Telegram |

When a run of `moex_vol_daily` or `moex_vol_retrain` fails, Airflow sends one Telegram message for the whole run: the DAG, the last failed task and a link to its log (see Alerts below).

The champion/challenger step by hand:

```bash
docker compose run --rm trainer train-model --model har --register --test-start "$(date -v-13w +%F)"
docker compose run --rm trainer promote-model        # --min-improvement 0.01 by default
```

Every candidate version gets tags in the MLflow registry: the comparison window, both metrics and the decision (`promoted` or `rejected`).

## Monitoring

A forecast made on day `as_of` can be checked once the next trading day has closed. `evaluate-forecasts` compares every stored forecast with the actual volatility of that day and with the naive forecast "tomorrow as today", using the training metric `rmse_to_naive` (below 1 means better than naive). The results go to the `forecast_quality` table, one row per day and model version; `rmse_to_naive_20d` is the metric over the last 20 days of the version.

```bash
docker compose run --rm predictor evaluate-forecasts                    # exit code 1 if the serving version's 20-day rmse_to_naive is above 1.0
docker compose run --rm predictor backfill-forecasts                    # add the champion's forecasts over its test period
docker compose run --rm predictor backfill-forecasts --model-version 1  # the same for any version
docker compose exec db psql -U moex -d moex -c "SELECT * FROM forecast_quality ORDER BY as_of DESC LIMIT 5"
```

The threshold of 1.0 was chosen on history. For the HAR model trained on data before 2024 (version 1), the 20-day `rmse_to_naive` from January 2024 to October 2026 stayed between 0.64 and 1.01 and was above 1.0 on one day, 13 February 2025. The one-day metric is above 1 on about every sixth day, so it is not used for alerts.

**Input drift.** `check-drift` compares the champion's input features with its training data. For every feature it computes the PSI (population stability index) of the last 20 trading days against the training distribution and counts tickers whose value is outside the training range. The PSI limit is not the textbook 0.25: twenty consecutive days share one market regime, so even inside the training period a 20-day window of a volatility feature usually has a PSI above 1. The limit is the 99th percentile of the PSI of all 20-day windows of the training period. Results go to the `feature_drift` table, one row per day, model version and feature.

```bash
docker compose run --rm predictor check-drift                                         # exit code 1 on drift above the limit or values outside the training range
docker compose run --rm predictor check-drift --model-version 2 --as-of 2024-07-16    # a past day for any version: VTBR volume after its reverse split
docker compose exec db psql -U moex -d moex -c "SELECT * FROM feature_drift ORDER BY as_of DESC, feature LIMIT 6"
```

**Dashboards.** The API exposes its metrics at `/metrics`: requests by path and status code, response time, the model version it serves and online forecasts. Prometheus scrapes them every 15 seconds and keeps 90 days of history. Grafana shows two dashboards provisioned from files in `monitoring/grafana`: *Модель: качество и дрейф* reads the `forecast_quality` and `feature_drift` tables from PostgreSQL through a read-only user, *Сервис API* reads Prometheus. Both are only reachable from this machine.

```bash
docker compose exec -T db psql -U moex -d moex < monitoring/grafana_user.sql  # once: read-only database user for Grafana
docker compose up -d prometheus grafana                                       # Grafana: http://localhost:3000, Prometheus: http://localhost:9090
curl -s http://localhost:8000/metrics | grep '^api_'                          # what Prometheus reads from the API
```

**Alerts.** Failures come to Telegram from two places. Airflow calls `dags/alerts.py` when a run of the daily or the retraining DAG fails. Grafana evaluates two rules every minute, provisioned from `monitoring/grafana/provisioning/alerting/alerts.yml`: *API не отвечает* (Prometheus cannot scrape the API for 2 minutes) and *Ежедневный конвейер не отработал* (no forecast was written by a scheduled nightly run, which means Airflow itself is not running). The bot token and the chat id live in `.env.alerts`, which is not in Git; without it everything works, only no messages are sent.

```bash
cp .env.alerts.example .env.alerts     # then put the bot token and the chat id into .env.alerts
./scripts/airflow.sh                   # restart Airflow to read .env.alerts
docker compose up -d grafana           # recreate Grafana to read .env.alerts
```

Trigger `moex_alert_test` in Airflow to check that a message arrives.
