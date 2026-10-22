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
- [ ] Stage 5: CI/CD
- [ ] Stage 6: monitoring (drift, Grafana)

## CI

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) checks every push to `main` and every pull request on a clean Ubuntu machine: `uv sync --locked`, `ruff check`, `ruff format --check` and `pytest`, including the database tests against a PostgreSQL 18 service container.

When the checks pass, the `image` job builds the Docker image for `linux/amd64` and `linux/arm64`. Builds from `main` are pushed to GitHub Container Registry as `ghcr.io/ayazsung/moex-vol` with the tags `main` (the latest `main`) and `sha-<commit>`; pull requests only build the image.

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

One image, `moex-vol:<version>` (the version from `pyproject.toml`), runs every job and service described in `compose.yaml`. The services listen on localhost only.

```bash
docker compose build                                        # build the image
docker compose run --rm loader                              # one-off job: fetch candles (settings from .env)
docker build --platform linux/amd64,linux/arm64 -t moex-vol:0.6.1 .   # also for typical servers (amd64)
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
| `moex_vol_daily` | 00:15, Tue–Sat | fetch the last week of candles for 21 tickers (3 at a time, 2 retries) → batch forecast → check that all 21 tickers got a fresh forecast |
| `moex_vol_retrain` | 02:00, Sat | train a candidate on all data except the last 13 weeks → compare it with the champion on those 13 weeks → move the `champion` alias if the candidate is at least 1% better → restart the API if it serves an old version → check the API |

The champion/challenger step by hand:

```bash
docker compose run --rm trainer train-model --model har --register --test-start "$(date -v-13w +%F)"
docker compose run --rm trainer promote-model        # --min-improvement 0.01 by default
```

Every candidate version gets tags in the MLflow registry: the comparison window, both metrics and the decision (`promoted` or `rejected`).
