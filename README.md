# moex-vol-forecast

Daily volatility forecast for MOEX stocks — end-to-end ML pipeline.

## Goal
Predict next-day realized volatility for MOEX index constituents.

## Data
MOEX ISS API (free, daily candles).

## Planned stack
Python, Docker, MLflow, FastAPI, Airflow, PostgreSQL, Prometheus/Grafana, GitHub Actions.

## Roadmap
- [x] Stage 0: project structure, configs, logging, tests
- [ ] Stage 1: Docker
- [ ] Stage 2: MLflow tracking and model registry
- [ ] Stage 3: batch scoring + FastAPI service
- [ ] Stage 4: Airflow orchestration
- [ ] Stage 5: CI/CD
- [ ] Stage 6: monitoring (drift, Grafana)

## Run locally

```bash
uv sync
uv run --env-file .env fetch-candles
```

Settings come from environment variables: `TICKER` (default `SBER`), `FROM_DATE` (default `2026-10-01`), `DATA_DIR` (default `./data`), `LOG_LEVEL` (default `INFO`).

## Docker

```bash
# image for both Mac (arm64) and typical servers (amd64)
docker build --platform linux/amd64,linux/arm64 -t moex-loader:0.2.0 .
docker run --rm -v "$(pwd)/data:/app/data" --env-file .env moex-loader:0.2.0
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


## Schedule

On a server the fetch job runs daily at 00:15 MSK, after the evening session closes, so the previous day's candle is final:

```
PATH=/path/to/python/bin:/usr/bin:/bin
P=/path/to/moex-vol-forecast
15 0 * * * $P/scripts/fetch_candles.sh >> $P/logs/fetch.log 2>&1
```
