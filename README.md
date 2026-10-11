# moex-vol-forecast

Daily volatility forecast for MOEX stocks — end-to-end ML pipeline.

## Goal
Predict next-day realized volatility for MOEX index constituents.

## Data
MOEX ISS API (free, daily candles).

## Planned stack
Python, Docker, MLflow, FastAPI, Airflow, PostgreSQL, Prometheus/Grafana, GitHub Actions.

## Roadmap
- [ ] Stage 0: project structure, configs, logging, tests
- [ ] Stage 1: Docker
- [ ] Stage 2: MLflow tracking and model registry
- [ ] Stage 3: batch scoring + FastAPI service
- [ ] Stage 4: Airflow orchestration
- [ ] Stage 5: CI/CD
- [ ] Stage 6: monitoring (drift, Grafana)

## Notes