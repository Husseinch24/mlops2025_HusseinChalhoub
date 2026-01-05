## Project structure 
```
MLOPS2025_HUSSEINCHALHOUB/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .pytest_cache/
├── .venv/
├── artifacts/
├── config/
├── mlruns/
├── outputs/
├── scripts/
├── src/
├── tests/
├── .gitignore
├── .python-version
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── uv.lock
├── run_batch_inference_pipeline.py
├── run_pipeline.py
└── run_training_pipeline.py
```

## How to Run Locally:
 ```
uv run .\run_pipeline.py
pytest tests/ --maxfail=1 -v (it helps checking all the test of the whole test and i can test each one alone by giving the path of the desired file)
```

## How to Run with Docker:
```
docker system prune -af --volumes
docker compose build --no-cache
docker compose up -d
docker compose run --rm app python -m scripts.pipeline run

```
## How to Run with Sagemaker:
```
python run_training_pipeline.py

```
## Models Selected :
```
GradientBoosting (gb) - RandomForest (rf) - ridge - linear

The best model is selected based on the configured metric and optionally registered with MLflow model registry is : rf 

```
## Metrics selected and justification :
```
RMSE — RMSE is suitable for measuring average prediction error in units of the target (seconds) and penalizes larger errors.
```

## Team Member:
```
Hussein Chalhoub - 252920 
```