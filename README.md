# VayuDrishti – AQI Forecasting for Delhi-NCR

Predicts the **next AQI reading (6 hours ahead)** for Delhi-NCR monitoring stations from current
sensor data plus recent history, and serves the model over a REST API.

```
raw CSV ──► step1-3 (verify, target, features) ──► step4-6 (baseline, train, diagnostics)
                                  └──► export_model.py ──► aqi_forecast_model.pkl ──► app.py (FastAPI)
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows PowerShell:  venv\Scripts\Activate.ps1
pip install -r requirements.txt   # add requirements-dev.txt to run the tests
```

`requirements.txt` pins `scikit-learn==1.9.1`: the committed model is a pickle and only loads
safely with the version that created it.

## 2. Run the API (works out of the box)

The trained model `aqi_forecast_model.pkl` is already in the repo.

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for interactive docs, or:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
```json
{"predicted_aqi_next": 393.1, "predicted_category": "Very Poor", "current_aqi": 360.0, "expected_change": 33.1}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check (also shows the model's saved metrics, once re-exported) |
| `POST /predict` | `current` reading + `history` (≥ 4 earlier readings) → next-step forecast |

**Input rules** (the model's lag features depend on them):
- `history` readings need `pm25, pm10, aqi, wind_speed, visibility`; `current` needs all sensor fields
  (see `sample_request.json`).
- Readings must be **exactly 6 hours apart**. Give every history reading a `datetime` and the API
  sorts them and checks the spacing; omit it and history is assumed to be ordered oldest → newest.
- Errors: `400` too little history, `422` missing/invalid fields or wrong spacing.

## 3. Retrain from scratch

The dataset is git-ignored, so place `delhi_ncr_aqi_dataset.csv` in the project folder. It needs these columns:
`city, station, latitude, longitude, datetime, year, hour, season, is_weekend, temperature, humidity,
wind_speed, visibility, pm25, pm10, no2, so2, co, o3, aqi`.

```bash
python run_pipeline.py              # runs all 7 steps in order, stops at the first failure
python run_pipeline.py --from-step 5   # resume part-way
python run_pipeline.py --list
```

| Step | Script | Output |
|---|---|---|
| 1 | `step1_verify.py` | integrity checks on the raw data |
| 2 | `step2_target.py` | verifies the `t+1` target |
| 3 | `step3_features.py` | `engineered_aqi_features.pkl` |
| 4 | `step4_split_and_baseline.py` | persistence baseline (train < 2025, test = 2025) |
| 5 | `step5_train.py` | benchmark model → `predictions_2025.pkl` |
| 6 | `step6_diagnostics.py` | seasonal / severe-episode errors, `aqi_model_diagnostics.png` |
| 7 | `export_model.py` | trains, **evaluates on 2025**, writes `aqi_forecast_model.pkl` |

Note: `step5` benchmarks a richer model (it also uses city/station/season). The model served by the
API is the one from `export_model.py`, which now prints its own 2025 score vs. persistence.
If you retrain with a different scikit-learn version, update the pin in `requirements.txt`.

## 4. Tests

```bash
pip install -r requirements-dev.txt
pytest
```
Covers the API, input validation, and a train/serve consistency check that the lag/rolling features
computed at inference match how `step3_features.py` computes them.
