import datetime as dt
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference_engine import MIN_HISTORY, STEP_HOURS, AQIPredictor

app = FastAPI(
    title="VayuDrishti AQI Predictor API",
    description=f"Predicts the next AQI reading ({STEP_HOURS} hours ahead) from current sensor "
                f"data plus at least {MIN_HISTORY} earlier readings.",
)

# Lets a browser front-end on another origin call the API. Restrict allow_origins before deploying.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = AQIPredictor()  # fails fast with a clear message if the model file is missing


class HistoryReading(BaseModel):
    # Include datetime on every history reading so order and 6-hour spacing can be verified.
    # Without it, history is assumed to be ordered oldest -> newest.
    datetime: Optional[dt.datetime] = None
    pm25: float
    pm10: float
    aqi: float
    wind_speed: float
    visibility: float


class CurrentReading(HistoryReading):
    datetime: dt.datetime
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    no2: float
    so2: float
    co: float
    o3: float


class SensorPayload(BaseModel):
    current: CurrentReading
    history: List[HistoryReading]


@app.get("/")
def root():
    return {"status": "online", "model": "HistGradientBoosting AQI Predictor", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", **predictor.metadata}


@app.post("/predict")
def predict(payload: SensorPayload):
    if len(payload.history) < MIN_HISTORY:
        raise HTTPException(
            status_code=400,
            detail=f"Provide at least {MIN_HISTORY} past readings for lag calculation.",
        )

    history_df = pd.DataFrame([r.model_dump() for r in payload.history])
    try:
        return predictor.predict_next(payload.current.model_dump(), history_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
