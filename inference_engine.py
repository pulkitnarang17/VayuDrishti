"""Turns raw sensor readings into the 37 features the model was trained on, then predicts."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Resolve relative to this file so the API works no matter where it is launched from.
MODEL_PATH = Path(__file__).resolve().parent / "aqi_forecast_model.pkl"

# The training data has one reading every 6 hours (4 per day), and the lag features
# (lag_1 = 6h ago, lag_2 = 12h ago, lag_4 = 24h ago) depend on that spacing.
STEP_HOURS = 6
MIN_HISTORY = 4

HISTORY_COLS = ["pm25", "pm10", "aqi", "wind_speed", "visibility"]
CURRENT_COLS = [
    "latitude", "longitude", "temperature", "humidity", "wind_speed", "visibility",
    "pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
]


class AQIPredictor:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Create it with `python export_model.py` (needs engineered_aqi_features.pkl; "
                "see README for the full pipeline)."
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        # Present only in bundles created by the current export_model.py.
        self.metadata = {k: bundle[k] for k in ("sklearn_version", "metrics") if k in bundle}

    @staticmethod
    def _get_category(aqi: float) -> str:
        if aqi <= 50: return "Good"
        if aqi <= 100: return "Satisfactory"
        if aqi <= 200: return "Moderate"
        if aqi <= 300: return "Poor"
        if aqi <= 400: return "Very Poor"
        return "Severe"

    @staticmethod
    def _prepare_history(history_df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        """Validate columns, put history in oldest-to-newest order, check the 6-hour cadence."""
        missing = [c for c in HISTORY_COLS if c not in history_df.columns]
        if missing:
            raise ValueError(f"Every history reading needs these fields: {missing}")
        if len(history_df) < MIN_HISTORY:
            raise ValueError(f"Provide at least {MIN_HISTORY} past readings for lag calculation.")

        history_df = history_df.reset_index(drop=True)
        has_times = "datetime" in history_df.columns and history_df["datetime"].notna().all()
        if not has_times:
            return history_df  # no timestamps: trust the caller's order (oldest first)

        try:
            history_df = history_df.assign(datetime=pd.to_datetime(history_df["datetime"]))
            history_df = history_df.sort_values("datetime").reset_index(drop=True)
            times = list(history_df["datetime"].iloc[-MIN_HISTORY:]) + [dt]
            gaps = [(b - a) for a, b in zip(times, times[1:])]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not compare timestamps (mixed timezone-aware and naive?): {exc}")

        expected = pd.Timedelta(hours=STEP_HOURS)
        if any(g != expected for g in gaps):
            raise ValueError(
                f"Readings must be exactly {STEP_HOURS} hours apart (the model's training cadence). "
                f"The last {MIN_HISTORY} history readings plus the current one must be consecutive "
                f"{STEP_HOURS}-hour steps; found gaps: {[str(g) for g in gaps]}"
            )
        return history_df

    def build_features(self, current_data: dict, history_df: pd.DataFrame) -> dict:
        missing = [c for c in ["datetime"] + CURRENT_COLS if c not in current_data]
        if missing:
            raise ValueError(f"'current' is missing required fields: {missing}")

        dt = pd.to_datetime(current_data["datetime"])
        history_df = self._prepare_history(history_df, dt)

        hour, doy = dt.hour, dt.day_of_year
        features = {
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        }
        features.update({c: float(current_data[c]) for c in CURRENT_COLS})

        # Lags from history (offset -1 = 6h ago, -2 = 12h ago, -4 = 24h ago)
        for lag_idx, offset in [(1, -1), (2, -2), (4, -4)]:
            past_row = history_df.iloc[offset]
            for col in HISTORY_COLS:
                features[f"{col}_lag_{lag_idx}"] = float(past_row[col])

        # Rolling metrics over the last 4 readings (the current one plus 3 before it)
        def window(col):
            return np.append(history_df[col].astype(float).values[-3:], float(current_data[col]))

        features["pm25_roll_mean_4"] = float(np.mean(window("pm25")))
        features["pm25_roll_max_4"] = float(np.max(window("pm25")))
        features["pm10_roll_mean_4"] = float(np.mean(window("pm10")))
        features["aqi_roll_mean_4"] = float(np.mean(window("aqi")))
        return features

    def predict_next(self, current_data: dict, history_df: pd.DataFrame) -> dict:
        features = self.build_features(current_data, history_df)
        feature_vector = pd.DataFrame([features])[self.feature_names]
        predicted_aqi = float(self.model.predict(feature_vector)[0])
        predicted_aqi = max(0.0, min(500.0, round(predicted_aqi, 1)))

        current_aqi = float(current_data["aqi"])
        return {
            "predicted_aqi_next": predicted_aqi,
            "predicted_category": self._get_category(predicted_aqi),
            "current_aqi": current_aqi,
            "expected_change": round(predicted_aqi - current_aqi, 1),
        }
