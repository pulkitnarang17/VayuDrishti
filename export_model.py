import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("[1/4] Loading engineered dataset...")
df = pd.read_pickle("engineered_aqi_features.pkl")

# Define model feature schema (must match inference_engine.build_features)
feature_cols = [
    "latitude", "longitude", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temperature", "humidity", "wind_speed", "visibility",
    "pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
    "pm25_lag_1", "pm10_lag_1", "aqi_lag_1", "wind_speed_lag_1", "visibility_lag_1",
    "pm25_lag_2", "pm10_lag_2", "aqi_lag_2", "wind_speed_lag_2", "visibility_lag_2",
    "pm25_lag_4", "pm10_lag_4", "aqi_lag_4", "wind_speed_lag_4", "visibility_lag_4",
    "pm25_roll_mean_4", "pm25_roll_max_4", "pm10_roll_mean_4", "aqi_roll_mean_4"
]

train_df = df[df["year"] < 2025]
test_df = df[df["year"] == 2025]
X_train, y_train = train_df[feature_cols], train_df["target_aqi_next"]

print(f"[2/4] Fitting model on {len(X_train):,} rows (years before 2025)...")
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    random_state=42
)
model.fit(X_train, y_train)

# step5_train.py benchmarks a different model (it also uses city/station/season), so measure
# the model that is actually shipped, on the same 2025 hold-out, against naive persistence.
print("[3/4] Evaluating the exported model on the 2025 hold-out...")
metrics = {}
if len(test_df):
    y_test = test_df["target_aqi_next"]
    preds = model.predict(test_df[feature_cols])
    naive = test_df["aqi"]
    metrics = {
        "test_year": 2025,
        "model_mae": float(mean_absolute_error(y_test, preds)),
        "model_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "model_r2": float(r2_score(y_test, preds)),
        "naive_mae": float(mean_absolute_error(y_test, naive)),
        "naive_rmse": float(np.sqrt(mean_squared_error(y_test, naive))),
        "naive_r2": float(r2_score(y_test, naive)),
    }
    print(f"{'':8}{'Naive persistence':>20}{'Exported model':>18}")
    for name in ("mae", "rmse", "r2"):
        print(f"{name.upper():<8}{metrics['naive_' + name]:>20.3f}{metrics['model_' + name]:>18.3f}")
else:
    print("No 2025 rows found; skipping evaluation.")

print("[4/4] Exporting model bundle...")
artifacts = {
    "model": model,
    "feature_names": feature_cols,
    "sklearn_version": sklearn.__version__,  # pickles only load safely with the same version
    "metrics": metrics,
}
joblib.dump(artifacts, "aqi_forecast_model.pkl")
print(f"Saved production model as 'aqi_forecast_model.pkl' (scikit-learn {sklearn.__version__}).")
print("If this version differs from requirements.txt, update the pin there.")
