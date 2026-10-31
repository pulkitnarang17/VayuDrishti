import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "engineered_aqi_features.pkl"

# 1. Load engineered features
print("[1/4] Loading engineered dataset...")
df = pd.read_pickle(INPUT_FILE)

# 2. Define Features and Chronological Train/Test Split
print("[2/4] Setting up feature matrix and train/test splits...")

# Convert categorical columns
for col in ["city", "station", "season"]:
    df[col] = df[col].astype("category")

train_df = df[df["year"] < 2025].reset_index(drop=True)
test_df = df[df["year"] == 2025].reset_index(drop=True)

feature_cols = [
    # Spatial & Temporal
    "city", "station", "latitude", "longitude", "season", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Current Meteorology
    "temperature", "humidity", "wind_speed", "visibility",
    # Current Pollutants & AQI
    "pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
    # Lagged Pollutants & Sensors (t-1, t-2, t-4)
    "pm25_lag_1", "pm10_lag_1", "aqi_lag_1", "wind_speed_lag_1", "visibility_lag_1",
    "pm25_lag_2", "pm10_lag_2", "aqi_lag_2", "wind_speed_lag_2", "visibility_lag_2",
    "pm25_lag_4", "pm10_lag_4", "aqi_lag_4", "wind_speed_lag_4", "visibility_lag_4",
    # 24-Hour Rolling Averages and Spikes
    "pm25_roll_mean_4", "pm25_roll_max_4", "pm10_roll_mean_4", "aqi_roll_mean_4"
]

X_train = train_df[feature_cols]
y_train = train_df["target_aqi_next"]

X_test = test_df[feature_cols]
y_test = test_df["target_aqi_next"]

print(f"Features count: {len(feature_cols)}")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

# 3. Train Gradient Boosted Decision Trees
print("\n[3/4] Training Gradient Boosted Regressor...")
try:
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model_name = "LightGBM Regressor"
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    # HistGradientBoosting supports categorical features natively
    categorical_mask = [c in ["city", "station", "season"] for c in feature_cols]
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=63,
        categorical_features=categorical_mask,
        random_state=42
    )
    model_name = "HistGradientBoostingRegressor"

print(f"Using algorithm: {model_name}")
model.fit(X_train, y_train)

# 4. Evaluation and Comparison Against Persistence Benchmark
print("\n[4/4] Evaluating predictions on 2025 unseen test data...")
preds = model.predict(X_test)
naive_preds = test_df["aqi"]

# Model metrics
model_mae = mean_absolute_error(y_test, preds)
model_rmse = np.sqrt(mean_squared_error(y_test, preds))
model_r2 = r2_score(y_test, preds)

# Naive baseline metrics
naive_mae = mean_absolute_error(y_test, naive_preds)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
naive_r2 = r2_score(y_test, naive_preds)

print("\n" + "=" * 62)
print(f"            FINAL PERFORMANCE BENCHMARK (TEST YEAR: 2025)")
print("=" * 62)
print(f"{'Metric':<10} | {'Naive Persistence':<18} | {'Trained Model':<15} | {'Improvement':<12}")
print("-" * 62)
print(f"{'MAE':<10} | {naive_mae:<18.2f} | {model_mae:<15.2f} | {((naive_mae - model_mae) / naive_mae * 100):+.1f}%")
print(f"{'RMSE':<10} | {naive_rmse:<18.2f} | {model_rmse:<15.2f} | {((naive_rmse - model_rmse) / naive_rmse * 100):+.1f}%")
print(f"{'R²':<10} | {naive_r2:<18.4f} | {model_r2:<15.4f} | {((model_r2 - naive_r2) / naive_r2 * 100):+.1f}%")
print("=" * 62)

# Save predictions for step 6 diagnostics
test_df["prediction_aqi"] = preds
test_df.to_pickle("predictions_2025.pkl")
print("Saved predictions to 'predictions_2025.pkl' for diagnostics.")