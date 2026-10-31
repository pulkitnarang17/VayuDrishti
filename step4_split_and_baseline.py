import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "engineered_aqi_features.pkl"

# 1. Load engineered features
print(f"[1/3] Loading dataset from '{INPUT_FILE}'...")
df = pd.read_pickle(INPUT_FILE)

# 2. Strict Out-of-Time Chronological Split
print("[2/3] Performing chronological train/test split...")
train_df = df[df["year"] < 2025].reset_index(drop=True)
test_df = df[df["year"] == 2025].reset_index(drop=True)

print(f"Training set (2020-2024): {train_df.shape[0]:,} rows ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
print(f"Testing set  (2025):      {test_df.shape[0]:,} rows ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

# 3. Naive Persistence Baseline Evaluation
# Baseline assumption: Next AQI (t+1) will equal current AQI (t)
print("\n[3/3] Evaluating Naive Persistence Baseline on 2025 test set...")
y_test = test_df["target_aqi_next"]
y_naive = test_df["aqi"]

naive_mae = mean_absolute_error(y_test, y_naive)
naive_rmse = np.sqrt(mean_squared_error(y_test, y_naive))
naive_r2 = r2_score(y_test, y_naive)

print("=" * 45)
print("       NAIVE PERSISTENCE BENCHMARK (2025)     ")
print("=" * 45)
print(f"  MAE  : {naive_mae:.2f}")
print(f"  RMSE : {naive_rmse:.2f}")
print(f"  R²   : {naive_r2:.4f}")
print("=" * 45)
print("The machine learning model in Step 5 must beat these metrics.")