import pandas as pd
import numpy as np

FILE_PATH = "delhi_ncr_aqi_dataset.csv"

# 1. Load data
print("[1/5] Loading and sorting dataset...")
df = pd.read_csv(FILE_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)

# 2. Forecasting Target (t+1)
print("[2/5] Creating forecast target (AQI at t+1)...")
df["target_aqi_next"] = df.groupby("station")["aqi"].shift(-1)

# 3. Temporal Lag Features per station
print("[3/5] Engineering lag features (lag 1, 2, 4)...")
lag_columns = ["pm25", "pm10", "aqi", "wind_speed", "visibility"]
for lag in [1, 2, 4]:
    for col in lag_columns:
        df[f"{col}_lag_{lag}"] = df.groupby("station")[col].shift(lag)

# 4. Rolling Window Features (4 intervals = 24 hours)
print("[4/5] Computing 24-hour rolling averages and peaks...")
df["pm25_roll_mean_4"] = df.groupby("station")["pm25"].transform(lambda x: x.rolling(4).mean())
df["pm25_roll_max_4"] = df.groupby("station")["pm25"].transform(lambda x: x.rolling(4).max())
df["pm10_roll_mean_4"] = df.groupby("station")["pm10"].transform(lambda x: x.rolling(4).mean())
df["aqi_roll_mean_4"] = df.groupby("station")["aqi"].transform(lambda x: x.rolling(4).mean())

# 5. Cyclical Encodings (Hour & Day of Year)
print("[5/5] Encoding cyclical temporal signals...")
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

df["day_of_year"] = df["datetime"].dt.day_of_year
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# Drop boundary rows (lags on the first 4 rows + lead target on last 1 row per station)
initial_rows = len(df)
df_features = df.dropna().reset_index(drop=True)
dropped_rows = initial_rows - len(df_features)

print(f"\nFeature engineering complete:")
print(f"Total features now: {df_features.shape[1]}")
print(f"Rows dropped at boundary edges: {dropped_rows}")
print(f"Clean modeling dataset shape: {df_features.shape[0]:,} rows, {df_features.shape[1]} columns")

# Save using native pandas pickle (no extra libraries required)
OUTPUT_FILE = "engineered_aqi_features.pkl"
df_features.to_pickle(OUTPUT_FILE)
print(f"Saved engineered dataset to '{OUTPUT_FILE}'.")