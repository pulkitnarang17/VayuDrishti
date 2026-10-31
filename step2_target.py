import pandas as pd

FILE_PATH = "delhi_ncr_aqi_dataset.csv"

# 1. Load data
print("[1/3] Loading dataset...")
df = pd.read_csv(FILE_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

# Sort chronologically per station to preserve continuous timelines
df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)

# 2. Generate forecasting target (AQI at t+1)
print("[2/3] Generating target: target_aqi_next (AQI at t+1)...")
df["target_aqi_next"] = df.groupby("station")["aqi"].shift(-1)

# Check missing targets (must be exactly 1 per station = 23 rows)
num_stations = df["station"].nunique()
missing_targets = df["target_aqi_next"].isnull().sum()
print(f"Total rows: {len(df):,}")
print(f"Rows with NaN target (expected 1 per station = {num_stations}): {missing_targets}")

# 3. Sanity check: verify row i target matches row i+1 current AQI
print("\n[3/3] Sanity check for 'Anand Vihar, Delhi':")
sample = df[df["station"] == "Anand Vihar, Delhi"][["datetime", "hour", "aqi", "target_aqi_next"]].head(5)
print(sample.to_string(index=False))

sample_station = df[df["station"] == "Anand Vihar, Delhi"]
is_aligned = (sample_station["target_aqi_next"].iloc[:-1].values == sample_station["aqi"].iloc[1:].values).all()
print(f"\nAlignment check passed? {is_aligned}")