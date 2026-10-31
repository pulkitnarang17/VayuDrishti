import pandas as pd
import numpy as np

FILE_PATH = "delhi_ncr_aqi_dataset.csv"
print(f"[1/4] Loading dataset from '{FILE_PATH}'...")
df = pd.read_csv(FILE_PATH)
print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

print("\n[2/4] Parsing datetime objects...")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)
print("Data sorted by ['station', 'datetime'].")

print("\n[3/4] Running data integrity checks...")
null_counts = df.isnull().sum()
total_nulls = null_counts.sum()
print(f"Total null values across all columns: {total_nulls}")

duplicate_records = df.duplicated(subset=["station", "datetime"]).sum()
print(f"Duplicate (station, datetime) entries: {duplicate_records}")

print("\n[4/4] Auditing monitoring stations and timestamps...")
num_stations = df["station"].nunique()
print(f"Total distinct monitoring stations: {num_stations}")

records_per_station = df.groupby("station")["datetime"].count()
expected_count = records_per_station.iloc[0]
is_balanced = (records_per_station == expected_count).all()

print(f"Records per station: {expected_count:,} readings each.")
print(f"Is grid fully balanced across all stations? {'Yes' if is_balanced else 'No'}")
print(f"Observed daily recording hours: {sorted(df['hour'].unique().tolist())}")
print(f"Timeline starts: {df['datetime'].min()}")
print(f"Timeline ends:   {df['datetime'].max()}")