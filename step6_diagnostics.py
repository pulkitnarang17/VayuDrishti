import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "predictions_2025.pkl"

# 1. Load predictions generated in Step 5
print(f"[1/4] Loading test predictions from '{INPUT_FILE}'...")
df = pd.read_pickle(INPUT_FILE)

y_true = df["target_aqi_next"]
y_pred = df["prediction_aqi"]

# 2. Seasonal Slice Breakdown
print("\n[2/4] Seasonal Error Diagnostics:")
seasonal_summary = []
for season, group in df.groupby("season", observed=True):
    mae = mean_absolute_error(group["target_aqi_next"], group["prediction_aqi"])
    rmse = np.sqrt(mean_squared_error(group["target_aqi_next"], group["prediction_aqi"]))
    r2 = r2_score(group["target_aqi_next"], group["prediction_aqi"])
    seasonal_summary.append({
        "Season": season,
        "Samples": len(group),
        "Mean Actual AQI": group["target_aqi_next"].mean(),
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    })

seasonal_df = pd.DataFrame(seasonal_summary).sort_values(by="MAE")
print(seasonal_df.to_string(index=False, formatters={
    "Mean Actual AQI": "{:.1f}".format,
    "MAE": "{:.2f}".format,
    "RMSE": "{:.2f}".format,
    "R²": "{:.4f}".format
}))

# 3. Severe Episode Audit (Severe AQI >= 400)
print("\n[3/4] High Pollution Episode Audit (AQI >= 400):")
severe_mask = df["target_aqi_next"] >= 400
severe_df = df[severe_mask]

if severe_df.empty:
    print("No hours with AQI >= 400 in the test set; skipping the severe-episode audit.")
else:
    sev_mae = mean_absolute_error(severe_df["target_aqi_next"], severe_df["prediction_aqi"])
    sev_rmse = np.sqrt(mean_squared_error(severe_df["target_aqi_next"], severe_df["prediction_aqi"]))
    bias = (severe_df["prediction_aqi"] - severe_df["target_aqi_next"]).mean()

    print(f"Total Severe Hours Evaluated: {len(severe_df):,}")
    print(f"Actual Mean AQI in Severe Events: {severe_df['target_aqi_next'].mean():.1f}")
    print(f"Predicted Mean AQI in Severe Events: {severe_df['prediction_aqi'].mean():.1f}")
    print(f"Severe Slice MAE: {sev_mae:.2f}")
    print(f"Severe Slice RMSE: {sev_rmse:.2f}")
    print(f"Average Model Bias (Under/Over-prediction): {bias:+.2f} AQI points")

# 4. Generate Diagnostic Plots
print("\n[4/4] Generating evaluation plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot A: Actual vs Predicted Scatter
axes[0].scatter(y_true, y_pred, alpha=0.15, s=10, color="#1f77b4")
axes[0].plot([0, 500], [0, 500], "r--", linewidth=2, label="Perfect Forecast (y = x)")
axes[0].set_title("Actual vs. Predicted Next-Step AQI (2025 Test Set)", fontsize=13)
axes[0].set_xlabel("Actual Next AQI (t+1)", fontsize=11)
axes[0].set_ylabel("Predicted Next AQI (t+1)", fontsize=11)
axes[0].set_xlim(0, 520)
axes[0].set_ylim(0, 520)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend()

# Plot B: Single Station Time-Series Tracking (Winter Peak 2025: Anand Vihar)
sample_station = "Anand Vihar, Delhi"
station_data = df[(df["station"] == sample_station) & 
                  (df["datetime"] >= "2025-11-01") & 
                  (df["datetime"] <= "2025-11-20")].sort_values("datetime")

axes[1].plot(station_data["datetime"], station_data["target_aqi_next"], 
             label="Actual AQI", color="black", linewidth=2)
axes[1].plot(station_data["datetime"], station_data["prediction_aqi"], 
             label="Model Predicted AQI", color="crimson", linestyle="--", linewidth=1.8)
axes[1].set_title(f"Next-Step Forecast: {sample_station} (Nov 2025)", fontsize=13)
axes[1].set_xlabel("Date", fontsize=11)
axes[1].set_ylabel("AQI", fontsize=11)
axes[1].tick_params(axis="x", rotation=30)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig("aqi_model_diagnostics.png", dpi=300)
print("Diagnostic plot saved as 'aqi_model_diagnostics.png'.")