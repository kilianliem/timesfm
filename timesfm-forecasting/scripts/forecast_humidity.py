import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timesfm import TimesFm

python forecast_humidity.py

# 1. Load your data first to see how much "Context" we actually have
file_path = "WeatherData_TG_Hourly_Jan_2026.csv"
df = pd.read_csv(file_path)

# Convert OBSTIME to actual date format so the graph looks nice
df['OBSTIME'] = pd.to_datetime(df['OBSTIME'])
total_rows = len(df)

# 2. Decide the Context Length (The 'Memory' size)
# TimesFM needs multiples of 32. We check how much data you have.
if total_rows >= 128:
    current_context = 128
elif total_rows >= 96:
    current_context = 96
elif total_rows >= 64:
    current_context = 64
else:
    current_context = 32

print(f"Using the last {current_context} hours from OBSTIME as context.")

# 3. Initialize the Model
tfm = TimesFm(
    context_len=current_context,
    horizon_len=48,              # We want to forecast 48 hours
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu",
)

# 4. Load Weights
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# 5. Prepare Input (Last X hours of Humidity)
# We take the humidity and the corresponding times
history_data = df['HUMIhe'].values[-current_context:]
history_times = df['OBSTIME'].values[-current_context:]

# 6. Run Forecast
point_forecast, _ = tfm.forecast([history_data])
forecast_values = point_forecast[0]

# 7. Create Future Timestamps for the 48-hour forecast
# This creates 48 new hourly timestamps starting after the last OBSTIME
last_time = df['OBSTIME'].iloc[-1]
forecast_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=48, freq='H')

# 8. Plotting
plt.figure(figsize=(12, 6))

# Plot historical context
plt.plot(history_times, history_data, label="Past Humidity (Actual)", color="#1f77b4")

# Plot forecast
plt.plot(forecast_times, forecast_values, label="48h Forecast (Predicted)", color="#ff7f0e", linestyle="--")

plt.title(f"Humidity Forecast starting from {last_time}")
plt.xlabel("Date & Time (OBSTIME)")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45) # Rotate dates so they don't overlap
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and Show
plt.savefig("humidity_final_forecast.png")
print("? Done! Check 'humidity_final_forecast.png' for your results.")
plt.show()


# Save the plot so you can view it in your Mac Finder
plt.savefig("humidity_48h_forecast.png")
print("? Forecast complete! Graph saved as 'humidity_48h_forecast.png'")
plt.show()
