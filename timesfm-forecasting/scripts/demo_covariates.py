#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import TimesFM 2.5 components
try:
    import timesfm
except ImportError:
    print("Error: TimesFM not found. Run: pip install timesfm[xreg]")

## --- CONFIGURATION ---
CSV_FILE = "WeatherData_TG_Hourly_Jan_2026.csv"
TARGET_COL = "HUMIhe"
TEMP_COL = "TEMPhe"
CONTEXT_LEN = 96  # 4 days of history
HORIZON_LEN = 24  # Predict next 24 hours
MODEL_ID = "google/timesfm-2.5-200m-pytorch"

def prepare_data(file_path):
    """Loads weather data and prepares arrays for TimesFM."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path} in current directory.")
    
    df = pd.read_csv(file_path)
    
    # Ensure numeric types and handle missing values
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').interpolate()
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors='coerce').interpolate()
    
    # TimesFM requires float32
    humidity = df[TARGET_COL].values.astype(np.float32)
    temperature = df[TEMP_COL].values.astype(np.float32)
    
    return humidity, temperature

def run_forecast():
    # 1. Load Data
    humidity, temperature = prepare_data(CSV_FILE)
    
    # Split into context (past) and horizon (future temperature we know)
    # Note: To predict humidity, we use 'known' future temperatures
    ctx_humidity = humidity[-CONTEXT_LEN:]
    ctx_temp = temperature[-(CONTEXT_LEN + HORIZON_LEN):] # Context + Horizon temps
    
    print(f"Data Loaded. Context: {len(ctx_humidity)} hours. Predicting: {HORIZON_LEN} hours.")

    # 2. Initialize TimesFM 2.5
    # Using CPU backend for MacBook Intel / Standard Linux compatibility
    hparams = timesfm.TimesFmHparams(
        backend="cpu", 
        per_core_batch_size=32, 
        horizon_len=HORIZON_LEN,
        context_len=CONTEXT_LEN
    )
    
    # In version 2.5, we use Checkpoint and Hparams separately
    ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=MODEL_ID)
    model = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)

    # 3. Forecast with Temperature as a Covariate (XReg)
    # xreg_mode="timesfm + xreg" is often better for weather physics
    point_fc, quant_fc = model.forecast_with_covariates(
        inputs=[ctx_humidity],
        dynamic_numerical_covariates={
            "temperature": [ctx_temp] 
        },
        xreg_mode="timesfm + xreg"
    )

    # 4. Simple Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(range(HORIZON_LEN), point_fc[0], label="Predicted Humidity", color='blue', marker='o')
    plt.title(f"24-Hour Humidity Forecast (Using {TARGET_COL})")
    plt.ylabel("Humidity %")
    plt.xlabel("Hours Ahead")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("humidity_prediction_output.png")
    print("Forecast complete. Graph saved as 'humidity_prediction_output.png'")

if __name__ == "__main__":
    run_forecast()
