import pandas as pd
import timesfm
import matplotlib.pyplot as plt # Keep this for a simple graph at the end

# Load the file
df = pd.read_csv('WeatherData_TG_Hourly_Jan_2026.csv')

# IMPORTANT: Make sure the time column is readable by the computer
df['OBSTIME'] = pd.to_datetime(df['OBSTIME'], format='%Y-%m-%d %H')

# Model set up
tfm = timesfm.TimesFm(
    context_len=96,       # Looks back at 96 hours of history
    horizon_len=24,       # Predicts 24 hours into the future
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu",        # Ensures it runs on your MacBook/Linux CPU
)

# This line downloads the 'brain' of the model
tfm.load_from_checkpoint(repo_id="google/timesfm-2.0-500m-pytorch")

# This one command does all the hard work
forecast_df = tfm.forecast_on_df(
    df=df,
    freq="H",             # "H" stands for Hourly
    value_name="HUMIhe",  # The column we want to predict
    time_name="OBSTIME",
    forecast_context_len=96
)

# Save the result to a new file so you can open it in Excel
forecast_df.to_csv('my_humidity_prediction.csv', index=False)
print("Forecast complete! Saved to my_humidity_prediction.csv")
