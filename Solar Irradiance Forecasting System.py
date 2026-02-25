import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

# --- INPUTS ---
latitude = float(input("Enter latitude (e.g., 19.0760): "))
longitude = float(input("Enter longitude (e.g., 72.8777): "))

start_date = input("Enter start date (YYYYMMDD, e.g., 20241001): ")
end_date = input("Enter end date (YYYYMMDD, e.g., 20250731): ")

arima_p = int(input("Enter ARIMA p (e.g., 5): "))
arima_d = int(input("Enter ARIMA d (e.g., 1): "))
arima_q = int(input("Enter ARIMA q (e.g., 0): "))

# --- FETCH DATA ---
url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point"
       f"?start={start_date}&end={end_date}"
       f"&latitude={latitude}&longitude={longitude}"
       f"&community=SB&parameters=ALLSKY_SFC_SW_DWN"
       f"&format=CSV")

print("\n-------------------------------------\n")
print("Fetching data from NASA POWER API...")
response = requests.get(url)
if not response.ok:
    raise Exception("Failed to fetch data from NASA POWER API")

with open('solar_data.csv', 'w') as f:
    f.write(response.text)

# --- LOAD DATA ---
with open('solar_data.csv', 'r') as f:
    lines = f.readlines()

header_line = next(i for i, line in enumerate(lines) if line.strip().startswith('YEAR'))
df = pd.read_csv('solar_data.csv', skiprows=header_line)

# Build datetime index
df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
df.set_index('DATE', inplace=True)
solar = df['ALLSKY_SFC_SW_DWN'].replace(-999, np.nan)

# --- REMOVE TRAILING MISSING DATA ---
last_valid_idx = solar.last_valid_index()
solar = solar.loc[:last_valid_idx]
solar = solar.interpolate()

plt.figure(figsize=(12, 5))
solar.plot(title="Hourly Solar Irradiance (W/m²)")
plt.ylabel("W/m²")
plt.show()

# --- TRAIN/TEST SPLIT ---
test_days = 7  # How many days for testing
train = solar.iloc[:-24 * test_days]
test = solar.iloc[-24 * test_days:]

# --- MODEL ---
print("Training ARIMA model...")
model = ARIMA(train, order=(arima_p, arima_d, arima_q))
model_fit = model.fit()
forecast = model_fit.forecast(steps=24 * test_days)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='dashed')
plt.title('ARIMA Solar Index Forecast (Next 7 Valid Days)')
plt.ylabel('Solar Irradiance (W/m²)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'ARIMA RMSE: {rmse:.2f} W/m²')
print("Last actual values:\n", test.tail())
print("Last forecasted values:\n", forecast.tail())
