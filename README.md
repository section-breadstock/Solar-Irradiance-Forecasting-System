# Solar Irradiance Forecasting

Time-series forecasting system to predict solar irradiance using ARIMA models and NASA satellite data.

## Installation

Install required packages:
```bash
pip install pandas numpy matplotlib requests statsmodels scikit-learn
```

## Usage

Run the script:
```bash
python solar_forecast.py
```

Enter the following inputs when prompted:
- **Latitude & Longitude** (e.g., 19.0760, 72.8777 for Mumbai)
- **Start Date** (YYYYMMDD format, e.g., 20241001)
- **End Date** (YYYYMMDD format, e.g., 20250731)
- **ARIMA Parameters:**
  - p: 5 (autoregressive order)
  - d: 1 (differencing order)
  - q: 0 (moving average order)

## How It Works

1. Fetches hourly solar irradiance data from NASA POWER API
2. Preprocesses data (handles missing values, interpolation)
3. Splits data into training and testing sets (7-day test period)
4. Trains ARIMA model on historical data
5. Generates 7-day forecast
6. Visualizes results and calculates RMSE

## Output

- Graph: Historical solar irradiance trends
- Graph: Actual vs forecasted values
- RMSE metric for model accuracy

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Statsmodels, Scikit-learn, NASA POWER API
