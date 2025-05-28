import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
 
df = pd.read_csv(r"C:\Users\acer\OneDrive\Documents\GitHub\Climate-data-analysis-forecasting\All_Feature_Data.csv", parse_dates=["Date"], dayfirst=True)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

for col in df.columns:
    if col != 'date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(method='ffill', inplace=True)

df.set_index('date', inplace=True)
target_series = df['global_avg_temp_anomaly_relative_to_19611990'].resample('M').mean()

plt.figure(figsize=(14, 6))
target_series.plot(title='Global Avg Temp Anomaly (up to Jan 2024)')
plt.xlabel("Year")
plt.ylabel("Temp Anomaly (°C)")
plt.grid(True)
plt.show()

model = SARIMAX(target_series, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)


forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(start=target_series.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
forecast_ci = forecast.conf_int()
forecast_ci.index = forecast_index

plt.figure(figsize=(14, 6))
ax = target_series.plot(label='Observed', color='blue')
forecast_series.plot(ax=ax, label='Forecast', color='red')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast of Global Avg Temp Anomaly (Feb 2024 - Jan 2025)')
plt.xlabel('Date')
plt.ylabel('Temp Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.show()

print("\n🔮 Forecast for Feb 2024 – Jan 2025:\n")
print(forecast_series)

forecast_series.to_csv("forecast_temp_anomaly_2024.csv", header=["forecast_temp_anomaly"])
