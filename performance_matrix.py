import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Assuming you use this for other plots or styling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = target_series[:-12]
test = target_series[-12:]

model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

forecast_obj = results.get_forecast(steps=12)
forecast_values = forecast_obj.predicted_mean

mae = mean_absolute_error(test, forecast_values)
rmse = np.sqrt(mean_squared_error(test, forecast_values))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    non_zero_indices = y_true != 0
    if not np.any(non_zero_indices): 
        return np.nan
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

mape = mean_absolute_percentage_error(test, forecast_values)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
if not np.isnan(mape):
    print(f"MAPE: {mape:.2f}%")
else:
    print("MAPE: Cannot be calculated due to zero or near-zero actual values in the test set.")

plt.figure(figsize=(14,6))
ax = test.plot(label='Actual')
forecast_values.plot(ax=ax, label='Forecast')
plt.fill_between(forecast_obj.conf_int().index,
                 forecast_obj.conf_int().iloc[:, 0],
                 forecast_obj.conf_int().iloc[:, 1], color='pink', alpha=0.3)
plt.title("SARIMA Forecast vs Actual")
plt.legend()
plt.show()