import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

df = pd.read_csv(r"C:\Users\acer\OneDrive\Documents\GitHub\Climate-data-analysis-forecasting\All_Feature_Data.csv", parse_dates=["Date"], dayfirst=True)

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

print("Basic Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nSummary Statistics:\n")
print(df.describe())

df['date'] = pd.to_datetime(df['date'], dayfirst=True)
for col in df.columns:
    if col != 'date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(method='ffill', inplace=True)

plt.figure()
sns.lineplot(x='date', y='global_avg_temp_anomaly_relative_to_19611990', data=df)
plt.title('Global Average Temperature Anomaly Over Time')
plt.xlabel('Year')
plt.ylabel('Temp Anomaly (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 12))
corr_matrix = df.drop(columns=['date']).corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Correlation Matrix of Variables')
plt.show()

plt.figure()
sns.scatterplot(x='co2_conc', y='global_avg_temp_anomaly_relative_to_19611990', data=df, alpha=0.5)
plt.title('CO2 Concentration vs Temperature Anomaly')
plt.xlabel('CO2 Concentration (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.show()

df_yearly = df.set_index('date').resample('Y').mean(numeric_only=True).reset_index()

plt.figure()
sns.lineplot(x='date', y='global_avg_temp_anomaly_relative_to_19611990', data=df_yearly)
plt.title('Yearly Avg Global Temperature Anomaly')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.grid(True)
plt.show()
