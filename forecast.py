import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==========================================
# BI PARAMETERS (KPI Definitions)
# ==========================================
ASSET_TICKER = "NVDA"   # The asset to analyze
FORECAST_HORIZON = 30   # Days to forecast (Future Window)
START_DATE = "2023-01-01"

print(f"--- STARTING BI ETL PIPELINE FOR {ASSET_TICKER} ---")

# 1. EXTRACT (Data Ingestion)
# Fetching raw OHLC data from external API
# ------------------------------------------
raw_data = yf.download(ASSET_TICKER, start=START_DATE)
df = raw_data[['Close']].copy()

# 2. TRANSFORM (Feature Engineering)
# Creating specific features for the model to "learn" from
# ------------------------------------------
# Target Variable: The price 'n' days into the future
df['Future_Target'] = df[['Close']].shift(-FORECAST_HORIZON)

# Feature Set (X): Current Price Data
X = np.array(df.drop(['Future_Target'], axis=1))[:-FORECAST_HORIZON]
# Target Set (y): Future Price Data
y = np.array(df['Future_Target'])[:-FORECAST_HORIZON]

# 3. MODELING (Predictive Analytics)
# Splitting data into "Training" (Historical) and "Testing" (Validation)
# ------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)

# KPI: R-Squared (Model Confidence Score)
confidence = model.score(x_test, y_test)
print(f"Model R² Confidence: {confidence:.2%}")

# 4. FORECASTING (Insight Generation)
# Predicting the 'Blind' data (The next 30 days)
# ------------------------------------------
x_forecast = np.array(df.drop(['Future_Target'], axis=1))[-FORECAST_HORIZON:]
lr_prediction = model.predict(x_forecast)

# 5. VISUALIZATION (Decision Support)
# Plotting the "Trend vs Reality" for the BI Dashboard
# ------------------------------------------
valid = df[X.shape[0]:]
valid['Forecast'] = lr_prediction

plt.figure(figsize=(14, 7))
plt.title(f'{ASSET_TICKER} Predictive Analysis | R²: {confidence:.2f}')
plt.xlabel('Fiscal Timeline')
plt.ylabel('Asset Valuation (USD)')
plt.plot(df['Close'], label='Historical Data', color='blue', alpha=0.6)
plt.plot(valid[['Close']], label='Actual Price (Validation)', color='green')
plt.plot(valid[['Forecast']], label='AI Forecast', color='red', linestyle='--')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# 6. EXPORT (Integration)
# Save to CSV for PowerBI / Tableau integration
# ------------------------------------------
valid.to_csv(f"{ASSET_TICKER}_forecast_data.csv")
print(f"Data exported to {ASSET_TICKER}_forecast_data.csv for Dashboard integration.")
