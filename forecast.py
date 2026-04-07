# pip install numpy pandas matplotlib scikit-learn yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# 📌 PARAMETERS (Dynamic)
# ==========================================
ASSET_TICKER = "NVDA"
FORECAST_HORIZON = 30
START_DATE = "2023-01-01"

print(f"\n🚀 Running BI Pipeline for: {ASSET_TICKER}")

# ==========================================
# 📌 1. EXTRACT (Data Fetch)
# ==========================================
df = yf.download(ASSET_TICKER, start=START_DATE)

# Keep only required column
df = df[['Close']].copy()

# Handle missing values
df.fillna(method='ffill', inplace=True)

# ==========================================
# 📌 2. TRANSFORM (Feature Engineering)
# ==========================================

# Moving averages (better prediction power)
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

# Daily returns
df['Returns'] = df['Close'].pct_change()

# Target variable
df['Future_Target'] = df['Close'].shift(-FORECAST_HORIZON)

# Drop NaN rows
df.dropna(inplace=True)

# Features and Labels
X = df[['Close', 'MA_10', 'MA_50', 'Returns']].values
y = df['Future_Target'].values

# ==========================================
# 📌 3. DATA SPLIT + SCALING
# ==========================================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ==========================================
# 📌 4. MODEL TRAINING
# ==========================================
model = LinearRegression()
model.fit(x_train, y_train)

# ==========================================
# 📌 5. MODEL EVALUATION (KPIs)
# ==========================================
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"R² Score   : {r2:.2%}")
print(f"MAE        : {mae:.2f}")
print(f"RMSE       : {rmse:.2f}")

# ==========================================
# 📌 6. FORECASTING FUTURE
# ==========================================
last_data = df[['Close', 'MA_10', 'MA_50', 'Returns']].tail(FORECAST_HORIZON)

last_data_scaled = scaler.transform(last_data)

forecast = model.predict(last_data_scaled)

# ==========================================
# 📌 7. VISUALIZATION
# ==========================================
df['Prediction'] = np.nan
df.iloc[-len(y_pred):, df.columns.get_loc('Prediction')] = y_pred

plt.figure(figsize=(14,7))

plt.plot(df['Close'], label='Historical Price')
plt.plot(df['Prediction'], label='Predicted Price')
plt.plot(df.index[-FORECAST_HORIZON:], forecast, 
         label='Future Forecast', linestyle='--')

plt.title(f"{ASSET_TICKER} Stock Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()

plt.show()

# ==========================================
# 📌 8. EXPORT FOR BI TOOLS
# ==========================================
output_df = pd.DataFrame({
    "Date": df.index[-FORECAST_HORIZON:],
    "Forecast": forecast
})

output_df.to_csv(f"{ASSET_TICKER}_forecast.csv", index=False)

print(f"\n✅ Exported: {ASSET_TICKER}_forecast.csv")
