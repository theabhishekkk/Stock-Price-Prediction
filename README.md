# Stock-Price-Prediction
# 📊 Stock Price Prediction & BI Analytics Pipeline

## 🚀 Project Overview

This project is an **end-to-end Business Intelligence (BI) and Machine Learning pipeline** designed to predict stock prices and generate actionable insights.

It integrates:

* 📥 Data Extraction (API-based)
* 🔄 Data Transformation (Feature Engineering)
* 🤖 Machine Learning (Linear Regression)
* 📈 Forecasting (Future Price Prediction)
* 📊 Visualization (Trend Analysis)
* 📤 Export (Power BI / Tableau Ready Data)

---

## 🎯 Objective

To build a system that:

* Predicts future stock prices using historical data
* Evaluates model performance using KPIs
* Provides data ready for BI dashboards

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **Libraries**:

  * `yfinance` → Stock data extraction
  * `pandas`, `numpy` → Data processing
  * `matplotlib` → Visualization
  * `scikit-learn` → Machine Learning

---

## 📂 Project Workflow

### 1️⃣ Extract (Data Collection)

* Fetch historical stock data using Yahoo Finance API

### 2️⃣ Transform (Feature Engineering)

* Moving Averages (MA_10, MA_50)
* Daily Returns
* Future Target Variable

### 3️⃣ Load / Model (Machine Learning)

* Linear Regression model
* Train-test split (Time-series aware)

### 4️⃣ Evaluate (KPIs)

* 📊 R² Score → Model accuracy
* 📉 MAE → Mean Absolute Error
* 📉 RMSE → Root Mean Squared Error

### 5️⃣ Forecast

* Predict next **30 days stock prices**

### 6️⃣ Visualize

* Historical vs Predicted vs Forecast trends

### 7️⃣ Export

* CSV file for:

  * Power BI
  * Tableau dashboards

---

## 📸 Output Example

* Line chart showing:

  * Historical Price
  * Predicted Price
  * Future Forecast

---


## 💡 Key Features

✔ End-to-end BI pipeline
✔ Real-time stock data
✔ Feature engineering for better accuracy
✔ KPI-based evaluation
✔ Dashboard-ready output

---

