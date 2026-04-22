import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# function to get next trading day
def next_trading_day(date):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:   # Skip Saturday & Sunday
        next_day += timedelta(days=1)
    return next_day

 
# input
print("\nExamples:")
print("RELIANCE.NS  -> Reliance Industries")
print("TCS.NS       -> Tata Consultancy Services")
print("INFY.NS      -> Infosys")
print("SBIN.NS      -> State Bank of India")
print("AAPL         -> Apple\n")

ticker = input("Enter Stock Symbol: ").upper()


# download data
today = pd.Timestamp.today().normalize()
end_date = today + pd.Timedelta(days=1)   # end-date exclusive

df = yf.download(
    ticker,
    start="2015-01-01",
    end=end_date.strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False
)

if df.empty:
    print(" Invalid stock symbol or no data available.")
    exit()

df.columns = df.columns.get_level_values(0)
df.reset_index(inplace=True)


# data cleaning
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)


# feature engineering
df["return"] = df["Close"].pct_change()
df["ma_5"] = df["Close"].rolling(5).mean()
df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()


#  Target Variable (Next Day Movement)
df["future_return"] = df["Close"].shift(-1) / df["Close"] - 1

threshold = 0.005  # 0.5%

df["movement"] = np.where(
    df["future_return"] > threshold, "UP",
    np.where(df["future_return"] < -threshold, "DOWN", "SIDEWAYS")
)

#  Drop NaNs & Safety Check
df.dropna(inplace=True)

if len(df) < 100:
    print(" Not enough historical data.")
    exit()


#  Train-Test Split (Time Based)
X = df[["return", "ma_5", "volume_ratio"]]
y = df["movement"]

split = int(0.8 * len(df))

X_train = X.iloc[:split]
X_test  = X.iloc[split:]
y_train = y.iloc[:split]
y_test  = y.iloc[split:]


#  Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, rf_pred))


#  Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {ticker}")
plt.show()


#  Classification Report
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


#  Next trading Day Prediction
latest_data = X.iloc[[-1]]

predicted_class = rf_model.predict(latest_data)[0]
probabilities = rf_model.predict_proba(latest_data)[0]
classes = rf_model.classes_

prob_dict = dict(zip(classes, probabilities))


#  Final date Logic
last_market_date = df["Date"].iloc[-1].date()
prediction_date = next_trading_day(last_market_date)


#  Final Output
print("\n==============================")
print(f" Data Used Till  : {last_market_date}")
print(f" Prediction For : {prediction_date}")
print(f" Stock          : {ticker}")
print("==============================")

print(f" Predicted Movement: {predicted_class}\n")

print(" Probability Chances:")
print(f"   UP        : {prob_dict.get('UP', 0)*100:.2f}%")
print(f"   DOWN      : {prob_dict.get('DOWN', 0)*100:.2f}%")
print(f"   SIDEWAYS  : {prob_dict.get('SIDEWAYS', 0)*100:.2f}%")


#  Stock Price Trend
plt.figure(figsize=(10, 4))
plt.plot(df["Date"], df["Close"])
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title(f"{ticker} Closing Price Trend")
plt.show()


#  Probability Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(prob_dict.keys(), prob_dict.values())
plt.ylabel("Probability")
plt.title(f"Prediction Probability Distribution - {ticker}")
plt.show()


#  Feature Importance
plt.figure(figsize=(6, 4))
plt.bar(X.columns, rf_model.feature_importances_)
plt.ylabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()


#  Disclaimer 
print("\n DISCLAIMER:")
print("This prediction is based on historical data and machine learning.")
print("It is for educational purposes only and not financial advice.")
