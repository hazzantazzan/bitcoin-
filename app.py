import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bitcoin Forecast", layout="wide")
st.title("📈 Bitcoin Price Forecasting with LSTM")

@st.cache_data
def load_data():
    df = yf.download('BTC-USD', start='2017-01-01', end='2024-12-31', progress=False)
    
    if df.empty:
        st.error("Failed to load BTC-USD data. Please check your internet or the ticker symbol.")
        return pd.DataFrame()
    
    if 'Close' not in df.columns:
        st.error("The 'Close' column was not found in the downloaded data.")
        return pd.DataFrame()

    df = df[['Close']]
    df.dropna(inplace=True)
    return df

def preprocess(df, seq_len=60):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

df = load_data()
st.subheader("📊 Historical BTC-USD Closing Prices")
st.line_chart(df)

st.write("Training model... this might take a minute ⏳")
X_train, y_train, X_test, y_test, scaler = preprocess(df)
model = build_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)
actual = scaler.inverse_transform(y_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual, label='Actual')
ax.plot(preds, label='Predicted')
ax.set_title("🔮 Predicted vs. Actual BTC Prices")
ax.legend()
st.pyplot(fig)
