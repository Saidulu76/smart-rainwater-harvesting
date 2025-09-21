
"""Train a simple LSTM to forecast rainfall/tank level.

Expected CSV columns: at minimum a 'date' column and a 'rainfall_mm' or 'rain_mm' column.
The script will attempt to infer columns but you can modify it for your dataset specifics.
"""
import argparse
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_preprocess(path, series_col=None, date_col=None, lookback=14):
    df = pd.read_csv(path, parse_dates=True)
    # try to infer date and series columns
    if date_col is None:
        for c in ['date','Date','DATE','timestamp','Timestamp']:
            if c in df.columns:
                date_col = c
                break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    if series_col is None:
        possible = [c for c in df.columns if 'rain' in c.lower() or 'mm' in c.lower() or 'precip' in c.lower()]
        if possible:
            series_col = possible[0]
        else:
            # fallback: numeric column except date
            numeric = df.select_dtypes('number').columns.tolist()
            if numeric:
                series_col = numeric[0]
            else:
                raise ValueError("No suitable series column found.")
    series = df[series_col].fillna(0).astype(float).values.reshape(-1,1)
    scaler = MinMaxScaler()
    series_s = scaler.fit_transform(series)
    X, y = [], []
    for i in range(lookback, len(series_s)):
        X.append(series_s[i-lookback:i,0])
        y.append(series_s[i,0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lookback', type=int, default=14)
    parser.add_argument('--out', default='models/lstm_model.h5')
    args = parser.parse_args()
    # Resolve relative paths against this script's directory
    if not os.path.isabs(args.data):
        args.data = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data)


    X, y, scaler = load_and_preprocess(args.data, lookback=args.lookback)
    model = build_model((X.shape[1], X.shape[2]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    checkpoint = ModelCheckpoint(args.out, save_best_only=True, monitor='loss')
    model.fit(X, y, epochs=args.epochs, batch_size=32, callbacks=[checkpoint])
    # save scaler for later
    joblib.dump(scaler, os.path.join(os.path.dirname(args.out), 'scaler.save'))
    print("Training complete. Model saved to", args.out)
