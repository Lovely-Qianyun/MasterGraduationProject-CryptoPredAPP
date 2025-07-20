import os
import pandas as pd
import numpy as np
from datetime import timedelta, date
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from abc import ABC, abstractmethod


class BasePredictor(ABC):
    def __init__(self, window_size=30, horizon=15, batch_size=32, epochs=50, patience=5, dropout_rate=0.2):
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout_rate = dropout_rate

        # Load macroeconomic data
        self.macro = self._load_macro_data()
        self.macro_cols = [
            'CPIAUCSL', 'DTWEXBGS', 'RIFSPFF_N.D', 'PCEPI',
            'SP_Close', 'SP_High', 'SP_Low', 'SP_Volume',
            'US3Y', 'US10Y', 'VIX_High', 'VIX_Close',
            'sen_svc_positive', 'sen_svc_neutral', 'sen_svc_negative',
            'sen_trans_positive', 'sen_trans_neutral', 'sen_trans_negative'
        ]

    def _load_macro_data(self):
        """Load and preprocess macroeconomic data"""
        macro_path = os.path.join(os.path.dirname(
            __file__), '..', 'data', 'full_macro.csv')
        macro = pd.read_csv(macro_path)
        macro['Date'] = pd.to_datetime(macro['Date']).dt.date
        macro = macro.rename(columns={
            'Close': 'SP_Close',
            'High': 'SP_High',
            'Low': 'SP_Low',
            'Volume': 'SP_Volume',
            'HIGH': 'VIX_High',
            'CLOSE': 'VIX_Close'
        })
        return macro

    def _load_coin_data(self, coin_name):
        """Load and preprocess coin data"""
        coin_path = os.path.join(os.path.dirname(
            __file__), '..', 'data', f'coin_{coin_name}.csv')
        df = pd.read_csv(coin_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.rename(columns={
            'Open': 'Coin_Open', 'High': 'Coin_High', 'Low': 'Coin_Low',
            'Close': 'Coin_Close', 'Volume': 'Coin_Volume', 'Marketcap': 'Coin_Marketcap'
        })
        return df

    def _prepare_data(self, coin_name):
        """Prepare data for training and prediction"""
        # Load coin data
        df = self._load_coin_data(coin_name)

        # Merge with macro data
        merged = pd.merge(df, self.macro, on='Date',
                          how='inner').sort_values('Date')

        # Define feature columns
        feature_cols = ['Coin_Open', 'Coin_High', 'Coin_Low',
                        'Coin_Volume', 'Coin_Marketcap'] + self.macro_cols
        target_col = 'Coin_Close'

        # Extract features and targets
        features = merged[feature_cols].values
        targets = merged[[target_col]].values

        # Scale features and targets
        feat_scaler = MinMaxScaler()
        targ_scaler = MinMaxScaler()
        features_scaled = feat_scaler.fit_transform(features)
        targets_scaled = targ_scaler.fit_transform(targets)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.window_size - self.horizon + 1):
            X.append(features_scaled[i:i+self.window_size])
            y.append(targets_scaled[i+self.window_size:i +
                     self.window_size+self.horizon].flatten())

        X = np.array(X)
        y = np.array(y)

        return X, y, feat_scaler, targ_scaler, merged

    @abstractmethod
    def build_model(self, input_shape, horizon):
        """Build the neural network model"""
        pass

    def train_and_predict(self, coin_name):
        """Train model and make predictions"""
        # Prepare data
        X, y, feat_scaler, targ_scaler, merged_data = self._prepare_data(
            coin_name)

        # Split data
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Build model
        model = self.build_model(X.shape[1:], self.horizon)

        # Train model
        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(monitor='val_loss',
                           patience=self.patience, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            verbose=0  # Suppress training output
        )

        # Make prediction
        last_window = X[-1:]  # Use the last sequence for prediction
        pred_scaled = model.predict(last_window, verbose=0).flatten()
        preds = targ_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten()

        # Generate prediction dates
        last_date = merged_data['Date'].max()
        pred_dates = [last_date + timedelta(days=i)
                      for i in range(1, self.horizon+1)]

        # Get recent historical data for plotting
        recent_data = merged_data.tail(30)  # Last 30 days
        historical_dates = recent_data['Date'].tolist()
        historical_prices = recent_data['Coin_Close'].tolist()

        return {
            'historical_dates': historical_dates,
            'historical_prices': historical_prices,
            'prediction_dates': pred_dates,
            'predicted_prices': preds.tolist(),
            'coin_name': coin_name,
            'model_name': self.__class__.__name__
        }
