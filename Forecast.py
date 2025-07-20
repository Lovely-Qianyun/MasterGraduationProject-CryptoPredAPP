import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D,
    MaxPooling1D, Flatten, Input, Bidirectional,
    RepeatVector, TimeDistributed, MultiHeadAttention,
    LayerNormalization
)
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False


# Data loading
def load_forecast_data(monitor=None):
    try:
        crypto = os.environ.get('CRYPTO_CURRENCY', 'Bitcoin')
        data_path = f"./data/coin_{crypto}.csv"

        if monitor:
            monitor.log_message(f"Loading {crypto} data: {data_path}")

        df = pd.read_csv(data_path)

        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df.sort_values('date')
        else:
            if monitor:
                monitor.log_message("'Date' column not found in data, using default index")

        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'value'})
        else:
            raise ValueError("Missing 'Close' column (target variable) in data")

        if monitor:
            monitor.log_message(f"{crypto} data loaded successfully, total {len(df)} records")

        return df

    except Exception as e:
        error_msg = f"Failed to load {crypto} data: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None


# Sentiment data loading
def load_sentiment_data(sentiment_type, crypto, monitor=None):
    try:
        if sentiment_type == 'None':
            if monitor:
                monitor.log_message("Not using sentiment data")
            return None

        base_path = "./data"
        sentiment_data = {}

        if sentiment_type in ['svc', 'both']:
            svc_path = f"{base_path}/sentiment_score_svc.csv"
            svc_df = pd.read_csv(svc_path)
            # Key modification: Keep date as datetime type, not converted to string
            svc_df['date'] = pd.to_datetime(svc_df[['year', 'month', 'day']])  # Removed dt.strftime
            svc_df = svc_df.set_index('date')  # Index is now datetime type
            svc_df["sum"] = svc_df.sum(axis=1)
            svc_df = svc_df.div(svc_df["sum"], axis=0).drop(columns=['sum'])
            svc_df = svc_df.rename(
                columns={"positive": "positive_svc", "neutral": "neutral_svc", "negative": "negative_svc"}
            )
            sentiment_data['svc'] = svc_df
            if monitor:
                monitor.log_message(f"Loaded SVC sentiment data: {len(svc_df)} records")

        if sentiment_type in ['trans', 'both']:
            trans_path = f"{base_path}/sentiment_score_trans.csv"
            trans_df = pd.read_csv(trans_path)
            # Key modification: Keep date as datetime type, not converted to string
            trans_df['date'] = pd.to_datetime(trans_df[['year', 'month', 'day']])  # Removed dt.strftime
            trans_df = trans_df.set_index('date')  # Index is now datetime type
            trans_df["sum"] = trans_df.sum(axis=1)
            trans_df = trans_df.div(trans_df["sum"], axis=0).drop(columns=['sum'])
            trans_df = trans_df.rename(
                columns={"positive": "positive_trans", "neutral": "neutral_trans", "negative": "negative_trans"}
            )
            sentiment_data['trans'] = trans_df
            if monitor:
                monitor.log_message(f"Loaded Transformer sentiment data: {len(trans_df)} records")

        if len(sentiment_data) == 0:
            return None

        merged_df = None
        for df in sentiment_data.values():
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='date', how='inner')

        return merged_df

    except Exception as e:
        error_msg = f"Failed to load sentiment data: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None


# Data preprocessing
def preprocess_data(df, sequence_length=14, forecast_steps=7, monitor=None, model_type=None, sentiment_df=None):
    try:
        if monitor:
            monitor.log_message(f"Preprocessing data, sequence length: {sequence_length}, forecast steps: {forecast_steps}")

        values = df['value'].values.reshape(-1, 1)

        if model_type == 'ARIMA':
            return values

        if sentiment_df is not None:
            df_with_date = df[['date', 'value']]
            df_with_date['date'] = pd.to_datetime(df_with_date['date'])

            merged = pd.merge(df_with_date, sentiment_df, on='date', how='inner')


            feature_cols = ['value']
            if 'positive_svc' in merged.columns and "neutral_svc" in merged.columns and "negative_svc" in merged.columns:
                feature_cols.append('positive_svc')
                feature_cols.append('neutral_svc')
                feature_cols.append('negative_svc')
            if 'positive_trans' in merged.columns and "neutral_trans" in merged.columns and "negative_trans" in merged.columns:
                feature_cols.append('positive_trans')
                feature_cols.append('neutral_trans')
                feature_cols.append('negative_trans')

            features = merged[feature_cols].values
            if monitor:
                monitor.log_message(f"Merged features: {feature_cols}, total {features.shape[0]} records")
        else:
            features = values
            feature_cols = ['value']

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(scaled_values) - sequence_length - forecast_steps + 1):
            X.append(scaled_values[i:i + sequence_length, :])
            y.append(scaled_values[i + sequence_length:i + sequence_length + forecast_steps, 0])

        X = np.array(X)
        y = np.array(y)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if monitor:
            monitor.log_message(f"Data preprocessing completed - training samples: {len(X_train)}, test samples: {len(X_test)}")

        return (X_train, X_test, y_train, y_test, scaler)

    except Exception as e:
        error_msg = f"Data preprocessing failed: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None


# Model construction
def create_forecast_model(model_type='LSTM', input_shape=None, output_steps=7, monitor=None):
    try:
        if monitor:
            monitor.log_message(f"Creating forecast model: {model_type}")

        if model_type == 'ARIMA':
            return "ARIMA"

        if input_shape is None:
            raise ValueError("Must provide input shape (input_shape)")

        if model_type == 'LSTM':
            model = Sequential([
                LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(output_steps)
            ])

        elif model_type == 'BD LSTM':
            model = Sequential([
                Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=input_shape),
                Dropout(0.2),
                Bidirectional(LSTM(32, activation='relu')),
                Dropout(0.2),
                Dense(output_steps)
            ])

        elif model_type == 'ED LSTM':
            model = Sequential([
                LSTM(64, activation='relu', input_shape=input_shape),
                Dropout(0.2),
                RepeatVector(output_steps),
                LSTM(32, activation='relu', return_sequences=True),
                Dropout(0.2),
                TimeDistributed(Dense(1)),
                Flatten(),
                Dense(output_steps)
            ])

        elif model_type == 'CNN':
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(output_steps)
            ])

        elif model_type == 'Convolutional LSTM':
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(output_steps)
            ])

        elif model_type == 'Transformer':
            inputs = Input(shape=input_shape)
            x = Dense(32, activation='relu')(inputs)
            attn_output = MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            ff_output = Dense(64, activation='relu')(x)
            ff_output = Dense(32)(ff_output)
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)
            x = Flatten()(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(output_steps)(x)
            model = Model(inputs=inputs, outputs=outputs)

        elif model_type == 'MLP':
            model = Sequential([
                Flatten(input_shape=input_shape),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(output_steps)
            ])

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[MeanAbsoluteError(name='mae'),
                     MeanSquaredError(name='mse')]
        )

        if monitor:
            monitor.log_message(f"{model_type} model created successfully")

        return model

    except Exception as e:
        error_msg = f"Failed to create forecast model: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None


# Model training
def train_forecast_model(model, model_type, train_data, test_data, epochs=50, monitor=None, scaler=None):
    try:
        if model_type == 'ARIMA':
            return train_arima_model(train_data, test_data, monitor)

        X_train, y_train = train_data
        X_test, y_test = test_data

        if monitor:
            monitor.log_message(f"Starting training {model_type} forecast model, total {epochs} epochs")

        for epoch in range(1, epochs + 1):
            history = model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )

            loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            mae = history.history['mae'][0]
            val_mae = history.history['val_mae'][0]

            if monitor:
                monitor.on_epoch_end(epoch, loss, val_loss, mae, val_mae, total_epochs=epochs)
                monitor.add_metric('mse', history.history['mse'][0])
                monitor.add_metric('val_mse', history.history['val_mse'][0])

        if monitor:
            monitor.log_message("Starting model evaluation...")

        eval_results = model.evaluate(X_test, y_test, verbose=0)
        loss, mae, mse = eval_results

        if monitor:
            monitor.log_message(f"Model evaluation completed:")
            monitor.log_message(f"Test loss: {loss:.6f}")
            monitor.log_message(f"Test MAE: {mae:.6f}")
            monitor.log_message(f"Test MSE: {mse:.6f}")

        save_training_plots(history, model_type)

        return model, y_test

    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None, None


# ARIMA training
def train_arima_model(train_data, test_data, monitor=None):
    try:
        from statsmodels.tsa.arima.model import ARIMA as statsARIMA

        if monitor:
            monitor.log_message("Starting ARIMA model training...")

        train_values = train_data.flatten()
        test_values = test_data.flatten()

        best_aic = float('inf')
        best_model = None

        for p in range(1, 4):
            for d in range(1, 2):
                for q in range(1, 4):
                    try:
                        model = statsARIMA(train_values, order=(p, d, q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_model = results
                            if monitor:
                                monitor.log_message(f"ARIMA({p},{d},{q}) - AIC: {results.aic:.2f}")
                    except:
                        continue

        if best_model is None:
            if monitor:
                monitor.log_message("ARIMA parameter search failed, using default parameters (2,1,2)")
            model = statsARIMA(train_values, order=(2, 1, 2))
            best_model = model.fit()

        forecast_steps = len(test_values)
        predictions = best_model.forecast(steps=forecast_steps)

        rmse = np.sqrt(mean_squared_error(test_values, predictions))
        mae = mean_absolute_error(test_values, predictions)

        if monitor:
            monitor.log_message("ARIMA model evaluation completed:")
            monitor.log_message(f"Test RMSE: {rmse:.6f}")
            monitor.log_message(f"Test MAE: {mae:.6f}")
            monitor.on_epoch_end(1, rmse, rmse, mae, mae, total_epochs=1)
            monitor.add_metric('rmse', rmse)
            monitor.add_metric('mae', mae)

        save_forecast_comparison(
            test_values.reshape(-1, 1),
            predictions.reshape(-1, 1),
            forecast_steps,
            "ARIMA"
        )

        return best_model, test_values

    except Exception as e:
        error_msg = f"ARIMA training failed: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None, None


# Forecast generation
def make_forecasts(model, X_test, y_test, scaler, forecast_steps=7, model_type='LSTM', monitor=None):
    try:
        if model_type == 'ARIMA':
            return None, None

        if monitor:
            monitor.log_message(f"Generating {model_type} forecast results...")

        predictions = model.predict(X_test, verbose=0)

        if scaler:
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y_test)
        else:
            actual = y_test

        save_forecast_comparison(actual, predictions, forecast_steps, model_type)

        if monitor:
            monitor.log_message(f"Forecast generation completed, total {len(predictions)} forecast samples")

        return predictions, actual

    except Exception as e:
        error_msg = f"Failed to generate forecasts: {str(e)}"
        if monitor:
            monitor.log_message(error_msg)
        else:
            print(error_msg)
        return None, None


# Visualization tools
def save_training_plots(history, model_type):
    try:
        results_dir = "../forecast_results"
        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title(f'{model_type} Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{model_type} MAE Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'{model_type}_training_curves.png')
        plt.savefig(plot_path)
        plt.close()

    except Exception as e:
        print(f"Failed to save training curves: {str(e)}")


def save_forecast_comparison(actual, predictions, forecast_steps, model_type):
    try:
        results_dir = "../forecast_results"
        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(15, 8))
        sample_indices = [0, min(5, len(actual) // 4), min(10, len(actual) // 2),
                          min(15, 3 * len(actual) // 4), min(20, len(actual) - 1)]

        for i, idx in enumerate(sample_indices[:5]):
            if idx >= len(actual):
                continue

            plt.subplot(1, 5, i + 1)
            plt.plot(range(min(forecast_steps, len(actual[idx]))), actual[idx][:forecast_steps], label='Actual value')
            plt.plot(range(min(forecast_steps, len(predictions[idx]))), predictions[idx][:forecast_steps], label='Predicted value')
            plt.title(f'Sample {idx}')
            plt.xlabel('Forecast step')
            plt.ylabel('Value')
            if i == 0:
                plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'{model_type}_forecast_comparison.png')
        plt.savefig(plot_path)
        plt.close()

    except Exception as e:
        print(f"Failed to save forecast comparison plot: {str(e)}")


# Main function
def main(monitor=None):
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

    model_type = os.environ.get('MODEL_TYPE', 'LSTM')
    epochs = int(os.environ.get('TRAIN_EPOCHS', 10))
    crypto = os.environ.get('CRYPTO_CURRENCY', 'Bitcoin')
    sentiment_type = os.environ.get('SENTIMENT_TYPE', 'None')

    sequence_length = 14
    forecast_steps = 7

    os.makedirs('../forecast_results', exist_ok=True)

    df = load_forecast_data(monitor=monitor)
    if df is None:
        if monitor:
            monitor.log_message("Data loading failed, cannot continue")
        return

    sentiment_df = load_sentiment_data(
        sentiment_type=sentiment_type,
        crypto=crypto,
        monitor=monitor
    )

    data = preprocess_data(
        df,
        sequence_length=sequence_length,
        forecast_steps=forecast_steps,
        monitor=monitor,
        model_type=model_type,
        sentiment_df=sentiment_df
    )

    if model_type == 'ARIMA':
        full_data = data
        split_idx = int(len(full_data) * 0.8)
        train_data = full_data[:split_idx]
        test_data = full_data[split_idx:]
        X_train, X_test, y_train, y_test, scaler = None, None, train_data, test_data, None
    else:
        if data is None:
            if monitor:
                monitor.log_message("Data preprocessing failed, cannot continue")
            return
        X_train, X_test, y_train, y_test, scaler = data

    if monitor:
        if model_type != 'ARIMA':
            feature_dim = X_train.shape[2]
        else:
            feature_dim = 1
        monitor.set_model_info(
            f"{crypto} {model_type} (Sentiment: {sentiment_type})",
            f"Features: {feature_dim}D, sequence={sequence_length}",
            epochs if model_type != 'ARIMA' else 1
        )

    if model_type != 'ARIMA':
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_forecast_model(
            model_type,
            input_shape=input_shape,
            output_steps=forecast_steps,
            monitor=monitor
        )

        if model is None:
            if monitor:
                monitor.log_message(f"Skipping {model_type} model")
            return
    else:
        model = "ARIMA"

    trained_model, test_labels = train_forecast_model(
        model,
        model_type,
        (X_train, y_train) if model_type != 'ARIMA' else train_data,
        (X_test, y_test) if model_type != 'ARIMA' else test_data,
        epochs=epochs,
        monitor=monitor,
        scaler=scaler
    )

    if trained_model:
        if model_type != 'ARIMA':
            make_forecasts(
                trained_model,
                X_test,
                test_labels,
                scaler,
                forecast_steps=forecast_steps,
                model_type=model_type,
                monitor=monitor
            )

    del model
    tf.keras.backend.clear_session()

    if monitor:
        monitor.log_message(f"{crypto} (Sentiment data: {sentiment_type}) {model_type} model training completed!")


if __name__ == '__main__':
    main()