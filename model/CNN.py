from base_predictor import BasePredictor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, Flatten, Dense
)


class CNNPredictor(BasePredictor):
    def __init__(self, window_size=30, horizon=15, batch_size=32,
                 epochs=50, patience=5, dropout_rate=0.2,
                 filters=64, kernel_size=3, pool_size=2):
        super().__init__(window_size, horizon, batch_size, epochs,
                         patience, dropout_rate)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def build_model(self, input_shape, horizon):
        """Build CNN model"""
        model = Sequential([
            Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                   activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=self.pool_size),
            Dropout(self.dropout_rate),
            Conv1D(filters=self.filters*2, kernel_size=self.kernel_size,
                   activation='relu'),
            MaxPooling1D(pool_size=self.pool_size),
            Dropout(self.dropout_rate),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(horizon)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
