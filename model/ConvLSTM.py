from base_predictor import BasePredictor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    ConvLSTM2D, BatchNormalization, Dense, Dropout, Flatten, Reshape
)
import numpy as np


class ConvLSTMPredictor(BasePredictor):
    def __init__(self, window_size=30, horizon=15, batch_size=32,
                 epochs=50, patience=5, dropout_rate=0.2,
                 filters=64, kernel_size=3, spatial_height=4, spatial_width=6):
        super().__init__(window_size, horizon, batch_size, epochs,
                         patience, dropout_rate)
        self.filters = filters
        self.kernel_size = kernel_size
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width

    def reshape_to_spatial(self, data, spatial_height, spatial_width):
        """Reshape 1D feature vector to spatial grid for ConvLSTM"""
        batch_size, time_steps, features = data.shape

        # Pad features if necessary to fit spatial dimensions
        target_features = spatial_height * spatial_width
        if features < target_features:
            padding = target_features - features
            pad_zeros = np.zeros((batch_size, time_steps, padding))
            data = np.concatenate([data, pad_zeros], axis=2)
        elif features > target_features:
            data = data[:, :, :target_features]

        # Reshape to spatial grid
        reshaped = data.reshape(batch_size, time_steps, spatial_height,
                                spatial_width, 1)
        return reshaped

    def build_model(self, input_shape, horizon):
        """Build ConvLSTM model"""
        # Calculate spatial features needed
        total_features = input_shape[1]  # feature dimension
        spatial_features = self.spatial_height * self.spatial_width

        model = Sequential([
            Reshape((input_shape[0], self.spatial_height,
                    self.spatial_width, 1)),
            ConvLSTM2D(filters=self.filters,
                       kernel_size=(self.kernel_size, self.kernel_size),
                       activation='relu', return_sequences=True,
                       padding='same'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            ConvLSTM2D(filters=self.filters//2,
                       kernel_size=(self.kernel_size, self.kernel_size),
                       activation='relu', return_sequences=False,
                       padding='same'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(horizon)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _prepare_data(self, coin_name):
        """Override to add spatial reshaping"""
        X, y, feat_scaler, targ_scaler, merged_data = super()._prepare_data(coin_name)

        # Reshape X for ConvLSTM
        X_reshaped = self.reshape_to_spatial(
            X, self.spatial_height, self.spatial_width)

        return X_reshaped, y, feat_scaler, targ_scaler, merged_data
