from base_predictor import BasePredictor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout


class EDLSTMPredictor(BasePredictor):
    def __init__(self, window_size=30, horizon=15, batch_size=32,
                 epochs=50, patience=5, dropout_rate=0.2,
                 encoder_units=128, decoder_units=64):
        super().__init__(window_size, horizon, batch_size, epochs,
                         patience, dropout_rate)
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

    def build_model(self, input_shape, horizon):
        """Build Encoder-Decoder LSTM model"""
        model = Sequential([
            # Encoder
            LSTM(self.encoder_units, activation='relu',
                 input_shape=input_shape, return_sequences=False),
            Dropout(self.dropout_rate),

            # Repeat vector to match horizon length
            RepeatVector(horizon),

            # Decoder
            LSTM(self.decoder_units, activation='relu', return_sequences=True),
            Dropout(self.dropout_rate),

            # Output layer
            TimeDistributed(Dense(1))
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
