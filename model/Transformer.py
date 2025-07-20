from base_predictor import BasePredictor
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import tensorflow as tf

# Try different import paths for MultiHeadAttention
try:
    from tensorflow.keras.layers import MultiHeadAttention
except ImportError:
    try:
        from tensorflow.keras.layers.experimental import MultiHeadAttention
    except ImportError:
        # For older TensorFlow versions, use a custom implementation
        MultiHeadAttention = None


class TransformerPredictor(BasePredictor):
    def __init__(self, window_size=30, horizon=15, batch_size=32,
                 epochs=50, patience=5, dropout_rate=0.1,
                 head_size=256, num_heads=4, ff_dim=4, num_blocks=2):
        super().__init__(window_size, horizon, batch_size, epochs,
                         patience, dropout_rate)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks

    def transformer_block(self, inputs, head_size, num_heads, ff_dim,
                          dropout=0.1):
        """Build a transformer block"""
        x = LayerNormalization(epsilon=1e-6)(inputs)

        # Use MultiHeadAttention if available, otherwise skip attention
        if MultiHeadAttention is not None:
            x = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        else:
            # Fallback: use a simple dense layer instead
            x = Dense(inputs.shape[-1], activation='relu')(x)

        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed forward
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return x + res

    def build_model(self, input_shape, horizon):
        """Build Transformer model"""
        inputs = Input(shape=input_shape)
        x = inputs

        # Apply transformer blocks
        for _ in range(self.num_blocks):
            x = self.transformer_block(x, self.head_size, self.num_heads,
                                       self.ff_dim, self.dropout_rate)

        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(horizon)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
