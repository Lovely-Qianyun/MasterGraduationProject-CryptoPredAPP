import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score


# 1. Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        if mask is not None:
            attention_mask = mask[:, tf.newaxis, tf.newaxis, :]
        else:
            attention_mask = None

        attn_output = self.mha(x, x, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


# 2. Transformer Classifier Model
class TransformerClassifier(tf.keras.Model):
    def __init__(self, bert_model, num_layers, d_model, num_heads, dff, max_len, rate=0.1):
        super(TransformerClassifier, self).__init__()

        self.bert = bert_model
        self.num_layers = num_layers
        self.d_model = d_model

        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.dropout = Dropout(rate)
        self.final_layer = Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # BERT embeddings
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        x = bert_output

        # Transformer Encoder Layers
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, attention_mask)

        # Global Average Pooling and Output
        x = tf.reduce_mean(x, axis=1)
        return self.final_layer(x)


# 3. Data Processing Functions
def labelencoder(train, valid, test):
    le = LabelEncoder()
    le.fit(train['sentiment_label'].values)
    train['sentiment_label'] = le.transform(train['sentiment_label'])
    valid['sentiment_label'] = le.transform(valid['sentiment_label'])
    test['sentiment_label'] = le.transform(test['sentiment_label'])
    return le, train, valid, test


def create_dataset(df, tokenizer, max_len, batch_size):
    texts = df['sentence'].tolist()
    labels = df['sentiment_label'].values

    encodings = tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        },
        labels
    )).batch(batch_size)

    return dataset


def macro_f1(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    return tf.py_function(
        lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        (y_true, y_pred),
        tf.float32
    )


# 4. Main Training Function
def main():
    # Set random seeds
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Device setup
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f'Using {device}')

    # Load data
    print('Reading Train, Test, Val')
    train_data = pd.read_csv('./Financial sentiment analysis data/train_set.csv')
    valid_data = pd.read_csv('./Financial sentiment analysis data/validation_set.csv')
    test_data = pd.read_csv('./Financial sentiment analysis data/test_set.csv')

    # Label encoding
    print('Encoding Labels')
    le, train_data, valid_data, test_data = labelencoder(train_data, valid_data, test_data)

    # Initialize tokenizer and BERT model
    bert_model_name = 'ProsusAI/finbert'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = TFBertModel.from_pretrained(bert_model_name)

    # Create datasets
    print('Data prep')
    BATCH_SIZE = 32
    MAX_LEN = 32
    train_dataset = create_dataset(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
    valid_dataset = create_dataset(valid_data, tokenizer, MAX_LEN, BATCH_SIZE)
    test_dataset = create_dataset(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

    # Model parameters
    NUM_LAYERS = 4
    D_MODEL = 768
    NUM_HEADS = 12
    DFF = 3072
    DROPOUT_RATE = 0.1

    # Create model
    print('Defining Model')
    model = TransformerClassifier(
        bert_model=bert_model,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        max_len=MAX_LEN,
        rate=DROPOUT_RATE
    )

    # Freeze BERT layers
    for layer in bert_model.layers:
        layer.trainable = False

    # Compile model
    optimizer = Adam(learning_rate=5e-5)
    loss_fn = SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[SparseCategoricalAccuracy(), macro_f1]
    )

    # Training
    print('Start model training')
    N_EPOCHS = 40
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # Train
        train_history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            verbose=1
        )

        # Evaluate
        valid_loss, valid_acc, valid_f1 = model.evaluate(valid_dataset, verbose=0)

        # Timing
        epoch_mins = int((time.time() - start_time) / 60)
        epoch_secs = int((time.time() - start_time) % 60)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_weights('./FinBERT-Transformer.h5')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_history.history["loss"][0]:.3f} | Train Acc: {train_history.history["sparse_categorical_accuracy"][0] * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% | Val. F1: {valid_f1 * 100:.2f}%')

    # Test evaluation
    print('Evaluating on test set...')
    test_loss, test_acc, test_f1 = model.evaluate(test_dataset)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test F1: {test_f1 * 100:.2f}%')

    # Generate classification report
    print('\nClassification Report:')
    y_true = []
    y_pred = []

    for batch in test_dataset:
        inputs, labels = batch
        preds = model.predict(inputs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    main()