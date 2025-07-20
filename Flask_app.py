from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import os
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
monitor = None  # Instantiated by train_monitor in actual use
training_thread = None
is_training = False

# Ensure English display is normal
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False


# Mock monitor class (should be imported from train_monitor in actual project)
class TrainingMonitor:
    def __init__(self, socketio):
        self.socketio = socketio

    def log_message(self, message):
        self.socketio.emit('log', {'message': message})

    def on_epoch_end(self, epoch, loss, val_loss, accuracy, val_accuracy, total_epochs=None):
        progress = (epoch / total_epochs) * 100 if total_epochs else 0
        self.socketio.emit('update', {
            'current_epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'val_accuracy': val_accuracy,
            'progress': progress,
            'total_epochs': total_epochs
        })

    def add_metric(self, name, value):
        self.socketio.emit('metric', {'name': name, 'value': value})

    def set_model_info(self, model_name, dataset, total_epochs):
        self.socketio.emit('model_info', {
            'model_name': model_name,
            'dataset': dataset,
            'total_epochs': total_epochs
        })


# Initialize the monitor
monitor = TrainingMonitor(socketio)


@app.route('/')
def index():
    return render_template('sentiment_monitor.html')


@app.route('/start_training', methods=['POST'])
def start_training():
    global training_thread, is_training

    if is_training:
        return jsonify({"status": "error", "message": "Training is already in progress"})

    data = request.json
    model_type = data.get('model_type', 'LSTM')
    epochs = int(data.get('epochs', 10))
    crypto = data.get('crypto', 'Bitcoin')
    sentiment_type = data.get('sentiment_type', 'None')

    training_thread = threading.Thread(
        target=run_training,
        args=(model_type, epochs, crypto, sentiment_type)
    )
    training_thread.daemon = True
    training_thread.start()

    return jsonify({"status": "success", "message": f"Started training {crypto} (Sentiment Data: {sentiment_type})"})


def run_training(model_type, epochs, crypto, sentiment_type):
    global is_training
    is_training = True

    try:
        from Forecast import main as forecast_main
        os.environ['MODEL_TYPE'] = model_type
        os.environ['TRAIN_EPOCHS'] = str(epochs)
        os.environ['CRYPTO_CURRENCY'] = crypto
        os.environ['SENTIMENT_TYPE'] = sentiment_type
        forecast_main(monitor=monitor)
    except Exception as e:
        monitor.log_message(f"Training error: {str(e)}")
    finally:
        is_training = False
        monitor.log_message("Training process has ended")


@socketio.on('check_status')
def check_training_status():
    emit('status_update', {
        'is_training': is_training,
        'message': "Training in progress..." if is_training else "Not training"
    })


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    socketio.run(app, host='0.0.0.0', port=5002, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)