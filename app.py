"""
Unified Cryptocurrency Analytics Platform
整合了三个模块：宏观指标分析、情感分析和价格预测
"""

import matplotlib
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import os
import sys
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# 添加各模块路径
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'macro-index'))
sys.path.append(os.path.join(current_dir, 'Furure Price Prediction'))
sys.path.append(os.path.join(current_dir, 'Sentiment Analysis'))
sys.path.append(os.path.join(current_dir, 'Furure Price Prediction', 'model'))

# 配置matplotlib
matplotlib.use('Agg')

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
socketio = SocketIO(app, async_mode='threading')

# 全局训练线程和监控器
training_thread = None
sentiment_training_thread = None
macro_monitor = None
sentiment_monitor = None

# 初始化监控器


def init_monitors():
    global macro_monitor, sentiment_monitor
    try:
        from train_monitor import TrainingMonitor as MacroTrainingMonitor
        macro_monitor = MacroTrainingMonitor(socketio)
    except ImportError:
        print("Warning: Could not import macro training monitor")

    try:
        # 这里使用简化的情感分析监控器
        class SentimentTrainingMonitor:
            def __init__(self, socketio):
                self.socketio = socketio

            def log_message(self, message):
                self.socketio.emit('sentiment_log', {'message': message})

            def on_epoch_end(self, epoch, loss, val_loss, accuracy, val_accuracy, total_epochs=None):
                progress = (epoch / total_epochs) * 100 if total_epochs else 0
                self.socketio.emit('sentiment_update', {
                    'current_epoch': epoch,
                    'loss': loss,
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'val_accuracy': val_accuracy,
                    'progress': progress,
                    'total_epochs': total_epochs
                })

        sentiment_monitor = SentimentTrainingMonitor(socketio)
    except Exception as e:
        print(f"Warning: Could not initialize sentiment monitor: {e}")


# 初始化预测模型
MODEL_CLASSES = {}


def init_prediction_models():
    global MODEL_CLASSES
    try:
        from CNN import CNNPredictor
        MODEL_CLASSES['CNN'] = CNNPredictor
        print("✓ CNN model loaded")
    except ImportError as e:
        print(f"✗ Error loading CNN: {e}")

    try:
        from ConvLSTM import ConvLSTMPredictor
        MODEL_CLASSES['ConvLSTM'] = ConvLSTMPredictor
        print("✓ ConvLSTM model loaded")
    except ImportError as e:
        print(f"✗ Error loading ConvLSTM: {e}")

    try:
        from EDLSTM import EDLSTMPredictor
        MODEL_CLASSES['EDLSTM'] = EDLSTMPredictor
        print("✓ EDLSTM model loaded")
    except ImportError as e:
        print(f"✗ Error loading EDLSTM: {e}")

    try:
        from Transformer import TransformerPredictor
        MODEL_CLASSES['Transformer'] = TransformerPredictor
        print("✓ Transformer model loaded")
    except ImportError as e:
        print(f"✗ Error loading Transformer: {e}")

# ============================================================================
# 主页路由
# ============================================================================


@app.route('/')
def home():
    """主页"""
    return render_template('index.html')

# ============================================================================
# 宏观指标分析模块路由
# ============================================================================


@app.route('/model-monitor')
def model_monitor():
    """宏观指标模型监控页面"""
    return render_template('monitor.html')


@app.route('/macro-correlation')
def macro_correlation():
    """宏观指标关联分析页面"""
    return render_template('macro_correlation.html')


@app.route('/macro-simulation')
def macro_simulation():
    """宏观经济冲击模拟页面"""
    return render_template('macro_simulation.html')


@app.route('/api/correlations')
def get_correlations():
    """获取相关系数数据"""
    period = request.args.get('period', 'all')
    try:
        from main import calculate_correlations
        correlations = calculate_correlations(period)
        return jsonify(correlations)
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({"error": "Data load failed"}), 500


@app.route('/api/historical-correlations')
def get_historical_correlations():
    """获取历史相关系数数据"""
    crypto = request.args.get('crypto')
    indicator = request.args.get('indicator')
    period = request.args.get('period', 'all')
    try:
        from main import calculate_historical_correlations
        data = calculate_historical_correlations(crypto, indicator, period)
        return jsonify(data)
    except Exception as e:
        print(f"Historical API error: {str(e)}")
        return jsonify({"error": "Failed to load historical data"}), 500


@app.route('/control/<action>')
def training_control(action):
    """宏观指标模型训练控制"""
    global training_thread, macro_monitor

    if not macro_monitor:
        return jsonify({'status': 'error', 'message': 'Monitor not initialized'})

    if action == 'start':
        if macro_monitor.training_stopped:
            return jsonify({'status': 'error', 'message': 'Training has been stopped and cannot be restarted'})

        if not training_thread or not training_thread.is_alive():
            training_thread = threading.Thread(target=start_macro_training)
            training_thread.daemon = True
            training_thread.start()
            return jsonify({'status': 'success', 'message': 'start to train'})
        elif macro_monitor.training_paused:
            macro_monitor.resume_training()
            return jsonify({'status': 'success', 'message': 'Training has resumed'})
        else:
            return jsonify({'status': 'info', 'message': 'Training is already working'})

    elif action == 'pause':
        if not training_thread or not training_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'There is no training running'})
        macro_monitor.pause_training()
        return jsonify({'status': 'success', 'message': 'Training has been suspended'})

    elif action == 'stop':
        if not training_thread or not training_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'There is no training running'})
        macro_monitor.stop_training()
        training_thread.join(timeout=5.0)
        return jsonify({'status': 'success', 'message': 'Training has stopped'})

    return jsonify({'status': 'error', 'message': 'Invalid operation'})


def start_macro_training():
    """启动宏观指标模型训练"""
    try:
        from main import main
        main(monitor=macro_monitor)
    except Exception as e:
        print(f"Training error: {e}")

# ============================================================================
# 价格预测模块路由
# ============================================================================


@app.route('/future-prediction')
def future_prediction():
    """价格预测主页"""
    return render_template('pred.html')


@app.route('/predict', methods=['POST'])
def predict():
    """执行价格预测"""
    try:
        data = request.json
        crypto_name = data.get('crypto_name')
        model_type = data.get('model_type')
        prediction_days = int(data.get('prediction_days', 30))

        if model_type not in MODEL_CLASSES:
            return jsonify({'error': f'Model {model_type} not available'}), 400

        # 获取数据文件路径
        data_file = f'data/coin_{crypto_name}.csv'
        if not os.path.exists(data_file):
            return jsonify({'error': f'Data file for {crypto_name} not found'}), 400

        # 创建并训练模型
        model_class = MODEL_CLASSES[model_type]
        predictor = model_class(
            data_file=data_file,
            target_column='close',
            sequence_length=60,
            prediction_days=prediction_days
        )

        # 训练模型
        print(f"Training {model_type} model for {crypto_name}...")
        train_loss, val_loss = predictor.train_model(epochs=50, batch_size=32)

        # 生成预测
        predictions, confidence_intervals = predictor.predict_future()

        # 生成图表
        img_buffer = io.BytesIO()
        predictor.plot_predictions(save_path=img_buffer, format='png')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode()

        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'confidence_intervals': confidence_intervals.tolist() if confidence_intervals is not None else None,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'plot_image': f'data:image/png;base64,{img_data}',
            'model_type': model_type,
            'crypto_name': crypto_name,
            'prediction_days': prediction_days
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/help')
def help_page():
    """帮助页面"""
    return render_template('help.html')

# ============================================================================
# 情感分析模块路由
# ============================================================================


@app.route('/sentiment-monitor')
def sentiment_monitor():
    """情感分析模型监控页面"""
    return render_template('sentiment_monitor.html')


@app.route('/start_sentiment_training', methods=['POST'])
def start_sentiment_training():
    """启动情感分析训练"""
    global sentiment_training_thread, sentiment_monitor

    if not sentiment_monitor:
        return jsonify({'status': 'error',
                       'message': 'Sentiment monitor not initialized'})

    if (sentiment_training_thread and
            sentiment_training_thread.is_alive()):
        return jsonify({'status': 'error',
                       'message': 'Training is already running'})

    sentiment_training_thread = threading.Thread(target=run_sentiment_training)
    sentiment_training_thread.daemon = True
    sentiment_training_thread.start()

    return jsonify({'status': 'success',
                   'message': 'Sentiment training started'})


def run_sentiment_training():
    """运行情感分析训练"""
    try:
        from Forecast import main as sentiment_main
        sentiment_main(monitor=sentiment_monitor)
    except Exception as e:
        print(f"Sentiment training error: {e}")

# ============================================================================
# 模型可视化路由
# ============================================================================


@app.route('/model-visualization')
def model_visualization():
    """模型可视化页面"""
    # 这里可以添加模型可视化的逻辑
    # 暂时重定向到宏观相关性分析
    return render_template('macro_correlation.html')


# ============================================================================
# 应用初始化和启动
# ============================================================================
if __name__ == '__main__':
    # 初始化各模块
    init_monitors()
    init_prediction_models()

    # 启动应用
    print("Starting Unified Cryptocurrency Analytics Platform...")
    print("Available models:", list(MODEL_CLASSES.keys()))
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=True, use_reloader=False)
