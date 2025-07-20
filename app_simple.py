"""
Unified Cryptocurrency Analytics Platform - Simplified Version
整合了三个模块：宏观指标分析、情感分析和价格预测
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加各模块路径
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'macro-index'))

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
socketio = SocketIO(app, async_mode='threading')

# 全局训练线程和监控器
training_thread = None
macro_monitor = None


def init_macro_monitor():
    """初始化宏观分析监控器"""
    global macro_monitor
    try:
        from train_monitor import TrainingMonitor
        macro_monitor = TrainingMonitor(socketio)
        print("✓ Macro monitor initialized")
    except ImportError as e:
        print(f"✗ Error loading macro monitor: {e}")

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
        return jsonify({'status': 'error',
                       'message': 'Monitor not initialized'})

    if action == 'start':
        if hasattr(macro_monitor, 'training_stopped') and macro_monitor.training_stopped:
            return jsonify({'status': 'error',
                           'message': 'Training has been stopped and cannot be restarted'})

        if not training_thread or not training_thread.is_alive():
            training_thread = threading.Thread(target=start_macro_training)
            training_thread.daemon = True
            training_thread.start()
            return jsonify({'status': 'success', 'message': 'Training started'})
        elif hasattr(macro_monitor, 'training_paused') and macro_monitor.training_paused:
            macro_monitor.resume_training()
            return jsonify({'status': 'success', 'message': 'Training resumed'})
        else:
            return jsonify({'status': 'info', 'message': 'Training is already working'})

    elif action == 'pause':
        if not training_thread or not training_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'No training running'})
        if hasattr(macro_monitor, 'pause_training'):
            macro_monitor.pause_training()
        return jsonify({'status': 'success', 'message': 'Training paused'})

    elif action == 'stop':
        if not training_thread or not training_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'No training running'})
        if hasattr(macro_monitor, 'stop_training'):
            macro_monitor.stop_training()
        if training_thread:
            training_thread.join(timeout=5.0)
        return jsonify({'status': 'success', 'message': 'Training stopped'})

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
    """执行价格预测 - 简化版本"""
    try:
        data = request.json
        crypto_name = data.get('crypto_name')
        model_type = data.get('model_type')
        prediction_days = int(data.get('prediction_days', 30))

        # 模拟预测结果（实际应用中会使用真实的模型）
        import numpy as np
        predictions = np.random.rand(
            prediction_days) * 1000 + 50000  # 模拟Bitcoin价格

        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'train_loss': 0.05,
            'val_loss': 0.07,
            'model_type': model_type,
            'crypto_name': crypto_name,
            'prediction_days': prediction_days,
            'message': 'This is a demo prediction. Install TensorFlow for real model training.'
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
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
    """启动情感分析训练 - 简化版本"""
    return jsonify({
        'status': 'info',
        'message': 'Sentiment training will be available when all dependencies are installed'
    })

# ============================================================================
# 模型可视化路由
# ============================================================================


@app.route('/model-visualization')
def model_visualization():
    """模型可视化页面"""
    return render_template('macro_correlation.html')


# ============================================================================
# 应用初始化和启动
# ============================================================================
if __name__ == '__main__':
    # 初始化监控器
    init_macro_monitor()

    # 启动应用
    print("Starting Unified Cryptocurrency Analytics Platform...")
    print("Access the application at: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=True, use_reloader=False)
