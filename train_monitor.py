from datetime import datetime


class TrainingMonitor:
    def __init__(self, socketio=None):
        self.socketio = socketio
        self.data = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'epochs': [],
            'start_time': datetime.now(),
            'current_epoch': 0,
            'total_epochs': 0,
            'progress': 0,
            'model_name': '',
            'dataset': ''
        }

    def set_model_info(self, model_name, dataset, total_epochs):
        """设置模型基本信息"""
        self.data['model_name'] = model_name
        self.data['dataset'] = dataset
        self.data['total_epochs'] = total_epochs
        self._emit('model_info', {
            'model_name': model_name,
            'dataset': dataset,
            'total_epochs': total_epochs
        })

    def on_epoch_end(self, epoch, loss, val_loss, accuracy, val_accuracy):
        """每轮训练结束时更新指标"""
        self.data['current_epoch'] = epoch
        self.data['loss'].append(loss)
        self.data['val_loss'].append(val_loss)
        self.data['accuracy'].append(accuracy)
        self.data['val_accuracy'].append(val_accuracy)
        self.data['epochs'].append(epoch)
        self.data['progress'] = int((epoch / self.data['total_epochs']) * 100)

        self._emit('update', {
            'current_epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'val_accuracy': val_accuracy,
            'progress': self.data['progress']
        })

    def add_metric(self, metric_name, value):
        """添加额外评估指标"""
        if metric_name not in self.data:
            self.data[metric_name] = []
        self.data[metric_name].append(value)

        self._emit('metric', {
            'name': metric_name,
            'value': value
        })

    def log_message(self, message):
        """记录日志信息"""
        self._emit('log', {'message': message})

    def _emit(self, event, data):
        """通过SocketIO发送数据"""
        if self.socketio:
            self.socketio.emit(event, data)