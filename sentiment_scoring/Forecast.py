# 读取数据
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D, RepeatVector, \
    TimeDistributed, LayerNormalization, Dropout, MultiHeadAttention, Input
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
from pmdarima import auto_arima
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def read_data(path, dim_type, sentiment_type='both',
              sentiment_svc_path=None, sentiment_trans_path=None, use_percentage=1):
    '''
    读取数据(支持选择情感数据类型)
    :param sentiment_type: 情感数据使用类型，可选值：'svc'（仅用SVC）、'trans'（仅用Transformer）、'both'（两者都用）
    :param sentiment_svc_path: sentiment_score_svc.csv的路径
    :param sentiment_trans_path: sentiment_score_trans.csv的路径
    '''
    df = pd.read_csv(path)
    data_len = df.shape[0]
    data = None

    if dim_type != 'Multi':
        data = df[dim_type].values.reshape((data_len, 1))
    else:
        # 1. 处理日期格式（用于匹配情感数据）
        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
        df["datetime"] = pd.to_datetime(df["Date"])
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["date_tuple"] = df.apply(lambda row: (row["year"], row["month"], row["day"]), axis=1)

        # 2. 基础价格特征（Close, Open, High, Low）
        open_data = df["Open"].values.reshape((data_len, 1))
        high_data = df["High"].values.reshape((data_len, 1))
        low_data = df["Low"].values.reshape((data_len, 1))
        close_data = df["Close"].values.reshape((data_len, 1))
        base_features = np.hstack((close_data, open_data, high_data, low_data))

        # 3. 根据sentiment_type选择情感特征
        if sentiment_svc_path is not None and sentiment_trans_path is not None:
            # 读取情感数据
            sentiment_svc = pd.read_csv(sentiment_svc_path).set_index(['year', 'month', 'day'])
            sentiment_svc["sum"] = sentiment_svc.sum(axis=1)
            sentiment_svc = sentiment_svc.div(sentiment_svc["sum"], axis=0).drop(columns=['sum'])
            sentiment_svc = sentiment_svc.rename(
                columns={"positive": "positive_svc", "neutral": "neutral_svc", "negative": "negative_svc"}
            )

            sentiment_trans = pd.read_csv(sentiment_trans_path).set_index(['year', 'month', 'day'])
            sentiment_trans["sum"] = sentiment_trans.sum(axis=1)
            sentiment_trans = sentiment_trans.div(sentiment_trans["sum"], axis=0).drop(columns=['sum'])
            sentiment_trans = sentiment_trans.rename(
                columns={"positive": "positive_trans", "neutral": "neutral_trans", "negative": "negative_trans"}
            )

            # 根据sentiment_type筛选情感列
            if sentiment_type == 'svc':
                sentiment_cols = ['positive_svc', 'neutral_svc', 'negative_svc']  # 仅SVC的3列
            elif sentiment_type == 'trans':
                sentiment_cols = ['positive_trans', 'neutral_trans', 'negative_trans']  # 仅Transformer的3列
            elif sentiment_type == 'both':
                sentiment_cols = ['positive_svc', 'neutral_svc', 'negative_svc',
                                 'positive_trans', 'neutral_trans', 'negative_trans']  # 两者共6列
            else:
                raise ValueError("sentiment_type must be 'svc', 'trans', or 'both'")

            # 匹配情感分数并填充缺失值
            for col in sentiment_cols:
                source = sentiment_svc if 'svc' in col else sentiment_trans
                df[col] = df['date_tuple'].apply(
                    lambda x: source.loc[x, col] if x in source.index else np.nan
                )
                df[col] = df[col].interpolate(limit_direction="both")  # 插值填充

            # 合并特征：基础价格 + 选中的情感特征
            sentiment_data = df[sentiment_cols].values.reshape((data_len, len(sentiment_cols)))
            data = np.hstack((base_features, sentiment_data))
        else:
            data = base_features

    return data[0:int(np.floor(data_len * use_percentage))], np.floor(data_len * use_percentage)


def split_sequence(sequence, dim_type, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of the input pattern
        end_ix = i + n_steps_in
        # find the end of the output pattern
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequence):
            break
        if dim_type == 'Multi':
            # gather input and output parts of the pattern
            seq_x = sequence[i:end_ix, 1:]
            seq_y = sequence[end_ix:out_end_ix, 0]
        else:
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def data_trasform(data, anti=False, scaler=None):
    '''
    说明以及例子
    MinMax data and anti MinMax data
    :param data: the data source
    :param model: MinMax and anti MinMax
    :param scaler: anti MinMax scaler
    :return: the transformed data

    '''
    if not anti:
        # 归一化
        # 创建一个空字典来存储每一列的 scaler
        scalers = {}
        # 归一化数据的容器
        normalized_data = np.zeros_like(data)
        # 循环每一列
        for i in range(data.shape[1]):  # data.shape[1] 是列的数量
            # 为每一列创建一个新的 MinMaxScaler
            scaler = MinMaxScaler()
            # 将列数据调整为正确的形状，即(-1, 1)
            column_data = data[:, i].reshape(-1, 1)
            # 拟合并转换数据
            normalized_column = scaler.fit_transform(column_data)
            # 将归一化的数据存回容器中
            normalized_data[:, i] = normalized_column.ravel()
            # 存储scaler以便后续使用
            scalers[i] = scaler
        # 现在 normalized_data 是完全归一化的数据
        # scalers 字典包含每一列的 MinMaxScaler 实例
        return normalized_data, scalers
    else:
        # 反归一化
        # 如果data是三维数组，去除最后一个维度
        if data.ndim == 3 and data.shape[2] == 1:
            data = data.squeeze(axis=2)

        restored_data = np.zeros_like(data)
        for i in range(data.shape[1]):  # 遍历所有列
            column_data = data[:, i].reshape(-1, 1)
            restored_data[:, i] = scaler.inverse_transform(column_data).ravel()
        return restored_data


def create_transformer_model(input_seq_length, output_seq_length, num_features, d_model, num_heads, ff_dim,
                             num_transformer_blocks, dropout_rate=0.1):
    inputs = Input(shape=(input_seq_length, num_features))

    x = Dense(d_model)(inputs)

    for _ in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)  # Dropout after attention
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(dropout_rate)(ff_output)  # Dropout after first dense layer
        ff_output = Dense(d_model)(ff_output)

        x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    outputs = Dense(output_seq_length)(x[:, -1, :])  # We take the last step's output for forecasting
    model = Model(inputs, outputs)
    return model


def create_model(model_type, n_features, n_steps_in, n_steps_out):
    '''
        create model
        :param model_type:  LSTM,BD LSTM(bidirectional LSTM),ED LSTM(Encoder-Decoder LSTM),CNN
        :param n_features:
        :param n_steps_in:
        :param n_steps_out:
        :return: the created model
    '''
    model = Sequential()
    adam_optimizer = Adam(learning_rate=0.001)
    if model_type == 'LSTM':
        # LSTM
        model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(Dense(n_steps_out))

    elif model_type == 'BD LSTM':
        # bidirectional LSTM
        model.add(Bidirectional(LSTM(50, activation='sigmoid'), input_shape=(n_steps_in, n_features)))
        model.add(Dense(n_steps_out))

    elif model_type == 'ED LSTM':
        # Encoder-Decoder LSTM
        # Encoder
        model.add(LSTM(100, activation='sigmoid', input_shape=(n_steps_in, n_features)))
        # Connector
        model.add(RepeatVector(n_steps_out))
        # Decoder
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))

    elif model_type == 'CNN':
        # CNN
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))

    elif model_type == 'Convolutional LSTM':
        # Convolutional LSTM
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(LSTM(20, activation='relu', return_sequences=False))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))

    elif model_type == 'Transformer':
        model = create_transformer_model(n_steps_in, n_steps_out, n_features, d_model=64,
                                         num_heads=12, ff_dim=64, num_transformer_blocks=3)

    elif model_type == 'MLP':
        # 多层感知机 (MLP)
        model.add(Dense(20, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))

    elif model_type == 'ARIMA':
        if n_features != 1:
            print("ARIMA model only supports univariate time series data")
            print("ARIMA model has no parameter n_steps_in")
            return None
        model = 'ARIMA'
        return model

    else:
        print("no model")
    model.compile(optimizer=adam_optimizer, loss='mse')
    return model


def train_and_forecast(model, n_features, dim_type, data_X, data_Y, n_steps_in, n_steps_out, ech):
    # 训练模型
    # 隐藏输出
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    X, y = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)
    # 对于多维数据，调整最后一个维度为特征数
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    #######################################################################################
    if model == 'ARIMA':
        # 检查数据是否平稳
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(data_X.squeeze(), lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(data_X, lags=40, ax=ax2)

        # # 自动确定 ARIMA 模型的参数
        # auto_model = auto_arima(data_X, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        # # 输出最佳 ARIMA 模型的参数
        # print(auto_model.summary())

        # 使用最佳参数拟合 ARIMA 模型
        # ARIMA 模型只接受单变量时间序列，这里假设 data_X 和 data_Y 是一维数组
        # order = auto_model.order
        order = (6, 1, 6)
        arma_model = ARIMA(data_X, order=order)
        model_fit = arma_model.fit()
        # 拟合结果
        fit_result = model_fit.fittedvalues
        # 使用模型进行滚动预测
        history = list(data_X)
        test_result = []
        for t in range(len(data_Y)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            output = model_fit.forecast(n_steps_out)
            test_result.append(output)
            history.append(data_Y[t])  # 更新历史数据
        test_result = np.array(test_result)
        fit_result = fit_result.reshape(len(fit_result), 1)
        return fit_result, test_result
    #######################################################################################

    # 训练模型
    model.fit(X, y, epochs=ech, batch_size=32, verbose=1)

    # 拟合结果
    fit_result = []
    for index, ele in enumerate(X):
        print(f'Fitting {index}th data')
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        fit_result.append(pred)
    fr = np.array(fit_result)
    fit_result = fr.reshape(len(fit_result), n_steps_out)
    # 测试结果
    test_x, test_y = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    test_result = []
    for index, ele in enumerate(test_x):
        print(f'Predicting {index}th data')
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        test_result.append(pred)
    tr = np.array(test_result)
    test_result = tr.reshape(len(test_result), n_steps_out)

    # 恢复输出
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return fit_result, test_result


def eval_result(result, n_steps_out, target, mode):
    '''
    evaluate the modl resule
    :param result:the model result
    :param n_steps_out:the days you predict
    :param target:the ground-true
    :param mode:the type of evaluation(you can choose 0：rmse,1：mape)
    :return:the evaluation result
    '''
    if mode == 0:
        # return rmse result
        # 归一化
        result, _ = data_trasform(result)
        target, _ = data_trasform(target)
        # 下面需要修改
        rmse = []
        for i in range(n_steps_out):
            rmse.append(np.sqrt(np.mean((result[:, i] - target[:, i]) ** 2)))
        return rmse

    elif mode == 1:
        # return MAPE result
        result = result + 0.0000001
        target = target + 0.0000001
        mape = []
        for i in range(n_steps_out):
            mape.append(np.mean(np.abs((target[:, i] - result[:, i]) / target[:, i])) * 100)
        return mape
    else:
        return None


def main():
    print("start")
    # -----------------参数设置-----------------
    model_hub = ['LSTM', 'BD LSTM', 'ED LSTM', 'CNN', 'Convolutional LSTM', 'Transformer', 'MLP', 'ARIMA']
    file_path = r"./Financial sentiment analysis data/coin_Dogecoin.csv"
    dim_type = 'Close' # 'Multi' or 'Open', 'High', 'Low', 'Close', 'Marketcap'
    use_percentage = 1

    n_steps_in = 6
    n_steps_out = 5

    percentage = 0.7
    epochs = 100
    rounds = 30
    random_seed = 42
    split_type = 0

    # ---------------情感数据设置---------------
    sentiment_svc_path = r"./sentiment_score_svc.csv"
    sentiment_trans_path = r"./sentiment_score_trans.csv"
    # 选择情感数据类型：'svc'（仅用SVC）、'trans'（仅用Transformer）、'both'（两者都用）
    sentiment_type = 'both'  # 可在此切换类型

    # ---------------读取数据---------------
    data, data_len = read_data(
        path=file_path,
        dim_type=dim_type,
        sentiment_type=sentiment_type,  # 传入选择的情感类型
        sentiment_svc_path=sentiment_svc_path,
        sentiment_trans_path=sentiment_trans_path,
        use_percentage=use_percentage
    )
    data, scalers = data_trasform(data)

    # ---------------划分训练集和测试集---------------
    if split_type == 0:
        train_set = data[0:int(np.floor(data_len * percentage))]
        test_set = data[int(np.floor(data_len * percentage)):]
    elif split_type == 1:
        np.random.seed(random_seed)
        rand_np = np.random.randint(data_len - int(np.floor(data_len * percentage)))
        train_set = np.concatenate([data[0:rand_np], data[rand_np + int(np.floor(data_len * percentage)):]])
        test_set = data[rand_np:rand_np + int(np.floor(data_len * percentage))]
    elif split_type == 2:
        train_set = data[int(np.floor(data_len * (1 - percentage))):]
        test_set = data[0:int(np.floor(data_len * (1 - percentage)))]
    elif split_type == 3:
        np.random.seed(random_seed)
        indices = np.arange(data_len, dtype=np.int64)
        np.random.shuffle(indices)
        train_size = int(np.floor(data_len * percentage))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        train_set = data[train_indices]
        test_set = data[test_indices]

    # ---------------特征数设置---------------
    n_features = data.shape[1] - 1 if data.shape[1] > 1 else 1  # 多维度时特征数=总列数-1（因Close为目标变量）

    # ---------------模型训练与评估---------------
    model_type = 'BD LSTM'  # 选择模型
    exp_result = pd.DataFrame({}, columns=['Train MINMAX RMSE', 'Test MINMAX RMSE', 'Train MAPE', 'Test MAPE'])

    for round in range(rounds):
        print(f"the {round}-th exp, total:{rounds} rounds")
        Model = create_model(model_type, n_features, n_steps_in, n_steps_out)

        train_result, test_result = train_and_forecast(
            Model, n_features, dim_type, train_set, test_set, n_steps_in, n_steps_out, epochs
        )

        # 反归一化与评估
        train_result = data_trasform(train_result, True, scalers[0])
        test_result = data_trasform(test_result, True, scalers[0])

        _, train = split_sequence(train_set, dim_type, n_steps_in, n_steps_out)
        train = data_trasform(train, True, scalers[0])
        _, test = split_sequence(test_set, dim_type, n_steps_in, n_steps_out)
        test = data_trasform(test, True, scalers[0])

        # 计算RMSE和MAPE
        per_n_steps_out = n_steps_out
        if Model == 'ARIMA':
            train = train_set
            train = data_trasform(train, True, scalers[0])
            test_result = test_result[0:len(test), :]
            n_steps_out = 1

        train_minmax_rmse = eval_result(train_result, n_steps_out, train, 0)
        if Model == 'ARIMA':
            n_steps_out = per_n_steps_out
        test_minmax_rmse = eval_result(test_result, n_steps_out, test, 0)

        print('Train MINMAX RMSE:', train_minmax_rmse)
        print('Test MINMAX RMSE:', test_minmax_rmse)

        if Model == 'ARIMA':
            n_steps_out = 1
        train_MAPE = eval_result(train_result, n_steps_out, train, 1)
        if Model == 'ARIMA':
            n_steps_out = per_n_steps_out
        test_MAPE = eval_result(test_result, n_steps_out, test, 1)

        print('Train MAPE:', train_MAPE)
        print('Test MAPE:', test_MAPE)

        # 保存结果
        exp_result.loc[len(exp_result.index)] = [
            train_minmax_rmse, test_minmax_rmse, train_MAPE, test_MAPE
        ]
        exp_result.to_excel("./results/results.xlsx")  # 结果文件名区分

    # 绘图
    if n_steps_out == 1:
        # 拟合结果
        plt.figure(figsize=(15, 6))
        plt.plot(train_result, label='Train')
        plt.plot(train, label='Actual')
        plt.title(f'{model_type} Train Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

        # 测试结果
        plt.figure(figsize=(15, 6))
        plt.plot(test_result, label='Predicted')
        plt.plot(test, label='Actual')
        plt.title(f'{model_type} Test Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()
if __name__ == '__main__':
    main()
