import json
import os
import sys
import time
import pandas as pd
import numpy as np
from flask import request, jsonify
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D, RepeatVector, \
    TimeDistributed, LayerNormalization, Dropout, MultiHeadAttention, Input, Concatenate, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
from scipy import signal
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import random
import datetime


from train_monitor import TrainingMonitor
from tensorflow.keras.layers import Lambda
from collections import Counter

# 忽略警告
warnings.filterwarnings('ignore')

# 宏观经济数据本地路径
MACRO_DATA_PATH = 'data/'
MACRO_FILES = {
    'fed_rate': 'Federal Funds Rate.csv',
    'treasury_yield': 'us_treasury_yields_daily.csv',
    'dxy': 'DXY.csv',
    'cpi': 'CPIAUCSL.csv',
    'vix': 'VIX_History.csv',
    'sp500': 'sap500.csv'
}


# ==================== 宏观经济数据获取与处理 ====================

def fetch_macro_data(start_date='2013-01-01', end_date='2021-07-07'):
    """
    从本地CSV文件获取宏观经济数据（时区统一为UTC）
    """
    print("Loading macro data from local files...")
    macro_df = pd.DataFrame()

    try:
        # 创建日期范围索引
        dates = pd.date_range(start=start_date, end=end_date, tz='UTC')
        macro_df = pd.DataFrame(index=dates)

        # 1. 联邦基金利率
        fed_rate = pd.read_csv(os.path.join(MACRO_DATA_PATH, MACRO_FILES['fed_rate']),
                               parse_dates=['Date'], index_col='Date')
        fed_rate = fed_rate.tz_localize('UTC').reindex(dates).ffill().bfill()
        macro_df['fed_rate'] = fed_rate['RIFSPFF_N.D']

        # 2. 国债收益率 (10年期)
        treasury = pd.read_csv(os.path.join(MACRO_DATA_PATH, MACRO_FILES['treasury_yield']),
                               parse_dates=['date'], index_col='date')
        treasury = treasury.tz_localize('UTC').reindex(dates).ffill().bfill()
        macro_df['treasury_yield'] = treasury['US10Y']

        # 3. 美元指数
        dxy = pd.read_csv(os.path.join(MACRO_DATA_PATH, MACRO_FILES['dxy']),
                          parse_dates=['observation_date'], index_col='observation_date')
        dxy = dxy.tz_localize('UTC').reindex(dates).ffill().bfill()
        macro_df['dxy'] = dxy['DTWEXBGS']

        # 4. CPI通胀指标
        cpi = pd.read_csv(os.path.join(MACRO_DATA_PATH, MACRO_FILES['cpi']),
                          parse_dates=['DATE'], index_col='DATE')
        cpi = cpi.tz_localize('UTC')
        cpi_daily = state_space_interpolation(cpi['CPIAUCSL'])
        macro_df['cpi'] = cpi_daily.reindex(dates).ffill().bfill()

        # 5. VIX恐慌指数
        vix = pd.read_csv(os.path.join(MACRO_DATA_PATH, MACRO_FILES['vix']),
                          parse_dates=['DATE'], index_col='DATE')
        vix = vix.tz_localize('UTC').reindex(dates).ffill().bfill()
        macro_df['vix'] = vix['CLOSE']

        # 6. 标普500指数
        sp500 = pd.read_csv(os.path.join(MACRO_DATA_PATH, MACRO_FILES['sp500']),
                            parse_dates=['Date'], index_col='Date')
        sp500 = sp500.tz_localize('UTC').reindex(dates).ffill().bfill()
        macro_df['sp500'] = sp500['Close']

        # 添加单位转换和频率调整
        try:
            # 转换利率和收益率为基点
            macro_df['fed_rate'] = macro_df['fed_rate'] * 100
            macro_df['treasury_yield'] = macro_df['treasury_yield'] * 100

            # CPI月度数据处理
            if 'cpi' in macro_df:
                # 前向填充直到新数据点
                macro_df['cpi'] = macro_df['cpi'].ffill()

            # 应用噪声抑制
            for col in macro_df.columns:
                if col == 'cpi':
                    macro_df[col] = apply_frequency_filter(macro_df[col])
                else:
                    macro_df[col] = volatility_adaptive_smoothing(macro_df[col], macro_df['vix'])

        except Exception as e:
            print(f"宏观经济数据处理失败: {e}")

        print("Macro data loaded successfully.")
        return macro_df

    except Exception as e:
        print(f"Failed to load macro data: {e}")
        # 生成模拟数据作为后备
        print("Using sample macro data as fallback.")
        dates = pd.date_range(start=start_date, end=end_date, tz='UTC')
        macro_df = pd.DataFrame(index=dates)
        macro_df['cpi'] = np.random.normal(250, 5, len(dates))
        macro_df['fed_rate'] = np.random.uniform(0.1, 2.5, len(dates))
        macro_df['treasury_yield'] = np.random.uniform(1.5, 3.5, len(dates))
        macro_df['vix'] = np.random.uniform(10, 30, len(dates))
        macro_df['dxy'] = np.random.uniform(90, 100, len(dates))
        macro_df['sp500'] = np.random.uniform(2500, 3500, len(dates))
        return macro_df


def state_space_interpolation(monthly_data):
    """
    使用状态空间模型进行智能插值（月频到日频）[6](@ref)
    """
    try:
        # 创建月度时间序列
        monthly_series = pd.Series(
            monthly_data.values,
            index=pd.date_range(start=monthly_data.index[0], periods=len(monthly_data), freq='MS', tz='UTC')
        )

        # 使用更稳健的季节性分解进行插值
        decomposition = sm.tsa.seasonal_decompose(monthly_series, model='additive', period=12)
        trend = decomposition.trend
        seasonal = decomposition.seasonal

        # 创建日频索引
        daily_index = pd.date_range(start=monthly_series.index[0],
                                    end=monthly_series.index[-1] + pd.DateOffset(months=1),
                                    freq='D', tz='UTC')

        # 插值趋势分量
        trend_daily = trend.interpolate(method='time').reindex(daily_index)

        # 创建季节性分量
        seasonal_daily = seasonal.resample('D').ffill().reindex(daily_index)

        # 重建日频序列
        interpolated = trend_daily + seasonal_daily
        return interpolated
    except Exception as e:
        print(f"State space interpolation failed: {e}")
        # 回退到线性插值
        return monthly_data.resample('D').interpolate(method='linear')


def apply_frequency_filter(series, cutoff_freq=0.1):
    """
    应用Butterworth低通滤波器[6,8](@ref)
    保留频率<0.1Hz的经济趋势信号
    """
    try:
        # 设计滤波器
        b, a = signal.butter(3, cutoff_freq, 'low')

        # 前向-后向滤波消除相位偏移
        filtered = signal.filtfilt(b, a, series.values)

        return pd.Series(filtered, index=series.index)
    except Exception as e:
        print(f"Frequency filtering failed: {e}")
        return series


def volatility_adaptive_smoothing(series, volatility_series, window=30):
    """
    基于市场波动率动态调整平滑强度[9](@ref)
    """
    try:
        # 计算波动率百分位
        vol_percentile = volatility_series.rolling(window, min_periods=1).rank(pct=True)

        # 动态平滑系数 (波动越大，平滑越强)
        alpha = 0.1 + 0.4 * vol_percentile

        # 应用指数平滑
        smoothed = series.copy()
        for i in range(1, len(series)):
            smoothed.iloc[i] = alpha.iloc[i] * series.iloc[i] + (1 - alpha.iloc[i]) * smoothed.iloc[i - 1]

        return smoothed
    except Exception as e:
        print(f"Volatility adaptive smoothing failed: {e}")
        return series


# ==================== 特征工程增强 ====================

def enhanced_feature_engineering(df, window_sizes=[7, 30, 90], add_noise=True):
    """
    改进的特征工程 - 添加关键衍生指标 + 高斯噪声抑制过拟合
    """
    # 数据验证
    if df.empty or 'Close' not in df.columns:
        print("错误：无效输入数据")
        return df

    # 创建结果的副本，避免修改原始数据
    results = pd.DataFrame(index=df.index)
    target = df['Close']

    # 1. 稳健的市场状态计算 (避免未来数据)
    try:
        # 使用滞后数据计算百分比变化
        pct_change = target.shift(30).pct_change(30).fillna(0)  # 30天前开始计算
        volatility = target.pct_change().rolling(30).std().fillna(0)

        # 基于价格变化和波动率定义市场状态
        market_status = np.where(
            (pct_change > 0.15) & (volatility < volatility.quantile(0.6)),
            2,  # 牛市
            np.where(
                (pct_change < -0.15) & (volatility > volatility.quantile(0.4)),
                0,  # 熊市
                1  # 中性
            )
        )
        results['market_status'] = market_status
    except Exception as e:
        print(f"市场状态计算失败: {e}")
        results['market_status'] = 1  # 默认中性

    # 2. 滚动相关性特征 - 使用滞后数据
    # macro_cols = ['vix', 'sp500', 'treasury_yield', 'fed_rate', 'dxy', 'cpi']
    # 修改权重计算中的特征列表
    macro_cols = ['vix', 'sp500', 'fed_rate', 'dxy', 'cpi', 'yield_curve', 'real_sp500']

    for col in macro_cols:
        if col not in df.columns:
            continue

        for window in window_sizes:
            try:
                # 使用滞后避免数据泄露
                shifted_target = target.shift(1)  # 滞后一期
                rolling_corr = shifted_target.rolling(
                    window=window,
                    min_periods=max(5, window // 2)
                ).corr(df[col]).fillna(0).clip(-1, 1)
                results[f'{col}_corr_{window}d'] = rolling_corr
            except:
                results[f'{col}_corr_{window}d'] = 0

    # 3. 关键衍生指标 - 使用滞后数据
    # 添加收益率曲线
    if 'treasury_yield' in df.columns and 'fed_rate' in df.columns:
        try:
            results['yield_curve'] = (df['treasury_yield'] - df['fed_rate']).fillna(0)
        except:
            results['yield_curve'] = 0

    # 添加实际收益率
    if 'treasury_yield' in df.columns and 'cpi' in df.columns:
        try:
            # 使用滞后值计算
            cpi_pct = df['cpi'].pct_change(12).fillna(0).clip(-0.1, 0.2)
            real_yield = (df['treasury_yield'].shift(1) - cpi_pct).clip(-5, 10)
            results['real_yield'] = real_yield.rolling(30, min_periods=1).mean()
        except:
            results['real_yield'] = 0

    # 添加实际标普500指数
    if 'sp500' in df.columns and 'cpi' in df.columns:
        try:
            results['real_sp500'] = (df['sp500'] / df['cpi']).fillna(0)
        except:
            results['real_sp500'] = 0

    # 4. 添加加密货币类型特定特征
    if 'Symbol' in df.columns:
        crypto_type = df['Symbol'].iloc[0]

        # 比特币特定特征
        if crypto_type == 'BTC':
            # 添加比特币主导指标
            if 'Marketcap' in df.columns:
                try:
                    total_cap = df.groupby('Date')['Marketcap'].transform('sum')
                    results['btc_dominance'] = (df['Marketcap'] / total_cap).fillna(0.5)
                except:
                    results['btc_dominance'] = 0.5

        # 以太坊特定特征
        elif crypto_type == 'ETH':
            # 添加DeFi相关波动特征
            if 'vix' in df.columns and 'treasury_yield' in df.columns:
                try:
                    results['defi_risk'] = (df['vix'] * (1 + df['treasury_yield'] / 100)).fillna(0)
                except:
                    results['defi_risk'] = 0

    # 5. 避免高度相关特征共存
    if 'fed_rate' in df and 'treasury_yield' in df:
        # 保留美联储利率作为主要指标
        df = df.drop(columns=['treasury_yield'])

    # 最终处理 - 保留所有特征
    results = results.fillna(0).replace([np.inf, -np.inf], 0)
    enhanced_df = pd.concat([df, results], axis=1)

    # 6. 添加高斯噪声抑制过拟合 - 关键改进
    if add_noise:
        try:
            print("添加高斯噪声抑制过拟合...")
            # 定义噪声参数
            NOISE_INTENSITY = 0.005  # 噪声强度（0.5%）

            # 获取所有数值型列
            numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns

            # 排除目标变量（Close）和市场状态
            exclude_cols = ['Close', 'market_status']
            noise_cols = [col for col in numeric_cols if col not in exclude_cols]

            # 对每个特征添加高斯噪声
            for col in noise_cols:
                if col in enhanced_df.columns:
                    # 计算特征的标准差
                    col_std = enhanced_df[col].std()

                    # 跳过标准差为0的常量特征
                    if col_std < 1e-7:
                        continue

                    # 生成高斯噪声（均值为0，标准差为特征标准差的0.5%）
                    noise = np.random.normal(0, col_std * NOISE_INTENSITY, len(enhanced_df))

                    # 应用噪声
                    enhanced_df[col] = enhanced_df[col] + noise

            print(f"已对 {len(noise_cols)} 个特征添加高斯噪声")
        except Exception as e:
            print(f"添加噪声失败: {str(e)}")

    return enhanced_df


def calculate_dynamic_weights(df, crypto_type='Bitcoin', lookback=90):
    """改进的动态权重计算 - 基于特征相关性动态调整"""
    # 数据验证
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()

    weights = pd.DataFrame(index=df.index, dtype=float)
    target = df['Close']

    # 特征相关性分析
    # macro_features = ['vix', 'sp500', 'fed_rate', 'treasury_yield', 'dxy', 'cpi']
    # 修改权重计算中的特征列表
    macro_features = ['vix', 'sp500', 'fed_rate', 'dxy', 'cpi', 'yield_curve', 'real_sp500']
    available_features = [f for f in macro_features if f in df.columns]

    # 为每个特征创建权重列
    for feature in available_features:
        weights[feature] = 0.0

    # 添加衍生指标
    if 'treasury_yield' in df.columns and 'fed_rate' in df.columns:
        df['yield_curve'] = df['treasury_yield'] - df['fed_rate']
        available_features.append('yield_curve')
        weights['yield_curve'] = 0.0

    if 'sp500' in df.columns and 'cpi' in df.columns:
        df['real_sp500'] = df['sp500'] / df['cpi']
        available_features.append('real_sp500')
        weights['real_sp500'] = 0.0

    # 动态计算特征重要性
    for i in range(lookback, len(df)):
        window_data = df.iloc[i - lookback:i]

        # 计算每个特征与目标的相关性
        corrs = {}
        for feature in available_features:
            try:
                # 使用Spearman秩相关，对非线性关系更稳健
                corr = window_data[feature].corr(window_data['Close'], method='spearman')
                corrs[feature] = corr if not np.isnan(corr) else 0
            except:
                corrs[feature] = 0

        # 计算相对重要性权重
        total_weight = sum(np.abs(corr) for corr in corrs.values())
        if total_weight < 1e-5:
            # 默认等权重分配
            weight_val = 1.0 / len(available_features) if available_features else 0
            for feature in available_features:
                weights.loc[df.index[i], feature] = weight_val
        else:
            for feature, corr in corrs.items():
                weight_val = np.abs(corr) / total_weight
                weights.loc[df.index[i], feature] = weight_val

    # 填充和平滑
    weights = weights.fillna(0)

    # 添加类型特定调整
    crypto_boost = {
        'Bitcoin': {'vix': 1.2, 'real_sp500': 1.1},
        'Ethereum': {'vix': 1.3, 'yield_curve': 1.2},
        'Litecoin': {'dxy': 1.2, 'fed_rate': 1.1},
        'Dogecoin': {'vix': 1.4, 'sp500': 0.8}
    }

    boost_factors = crypto_boost.get(crypto_type, {})
    for feature, factor in boost_factors.items():
        if feature in weights.columns:
            weights[feature] = weights[feature] * factor

    # 归一化确保权重和为1
    for i in range(len(weights)):
        row = weights.iloc[i]
        total = row.sum()
        if total > 0:
            weights.iloc[i] = row / total

    # 指数平滑
    for col in weights.columns:
        weights[col] = weights[col].ewm(span=7).mean().fillna(0)

    # 可视化权重分布
    plt.figure(figsize=(12, 6))
    for feature in available_features[:5]:  # 最多显示5个特征
        plt.plot(weights.index, weights[feature], label=feature)
    plt.title(f'{crypto_type} Dynamic Feature Weights')  # 英文标题
    plt.xlabel('Date')  # 英文标签
    plt.ylabel('Weight')  # 英文标签
    plt.legend()
    plt.savefig(f'{crypto_type}_feature_weights.png')
    plt.close()

    return weights


def create_transformer_model(input_seq_length, output_seq_length, num_features,
                             d_model=48, num_heads=6, ff_dim=128,
                             num_transformer_blocks=2, dropout_rate=0.1):
    """优化Transformer架构 - 减少复杂度增强稳定性"""
    inputs = Input(shape=(input_seq_length, num_features))

    # 输入嵌入层
    x = Dense(d_model)(inputs)

    # 简化的Transformer块
    for _ in range(num_transformer_blocks):
        # 注意力机制
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )(x, x)

        # 残差连接+层归一化
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # 前馈网络
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(d_model)(ff_output)

        # 残差连接+层归一化
        x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    # 输出处理
    x = x[:, -1, :]  # 只取最后时间步
    outputs = Dense(64, activation='relu')(x)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(output_seq_length)(outputs)

    return Model(inputs, outputs)


def read_data(path, dim_type, use_percentage=1, include_macro=True):
    '''
    读取数据并整合宏观经济指标 - 修复日期格式问题
    '''
    # 读取加密货币数据
    try:
        df = pd.read_csv(path)
        print(f"原始数据行数: {len(df)}")
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        return None, 0, None

    # 检查日期列
    if 'Date' not in df.columns:
        print("错误: 数据中缺少'Date'列")
        return None, 0, None

    try:
        # 关键修复：移除时间部分，保留日期
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
        df["Date"] = pd.to_datetime(df["Date"], utc=True)  # 重新转换为Timestamp类型
    except Exception as e:
        print(f"日期解析失败: {str(e)}")
        return None, 0, None

    # 确保数据按日期排序
    df = df.sort_values('Date')

    # 提取日期范围
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')
    print(f"数据日期范围: {start_date} 到 {end_date}")

    # 获取宏观经济数据
    if include_macro:
        try:
            macro_df = fetch_macro_data(start_date, end_date)
            macro_df = macro_df.reset_index().rename(columns={'index': 'Date'})

            # 合并宏观经济数据
            df = df.merge(macro_df, on='Date', how='left')
            print(f"合并宏观经济数据后行数: {len(df)}")

            # 新增：检查宏观列是否存在空值
            macro_cols = ['vix', 'sp500', 'fed_rate', 'treasury_yield', 'dxy', 'cpi']
            for col in macro_cols:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    print(f"{col}空值数量: {null_count}/{len(df)}")
                else:
                    print(f"警告: 列 {col} 不存在")

            # 应用增强特征工程
            try:
                df = enhanced_feature_engineering(df)
                print(f"特征工程后行数: {len(df)}")
            except Exception as e:
                print(f"特征工程失败: {str(e)}")
                return None, 0, None
        except Exception as e:
            print(f"宏观经济数据合并失败: {str(e)}")
            include_macro = False

    # 填充缺失值（向前填充）
    df = df.ffill().bfill()

    # 检查目标变量中的零值或接近零值
    close_values = df['Close'].values
    zero_close_count = np.sum(close_values < 1e-5)
    if zero_close_count > 0:
        print(f"警告: 发现 {zero_close_count} 个接近零的收盘价")
        # 移除这些行
        df = df[df['Close'] >= 1e-5]
        print(f"移除接近极值后行数: {len(df)}")

    # 移除全零列（修改版：只处理数值列）
    cols_to_drop = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].values
        # 检查列是否全为零（使用容差）
        if np.all(np.abs(col_data) < 1e-7):
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"移除全零列: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 根据dim_type选择数据
    data_len = df.shape[0]
    if data_len == 0:
        print("错误: 处理后数据为空")
        return None, 0, None

    data = None
    feature_names = []  # 创建空的特征名称列表

    if dim_type != 'Multi':
        # 单维数据
        data = df[dim_type].values.reshape((data_len, 1))
        feature_names = [dim_type]  # 设置特征名称
    else:
        # 多维数据 - 包括所有特征
        feature_cols = [col for col in df.columns if col not in ['SNo', 'Name', 'Symbol', 'Date']]
        data = df[feature_cols].values
        feature_names = feature_cols  # 保存特征名称

    # 截断数据到指定百分比
    truncated_len = int(np.floor(data_len * use_percentage))
    data = data[0:truncated_len]

    # 打印关键列统计信息
    if 'Close' in feature_names:
        close_idx = feature_names.index('Close')
        close_data = data[:, close_idx]
        print(
            f"Close列统计: min={np.min(close_data):.2f}, max={np.max(close_data):.2f}, mean={np.mean(close_data):.2f}")

    return data, truncated_len, feature_names  # 返回数据和特征名称


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def data_trasform(data, anti=False, scalers=None, target_index=0, feature_names=None):
    '''
    分层归一化和反归一化 - 增强版：针对不同特征类型优化归一化策略
    '''
    # 特征类型定义
    PRICE_FEATURES = ['High', 'Low', 'Open', 'Close', 'sp500']
    VOLUME_FEATURES = ['Volume', 'Marketcap']
    MACRO_RATE = ['fed_rate', 'treasury_yield', 'real_rate']
    MACRO_LEVEL = ['dxy', 'cpi']
    VOLATILITY_FEATURES = ['vix']

    # 如果没有提供特征名称，创建默认列表
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        print(f"警告: 未提供特征名称，使用默认名称: {feature_names}")

    def safe_normalize(col_data, feature_name):
        """安全处理包含NaN和inf的归一化"""
        # 替换NaN和inf
        col_data = np.nan_to_num(col_data, nan=0.0, posinf=np.nanmax(col_data), neginf=np.nanmin(col_data))

        # 计算统计信息
        data_min = np.min(col_data)
        data_max = np.max(col_data)
        data_range = data_max - data_min

        # 处理全零列
        if np.allclose(col_data, 0, atol=1e-7):
            print(f"特征 '{feature_name}' 为全零，跳过归一化")
            return col_data, None

        # 处理常数列
        if data_range < 1e-7:
            print(f"特征 '{feature_name}' 为常数值 ({data_min:.6f})，跳过归一化")
            return col_data, None

        try:
            # 根据特征类型选择归一化策略
            if feature_name in PRICE_FEATURES:
                # 价格特征：对数归一化应对指数变化
                signed_data = np.sign(col_data) * np.log1p(np.abs(col_data))
                scaler = MinMaxScaler(feature_range=(-1, 1))
                normalized = scaler.fit_transform(signed_data.reshape(-1, 1))
            elif feature_name in VOLUME_FEATURES:
                # 交易量特征：RobustScaler应对离群值
                scaler = RobustScaler(quantile_range=(5, 95))
                normalized = scaler.fit_transform(col_data.reshape(-1, 1))
            elif feature_name in VOLATILITY_FEATURES:
                # 波动率特征：标准归一化
                scaler = StandardScaler()
                normalized = scaler.fit_transform(col_data.reshape(-1, 1))
            elif feature_name in MACRO_RATE:
                # 利率特征：转换为基点后归一化
                col_data = col_data * 100
                scaler = MinMaxScaler(feature_range=(-1, 1))
                normalized = scaler.fit_transform(col_data.reshape(-1, 1))
            elif feature_name in MACRO_LEVEL:
                # 宏观水平特征：标准归一化
                scaler = StandardScaler()
                normalized = scaler.fit_transform(col_data.reshape(-1, 1))
            else:
                # 默认处理
                scaler = MinMaxScaler(feature_range=(-1, 1))
                normalized = scaler.fit_transform(col_data.reshape(-1, 1))

            return normalized, scaler
        except Exception as e:
            print(f"特征 '{feature_name}' 归一化失败: {str(e)}")
            return col_data, None

    if not anti:
        normalized_data = np.zeros_like(data, dtype=np.float32)
        scalers_dict = {}

        print(f"开始增强归一化 {len(feature_names)} 个特征...")
        for i, col_name in enumerate(feature_names):
            col_data = data[:, i]
            normalized_col, scaler = safe_normalize(col_data, col_name)
            normalized_data[:, i] = normalized_col.ravel()
            scalers_dict[i] = scaler

        return normalized_data, scalers_dict

    else:
        # 反归一化 - 针对特定目标变量
        if scalers is None or target_index not in scalers or scalers[target_index] is None:
            print(f"警告: 缺少目标变量（索引{target_index}）的归一化器，使用原始数据")
            return data, None

        target_scaler = scalers[target_index]
        try:
            # 确保输入形状正确
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            restored_data = target_scaler.inverse_transform(data)

            # 特殊处理：如果是价格特征，需要指数变换
            if feature_names and target_index < len(feature_names):
                feature_name = feature_names[target_index]
                if feature_name in PRICE_FEATURES:
                    restored_data = np.sign(restored_data) * (np.exp(np.abs(restored_data)) - 1)

            return restored_data, scalers
        except Exception as e:
            print(f"反归一化失败: {str(e)}，使用原始数据")
            return data, scalers

# ==================== 训练与评估函数 ====================
# ==================== 完整的 train_and_forecast 函数 ====================
def train_and_forecast(model, n_features, data_X, data_Y, n_steps_in, n_steps_out, ech,
                       dynamic_weights=None, n_macro_features=0, scalers=None, target_index=0, feature_names=None,
                       round_seed=None, monitor=None):
    """
    训练模型并进行预测 - 完整修复版
    """

    # 日志记录辅助函数
    def log_message(message):
        print(message)
        if monitor:
            monitor.log_message(message)

    # 设置随机种子确保可复现性
    if round_seed is not None:
        log_message(f"设置本轮随机种子: {round_seed}")
        tf.random.set_seed(round_seed)
        np.random.seed(round_seed)
        random.seed(round_seed)
    else:
        log_message("警告: 未提供本轮随机种子，使用默认随机状态")

    log_message("Splitting sequences...")
    try:
        # 使用带目标索引的分割函数
        X, y = split_sequence(data_X, n_steps_in, n_steps_out, target_index)
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        test_x, test_y = split_sequence(data_Y, n_steps_in, n_steps_out, target_index)
        test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))

        log_message(f"训练序列数: {X.shape[0]}, 测试序列数: {test_x.shape[0]}")
        log_message(f"输入数据形状: {X.shape}, 目标数据形状: {y.shape}")
    except Exception as e:
        log_message(f"序列分割失败: {str(e)}")
        return None, None, None, None

    # 检查数据有效性 - 关键修复
    def sanitize_data(data, name):
        """处理NaN和无穷值"""
        original_shape = data.shape
        nan_count_before = np.isnan(data).sum()
        inf_count_before = np.isinf(data).sum()

        # 替换NaN和无穷值
        data = np.nan_to_num(data, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

        nan_count_after = np.isnan(data).sum()
        inf_count_after = np.isinf(data).sum()

        log_message(
            f"清理 {name}: 原始形状={original_shape}, NaN: {nan_count_before}->{nan_count_after}, Inf: {inf_count_before}->{inf_count_after}")

        return data.astype(np.float32)

    X = sanitize_data(X, "X")
    y = sanitize_data(y, "y")
    test_x = sanitize_data(test_x, "test_x")
    test_y = sanitize_data(test_y, "test_y")

    # 打印数据统计信息
    log_message("输入数据统计:")
    log_message(
        f"  X - 形状: {X.shape}, 均值: {np.mean(X):.6f}, 标准差: {np.std(X):.6f}, 范围: {np.min(X):.6f}-{np.max(X):.6f}")
    log_message(
        f"  y - 形状: {y.shape}, 均值: {np.mean(y):.6f}, 标准差: {np.std(y):.6f}, 范围: {np.min(y):.6f}-{np.max(y):.6f}")

    # 检查模型是否是融合模型
    is_fusion_model = isinstance(model.input, list) and len(model.input) > 1

    # 准备输入数据
    inputs = X
    test_inputs = test_x
    X_macro = None
    test_X_macro = None

    # 处理宏观经济特征输入
    if n_macro_features > 0:
        if is_fusion_model:
            log_message("准备宏观经济输入 (融合模型)...")
            try:
                # 从序列中提取宏观经济特征（每个样本的最后一个时间步）
                X_macro = X[:, -1, -n_macro_features:]
                test_X_macro = test_x[:, -1, -n_macro_features:]

                # 创建输入元组
                inputs = [X, X_macro]
                test_inputs = [test_x, test_X_macro]
            except Exception as e:
                log_message(f"宏观经济输入准备失败: {str(e)}")
                # 回退到单输入
                inputs = X
                test_inputs = test_x
        else:
            log_message("注意: 宏观经济特征已包含在常规输入中 (非融合模型)")
            # 对于非融合模型，所有特征已经包含在X中
            inputs = X
            test_inputs = test_x

    # ==================== 添加监控回调 ====================
    class MonitorCallback(tf.keras.callbacks.Callback):
        def __init__(self, monitor, total_epochs):
            self.monitor = monitor
            self.total_epochs = total_epochs
            self.should_pause = False
            self.should_stop = False

        def on_epoch_begin(self, epoch, logs=None):
            # 检查暂停状态
            while self.monitor.training_paused and not self.monitor.training_stopped:
                time.sleep(1)  # 等待1秒再检查
                print("训练暂停中...")
                if self.monitor:
                    self.monitor.log_message("训练暂停中...")

            # 检查停止状态
            if self.monitor.training_stopped:
                self.model.stop_training = True
                return

        def on_epoch_end(self, epoch, logs=None):
            if self.monitor:
                # 更新当前轮次
                self.monitor.data['current_epoch'] = epoch + 1
                # 计算进度百分比
                progress = int(((epoch + 1) / self.total_epochs) * 100)
                self.monitor.data['progress'] = progress

                # 记录损失指标
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)

                # 添加指标到监控器
                self.monitor.add_metric('loss', loss)
                self.monitor.add_metric('val_loss', val_loss)

                # 发送更新到前端
                self.monitor._emit('update', {
                    'current_epoch': epoch + 1,
                    'loss': loss,
                    'val_loss': val_loss,
                    'progress': progress
                })

    # 添加早停和模型检查点
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )

    # 添加监控回调
    callbacks = [early_stopping, checkpoint]
    if monitor:
        monitor_callback = MonitorCallback(monitor, ech)
        callbacks.append(monitor_callback)
        # 发送模型信息到前端
        monitor.set_model_info(monitor.data.get('crypto', 'Unknown'),
                               monitor.data.get('model', 'Unknown'),
                               ech)

    # 训练模型
    log_message(f"使用 {X.shape[0]} 个样本训练模型...")

    # 添加输入数据验证
    log_message("输入数据验证:")
    if isinstance(inputs, list):
        for i, inp in enumerate(inputs):
            log_message(
                f"  输入 {i}: 形状={inp.shape}, 均值={np.mean(inp):.6f}, 范围={np.min(inp):.6f}-{np.max(inp):.6f}")
    else:
        log_message(
            f"  输入: 形状={inputs.shape}, 均值={np.mean(inputs):.6f}, 范围={np.min(inputs):.6f}-{np.max(inputs):.6f}")
    log_message(f"  目标: 形状={y.shape}, 均值={np.mean(y):.6f}, 范围={np.min(y):.6f}-{np.max(y):.6f}")

    try:
        history = None
        if dynamic_weights is not None and len(dynamic_weights) >= len(y):
            log_message("使用动态权重")
            # 确保权重长度匹配
            sample_weights = dynamic_weights[:len(y)]
            history = model.fit(
                inputs, y,
                epochs=ech,
                batch_size=16,
                validation_split=0.2,
                callbacks=callbacks,  # 使用添加了监控的回调
                verbose=0,  # 关闭控制台输出，由监控回调处理
                sample_weight=sample_weights
            )
        else:
            history = model.fit(
                inputs, y,
                epochs=ech,
                batch_size=16,
                validation_split=0.2,
                callbacks=callbacks,  # 使用添加了监控的回调
                verbose=0  # 关闭控制台输出，由监控回调处理
            )

        # 加载最佳权重
        model.load_weights('best_model.h5')
        if history:
            log_message(f"最终训练损失: {history.history['loss'][-1]:.6f}")
            if 'val_loss' in history.history:
                log_message(f"最佳验证损失: {min(history.history['val_loss']):.6f}")

                # 绘制损失曲线
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='train_loss')
                plt.plot(history.history['val_loss'], label='val_loss')
                plt.title('model training loss')
                plt.ylabel('loss')
                plt.xlabel('round')
                plt.legend()
                plt.savefig('training_loss.png')
                plt.close()
    except Exception as e:
        log_message(f"模型训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # 拟合结果 - 生成所有样本的预测
    log_message("生成训练预测...")
    fit_result = []
    for i in range(X.shape[0]):  # 预测所有训练样本
        try:
            if is_fusion_model and X_macro is not None:
                sample_input = [X[i:i + 1], X_macro[i:i + 1]]
            else:
                sample_input = X[i:i + 1]

            pred = model.predict(sample_input, verbose=0)
            fit_result.append(pred[0])  # 取第一个预测结果
        except Exception as e:
            log_message(f"样本 {i} 预测失败: {str(e)}")
            fit_result.append(np.zeros(n_steps_out))

    fit_result = np.array(fit_result)

    # 测试结果 - 生成所有样本的预测
    log_message("生成测试预测...")
    test_result = []
    for i in range(test_x.shape[0]):  # 预测所有测试样本
        try:
            if is_fusion_model and test_X_macro is not None:
                sample_input = [test_x[i:i + 1], test_X_macro[i:i + 1]]
            else:
                sample_input = test_x[i:i + 1]

            pred = model.predict(sample_input, verbose=0)
            test_result.append(pred[0])  # 取第一个预测结果
        except Exception as e:
            log_message(f"测试样本 {i} 预测失败: {str(e)}")
            test_result.append(np.zeros(n_steps_out))

    test_result = np.array(test_result)

    # 打印预测结果形状
    log_message(f"训练预测形状: {fit_result.shape}, 测试预测形状: {test_result.shape}")

    # 反归一化结果 - 关键修复
    log_message("反归一化结果...")
    try:
        # 检查归一化器有效性
        if scalers is None or target_index not in scalers or scalers[target_index] is None:
            log_message(f"错误: 缺少目标变量（索引{target_index}）的归一化器")
            return None, None, None, None

        target_scaler = scalers[target_index]
        log_message(f"使用目标归一化器（索引{target_index}）进行反归一化")

        # ============= 新增的反归一化处理逻辑 =============
        def flatten_and_restore(scaler, data):
            """处理任意维度的预测结果：展平->反归一化->恢复原始形状"""
            original_shape = data.shape
            # 展平为二维数组 (n_samples, 1)
            flattened = data.reshape(-1, 1)
            # 执行反归一化
            restored = scaler.inverse_transform(flattened)
            # 恢复原始形状
            return restored.reshape(original_shape)

        # 使用新的函数处理所有结果
        train_result = flatten_and_restore(target_scaler, fit_result)
        test_result_final = flatten_and_restore(target_scaler, test_result)
        train_actual = flatten_and_restore(target_scaler, y)
        test_actual = flatten_and_restore(target_scaler, test_y)
        # ============= 新增结束 =============

        # 确保形状匹配
        train_result = train_result.reshape(-1, n_steps_out)
        test_result_final = test_result_final.reshape(-1, n_steps_out)
        train_actual = train_actual.reshape(-1, n_steps_out)
        test_actual = test_actual.reshape(-1, n_steps_out)

        # 打印样本结果验证
        log_message("反归一化样本验证:")
        log_message(f"训练预测样本: {train_result[0]}")
        log_message(f"训练实际样本: {train_actual[0]}")
        log_message(f"测试预测样本: {test_result_final[0]}")
        log_message(f"测试实际样本: {test_actual[0]}")

        # 打印数据范围验证
        log_message("反归一化后范围验证:")
        log_message(f"训练预测范围: {np.min(train_result):.2f} - {np.max(train_result):.2f}")
        log_message(f"训练实际范围: {np.min(train_actual):.2f} - {np.max(train_actual):.2f}")
        log_message(f"测试预测范围: {np.min(test_result_final):.2f} - {np.max(test_result_final):.2f}")
        log_message(f"测试实际范围: {np.min(test_actual):.2f} - {np.max(test_actual):.2f}")

        return train_result, test_result_final, train_actual, test_actual

    except Exception as e:
        log_message(f"反归一化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def split_sequence(sequence, n_steps_in, n_steps_out, target_index=0, step_size=1):
    """
    分割序列为输入和输出 - 使用重叠窗口提高数据利用率
    """
    X, y = [], []

    # 计算可生成的样本数量
    num_samples = (len(sequence) - n_steps_in - n_steps_out) // step_size + 1

    for i in range(0, num_samples * step_size, step_size):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(sequence):
            break

        seq_x = sequence[i:end_ix, :]

        # 使用指定的目标索引获取目标值
        if sequence.shape[1] > 1 and target_index < sequence.shape[1]:
            seq_y = sequence[end_ix:out_end_ix, target_index]
        else:
            seq_y = sequence[end_ix:out_end_ix, 0]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

# ==================== 模型构建函数 ====================
from tensorflow.keras import regularizers


def create_macro_fusion_model(n_steps_in, n_steps_out, n_features, n_macro_features):
    """
    创建融合宏观经济因素的模型 - 添加正则化抑制过拟合
    """
    # 增加L2正则化强度
    l2_reg = 0.001

    # 时间序列输入
    ts_input = Input(shape=(n_steps_in, n_features))

    # 宏观因子输入
    macro_input = Input(shape=(n_macro_features,))

    # 添加Dropout层
    macro_processed = Dense(32, activation='relu')(macro_input)
    macro_processed = Dropout(0.3)(macro_processed)  # 增加Dropout比例
    macro_processed = Dense(32, activation='relu')(macro_processed)
    macro_processed = Dropout(0.3)(macro_processed)  # 增加Dropout比例

    # 时间特征处理 - 添加L2正则化
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_reg))(ts_input)  # 添加L2正则
    x = LayerNormalization()(x)

    # 动态特征加权
    macro_gate = Dense(64, activation='sigmoid')(macro_processed)
    macro_gate = RepeatVector(n_steps_in)(macro_gate)
    macro_gate = TimeDistributed(Dense(64))(macro_gate)

    # 特征融合
    fused = Multiply()([x, macro_gate])

    # Transformer编码层 - 添加正则化
    for _ in range(2):  # 保持层数不变
        # 注意力机制
        attn_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64 // 8,  # 减少key维度
            dropout=0.2  # 增加注意力dropout
        )(fused, fused)

        # 残差连接+层归一化
        attn_output = Dropout(0.3)(attn_output)  # 添加额外Dropout
        fused = LayerNormalization(epsilon=1e-6)(fused + attn_output)

        # 前馈网络
        ff_output = Dense(128, activation="relu",
                          kernel_regularizer=regularizers.l2(l2_reg))(fused)  # 添加L2正则
        ff_output = Dropout(0.4)(ff_output)  # 增加Dropout比例
        ff_output = Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(ff_output)  # 添加L2正则

        # 残差连接+层归一化
        ff_output = Dropout(0.3)(ff_output)  # 添加额外Dropout
        fused = LayerNormalization(epsilon=1e-6)(fused + ff_output)

    # 输出层 - 添加正则化
    x = fused[:, -1, :]
    outputs = Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)  # 添加L2正则
    outputs = Dropout(0.3)(outputs)  # 增加Dropout比例
    outputs = Dense(n_steps_out)(outputs)

    return Model(inputs=[ts_input, macro_input], outputs=outputs)


# ==================== 模型构建函数 ====================
def create_model(model_type, n_features, n_steps_in, n_steps_out, n_macro_features=0):
    '''
    创建模型 - 增强版：支持宏观经济因素融合和优化器兼容性
    '''
    # 使用带梯度裁剪和AMSGrad的优化器
    try:
        # 尝试导入legacy优化器
        from tensorflow.keras.optimizers.legacy import Adam
        optimizer = Adam(
            learning_rate=0.001,
            clipvalue=1.0,  # 梯度裁剪
            amsgrad=True  # 稳定训练
        )
        print("使用legacy Adam优化器 (带梯度裁剪和AMSGrad)")
    except ImportError:
        # 如果legacy不可用，则使用普通Adam
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(
            learning_rate=0.001,
            clipvalue=1.0,
            amsgrad=True
        )
        print("使用标准Adam优化器 (带梯度裁剪和AMSGrad)")

    # 创建宏观经济融合模型（当用户选择此模型类型时）
    if model_type == 'MacroFusion':
        model = create_macro_fusion_model(n_steps_in, n_steps_out, n_features, n_macro_features)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    # 创建其他类型的模型（所有模型都将使用宏观经济特征）
    if model_type == 'LSTM':
        model = Sequential()
        model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(n_steps_out))
    elif model_type == 'BD LSTM':
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='sigmoid'), input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_steps_out))
    elif model_type == 'ED LSTM':
        model = Sequential()
        model.add(LSTM(100, activation='sigmoid', input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.2))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(1)))
    elif model_type == 'CNN':
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_steps_out))
    elif model_type == 'Convolutional LSTM':
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(20, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))
    elif model_type == 'Transformer':
        model = create_transformer_model(n_steps_in, n_steps_out, n_features, d_model=64,
                                         num_heads=12, ff_dim=64, num_transformer_blocks=3)
    elif model_type == 'MLP':
        model = Sequential()
        model.add(Dense(20, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_steps_out))
    elif model_type == 'Informer':
        # 简化的Informer模型实现
        inputs = Input(shape=(n_steps_in, n_features))

        # 嵌入层
        x = Dense(64)(inputs)

        # 自注意力机制
        attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization()(x + attn_output)

        # 前馈网络
        ff_output = Dense(128, activation='relu')(x)
        ff_output = Dense(64)(ff_output)
        x = LayerNormalization()(x + ff_output)

        # 输出层
        x = x[:, -1, :]  # 只取最后时间步
        outputs = Dense(n_steps_out)(x)

        model = Model(inputs, outputs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.compile(optimizer=optimizer, loss='mse')
    return model


def eval_result(result, target, mode):
    '''
    改进的评估模型结果 - 使用sMAPE和方向精度
    '''
    # 确保预测和实际值维度一致
    if result.shape != target.shape:
        min_samples = min(result.shape[0], target.shape[0])
        result = result[:min_samples]
        target = target[:min_samples]
        print(f"警告: 结果与实际值形状不匹配，已截断为 {min_samples} 个样本")

    # 添加严格的数据验证
    result = np.nan_to_num(result, nan=0.0, posinf=np.nanmax(result), neginf=np.nanmin(result))
    target = np.nan_to_num(target, nan=0.0, posinf=np.nanmax(target), neginf=np.nanmin(target))

    # 避免除零错误
    epsilon = 1e-7

    if mode == 0:  # RMSE
        rmse = []
        for i in range(result.shape[1]):
            # 添加容差检查
            if np.ptp(target[:, i]) < epsilon:  # 如果变化范围太小
                rmse.append(0)
            else:
                # 计算该步长的RMSE
                rmse.append(np.sqrt(np.mean((result[:, i] - target[:, i]) ** 2)))
        return rmse

    elif mode == 1:  # sMAPE
        smape = []
        for i in range(result.shape[1]):
            # 对称平均绝对百分比误差
            abs_error = np.abs(result[:, i] - target[:, i])
            denominator = (np.abs(result[:, i]) + np.abs(target[:, i]) + epsilon)
            smape.append(np.mean(200 * abs_error / denominator))
        return smape

    elif mode == 2:  # 方向精度
        direction_acc = []
        for i in range(result.shape[1]):
            # 预测方向正确性
            pred_direction = np.sign(result[1:, i] - result[:-1, i])
            actual_direction = np.sign(target[1:, i] - target[:-1, i])
            acc = np.mean(pred_direction == actual_direction) * 100
            direction_acc.append(acc)
        return direction_acc

    else:
        return None


# 宏观指标关联性分析器开发

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 宏观指标文件映射
MACRO_FILES = {
    'fed_rate': 'Federal Funds Rate.csv',
    'treasury_yield': 'us_treasury_yields_daily.csv',
    'dxy': 'DXY.csv',
    'cpi': 'CPIAUCSL.csv',
    'vix': 'VIX_History.csv',
    'sp500': 'sap500.csv'
}

CRYPTO_FILES = {
    'Bitcoin': 'coin_Bitcoin.csv',
    'Ethereum': 'coin_Ethereum.csv',
    'Dogecoin': 'coin_Dogecoin.csv',
    'Litecoin': 'coin_Litecoin.csv'
}


def load_macro_data(period='all'):
    """加载并处理宏观数据"""
    macro_data = {}

    # 计算时间范围
    end_date = datetime.now()
    if period == '1y':
        start_date = end_date - timedelta(days=365)
    elif period == '3y':
        start_date = end_date - timedelta(days=1095)
    elif period == '5y':
        start_date = end_date - timedelta(days=1825)
    else:
        start_date = datetime(2013, 1, 1)  # 所有可用数据

    # 加载每个宏观指标
    for indicator, filename in MACRO_FILES.items():
        df = pd.read_csv(f'macro_data/{filename}')

        # 统一日期格式处理
        if 'DATE' in df.columns:
            df['Date'] = pd.to_datetime(df['DATE'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'observation_date' in df.columns:
            df['Date'] = pd.to_datetime(df['observation_date'])

        # 筛选时间范围
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # 提取相关列
        if indicator == 'fed_rate':
            df = df[['Date', 'RIFSPFF_N.D']].rename(columns={'RIFSPFF_N.D': 'value'})
        elif indicator == 'treasury_yield':
            df = df[['Date', 'US10Y']].rename(columns={'US10Y': 'value'})
        elif indicator == 'dxy':
            df = df[['Date', 'DTWEXBGS']].rename(columns={'DTWEXBGS': 'value'})
        elif indicator == 'cpi':
            df = df[['Date', 'CPIAUCSL']].rename(columns={'CPIAUCSL': 'value'})
        elif indicator == 'vix':
            df = df[['Date', 'CLOSE']].rename(columns={'CLOSE': 'value'})
        elif indicator == 'sp500':
            df = df[['Date', 'Close']].rename(columns={'Close': 'value'})

        macro_data[indicator] = df.set_index('Date')

    return macro_data


def load_crypto_data(period='all'):
    """加载并处理加密货币数据"""
    crypto_data = {}

    # 计算时间范围
    end_date = datetime.now()
    if period == '1y':
        start_date = end_date - timedelta(days=365)
    elif period == '3y':
        start_date = end_date - timedelta(days=1095)
    elif period == '5y':
        start_date = end_date - timedelta(days=1825)
    else:
        start_date = datetime(2013, 1, 1)  # 所有可用数据

    for crypto, filename in CRYPTO_FILES.items():
        df = pd.read_csv(f'crypto_data/{filename}')

        # 统一日期格式处理
        df['Date'] = pd.to_datetime(df['Date'])

        # 筛选时间范围
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # 提取收盘价
        df = df[['Date', 'Close']].rename(columns={'Close': 'value'})
        crypto_data[crypto] = df.set_index('Date')

    return crypto_data


# 定义全局数据集时间范围
DATA_START = pd.Timestamp('2013-04-29')
DATA_END = pd.Timestamp('2021-07-07')


def get_date_range(period):
    """根据时间段获取日期范围（返回数据集时间范围内的日期）"""
    # 根据时间段计算起始日期
    if period == '1y':
        # 1年时间范围：结束日期前推1年，但不早于数据集最早日期
        start_date = max(DATA_END - pd.Timedelta(days=365), DATA_START)
    elif period == '3y':
        # 3年时间范围：结束日期前推3年，但不早于数据集最早日期
        start_date = max(DATA_END - pd.Timedelta(days=1095), DATA_START)
    elif period == '5y':
        # 5年时间范围：结束日期前推5年，但不早于数据集最早日期
        start_date = max(DATA_END - pd.Timedelta(days=1825), DATA_START)
    else:  # all
        # 全部数据：使用数据集的最早日期
        start_date = DATA_START

    # 结束日期使用数据集结束日期
    end_date = DATA_END

    print(f"实际使用时间范围: {start_date} 到 {end_date}")
    return start_date, end_date


def calculate_correlations(period='all'):
    """完整的相关系数计算实现"""
    try:
        # 获取日期范围
        start_date, end_date = get_date_range(period)
        print(f"计算相关系数 - 时间范围: {start_date} 到 {end_date}")

        # 获取宏观经济数据
        macro_df = fetch_macro_data(start_date, end_date)

        # 确保所有索引都是时区无关的
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)

        # 获取加密货币数据
        results = {}

        # 加密货币列表
        cryptos = ['Bitcoin', 'Ethereum', 'Litecoin', 'Dogecoin']

        for crypto in cryptos:
            path = f'data/coin_{crypto}.csv'
            if os.path.exists(path):
                print(f"处理加密货币: {crypto}")
                # 读取加密货币数据
                df = pd.read_csv(path)

                # 统一处理日期格式 - 移除时区信息
                df['Date'] = pd.to_datetime(df['Date'], utc=False)  # utc=False 确保时区无关
                df = df.set_index('Date')
                df = df.sort_index()

                # 筛选日期范围 - 确保在数据集范围内
                df = df.loc[start_date:end_date]

                # 检查是否有数据
                if df.empty:
                    print(f"警告: {crypto} 在时间范围内无数据")
                    results[crypto] = {}
                    continue

                # 确保索引是时区无关的
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # 合并宏观数据 - 使用外部连接确保数据对齐
                merged = pd.merge(df[['Close']], macro_df, left_index=True, right_index=True, how='outer')

                # 前向填充和后向填充处理缺失值
                merged = merged.ffill().bfill()

                # 添加调试信息：打印合并后的数据样本
                print(f"{crypto} 数据样本（前5行）:")
                print(merged.head())

                # 计算相关系数
                correlations = {}
                for macro_col in macro_df.columns:
                    try:
                        # 检查是否有足够的数据点计算相关系数
                        valid_data = merged[['Close', macro_col]].dropna()

                        if len(valid_data) < 2:
                            print(f"警告: {macro_col} 数据点不足（{len(valid_data)}），无法计算相关系数")
                            correlations[macro_col] = 0.0
                            continue

                        # 计算Pearson相关系数
                        corr = valid_data['Close'].corr(valid_data[macro_col])

                        if pd.isna(corr):
                            print(f"警告: {macro_col} 相关系数计算为NaN")
                            correlations[macro_col] = 0.0
                        else:
                            correlations[macro_col] = corr
                            print(f"{crypto}与{macro_col}相关系数: {corr:.4f}")
                    except Exception as e:
                        print(f"计算 {macro_col} 相关性出错: {e}")
                        correlations[macro_col] = 0.0

                results[crypto] = correlations
            else:
                print(f"文件不存在: {path}")
                results[crypto] = {}

        return results
    except Exception as e:
        print(f"相关系数计算失败: {str(e)}")
        return {}


def calculate_historical_correlations(crypto, indicator, period='all'):
    """计算历史滚动相关系数（按月） - 修复版"""
    # 获取日期范围
    start_date, end_date = get_date_range(period)
    print(f"计算历史相关性 - {crypto} 和 {indicator}, 时间范围: {start_date} 到 {end_date}")

    # 获取宏观经济数据
    macro_df = fetch_macro_data(start_date, end_date)

    # 确保索引是时区无关的
    if macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_localize(None)

    # 获取加密货币数据
    crypto_path = f'data/coin_{crypto}.csv'
    if not os.path.exists(crypto_path):
        print(f"文件不存在: {crypto_path}")
        return []

    try:
        # 读取加密货币数据
        crypto_df = pd.read_csv(crypto_path)

        # 统一处理日期格式 - 移除时区信息
        crypto_df['Date'] = pd.to_datetime(crypto_df['Date'], utc=False)
        crypto_df = crypto_df.set_index('Date')
        crypto_df = crypto_df.sort_index()

        # 筛选日期范围
        crypto_df = crypto_df.loc[start_date:end_date]

        # 检查是否有数据
        if crypto_df.empty:
            print(f"警告: {crypto} 在时间范围内无数据")
            return []

        # 确保索引是时区无关的
        if crypto_df.index.tz is not None:
            crypto_df.index = crypto_df.index.tz_localize(None)

        # 合并数据 - 使用外部连接确保数据对齐
        merged = pd.merge(crypto_df[['Close']], macro_df[[indicator]],
                          left_index=True, right_index=True, how='outer')

        # 前向填充和后向填充处理缺失值
        merged = merged.ffill().bfill()

        # 添加调试信息：打印合并后的数据样本
        print(f"历史相关性数据样本（前5行）:")
        print(merged.head())

        # 按月重采样并计算相关系数
        monthly_data = merged.resample('M').mean()

        # 检查是否有足够的数据点
        if len(monthly_data) < 3:
            print(f"警告: 数据点不足（{len(monthly_data)}），无法计算滚动相关系数")
            return []

        # 计算滚动相关系数（3个月窗口）
        monthly_data['correlation'] = monthly_data['Close'].rolling(window=3).corr(monthly_data[indicator])

        # 构建返回数据
        historical_correlations = []
        for date, row in monthly_data.iterrows():
            if not pd.isna(row['correlation']):
                historical_correlations.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'correlation': round(row['correlation'], 4)
                })

        print(f"历史相关性数据点数量: {len(historical_correlations)}")
        return historical_correlations

    except Exception as e:
        print(f"历史相关性计算失败: {str(e)}")
        return []


# 添加历史相关系数路由
# @app.route('/api/historical-correlations')
# def get_historical_correlations():
#     """获取历史相关系数数据"""
#     crypto = request.args.get('crypto')
#     indicator = request.args.get('indicator')
#     period = request.args.get('period', 'all')
#     data = calculate_historical_correlations(crypto, indicator, period)
#     return jsonify(data)
#
# @app.route('/api/correlations')
# def get_correlations():
#     """获取相关系数数据"""
#     period = request.args.get('period', 'all')
#     correlations = calculate_correlations(period)
#     return jsonify(correlations)

# ==================== 主函数 ====================
# def main():
#     print("启动增强版加密货币预测系统...")
#
#     # 设置全局随机种子
#     tf.random.set_seed(917304)
#     np.random.seed(917304)
#     random.seed(917304)
#     os.environ['PYTHONHASHSEED'] = '917304'
#
#     # ----------------- 参数设置 -----------------
#     file_path = "data/coin_Bitcoin.csv"
#     dim_type = 'Multi'
#     use_percentage = 1
#     include_macro = True
#     n_steps_in = 7
#     n_steps_out = 5
#     percentage = 0.8  # 增加训练集比例到80%
#     epochs = 50  # 增加训练轮数
#     rounds = 3  # 增加实验轮次
#     split_type = 0  # 使用时间顺序划分，避免随机性
#
#     # 关键修改：定义可配置的模型类型
#     model_type = 'Transformer'  # 可修改为: 'LSTM', 'ED LSTM', 'CNN','MacroFusion'
#     # 'Convolutional LSTM', 'Transformer', 'MLP'，'BD LSTM'等
#
#     # 创建结果目录
#     results_dir = "results"
#     os.makedirs(results_dir, exist_ok=True)
#
#     try:
#         # ----------------- 数据处理 -----------------
#         print("加载数据和宏观经济指标...")
#         data, data_len, feature_names = read_data(file_path, dim_type, use_percentage, include_macro)
#         if data is None:
#             print("数据处理失败，程序终止")
#             return
#
#         print(f"数据加载成功. 形状: {data.shape}")
#         print(f"使用的模型类型: {model_type}")  # 显示当前使用的模型类型
#         print(f"特征名称: {feature_names}")  # 打印特征名称
#
#         # 归一化数据
#         print("数据归一化...")
#         # 传递特征名称给归一化函数
#         data, scalers = data_trasform(data, feature_names=feature_names)
#         if data is None:
#             print("数据归一化失败，程序终止")
#             return
#
#         # 添加关键诊断：检查归一化后数据
#         print("归一化数据统计:")
#         print(f"NaN数量: {np.isnan(data).sum()}, Inf数量: {np.isinf(data).sum()}")
#         print(f"均值: {np.nanmean(data):.6f}, 标准差: {np.nanstd(data):.6f}")
#         print(f"最小值: {np.nanmin(data):.6f}, 最大值: {np.nanmax(data):.6f}")
#
#         # 检查归一化后的Close列
#         if 'Close' in feature_names:
#             close_idx = feature_names.index('Close')
#             close_data = data[:, close_idx]
#             print(
#                 f"归一化后Close列统计: min={np.min(close_data):.6f}, max={np.max(close_data):.6f}, mean={np.mean(close_data):.6f}")
#         else:
#             print("警告: 归一化数据中找不到'Close'列")
#
#         # 分割训练集和测试集
#         print("分割数据集...")
#         train_set = data[0:int(np.floor(data_len * percentage))]
#         test_set = data[int(np.floor(data_len * percentage)):]
#
#         # 确保训练/测试集大小合理
#         min_samples = n_steps_in + n_steps_out + 1
#         if len(train_set) < min_samples:
#             raise ValueError(f"训练集太小 ({len(train_set)} 样本). 需要至少 {min_samples} 个样本")
#         if len(test_set) < min_samples:
#             raise ValueError(f"测试集太小 ({len(test_set)} 样本). 需要至少 {min_samples} 个样本")
#
#         print(f"训练集形状: {train_set.shape}, 测试集形状: {test_set.shape}")
#
#         # 定义宏观经济特征列表
#         macro_features = ['vix', 'sp500', 'fed_rate', 'treasury_yield', 'dxy', 'cpi']
#
#         # 计算动态权重
#         dynamic_weights = None
#         if include_macro:
#             print("计算动态特征权重...")
#             try:
#                 # 获取完整数据用于权重计算
#                 full_df = pd.read_csv(file_path)
#                 full_df["Date"] = pd.to_datetime(full_df["Date"], utc=True)
#
#                 # 获取宏观经济数据
#                 macro_df = fetch_macro_data(
#                     start_date=full_df['Date'].min().strftime('%Y-%m-%d'),
#                     end_date=full_df['Date'].max().strftime('%Y-%m-%d')
#                 )
#                 macro_df = macro_df.reset_index().rename(columns={'index': 'Date'})
#
#                 # 合并数据
#                 full_df = full_df.merge(macro_df, on='Date', how='left')
#
#                 # 应用特征工程
#                 full_df = enhanced_feature_engineering(full_df)
#
#                 # 计算动态权重
#                 crypto_type = 'Bitcoin'  # 从文件名推断
#                 if 'Ethereum' in file_path:
#                     crypto_type = 'Ethereum'
#                 elif 'Litecoin' in file_path:
#                     crypto_type = 'Litecoin'
#
#                 dynamic_weights = calculate_dynamic_weights(full_df, crypto_type)
#
#                 # 提取宏观特征权重
#                 train_weights = dynamic_weights[macro_features].mean(axis=1).iloc[:len(train_set)].values
#
#                 # 打印权重统计
#                 print(
#                     f"动态权重统计: min={np.min(train_weights):.4f}, max={np.max(train_weights):.4f}, mean={np.mean(train_weights):.4f}")
#             except Exception as e:
#                 print(f"动态权重计算失败: {str(e)}")
#                 train_weights = None
#         else:
#             train_weights = None
#
#         # 确定特征数量
#         n_features = train_set.shape[1]
#         n_macro_features = len(macro_features) if include_macro else 0
#         print(f"特征数量: {n_features}, 宏观经济特征数量: {n_macro_features}")
#
#         # 添加关键修复：获取目标列索引
#         target_index = 0  # 默认值
#         if feature_names:  # 使用独立的特征名称列表
#             try:
#                 # 查找Close列位置
#                 if 'Close' in feature_names:
#                     target_index = feature_names.index('Close')
#                     print(f"目标列 'Close' 的索引位置: {target_index}")
#
#                     # 验证目标列数据
#                     close_col = data[:, target_index]
#                     print(
#                         f"归一化后Close列统计: min={np.min(close_col):.6f}, max={np.max(close_col):.6f}, mean={np.mean(close_col):.6f}")
#                 else:
#                     # 尝试自动查找
#                     found = False
#                     for i, name in enumerate(feature_names):
#                         if 'close' in name.lower():
#                             target_index = i
#                             print(f"找到类似列 '{name}' 作为目标，索引={target_index}")
#                             found = True
#                             break
#
#                     if not found:
#                         target_index = 0
#                         print(f"警告: 无法确定目标列，使用默认索引0")
#             except Exception as e:
#                 print(f"目标索引获取失败: {str(e)}，使用默认索引0")
#                 target_index = 0
#         else:
#             target_index = 0
#             print("警告: 缺少特征名称列表，使用默认目标索引0")
#
#         # 打印目标列在训练集中的统计
#         if len(train_set) > 0:
#             target_col = train_set[:, target_index]
#             print(
#                 f"训练集目标列统计: min={np.min(target_col):.6f}, max={np.max(target_col):.6f}, mean={np.mean(target_col):.6f}")
#
#         # ----------------- 模型训练与评估 -----------------
#         exp_result = pd.DataFrame(columns=['Model', 'Train MINMAX RMSE', 'Test MINMAX RMSE', 'Train MAPE', 'Test MAPE'])
#
#         for round_idx in range(rounds):
#             print(f"\n开始第 {round_idx + 1}/{rounds} 轮...")
#             print(f"使用模型: {model_type}")  # 确认当前轮次使用的模型
#
#             try:
#                 model = create_model(model_type, n_features, n_steps_in, n_steps_out, n_macro_features)
#                 print(f"创建 {model_type} 模型...")
#
#                 # 打印模型摘要
#                 print("模型架构摘要:")
#                 model.summary()
#             except ValueError as e:
#                 print(f"模型创建错误: {str(e)}")
#                 continue
#
#             if model is None:
#                 print(f"{model_type} 模型创建失败")
#                 continue
#
#             # 训练和预测
#             print("训练模型并生成预测...")
#             try:
#                 # 添加关键修复：传递target_index和feature_names参数
#                 train_result, test_result, train_actual, test_actual = train_and_forecast(
#                     model, n_features, train_set, test_set,
#                     n_steps_in, n_steps_out, epochs,
#                     dynamic_weights=train_weights,
#                     n_macro_features=n_macro_features,
#                     scalers=scalers,  # 传递归一化器
#                     target_index=target_index,  # 关键修复
#                     feature_names=feature_names  # 传递特征名称
#                 )
#
#                 if train_result is None or test_result is None:
#                     print("训练失败，跳过本轮")
#                     continue
#
#                 # 打印预测结果统计
#                 print("预测结果统计:")
#                 print(f"训练预测范围: {np.min(train_result):.2f} - {np.max(train_result):.2f}")
#                 print(f"测试预测范围: {np.min(test_result):.2f} - {np.max(test_result):.2f}")
#                 print(f"训练实际范围: {np.min(train_actual):.2f} - {np.max(train_actual):.2f}")
#                 print(f"测试实际范围: {np.min(test_actual):.2f} - {np.max(test_actual):.2f}")
#
#                 # 检查预测结果合理性
#                 if np.max(train_result) > 1e6 or np.max(test_result) > 1e6:
#                     print(f"警告: 预测值异常大，可能存在问题")
#
#                 if np.any(np.isnan(train_result)) or np.any(np.isnan(test_result)):
#                     print(f"警告: 预测结果包含NaN值")
#             except Exception as e:
#                 print(f"训练失败: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
#
#             # 计算评估指标
#             print("计算指标...")
#             try:
#                 train_rmse = eval_result(train_result, train_actual, 0)
#                 test_rmse = eval_result(test_result, test_actual, 0)
#
#                 # 计算MAPE
#                 train_mape = eval_result(train_result, train_actual, 1)
#                 test_mape = eval_result(test_result, test_actual, 1)
#
#                 # 打印指标统计
#                 print(f"训练RMSE: 均值={np.mean(train_rmse):.2f}, 最大={np.max(train_rmse):.2f}")
#                 print(f"测试RMSE: 均值={np.mean(test_rmse):.2f}, 最大={np.max(test_rmse):.2f}")
#                 print(f"训练MAPE: 均值={np.mean(train_mape):.2f}%, 最大={np.max(train_mape):.2f}%")
#                 print(f"测试MAPE: 均值={np.mean(test_mape):.2f}%, 最大={np.max(test_mape):.2f}%")
#             except Exception as e:
#                 print(f"评估失败: {str(e)}")
#                 # 使用默认值继续
#                 train_rmse = [0] * n_steps_out
#                 test_rmse = [0] * n_steps_out
#                 train_mape = [0] * n_steps_out
#                 test_mape = [0] * n_steps_out
#
#             print(f'训练集 MINMAX RMSE: {train_rmse}')
#             print(f'测试集 MINMAX RMSE: {test_rmse}')
#             print(f'训练集 MAPE: {train_mape}')
#             print(f'测试集 MAPE: {test_mape}')
#
#             # 保存结果 - 包括模型类型
#             exp_result.loc[len(exp_result.index)] = [
#                 model_type,  # 记录模型类型
#                 train_rmse,
#                 test_rmse,
#                 train_mape,
#                 test_mape
#             ]
#
#             # # 保存结果到Excel文件
#             # results_file = os.path.join(results_dir, f"results_round_{round_idx + 1}.xlsx")
#             # exp_result.to_excel(results_file, index=False)
#             # print(f"第 {round_idx + 1} 轮结果已保存到: {os.path.abspath(results_file)}")
#
#         # 保存最终结果
#         final_results_file = os.path.join(results_dir, "final_results.xlsx")
#         exp_result.to_excel(final_results_file, index=False)
#         print(f"最终结果已保存到: {os.path.abspath(final_results_file)}")
#
#         # 保存完整结果
#         print("所有实验完成. 结果已保存到 results 目录")
#
#         # 可视化结果
#         if n_steps_out == 1 and train_result is not None and test_result is not None:
#             print("生成可视化结果...")
#             try:
#                 plt.figure(figsize=(15, 6))
#                 plt.plot(train_result, label=f'{model_type} 训练预测')
#                 plt.plot(train_actual, label='实际值')
#                 plt.title(f'{model_type} 训练结果')
#                 plt.xlabel('时间步')
#                 plt.ylabel('收盘价')
#                 plt.legend()
#                 train_plot_file = os.path.join(results_dir, f'train_results_{model_type}.png')
#                 plt.savefig(train_plot_file)
#                 plt.close()
#                 print(f"训练结果图已保存: {os.path.abspath(train_plot_file)}")
#
#                 plt.figure(figsize=(15, 6))
#                 plt.plot(test_result, label=f'{model_type} 测试预测')
#                 plt.plot(test_actual, label='实际值')
#                 plt.title(f'{model_type} 测试结果')
#                 plt.xlabel('时间步')
#                 plt.ylabel('收盘价')
#                 plt.legend()
#                 test_plot_file = os.path.join(results_dir, f'test_results_{model_type}.png')
#                 plt.savefig(test_plot_file)
#                 plt.close()
#                 print(f"测试结果图已保存: {os.path.abspath(test_plot_file)}")
#
#                 print("可视化结果已保存")
#             except Exception as e:
#                 print(f"可视化失败: {str(e)}")
#
#     except Exception as e:
#         print(f"发生严重错误: {str(e)}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # 清理资源
#         tf.keras.backend.clear_session()
#         print("清理完成，程序结束")




# #开始做前端，这里注释掉
# def main():
#     # ======== 字体问题终极解决方案 ========
#     import matplotlib
#     import matplotlib.font_manager as fm
#
#     # 确保使用支持字体的后端
#     matplotlib.use('Agg')  # 强制使用无头渲染器
#     plt.switch_backend('Agg')  # 确保所有绘图使用正确后端
#
#     # 获取可用字体列表
#     font_list = [f.name for f in fm.fontManager.ttflist]
#
#     # 选择最适合的英文无衬线字体
#     fallback_fonts = [
#         'DejaVu Sans', 'Arial', 'Liberation Sans',
#         'Bitstream Vera Sans', 'sans-serif'
#     ]
#
#     # 设置优先字体选项
#     for font in fallback_fonts:
#         if font in font_list:
#             plt.rcParams['font.family'] = font
#             print(f"✅ 使用字体: {font}")
#             break
#     else:
#         plt.rcParams['font.family'] = 'sans-serif'
#         print("⚠️ 使用备用无衬线字体")
#
#     # 确保正确显示负号
#     matplotlib.rcParams['axes.unicode_minus'] = False
#
#     # 设置全局字体参数
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.titlesize'] = 15
#     plt.rcParams['axes.labelsize'] = 12
#
#     # ======== 在训练函数中检查字体设置 ========
#     print("\n=== 当前字体状态验证 ===")
#     print("活动后端:", matplotlib.get_backend())
#     print("当前字体:", plt.rcParams['font.family'])
#     print("可用字体示例:", font_list[:3])
#
#     print("Starting enhanced cryptocurrency prediction system...")
#
#     # 设置全局随机种子
#     tf.random.set_seed(917304)
#     np.random.seed(917304)
#     random.seed(917304)
#     os.environ['PYTHONHASHSEED'] = '917304'
#
#     # ----------------- 参数设置 -----------------
#     crypto_files = [
#         "data/coin_Bitcoin.csv",
#         "data/coin_Dogecoin.csv",
#         "data/coin_Ethereum.csv",
#         "data/coin_Litecoin.csv"
#     ]
#
#     model_types = [
#         'LSTM',
#         'BD LSTM',
#         'ED LSTM',
#         'CNN',
#         'Convolutional LSTM',
#         'Transformer',
#         'MLP',
#         'MacroFusion'
#     ]
#
#     # 存储所有结果的字典
#     all_results = {}
#
#     # 创建结果目录
#     results_dir = "results"
#     os.makedirs(results_dir, exist_ok=True)
#
#     # 循环处理每个加密货币
#     for crypto_path in crypto_files:
#         crypto_name = os.path.basename(crypto_path).split('.')[0].split('_')[-1]
#         print(f"\n{'=' * 50}")
#         print(f"Processing cryptocurrency: {crypto_name}")
#         print(f"{'=' * 50}")
#
#         # 创建该币种的目录
#         crypto_dir = os.path.join(results_dir, crypto_name)
#         os.makedirs(crypto_dir, exist_ok=True)
#
#         # 存储该币种的结果
#         crypto_results = []
#         crypto_performance = []
#
#         # 循环处理每个模型类型
#         for model_type in model_types:
#             print(f"\n{'*' * 40}")
#             print(f"Training model: {model_type}")
#             print(f"{'*' * 40}")
#
#             # 参数设置
#             dim_type = 'Multi'
#             use_percentage = 1
#             include_macro = True
#             include_macro = True
#             n_steps_in = 7
#             n_steps_out = 5
#             percentage = 0.8
#             epochs = 50
#             rounds = 3
#             split_type = 0
#
#             try:
#                 # ----------------- 数据处理 -----------------
#                 print("Loading data and macroeconomic indicators...")
#                 data, data_len, feature_names = read_data(crypto_path, dim_type, use_percentage, include_macro)
#                 if data is None:
#                     print("Data processing failed, skipping")
#                     continue
#
#                 print(f"Data loaded successfully. Shape: {data.shape}")
#                 print(f"Model type: {model_type}")
#                 print(f"Feature names: {feature_names}")
#
#                 # 归一化数据
#                 print("Normalizing data...")
#                 data, scalers = data_trasform(data, feature_names=feature_names)
#                 if data is None:
#                     print("Data normalization failed, skipping")
#                     continue
#
#                 # 分割训练集和测试集
#                 print("Splitting dataset...")
#                 train_set = data[0:int(np.floor(data_len * percentage))]
#                 test_set = data[int(np.floor(data_len * percentage)):]
#
#                 # 确保训练/测试集大小合理
#                 min_samples = n_steps_in + n_steps_out + 1
#                 if len(train_set) < min_samples:
#                     print(f"Training set too small ({len(train_set)} samples). Need at least {min_samples} samples")
#                     continue
#                 if len(test_set) < min_samples:
#                     print(f"Test set too small ({len(test_set)} samples). Need at least {min_samples} samples")
#                     continue
#
#                 print(f"Training set shape: {train_set.shape}, Test set shape: {test_set.shape}")
#
#                 # 定义宏观经济特征列表
#                 macro_features = ['vix', 'sp500', 'fed_rate', 'treasury_yield', 'dxy', 'cpi']
#
#                 # 计算动态权重
#                 dynamic_weights = None
#                 if include_macro:
#                     print("Calculating dynamic feature weights...")
#                     try:
#                         # 获取完整数据用于权重计算
#                         full_df = pd.read_csv(crypto_path)
#                         full_df["Date"] = pd.to_datetime(full_df["Date"], utc=True)
#
#                         # 获取宏观经济数据
#                         macro_df = fetch_macro_data(
#                             start_date=full_df['Date'].min().strftime('%Y-%m-%d'),
#                             end_date=full_df['Date'].max().strftime('%Y-%m-%d')
#                         )
#                         macro_df = macro_df.reset_index().rename(columns={'index': 'Date'})
#
#                         # 合并数据
#                         full_df = full_df.merge(macro_df, on='Date', how='left')
#
#                         # 应用特征工程
#                         full_df = enhanced_feature_engineering(full_df)
#
#                         # 计算动态权重
#                         dynamic_weights = calculate_dynamic_weights(full_df, crypto_name)
#
#                         # 提取宏观特征权重
#                         train_weights = dynamic_weights[macro_features].mean(axis=1).iloc[:len(train_set)].values
#                     except Exception as e:
#                         print(f"Dynamic weight calculation failed: {str(e)}")
#                         train_weights = None
#                 else:
#                     train_weights = None
#
#                 # 确定特征数量
#                 n_features = train_set.shape[1]
#                 n_macro_features = len(macro_features) if include_macro else 0
#                 print(f"Number of features: {n_features}, Macro features: {n_macro_features}")
#
#                 # 添加关键修复：获取目标列索引
#                 target_index = 0
#                 if feature_names:
#                     if 'Close' in feature_names:
#                         target_index = feature_names.index('Close')
#                         print(f"Target column 'Close' index: {target_index}")
#                     else:
#                         # 尝试自动查找
#                         found = False
#                         for i, name in enumerate(feature_names):
#                             if 'close' in name.lower():
#                                 target_index = i
#                                 print(f"Found similar column '{name}' as target, index={target_index}")
#                                 found = True
#                                 break
#
#                         if not found:
#                             print(f"Warning: Unable to determine target column, using default index 0")
#                 else:
#                     print("Warning: Missing feature names, using default target index 0")
#
#                 # ----------------- 模型训练与评估 -----------------
#                 model_results = []
#
#                 # ======== 关键修复：修改训练循环 ========
#                 for round_idx in range(rounds):
#                     print(f"\n开始第 {round_idx + 1}/{rounds} 轮...")
#                     print(f"使用模型: {model_type}")
#
#                     if monitor:
#                         monitor.set_model_info(crypto_name, model_type, epochs)
#                         monitor.log_message(f"开始训练 {model_type} 模型, 轮次: {round_idx + 1}/{rounds}")
#
#
#                     # 每轮前清除计算图
#                     tf.keras.backend.clear_session()
#
#                     # 生成本轮唯一种子 (基础种子+轮次偏移)
#                     round_seed = 917304 + round_idx * 100
#
#                     try:
#                         model = create_model(model_type, n_features, n_steps_in, n_steps_out, n_macro_features)
#                         print(f"Created {model_type} model...")
#
#                         # 传递本轮种子到训练函数
#                         train_result, test_result, train_actual, test_actual = train_and_forecast(
#                             model, n_features, train_set, test_set,
#                             n_steps_in, n_steps_out, epochs,
#                             dynamic_weights=train_weights,
#                             n_macro_features=n_macro_features,
#                             scalers=scalers,
#                             target_index=target_index,
#                             feature_names=feature_names,
#                             round_seed=round_seed  # 新增参数
#                         )
#
#                         if train_result is None or test_result is None:
#                             print("Training failed, skipping round")
#                             continue
#                     except Exception as e:
#                         print(f"Training failed: {str(e)}")
#                         continue
#
#                     # 计算评估指标
#                     print("Calculating metrics...")
#                     try:
#                         train_rmse = eval_result(train_result, train_actual, 0)
#                         test_rmse = eval_result(test_result, test_actual, 0)
#                         train_mape = eval_result(train_result, train_actual, 1)
#                         test_mape = eval_result(test_result, test_actual, 1)
#                     except Exception as e:
#                         print(f"Evaluation failed: {str(e)}")
#                         continue
#
#                     # 打印指标
#                     print(f'Training RMSE: {train_rmse}')
#                     print(f'Testing RMSE: {test_rmse}')
#                     print(f'Training MAPE: {train_mape}')
#                     print(f'Testing MAPE: {test_mape}')
#
#                     # 记录本轮结果
#                     model_results.append({
#                         'round': round_idx + 1,
#                         'model': model_type,
#                         'train_rmse': train_rmse,
#                         'test_rmse': test_rmse,
#                         'train_mape': train_mape,
#                         'test_mape': test_mape
#                     })
#
#                     # 计算平均MAPE和RMSE作为性能指标
#                     avg_test_mape = np.mean([np.mean(m) for m in test_mape])
#                     avg_test_rmse = np.mean([np.mean(m) for m in test_rmse])
#
#                     crypto_performance.append({
#                         'model': model_type,
#                         'round': round_idx + 1,
#                         'avg_mape': avg_test_mape,
#                         'avg_rmse': avg_test_rmse
#                     })
#
#                 # 保存该模型的结果
#                 model_df = pd.DataFrame(model_results)
#                 model_file = os.path.join(crypto_dir, f"{model_type}_results.csv")
#                 model_df.to_csv(model_file, index=False)
#                 print(f"{model_type} model results saved to: {model_file}")
#
#                 # 保存训练过程的图片
#                 loss_file = os.path.join(crypto_dir, f"{model_type}_training_loss.png")
#                 if os.path.exists('training_loss.png'):
#                     os.rename('training_loss.png', loss_file)
#                     print(f"Training loss plot saved to: {loss_file}")
#
#                 # 保存预测结果图片
#                 if n_steps_out == 1 and train_result is not None and test_result is not None:
#                     # 训练结果图
#                     plt.figure(figsize=(15, 6))
#                     plt.plot(train_result, label='Train Prediction')
#                     plt.plot(train_actual, label='Actual Value')
#                     plt.title(f'{model_type} Training Results - {crypto_name}')
#                     plt.xlabel('Time Step')
#                     plt.ylabel('Close Price')
#                     plt.legend()
#                     train_plot_file = os.path.join(crypto_dir, f'{model_type}_train_results.png')
#                     plt.savefig(train_plot_file)
#                     plt.close()
#                     print(f"Training results plot saved: {train_plot_file}")
#
#                     # 测试结果图
#                     plt.figure(figsize=(15, 6))
#                     plt.plot(test_result, label='Test Prediction')
#                     plt.plot(test_actual, label='Actual Value')
#                     plt.title(f'{model_type} Test Results - {crypto_name}')
#                     plt.xlabel('Time Step')
#                     plt.ylabel('Close Price')
#                     plt.legend()
#                     test_plot_file = os.path.join(crypto_dir, f'{model_type}_test_results.png')
#                     plt.savefig(test_plot_file)
#                     plt.close()
#                     print(f"Test results plot saved: {test_plot_file}")
#
#             except Exception as e:
#                 print(f"Error processing model {model_type}: {str(e)}")
#                 continue
#
#             finally:
#                 # 清理资源
#                 tf.keras.backend.clear_session()
#
#         # 保存该币种的性能数据
#         performance_df = pd.DataFrame(crypto_performance)
#         performance_file = os.path.join(crypto_dir, f"{crypto_name}_performance.csv")
#         performance_df.to_csv(performance_file, index=False)
#         print(f"Performance data saved to: {performance_file}")
#
#         # 可视化该币种的模型性能比较
#         plt.figure(figsize=(14, 10))
#
#         # MAPE性能比较图
#         plt.subplot(2, 1, 1)
#         for model in model_types:
#             model_data = performance_df[performance_df['model'] == model]
#             if not model_data.empty:
#                 plt.plot(model_data['round'], model_data['avg_mape'], 'o-', label=model)
#         plt.title(f"{crypto_name} - MAPE Performance Comparison")
#         plt.xlabel('Training Round')
#         plt.ylabel('Average Test MAPE (%)')
#         plt.legend()
#         plt.grid(True)
#
#         # RMSE性能比较图
#         plt.subplot(2, 1, 2)
#         for model in model_types:
#             model_data = performance_df[performance_df['model'] == model]
#             if not model_data.empty:
#                 plt.plot(model_data['round'], model_data['avg_rmse'], 'o-', label=model)
#         plt.title(f"{crypto_name} - RMSE Performance Comparison")
#         plt.xlabel('Training Round')
#         plt.ylabel('Average Test RMSE')
#         plt.legend()
#         plt.grid(True)
#
#         plt.tight_layout()
#
#         # 保存性能比较图
#         comparison_file = os.path.join(results_dir, f"{crypto_name}_performance_comparison.png")
#         plt.savefig(comparison_file)
#         plt.close()
#         print(f"Performance comparison plot saved: {comparison_file}")
#
#         # 存储到总结果
#         all_results[crypto_name] = performance_df
#
#     # 保存所有结果
#     summary_file = os.path.join(results_dir, "all_performance.csv")
#     summary_df = pd.concat(all_results.values(), keys=all_results.keys())
#     summary_df.to_csv(summary_file)
#     print(f"All results saved to: {summary_file}")
#
#     print("All cryptocurrencies processed successfully!")

def main(monitor=None):
    # ======== 字体问题终极解决方案 ========
    import matplotlib
    import matplotlib.font_manager as fm

    # 确保使用支持字体的后端
    matplotlib.use('Agg')  # 强制使用无头渲染器
    plt.switch_backend('Agg')  # 确保所有绘图使用正确后端

    # 获取可用字体列表
    font_list = [f.name for f in fm.fontManager.ttflist]

    # 选择最适合的英文无衬线字体
    fallback_fonts = [
        'DejaVu Sans', 'Arial', 'Liberation Sans',
        'Bitstream Vera Sans', 'sans-serif'
    ]

    # 设置优先字体选项
    for font in fallback_fonts:
        if font in font_list:
            plt.rcParams['font.family'] = font
            print(f"✅ 使用字体: {font}")
            if monitor:
                monitor.log_message(f"✅ 使用字体: {font}")
            break
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        print("⚠️ 使用备用无衬线字体")
        if monitor:
            monitor.log_message("⚠️ 使用备用无衬线字体")

    # 确保正确显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置全局字体参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 12

    # ======== 在训练函数中检查字体设置 ========
    print("\n=== 当前字体状态验证 ===")
    print("活动后端:", matplotlib.get_backend())
    print("当前字体:", plt.rcParams['font.family'])
    print("可用字体示例:", font_list[:3])

    if monitor:
        monitor.log_message("=== 当前字体状态验证 ===")
        monitor.log_message(f"活动后端: {matplotlib.get_backend()}")
        monitor.log_message(f"当前字体: {plt.rcParams['font.family']}")
        monitor.log_message(f"可用字体示例: {font_list[:3]}")

    print("Starting enhanced cryptocurrency prediction system...")
    if monitor:
        monitor.log_message("Starting enhanced cryptocurrency prediction system...")

    # 设置全局随机种子
    tf.random.set_seed(917304)
    np.random.seed(917304)
    random.seed(917304)
    os.environ['PYTHONHASHSEED'] = '917304'

    # ----------------- 参数设置 -----------------
    crypto_files = [
        "data/coin_Bitcoin.csv",
        "data/coin_Dogecoin.csv",
        "data/coin_Ethereum.csv",
        "data/coin_Litecoin.csv"
    ]

    model_types = [
        'LSTM',
        'BD LSTM',
        'ED LSTM',
        'CNN',
        'Convolutional LSTM',
        'Transformer',
        'MLP',
        'MacroFusion'
    ]

    # 存储所有结果的字典
    all_results = {}

    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    if monitor:
        monitor.log_message(f"创建结果目录: {results_dir}")

    # 循环处理每个加密货币
    for crypto_path in crypto_files:
        crypto_name = os.path.basename(crypto_path).split('.')[0].split('_')[-1]
        print(f"\n{'=' * 50}")
        print(f"Processing cryptocurrency: {crypto_name}")
        print(f"{'=' * 50}")

        if monitor:
            monitor.log_message(f"\n{'=' * 50}")
            monitor.log_message(f"Processing cryptocurrency: {crypto_name}")
            monitor.log_message(f"{'=' * 50}")

        # 创建该币种的目录
        crypto_dir = os.path.join(results_dir, crypto_name)
        os.makedirs(crypto_dir, exist_ok=True)
        if monitor:
            monitor.log_message(f"创建币种目录: {crypto_dir}")

        # 存储该币种的性能数据
        crypto_performance = []

        # 循环处理每个模型类型
        for model_type in model_types:
            print(f"\n{'*' * 40}")
            print(f"Training model: {model_type}")
            print(f"{'*' * 40}")

            if monitor:
                monitor.log_message(f"\n{'*' * 40}")
                monitor.log_message(f"Training model: {model_type}")
                monitor.log_message(f"{'*' * 40}")

            # 参数设置
            dim_type = 'Multi'
            use_percentage = 1
            include_macro = True
            n_steps_in = 7
            n_steps_out = 5
            percentage = 0.8
            epochs = 50
            rounds = 3
            split_type = 0

            # 为每个模型创建独立的数据容器
            model_results = []  # 每个模型的详细结果
            model_performance = []  # 每个模型的性能指标

            try:
                # ----------------- 数据处理 -----------------
                print("Loading data and macroeconomic indicators...")
                if monitor:
                    monitor.log_message("Loading data and macroeconomic indicators...")

                data, data_len, feature_names = read_data(crypto_path, dim_type, use_percentage, include_macro)
                if data is None:
                    print("Data processing failed, skipping")
                    if monitor:
                        monitor.log_message("Data processing failed, skipping")
                    continue

                print(f"Data loaded successfully. Shape: {data.shape}")
                print(f"Model type: {model_type}")
                print(f"Feature names: {feature_names}")

                if monitor:
                    monitor.log_message(f"Data loaded successfully. Shape: {data.shape}")
                    monitor.log_message(f"Model type: {model_type}")
                    monitor.log_message(f"Feature names: {feature_names}")

                # 归一化数据
                print("Normalizing data...")
                if monitor:
                    monitor.log_message("Normalizing data...")

                data, scalers = data_trasform(data, feature_names=feature_names)
                if data is None:
                    print("Data normalization failed, skipping")
                    if monitor:
                        monitor.log_message("Data normalization failed, skipping")
                    continue

                # 分割训练集和测试集
                print("Splitting dataset...")
                if monitor:
                    monitor.log_message("Splitting dataset...")

                train_set = data[0:int(np.floor(data_len * percentage))]
                test_set = data[int(np.floor(data_len * percentage)):]

                # 确保训练/测试集大小合理
                min_samples = n_steps_in + n_steps_out + 1
                if len(train_set) < min_samples:
                    print(f"Training set too small ({len(train_set)} samples). Need at least {min_samples} samples")
                    if monitor:
                        monitor.log_message(
                            f"Training set too small ({len(train_set)} samples). Need at least {min_samples} samples")
                    continue
                if len(test_set) < min_samples:
                    print(f"Test set too small ({len(test_set)} samples). Need at least {min_samples} samples")
                    if monitor:
                        monitor.log_message(
                            f"Test set too small ({len(test_set)} samples). Need at least {min_samples} samples")
                    continue

                print(f"Training set shape: {train_set.shape}, Test set shape: {test_set.shape}")
                if monitor:
                    monitor.log_message(f"Training set shape: {train_set.shape}, Test set shape: {test_set.shape}")

                # 定义宏观经济特征列表
                macro_features = ['vix', 'sp500', 'fed_rate', 'treasury_yield', 'dxy', 'cpi']

                # 计算动态权重
                dynamic_weights = None
                if include_macro:
                    print("Calculating dynamic feature weights...")
                    if monitor:
                        monitor.log_message("Calculating dynamic feature weights...")
                    try:
                        # 获取完整数据用于权重计算
                        full_df = pd.read_csv(crypto_path)
                        full_df["Date"] = pd.to_datetime(full_df["Date"], utc=True)

                        # 获取宏观经济数据
                        macro_df = fetch_macro_data(
                            start_date=full_df['Date'].min().strftime('%Y-%m-%d'),
                            end_date=full_df['Date'].max().strftime('%Y-%m-%d')
                        )
                        macro_df = macro_df.reset_index().rename(columns={'index': 'Date'})

                        # 合并数据
                        full_df = full_df.merge(macro_df, on='Date', how='left')

                        # 应用特征工程
                        full_df = enhanced_feature_engineering(full_df)

                        # 计算动态权重
                        dynamic_weights = calculate_dynamic_weights(full_df, crypto_name)

                        # 提取宏观特征权重
                        train_weights = dynamic_weights[macro_features].mean(axis=1).iloc[:len(train_set)].values
                    except Exception as e:
                        print(f"Dynamic weight calculation failed: {str(e)}")
                        if monitor:
                            monitor.log_message(f"Dynamic weight calculation failed: {str(e)}")
                        train_weights = None
                else:
                    train_weights = None

                # 确定特征数量
                n_features = train_set.shape[1]
                n_macro_features = len(macro_features) if include_macro else 0
                print(f"Number of features: {n_features}, Macro features: {n_macro_features}")
                if monitor:
                    monitor.log_message(f"Number of features: {n_features}, Macro features: {n_macro_features}")

                # 添加关键修复：获取目标列索引
                target_index = 0
                if feature_names:
                    if 'Close' in feature_names:
                        target_index = feature_names.index('Close')
                        print(f"Target column 'Close' index: {target_index}")
                        if monitor:
                            monitor.log_message(f"Target column 'Close' index: {target_index}")
                    else:
                        # 尝试自动查找
                        found = False
                        for i, name in enumerate(feature_names):
                            if 'close' in name.lower():
                                target_index = i
                                print(f"Found similar column '{name}' as target, index={target_index}")
                                if monitor:
                                    monitor.log_message(
                                        f"Found similar column '{name}' as target, index={target_index}")
                                found = True
                                break

                        if not found:
                            print(f"Warning: Unable to determine target column, using default index 0")
                            if monitor:
                                monitor.log_message("Warning: Unable to determine target column, using default index 0")
                else:
                    print("Warning: Missing feature names, using default target index 0")
                    if monitor:
                        monitor.log_message("Warning: Missing feature names, using default target index 0")

                # ----------------- 模型训练与评估 -----------------
                best_model = None
                best_test_mape = float('inf')
                best_round_idx = -1

                # ======== 关键修复：修改训练循环 ========
                for round_idx in range(rounds):
                    print(f"\nStarting round {round_idx + 1}/{rounds}...")
                    if monitor:
                        monitor.log_message(f"\nStarting round {round_idx + 1}/{rounds}...")
                        monitor.set_model_info(crypto_name, model_type, epochs)

                    # 每轮前清除计算图
                    tf.keras.backend.clear_session()

                    # 生成本轮唯一种子 (基础种子+轮次偏移)
                    round_seed = 917304 + round_idx * 100

                    try:
                        model = create_model(model_type, n_features, n_steps_in, n_steps_out, n_macro_features)
                        print(f"Created {model_type} model...")
                        if monitor:
                            monitor.log_message(f"Created {model_type} model...")
                            # 打印模型摘要
                            model.summary(print_fn=lambda x: monitor.log_message(x) if monitor else None)

                    except Exception as e:
                        print(f"Model creation failed: {str(e)}")
                        if monitor:
                            monitor.log_message(f"Model creation failed: {str(e)}")
                        continue

                    try:
                        # 传递本轮种子到训练函数
                        train_result, test_result, train_actual, test_actual = train_and_forecast(
                            model, n_features, train_set, test_set,
                            n_steps_in, n_steps_out, epochs,
                            dynamic_weights=train_weights,
                            n_macro_features=n_macro_features,
                            scalers=scalers,
                            target_index=target_index,
                            feature_names=feature_names,
                            round_seed=round_seed,  # 新增参数
                            monitor=monitor  # 传递监控对象
                        )

                        if train_result is None or test_result is None:
                            print("Training failed, skipping round")
                            if monitor:
                                monitor.log_message("Training failed, skipping round")
                            continue
                    except Exception as e:
                        print(f"Training failed: {str(e)}")
                        if monitor:
                            monitor.log_message(f"Training failed: {str(e)}")
                        continue

                    # 计算评估指标
                    print("Calculating metrics...")
                    if monitor:
                        monitor.log_message("Calculating metrics...")
                    try:
                        train_rmse = eval_result(train_result, train_actual, 0)
                        test_rmse = eval_result(test_result, test_actual, 0)
                        train_mape = eval_result(train_result, train_actual, 1)
                        test_mape = eval_result(test_result, test_actual, 1)
                    except Exception as e:
                        print(f"Evaluation failed: {str(e)}")
                        if monitor:
                            monitor.log_message(f"Evaluation failed: {str(e)}")
                        continue

                    # 计算平均MAPE和RMSE
                    avg_train_mape = np.mean([np.mean(m) for m in train_mape]) if isinstance(train_mape,
                                                                                             list) else train_mape
                    avg_test_mape = np.mean([np.mean(m) for m in test_mape]) if isinstance(test_mape,
                                                                                           list) else test_mape
                    avg_train_rmse = np.mean([np.mean(m) for m in train_rmse]) if isinstance(train_rmse,
                                                                                             list) else train_rmse
                    avg_test_rmse = np.mean([np.mean(m) for m in test_rmse]) if isinstance(test_rmse,
                                                                                           list) else test_rmse

                    # 打印指标
                    print(f'Training RMSE: {train_rmse}')
                    print(f'Testing RMSE: {test_rmse}')
                    print(f'Training MAPE: {train_mape}')
                    print(f'Testing MAPE: {test_mape}')
                    print(f'Average Test MAPE: {avg_test_mape:.4f}%')
                    print(f'Average Test RMSE: {avg_test_rmse:.4f}')

                    if monitor:
                        monitor.log_message(f'Training RMSE: {train_rmse}')
                        monitor.log_message(f'Testing RMSE: {test_rmse}')
                        monitor.log_message(f'Training MAPE: {train_mape}')
                        monitor.log_message(f'Testing MAPE: {test_mape}')
                        monitor.log_message(f'Average Test MAPE: {avg_test_mape:.4f}%')
                        monitor.log_message(f'Average Test RMSE: {avg_test_rmse:.4f}')

                    # 记录本轮详细结果
                    round_result = {
                        'round': round_idx + 1,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_mape': train_mape,
                        'test_mape': test_mape,
                        'avg_train_rmse': avg_train_rmse,
                        'avg_test_rmse': avg_test_rmse,
                        'avg_train_mape': avg_train_mape,
                        'avg_test_mape': avg_test_mape,
                        'train_predictions': train_result.tolist() if train_result is not None else [],
                        'test_predictions': test_result.tolist() if test_result is not None else [],
                        'train_actual': train_actual.tolist() if train_actual is not None else [],
                        'test_actual': test_actual.tolist() if test_actual is not None else []
                    }

                    # 添加到模型结果列表
                    model_results.append(round_result)

                    # 添加到性能指标列表
                    model_performance.append({
                        'round': round_idx + 1,
                        'avg_mape': avg_test_mape,
                        'avg_rmse': avg_test_rmse
                    })

                    # 添加到加密货币级别的性能列表
                    crypto_performance.append({
                        'model': model_type,
                        'round': round_idx + 1,
                        'avg_mape': avg_test_mape,
                        'avg_rmse': avg_test_rmse
                    })

                    # 检查是否为最佳模型
                    if avg_test_mape < best_test_mape:
                        best_test_mape = avg_test_mape
                        best_model = model
                        best_round_idx = round_idx + 1
                        best_train_result = train_result
                        best_test_result = test_result
                        best_train_actual = train_actual
                        best_test_actual = test_actual

                # ==================== 保存模型结果 ====================
                # 保存该模型的详细结果
                model_results_df = pd.DataFrame(model_results)
                model_results_file = os.path.join(crypto_dir, f"{model_type}_detailed_results.csv")
                model_results_df.to_csv(model_results_file, index=False)
                print(f"{model_type} detailed results saved to: {model_results_file}")
                if monitor:
                    monitor.log_message(f"{model_type} detailed results saved to: {model_results_file}")

                # 保存该模型的性能指标
                model_perf_df = pd.DataFrame(model_performance)
                model_perf_file = os.path.join(crypto_dir, f"{model_type}_performance.csv")
                model_perf_df.to_csv(model_perf_file, index=False)
                print(f"{model_type} performance metrics saved to: {model_perf_file}")
                if monitor:
                    monitor.log_message(f"{model_type} performance metrics saved to: {model_perf_file}")

                # 保存训练过程的图片
                loss_file = os.path.join(crypto_dir, f"{model_type}_training_loss.png")
                if os.path.exists('training_loss.png'):
                    os.rename('training_loss.png', loss_file)
                    print(f"Training loss plot saved to: {loss_file}")
                    if monitor:
                        monitor.log_message(f"Training loss plot saved to: {loss_file}")

                # 保存预测结果图片
                if n_steps_out == 1 and best_test_result is not None and best_test_actual is not None:
                    # 训练结果图
                    plt.figure(figsize=(15, 6))
                    plt.plot(best_train_result, label='Train Prediction')
                    plt.plot(best_train_actual, label='Actual Value')
                    plt.title(f'{model_type} Training Results - {crypto_name} (Best Round: {best_round_idx})')
                    plt.xlabel('Time Step')
                    plt.ylabel('Close Price')
                    plt.legend()
                    train_plot_file = os.path.join(crypto_dir, f'{model_type}_train_results.png')
                    plt.savefig(train_plot_file)
                    plt.close()
                    print(f"Training results plot saved: {train_plot_file}")
                    if monitor:
                        monitor.log_message(f"Training results plot saved: {train_plot_file}")

                    # 测试结果图
                    plt.figure(figsize=(15, 6))
                    plt.plot(best_test_result, label='Test Prediction')
                    plt.plot(best_test_actual, label='Actual Value')
                    plt.title(f'{model_type} Test Results - {crypto_name} (Best Round: {best_round_idx})')
                    plt.xlabel('Time Step')
                    plt.ylabel('Close Price')
                    plt.legend()
                    test_plot_file = os.path.join(crypto_dir, f'{model_type}_test_results.png')
                    plt.savefig(test_plot_file)
                    plt.close()
                    print(f"Test results plot saved: {test_plot_file}")
                    if monitor:
                        monitor.log_message(f"Test results plot saved: {test_plot_file}")

            except Exception as e:
                print(f"Error processing model {model_type}: {str(e)}")
                if monitor:
                    monitor.log_message(f"Error processing model {model_type}: {str(e)}")
                continue

            finally:
                # 清理资源
                tf.keras.backend.clear_session()

        # 保存该币种的性能数据
        performance_df = pd.DataFrame(crypto_performance)
        performance_file = os.path.join(crypto_dir, f"{crypto_name}_performance.csv")
        performance_df.to_csv(performance_file, index=False)
        print(f"Performance data saved to: {performance_file}")
        if monitor:
            monitor.log_message(f"Performance data saved to: {performance_file}")

        # 可视化该币种的模型性能比较
        plt.figure(figsize=(14, 10))

        # MAPE性能比较图
        plt.subplot(2, 1, 1)
        for model in model_types:
            model_data = performance_df[performance_df['model'] == model]
            if not model_data.empty:
                plt.plot(model_data['round'], model_data['avg_mape'], 'o-', label=model)
        plt.title(f"{crypto_name} - MAPE Performance Comparison")
        plt.xlabel('Training Round')
        plt.ylabel('Average Test MAPE (%)')
        plt.legend()
        plt.grid(True)

        # RMSE性能比较图
        plt.subplot(2, 1, 2)
        for model in model_types:
            model_data = performance_df[performance_df['model'] == model]
            if not model_data.empty:
                plt.plot(model_data['round'], model_data['avg_rmse'], 'o-', label=model)
        plt.title(f"{crypto_name} - RMSE Performance Comparison")
        plt.xlabel('Training Round')
        plt.ylabel('Average Test RMSE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 保存性能比较图
        comparison_file = os.path.join(crypto_dir, f"{crypto_name}_performance_comparison.png")
        plt.savefig(comparison_file)
        plt.close()
        print(f"Performance comparison plot saved: {comparison_file}")
        if monitor:
            monitor.log_message(f"Performance comparison plot saved: {comparison_file}")

        # 存储到总结果
        all_results[crypto_name] = performance_df

    # 保存所有结果
    summary_file = os.path.join(results_dir, "all_performance.csv")
    summary_df = pd.concat(all_results.values(), keys=all_results.keys())
    summary_df.to_csv(summary_file)
    print(f"All results saved to: {summary_file}")
    if monitor:
        monitor.log_message(f"All results saved to: {summary_file}")

    print("All cryptocurrencies processed successfully!")
    if monitor:
        monitor.log_message("All cryptocurrencies processed successfully!")



if __name__ == '__main__':
    # 测试关联分析函数
    print("测试关联分析功能...")
    correlations = calculate_correlations('1y')
    print("比特币关联分析结果:", correlations.get('Bitcoin', {}))

    historical = calculate_historical_correlations('Bitcoin', 'vix', '1y')
    print("历史关联分析结果样本:", historical[:3] if historical else "无数据")
    main()

