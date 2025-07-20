import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA

plt.switch_backend('Agg')
plt.rcParams['axes.unicode_minus'] = False

file_path = r"./Financial sentiment analysis data/coin_Litecoin.csv"
data_coin = pd.read_csv(file_path).set_index('Date')
data_coin.index = pd.to_datetime(data_coin.index)
data_coin = data_coin.asfreq('D')
data_coin['year'] = data_coin.index.year
data_coin['month'] = data_coin.index.month
data_coin['day'] = data_coin.index.day
data_coin = data_coin.drop(columns=['SNo', 'Name', 'Symbol', 'Volume'])
print(data_coin)


# Bitcoin: p=1; Dogecoin: p=1; Litecoin: p=1; Ethereum: p=1
df = data_coin['Close']
stationarityTest_Close = adfuller(df, autolag='AIC')
print("P-value (Raw Data Stationarity Test): ", stationarityTest_Close[1])

fig_acf = plt.figure(figsize=(10, 4))
plot_acf(df, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.tight_layout()
plt.savefig('./arima/acf.png')
plt.close(fig_acf)

fig_pacf = plt.figure(figsize=(10, 4))
plot_pacf(df, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.tight_layout()
plt.savefig('./arima/pacf.png')
plt.close(fig_pacf)


# Bitcoin: d=1; Dogecoin: d=1; Litecoin: d=1; Ethereum: d=1
df_diff = df.diff().dropna()
stationarityTest_diff = adfuller(df_diff, autolag='AIC')
print("\nP-value (Differenced Data Stationarity Test): ", stationarityTest_diff[1])

fig_acf_diff = plt.figure(figsize=(12, 4))
plot_acf(df_diff, ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function (ACF) - Differenced Data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.tight_layout()
plt.savefig('./arima/acf_diff.png')
plt.close(fig_acf_diff)

fig_pacf_diff = plt.figure(figsize=(12, 4))
plot_pacf(df_diff, ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function (PACF) - Differenced Data')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.tight_layout()
plt.savefig('./arima/pacf_diff.png')
plt.close(fig_pacf_diff)


# Bitcoin: q=2; Dogecoin: q=5+; Litecoin: q=4; Ethereum: q=5+
best_q = None
min_aic = float('inf')
for q_candidate in range(0, 6):
    try:
        model = ARIMA(df, order=(2, 1, q_candidate))
        results = model.fit()
        print(f"ARIMA(1,1,{q_candidate}) - AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")

        if results.aic < min_aic:
            min_aic = results.aic
            best_q = q_candidate
    except Exception as e:
        print(f"Failed to fit ARIMA(1,1,{q_candidate}): {e}")

print(f"\nBased on the minimum AIC principle, the optimal MA order q = {best_q}")