import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

import datetime

gold_df = pd.read_csv('/Users/precious/Desktop/test/data/LBMA-GOLD.csv')
bitcoin_df = pd.read_csv('/Users/precious/Desktop/test/data/BCHAIN-MKPRU.csv')

combination_df = pd.merge(bitcoin_df, gold_df, on=['Date'], how='outer')

# 假设第2天交易者进行如下分配资产
crash = 500.0
gold = 250.0 / 1324.6
bitcoin = 250.0 / 609.67
holdings = [crash, gold, bitcoin]

crash_all = 1000.0
beginning = 10

crash_rolling = pd.read_csv('/Users/precious/Desktop/test/data/BCHAIN-MKPRU.csv')
crash_rolling.drop(labels='Value',axis=1,inplace=True)
for i in range(beginning):
        crash_rolling.loc[i].at['crash_all'] = crash_all

# i是交易的时间（days）
for i in range(beginning, 1825):

    print('The ', i, 'th has begun !')
    print()

    # combination_df_temp = combination_df.fillna(value=0, inplace=True)
    if combination_df.iloc[i].at['USD (PM)'] == 0:
        continue

    gold_df = pd.concat([pd.DataFrame(combination_df[:i]['Date']),
                         pd.DataFrame(combination_df[:i]['USD (PM)'])], axis=1, join='outer')
    bitcoin_df = pd.concat([pd.DataFrame(combination_df[:i]['Date']),
                            pd.DataFrame(combination_df[:i]['Value'])], axis=1, join='outer')
    gold_df = gold_df.dropna()
    bitcoin_df = bitcoin_df.dropna()

    # 获得移动平均值
    gold_mavg = gold_df['USD (PM)'].shift(1).rolling(window=i+1).mean()

    bitcoin_mavg = bitcoin_df['Value'].shift(1).rolling(window=i+1).mean()

    gold_df = gold_df.dropna()
    bitcoin_df = bitcoin_df.dropna()

    # 收益率
    earnings_gold = gold_df['USD (PM)'] / gold_df['USD (PM)'].shift(1) - 1
    earnings_bit = bitcoin_df['Value'] / bitcoin_df['Value'].shift(1) - 1

    dfreg_gold = gold_df.loc[:, ['USD (PM)']]
    dfreg_gold['PCT_change'] = gold_df['USD (PM)'].pct_change() * 10000
    for j in list(gold_df.index):
        dfreg_gold.loc[j, 'HL_PCT_gold'] = float(dfreg_gold[['USD (PM)']].max(
        )) - float(dfreg_gold[['USD (PM)']].min()) / dfreg_gold.loc[j, 'USD (PM)'] * 1000

    dfreg_bit = bitcoin_df.loc[:, ['Value']]
    dfreg_bit['PCT_change'] = bitcoin_df['Value'].pct_change() * 100000
    for j in list(bitcoin_df.index):
        dfreg_bit.loc[j, 'HL_PCT_bit'] = float(dfreg_bit[['Value']].max(
        )) - float(dfreg_bit[['Value']].min()) / dfreg_bit.loc[j, 'Value'] * 100000

    dfreg_gold['Date'] = combination_df[['Date']]
    dfreg_bit['Date'] = combination_df[['Date']]

    # dfreg_gold.plot()

    # 去掉空值的影响

    # Drop missing value
    dfreg_gold.fillna(value=-99999, inplace=True)
    dfreg_bit.fillna(value=-99999, inplace=True)

    # 分离数据（training：testing = 9:1）
    # We want to separate 10 percent of the data to forecast
    forecast_out_gold = int(math.ceil(0.1 * len(dfreg_gold)))
    forecast_out_bit = int(math.ceil(0.1 * len(dfreg_bit)))

    forecast_col_gold = 'USD (PM)'
    forecast_col_bit = 'Value'

    dfreg_gold['label'] = dfreg_gold[forecast_col_gold].shift(
        -forecast_out_gold)
    dfreg_bit['label'] = dfreg_bit[forecast_col_bit].shift(-forecast_out_bit)

    X_gold = np.array(dfreg_gold.drop(['label', 'Date'], 1))
    X_bit = np.array(dfreg_bit.drop(['label', 'Date'], 1))

    # 训练
    # Scale the X so that everyone can have the same distribution for linear regression
    X_gold = sklearn.preprocessing.scale(X_gold)
    X_bit = sklearn.preprocessing.scale(X_bit)

    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately_gold = X_gold[-forecast_out_gold:]
    X_lately_bit = X_bit[-forecast_out_bit:]

    X_gold = X_gold[:-forecast_out_gold]
    X_bit = X_bit[:-forecast_out_bit]

    # Separate label and identify it as y
    y_gold = np.array(dfreg_gold['label'])
    y_gold = y_gold[:-forecast_out_gold]

    y_bit = np.array(dfreg_bit['label'])
    y_bit = y_bit[:-forecast_out_bit]

    X_gold_test = X_gold[-forecast_out_gold:]
    X_bit_test = X_bit[-forecast_out_bit:]

    y_gold_test = y_gold[-forecast_out_gold:]
    y_bit_test = y_bit[-forecast_out_bit:]

    # Linear regression
    clfreg_gold = LinearRegression(n_jobs=-1)
    clfreg_gold.fit(X_gold, y_gold)

    clfreg_bit = LinearRegression(n_jobs=-1)
    clfreg_bit.fit(X_bit, y_bit)

    # Quadratic Regression 2
    clfpoly2_gold = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2_gold.fit(X_gold, y_gold)

    clfpoly2_bit = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2_bit.fit(X_bit, y_bit)

    # Quadratic Regression 3
    clfpoly3_gold = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3_gold.fit(X_gold, y_gold)

    clfpoly3_bit = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3_bit.fit(X_bit, y_bit)

    clfknn_gold = KNeighborsRegressor(n_neighbors=2)
    clfknn_gold.fit(X_gold, y_gold)

    clfknn_bit = KNeighborsRegressor(n_neighbors=2)
    clfknn_bit.fit(X_bit, y_bit)

    # 预测后day天价格

    days = 15

    last_date_gold = pd.to_datetime(gold_df['Date'])

    last_unix_gold = last_date_gold

    next_unix_gold = last_unix_gold + datetime.timedelta(days=1)

    for j in range(1, days):

        next_date_gold = next_unix_gold
        next_unix_gold += datetime.timedelta(days=1)
        dfreg_gold.loc['Date'] = [
            np.nan for _ in range(len(dfreg_gold.columns)-1)]+[j]

    dfreg_gold = dfreg_gold.drop('Date')
    # ax = dfreg_gold['USD (PM)'].tail(500).plot()
    # dfreg_gold[['USD (PM)']][len(dfreg_gold)-15:].tail(500).plot(ax=ax)
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('USD (PM)')
    # plt.show()

    bitcoin_df = combination_df.dropna()
    bitcoin_df = bitcoin_df.drop('USD (PM)', 1)
    bitcoin_df = bitcoin_df.reset_index()
    bitcoin_df = bitcoin_df.drop('index', 1)
    dfreg_bit = bitcoin_df

    last_date_bit = pd.to_datetime(bitcoin_df['Date'])

    last_unix_bit = last_date_bit

    next_unix_bit = last_unix_bit + datetime.timedelta(days=1)

    for j in range(1, days):

        next_date_bit = next_unix_bit
        next_unix_bit += datetime.timedelta(days=1)
        dfreg_bit.loc['Date'] = [
            np.nan for _ in range(len(dfreg_bit.columns)-1)]+[j]

    dfreg_bit = dfreg_bit.drop('Date')
    # ax = dfreg_bit['Value'].tail(500).plot()
    # dfreg_bit[['Value']][len(dfreg_bit)-15:].tail(500).plot(ax=ax)
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.show()

    combination_df_temp = combination_df.fillna(value=0.0)

    if combination_df_temp.iloc[i].at['USD (PM)'] == 0.0:
        continue

    # 计算每日变化

    avarage_gold = dfreg_gold[['USD (PM)']][len(dfreg_gold)-days:].mean()
    avarage_bit = dfreg_bit[['Value']][len(dfreg_bit)-days:].mean()

    avarage_gold = avarage_gold['USD (PM)']
    avarage_bit = avarage_bit['Value']

    # Define the logistic function

    def logistic(z):

        return 1 / (1 + np.exp(-z))

    # 增加价格变化权重(Logistics)
    # delta = avarage_gold + avarage_bit
    # avarage_gold = logistic(
    #     avarage_gold - combination_df.iloc[i].at['USD (PM)'])
    # avarage_bit = logistic(avarage_bit - combination_df.iloc[i].at['Value'])

    priceOfGold = combination_df.iloc[i].at['USD (PM)']
    priceOfBit = combination_df.iloc[i].at['Value']

    valueOfDallor = crash
    valueOfGold = gold * priceOfGold
    valueOfBit = bitcoin * priceOfBit

    priceOfGold = combination_df.iloc[i].at['USD (PM)']
    priceOfBit = combination_df.iloc[i].at['Value']

    # y2,y2是量
    # present_valueOfDallor = ((priceOfGold * priceOfBit * (valueOfDallor + valueOfGold + valueOfBit)) - (0.01 * valueOfGold * priceOfBit + 0.02 * valueOfBit * priceOfGold)) / (
    #     ((avarage_bit * priceOfBit + avarage_gold * priceOfGold + 1) * priceOfGold * priceOfBit) - (avarage_bit * priceOfBit * priceOfGold + avarage_gold * priceOfGold * priceOfBit))

    # y2,y3 is value
    present_valueOfDallor = ((priceOfGold * priceOfBit * (valueOfDallor + valueOfGold + valueOfBit)) - (0.01 * valueOfGold * priceOfBit + 0.02 * valueOfBit * priceOfGold)) / (
        ((avarage_bit + avarage_gold + 1) * priceOfGold * priceOfBit) - (avarage_bit * priceOfBit + avarage_gold * priceOfGold))

    # print(present_valueOfDallor)
    # input()

    # y2,y3是量
    # presnet_valueOfGold = avarage_gold * present_valueOfDallor * priceOfGold
    # present_valueOfBit = avarage_bit * present_valueOfDallor * priceOfBit

    # y2,y3 is value
    presnet_valueOfGold = avarage_gold * present_valueOfDallor
    present_valueOfBit = avarage_bit * present_valueOfDallor

    avarage_gold = dfreg_gold[['USD (PM)']][len(dfreg_gold)-days:].mean()
    avarage_bit = dfreg_bit[['Value']][len(dfreg_bit)-days:].mean()

    avarage_gold = avarage_gold['USD (PM)']
    avarage_bit = avarage_bit['Value']

    # if crash_all > present_valueOfDallor + avarage_gold * gold + avarage_bit * bitcoin:
    #     continue

    # 每日持有总价值（美元）

    crash_all = present_valueOfDallor + presnet_valueOfGold + present_valueOfBit

    crash = present_valueOfDallor
    gold = presnet_valueOfGold / combination_df.iloc[i].at['USD (PM)']
    bitcoin = present_valueOfBit / combination_df.iloc[i].at['Value']

    print('You have ', crash_all, ' !')
    print()

    crash_rolling.iloc[i].at['crash_all'] = crash_all

    # combination_df = combination_df_temp

holdings = [crash, gold, bitcoin]
print(holdings)

crash_rolling.plot()
plt.xlabel('Date')
plt.ylabel('crash_all')