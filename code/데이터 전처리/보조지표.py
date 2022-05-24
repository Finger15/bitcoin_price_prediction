import csv
import numpy as np
import talib
import pandas as pd

df = pd.read_csv("./BTC_KRW_60min_17_21.csv") #비트코인 가격데이터가 있는 곳으로 지정
df.head(20)

TradePrice = np.array(df[['TradePrice']])
TradePrice

df['sma5'] = talib.SMA(np.asarray(df['TradePrice']), 5)  #SMA(Simple Moving Average)
df['sma20'] = talib.SMA(np.asarray(df['TradePrice']), 20)
df['sma120'] = talib.SMA(np.asarray(df['TradePrice']), 120)
df['ema12'] = talib.SMA(np.asarray(df['TradePrice']), 12)    #EMA(Exponential Moving Average)
df['ema26'] = talib.SMA(np.asarray(df['TradePrice']), 26)

upper, middle, lower = talib.BBANDS(np.asarray(df['TradePrice']), timeperiod = 20, nbdevup = 2, nbdevdn = 2, matype = 0 )
df['dn'] = lower
df['mavg'] = middle
df['up'] = upper
df['pctB'] = (df.TradePrice - df.dn) / (df.up - df.dn)

rsi14 = talib.RSI(np.asarray(df['TradePrice']), 14)
df['rsi14'] = rsi14

macd, macdsignal, macdhist = talib.MACD(np.asarray(df['TradePrice']), 12, 26, 9)
df['macd'] = macd
df['sigmal'] = macdsignal

df.to_csv('./중복제거최종데이터_전처리전.csv', sep=',', na_rep='NaN')