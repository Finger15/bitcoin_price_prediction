import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import os

def data_reader():
    stock_file_name = 'bitcoin_price_1hours.csv'
    encoding = 'euc-kr'
    raw_dataframe = pd.read_csv(stock_file_name, header=0, encoding=encoding)
    raw_dataframe.info()
    return raw_dataframe

def min_max_scaling(x, d_min, d_max):
    re_x = x.values.astype(np.float)

    data_min = np.array(d_min)
    data_max = np.array(d_max)
    data_min = data_min.reshape(len(data_min), 1)
    data_max = data_max.reshape(len(data_max), 1)
    reg_data = (re_x - data_min) / (data_max - data_min + 1e-7)

    return reg_data, data_min, data_max

def fnMA(m_DF):
    m_DF['MA5'] = m_DF['closing_price'].rolling(window=5).mean()
    m_DF['MA20'] = m_DF['closing_price'].rolling(window=20).mean()
    m_DF['MA60'] = m_DF['closing_price'].rolling(window=60).mean()
    m_DF['MA120'] = m_DF['closing_price'].rolling(window=120).mean()
    return m_DF


def fnStoch(m_Df, n=10):
    sz = len(m_Df['closing_price'])
    if sz < n:
        raise SystemExit('입력값이 기간보다 작음')
    tempSto_K = []
    for i in range(sz):
        if i >= n - 1:
            tempUp = m_Df['closing_price'][i] - min(m_Df['low_price'][i - n + 1:i + 1])
            tempDown = max(m_Df['high_price'][i - n + 1:i + 1]) - min(
                m_Df['low_price'][i - n + 1:i + 1])
            tempSto_K.append(tempUp / tempDown)
        else:
            tempSto_K.append(0)

    m_Df['Sto_K'] = pd.Series(tempSto_K, index=m_Df.index)
    m_Df['Sto_D'] = pd.Series(m_Df['Sto_K'].rolling(window=6, center=False).mean())
    m_Df['Sto_SlowD'] = pd.Series(m_Df['Sto_D'].rolling(window=6, center=False).mean())
    return m_Df

def fnMACD(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):

    m_Df['EMAFast'] = m_Df['closing_price'].ewm(span=m_NumFast, min_periods=m_NumFast - 1).mean()
    m_Df['EMASlow'] = m_Df['closing_price'].ewm(span=m_NumSlow, min_periods=m_NumSlow - 1).mean()
    m_Df['MACD'] = m_Df['EMAFast'] - m_Df['EMASlow']
    m_Df['MACDSignal'] = m_Df['MACD'].ewm(span=m_NumSignal, min_periods=m_NumSignal - 1).mean()
    m_Df['MACDDiff'] = m_Df['MACD'] - m_Df['MACDSignal']
    return m_Df

def fnBolingerBand(m_DF, n=20, k=2):

    m_DF['Bol_upper'] = m_DF['MA20'] + k * m_DF['closing_price'].rolling(window=n).std()
    m_DF['Bol_lower'] = m_DF['MA20'] - k * m_DF['closing_price'].rolling(window=n).std()
    return m_DF


def data_reg(data):
    a = []

    price_df = data[['opening_price', 'closing_price', 'high_price', 'low_price',
                     'trade_price',
                     'MA5', 'MA20', 'MA60', 'MA120', 'EMAFast', 'EMASlow', 'Bol_upper', 'Bol_lower']]
    price_df_min = price_df.min(axis=1).values
    price_df_max = price_df.max(axis=1).values
    price_reg_data, price_reg_min, price_reg_max = min_max_scaling(price_df, price_df_min, price_df_max)

    macd_df = data[['MACD', 'MACDSignal', 'MACDDiff']]
    macd_df_min = macd_df.min(axis=1).values
    macd_df_max = macd_df.max(axis=1).values
    macd_reg_data, macd_reg_min, macd_reg_max = min_max_scaling(macd_df, macd_df_min, macd_df_max)

    sto_df = data[['Sto_K', 'Sto_D', 'Sto_SlowD']]
    sto_df_min = sto_df.min(axis=1).values
    sto_df_max = sto_df.max(axis=1).values
    sto_reg_data, sto_reg_min, sto_reg_max = min_max_scaling(sto_df, sto_df_min, sto_df_max)

    volume_df = data[['trade_volume']]
    volume_df_min = volume_df.min().values
    volume_df_max = volume_df.max().values
    volume_reg_data, volume_reg_min, volume_reg_max = min_max_scaling(volume_df, volume_df_min, volume_df_max)

    output_df = data[['output']]
    output_reg_data, output_reg_min, output_reg_max = min_max_scaling(output_df, price_df_min, price_df_max)

    tmp_1 = np.concatenate((price_reg_data, macd_reg_data), axis=1)
    tmp_2 = np.concatenate((tmp_1, sto_reg_data), axis=1)
    data_x = np.concatenate((tmp_2, volume_reg_data), axis=1)

    print(data_x.shape)
    data_y = output_reg_data
    print(data_y.shape)

    return data_x, data_y, price_reg_min, price_reg_max


def data_split(data_x, data_y):
    train_size = int(len(data_y) * 0.9)
    print('tr size : ', train_size)

    trX = np.array(data_x[:train_size])
    trY = np.array(data_y[:train_size])

    vaX = np.array(data_x[train_size:])
    vaY = np.array(data_y[train_size:])

    print('trX shape : ', trX.shape)
    print('trY shape : ', trY.shape)

    return trX, trY, vaX, vaY

def data_preprocessing(raw_dataframe, seq_length):
    fnMA(raw_dataframe)
    fnMACD(raw_dataframe)
    fnBolingerBand(raw_dataframe)
    fnStoch(raw_dataframe)

    output_df = DataFrame(output, columns=['output'])

    r2p = pd.concat([raw_dataframe, output_df], axis=1)
    r2p = r2p.dropna(axis=0)
    r2p.to_csv('preprocessing_data.csv', index=False)

    del r2p['opening_date(kor)']
    del r2p['closing_date(kor)']

    data_x, data_y, reg_min, reg_max = data_reg(r2p)

    dataX = []
    dataY = []
    for i in range(0, len(data_y) - seq_length):
        _x = data_x[i: i + seq_length]
        _y = data_y[i + 1: i + seq_length + 1]
        dataX.append(_x)
        dataY.append(_y)

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    reg_min = reg_min[seq_length:]
    reg_max = reg_max[seq_length:]

    return dataX, dataY, reg_min, reg_max


def batch_iterator(dataX, dataY, batch_size, num_steps):
    data_len = len(dataY)
    epoch_size = int((data_len) / batch_size)

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        input_x = dataX[i * batch_size: (i + 1) * batch_size]
        input_y = dataY[i * batch_size: (i + 1) * batch_size]
        yield (input_x, input_y)


def main():
    raw_data = data_reader()
    seq_time_step = 14
    save_path = 'model\\'
    dataX, dataY, reg_min, reg_max = data_preprocessing(raw_data, seq_time_step)

if __name__ == "__main__":
    main()


