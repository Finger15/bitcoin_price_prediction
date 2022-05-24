import requests
import pandas as pd
from bs4 import BeautifulSoup

coin_list = ['BTC']
time_units = ['days', 'weeks']
minutes_units = [60]


for coin in coin_list:
    '''
    for time_units in time_units:

        req = requests.get(f'https://crix-api-endpoint.upbit.com/v1/crix/candles/{time_units}?code=CRIX.UPBIT.KRW-{coin}&count=100&')
        data = req.json()
        result = []

        for i, candle in enumerate(data):
            result.append({
                'Time' : data[i]["candleDateTimeKst"],
                'OpeningPrice' : data[i]["openingPrice"],
                'HighPrice' : data[i]["highPrice"],
                'LowPrice' : data[i]["lowPrice"],
                'TradePrice' : data[i]["tradePrice"],
                'CandleAccTradeVolume' : data[i]["candleAccTradeVolume"],
                "CandleAccTradePrice" : data[i]["candleAccTradePrice"]
            })

        coin_data = pd.DataFrame(result)
        coin_data.to_csv(f'{coin}_KRW_{time_units}.csv')
    '''

    for minutes_units in minutes_units:
        req = requests.get(f'https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/{minutes_units}?code=CRIX.UPBIT.KRW-{coin}&count=400&to=2021-06-12%2010:00:00') #&to=해당년도날짜&20시간
        data = req.json()
        result = []

        for i, candle in enumerate(data):
            result.append({
                'Time': data[i]["candleDateTimeKst"],
                'OpeningPrice': data[i]["openingPrice"],
                'HighPrice': data[i]["highPrice"],
                'LowPrice': data[i]["lowPrice"],
                'TradePrice': data[i]["tradePrice"],
                'CandleAccTradeVolume': data[i]["candleAccTradeVolume"],
                "CandleAccTradePrice": data[i]["candleAccTradePrice"]
            })

        coin_data = pd.DataFrame(result)
        coin_data.to_csv(f'{coin}_KRW_{minutes_units}min.csv')