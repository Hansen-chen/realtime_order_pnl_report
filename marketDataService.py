# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""
from datetime import datetime as dt
import time
import random
import os
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels
import datetime
import pandas as pd

class MarketDataService:

    def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q):
        print("[%d]<<<<< call MarketDataService.init" % (os.getpid(),))
        time.sleep(3)
        self.future2stock = {'JBF': 3443, 'QWF': 2388, 'HCF': 2498, 'DBF': 2610, 'EHF': 1319, 'IPF': 3035, 'IIF': 3006, 'QXF': 2615, 'PEF': 5425, 'NAF': 3105}
        self.future_path = './processedData_2023/futures/'
        self.stock_path = './processedData_2023/stocks/'
        self.suffix = '_md_202306_202306.csv.gz'
        self.counter = 0
        # Get all the future tickers
        self.future_data = {}
        self.stock_data = {}
        for future_ticker, stock_ticker in self.future2stock.items():
            self.future_data[future_ticker] = pd.DataFrame()
            self.stock_data[stock_ticker] = pd.DataFrame()

            tmp = pd.read_csv(self.future_path + future_ticker + "1"+self.suffix , compression='gzip')
            tmp = tmp[tmp['askPrice1'] > 0]
            tmp = tmp[tmp['bidPrice1'] > 0]
            tmp = tmp[tmp['askPrice1'] > tmp['bidPrice1']]
            self.future_data[future_ticker] = pd.concat([self.future_data[future_ticker], tmp])


            tmp = pd.read_csv(self.stock_path + str(stock_ticker) + self.suffix, compression='gzip')
            tmp = tmp[tmp['SP1'] > 0]
            tmp = tmp[tmp['BP1'] > 0]
            tmp = tmp[tmp['SP1'] > tmp['BP1']]
            self.stock_data[stock_ticker] = pd.concat([self.stock_data[stock_ticker], tmp])
        self.produce_market_data(marketData_2_exchSim_q, marketData_2_platform_q)

    def produce_market_data(self, marketData_2_exchSim_q, marketData_2_platform_q):
        for i in range(500):
            self.produce_quote(marketData_2_exchSim_q, marketData_2_platform_q)
            time.sleep(1)

    def produce_quote(self, marketData_2_exchSim_q, marketData_2_platform_q):


        for future_ticker, stock_ticker in self.future2stock.items():

            if self.counter >= len(self.future_data[future_ticker]):
                break

            bidPrice, askPrice, bidSize, askSize = [], [], [], []
            tmp = self.future_data[future_ticker].iloc[self.counter]
            bidPrice.extend(tmp[['bidPrice1','bidPrice2','bidPrice3','bidPrice4','bidPrice5']].values.tolist())
            askPrice.extend(tmp[['askPrice1','askPrice2','askPrice3','askPrice4','askPrice5']].values.tolist())
            bidSize.extend(tmp[['bidSize1','bidSize2','bidSize3','bidSize4','bidSize5']].values.tolist())
            askSize.extend(tmp[['askSize1','askSize2','askSize3','askSize4','askSize5']].values.tolist())
            quoteSnapshot = OrderBookSnapshot_FiveLevels(future_ticker,tmp['date'] ,
                                                          dt.fromtimestamp(tmp['time']), bidPrice, askPrice, bidSize, askSize)
            print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
            print(quoteSnapshot.outputAsDataFrame())
            marketData_2_exchSim_q.put(quoteSnapshot)
            marketData_2_platform_q.put(quoteSnapshot)

            bidPrice, askPrice, bidSize, askSize = [], [], [], []
            tmp = self.stock_data[stock_ticker].iloc[self.counter]
            bidPrice.extend(tmp[['BP1','BP2','BP3','BP4','BP5']].values.tolist())
            askPrice.extend(tmp[['SP1','SP2','SP3','SP4','SP5']].values.tolist())
            bidSize.extend(tmp[['BV1','BV2','BV3','BV4','BV5']].values.tolist())
            askSize.extend(tmp[['SV1','SV2','SV3','SV4','SV5']].values.tolist())
            quoteSnapshot = OrderBookSnapshot_FiveLevels(str(stock_ticker),tmp['date'] ,
                                                         dt.fromtimestamp(tmp['time']), bidPrice, askPrice, bidSize, askSize)
            print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
            print(quoteSnapshot.outputAsDataFrame())
            marketData_2_exchSim_q.put(quoteSnapshot)
            marketData_2_platform_q.put(quoteSnapshot)

            time.sleep(1)
            self.counter += 1