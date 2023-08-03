# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import time
import random
import os
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels
import datetime

class MarketDataService:

    def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q):
        print("[%d]<<<<< call MarketDataService.init" % (os.getpid(),))
        time.sleep(2)
        self.stock_midquote = {3443: 124905.336,
                               2388: 8040.317,
                               2498: 6002.225,
                               2610: 2375.1636,
                               1319: 5017.527,
                               3035: 18295.475,
                               3006: 8452.317,
                               2615: 6256.2876,
                               5425: 9768.326,
                               3105: 16809.557}

        self.produce_market_data(marketData_2_exchSim_q, marketData_2_platform_q)


    def produce_market_data(self, marketData_2_exchSim_q, marketData_2_platform_q):
        while True:
            self.produce_quote(marketData_2_exchSim_q, marketData_2_platform_q)
            time.sleep(1)

    def produce_quote(self, marketData_2_exchSim_q, marketData_2_platform_q):
        for stock_ticker, midquote in self.stock_midquote.items():
            bidPrice, askPrice, bidSize, askSize = [], [], [], []
            bidPrice1 = midquote + random.randint(0, 100) / 10
            askPrice1 = bidPrice1 + 0.01
            for i in range(5):
                bidPrice.append(bidPrice1 - i * 0.01)
                askPrice.append(askPrice1 + i * 0.01)
                bidSize.append(bidPrice1 + random.randint(0, 100) * 100)
                askSize.append(askPrice1 + random.randint(0, 100) * 100)
            quoteSnapshot = OrderBookSnapshot_FiveLevels(str(stock_ticker), datetime.datetime.now().strftime('%Y-%m-%d'),
                                                         datetime.datetime.now(), bidPrice, askPrice, bidSize, askSize)
            print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
            print(quoteSnapshot.outputAsDataFrame())
            marketData_2_exchSim_q.put(quoteSnapshot)
            time.sleep(2)
            marketData_2_platform_q.put(quoteSnapshot)
            time.sleep(2)

