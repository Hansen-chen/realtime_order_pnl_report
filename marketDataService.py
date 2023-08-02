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
        time.sleep(3)
        self.produce_market_data(marketData_2_exchSim_q, marketData_2_platform_q)

    def produce_market_data(self, marketData_2_exchSim_q, marketData_2_platform_q):
        for i in range(200):
            self.produce_quote(marketData_2_exchSim_q, marketData_2_platform_q)
            time.sleep(1)

    def produce_quote(self, marketData_2_exchSim_q, marketData_2_platform_q):
        #TODO: read from file
        bidPrice, askPrice, bidSize, askSize = [], [], [], []
        bidPrice1 = 20+random.randint(0,100)/10
        askPrice1 = bidPrice1 + 0.01
        for i in range(5):
            bidPrice.append(bidPrice1-i*0.01)
            askPrice.append(askPrice1+i*0.01)
            bidSize.append(100+random.randint(0,100)*100)
            askSize.append(100+random.randint(0,100)*100)
        quoteSnapshot = OrderBookSnapshot_FiveLevels('testTicker_1', datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(),bidPrice, askPrice, bidSize, askSize)
        print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
        print(quoteSnapshot.outputAsDataFrame())
        marketData_2_exchSim_q.put(quoteSnapshot)
        time.sleep(2)
        marketData_2_platform_q.put(quoteSnapshot)
        time.sleep(2)



        bidPrice, askPrice, bidSize, askSize = [], [], [], []
        bidPrice1 = 100+random.randint(0,100)/10
        askPrice1 = bidPrice1 + 0.01
        for i in range(5):
            bidPrice.append(bidPrice1-i*0.01)
            askPrice.append(askPrice1+i*0.01)
            bidSize.append(100+random.randint(0,100)*100)
            askSize.append(100+random.randint(0,100)*100)
        quoteSnapshot = OrderBookSnapshot_FiveLevels('testTicker_2', datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(),bidPrice, askPrice, bidSize, askSize)
        print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
        print(quoteSnapshot.outputAsDataFrame())
        marketData_2_exchSim_q.put(quoteSnapshot)
        time.sleep(2)
        marketData_2_platform_q.put(quoteSnapshot)
        time.sleep(2)