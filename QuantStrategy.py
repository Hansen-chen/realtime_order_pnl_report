#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou
"""

import os
import time
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels
from common.Strategy import Strategy
from common.SingleStockOrder import SingleStockOrder
from common.SingleStockExecution import SingleStockExecution
import pandas as pd
#TODO: use dash plotly to plot realtime networth

class QuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day):
        super(QuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        #networth is a dataframe with columns: date, timestamp, networth
        self.networth = pd.DataFrame(columns=['date','timestamp','networth'])
        #cash is a dataframe with columns: date, timestamp, cash
        self.cash = pd.DataFrame(columns=['date','timestamp','cash'])
        # position_price is a dataframe with columns: date, timestamp, ticker, price
        self.position_price = pd.DataFrame(columns=['date','timestamp','ticker','price'])
        # submitted_order is a dataframe with columns: date, submissionTime, ticker, orderID, currStatus, currStatusTime, direction, price, size, type
        self.submitted_order = pd.DataFrame(columns=['date','submissionTime','ticker','orderID','currStatus','currStatusTime','direction','price','size','type'])
        # executed_order is a dataframe with columns: date, ticker, timeStamp, execID, orderID, direction, price, size, comm
        self.executed_order = pd.DataFrame(columns=['date','ticker','timeStamp','execID','orderID','direction','price','size','comm'])
        self.current_position = {}  # {'ticker':quantity}

    def getStratDay(self):
        return self.day
    
    def run(self, marketData, execution):
        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            print('[%d] Strategy.handle_execution' % (os.getpid()))
            # TODO: save executions order to a local csv file, path is hardcoded "./"
            # TODO: update current position and save to a local csv file, path is hardcoded "./"
            # TODO: update current PnL and save to a local csv file, path is hardcoded "./"
            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            current_market_data = marketData.outputAsDataFrame()
            #handle new market data, then create a new order and send it via quantTradingPlatform if needed
            #If it is the first time to receive market data, initialize the networth, cash
            if self.networth.empty:
                self.networth = self.networth.append({'date':current_market_data.iloc[0]['date'], 'timestamp':marketData.iloc[0]['time'], 'networth':1000000}, ignore_index=True)
            if self.cash.empty:
                self.cash = self.cash.append({'date':current_market_data.iloc[0]['date'], 'timestamp':marketData.iloc[0]['time'], 'cash':1000000}, ignore_index=True)

            #TODO: update PnL if there is an open position, and save to a local csv file, path is hardcoded "./"
            #if len(self.current_position) > 0:


            #TODO: if the strategy submits an order, save submitted order to a local csv file, path is hardcoded "./"
            return SingleStockOrder('testTicker','2019-07-05',time.asctime(time.localtime(time.time())))
        else:
            return None
                
        