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
from bottle import route, run, template

@route('/main')
def index():
    return template('<b>trading {{name}}</b>! <br> Graph Here')

@route('/latest_data')
def index():
    #TODO: read from local json file, path is hardcoded "./"
    return template('<b>update graphs</b>')

class QuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day):
        super(QuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        run(host='localhost', port=8081)

    def getStratDay(self):
        return self.day
    
    def run(self, marketData, execution):
        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            print('[%d] Strategy.handle_execution' % (os.getpid()))
            # TODO: save executions order to a local json file, path is hardcoded "./"
            # TODO: update current position and save to a local json file, path is hardcoded "./"
            # TODO: update current PnL and save to a local json file, path is hardcoded "./"
            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            #handle new market data, then create a new order and send it via quantTradingPlatform.
            #TODO: update PnL if there is an open position, and save to a local json file, path is hardcoded "./"
            #TODO: save submitted order to a local json file, path is hardcoded "./"
            return SingleStockOrder('testTicker','2019-07-05',time.asctime(time.localtime(time.time())))
        else:
            return None
                
        