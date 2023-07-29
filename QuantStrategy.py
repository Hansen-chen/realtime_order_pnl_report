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
import random

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
        #Save all the dataframes to local csv files, path is hardcoded "./"
        self.networth.to_csv('./networth.csv', index=False)
        self.cash.to_csv('./cash.csv', index=False)
        self.position_price.to_csv('./position_price.csv', index=False)
        self.submitted_order.to_csv('./submitted_order.csv', index=False)
        self.executed_order.to_csv('./executed_order.csv', index=False)

        #TODO: initiate dash plotly app

    def getStratDay(self):
        return self.day
    
    def run(self, marketData, execution):
        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            print('[%d] Strategy.handle_execution' % (os.getpid()))
            date, ticker, timeStamp, execID, orderID, direction, price, size, comm = execution.outputAsArray()

            # locate the row in self.submitted_order with orderID, then update the currStatus and currStatusTime, check if the size of executed order is the same as the size of submitted order, if less than, the status is PartiallyFilled, if equal, the status is Filled, if more than, return Non and print error

            #check if the orderID is in self.submitted_order
            if orderID not in self.submitted_order['orderID'].values:
                print('Error: orderID not in submitted_order')
                return None
            else:
                #locate the row in self.submitted_order with orderID
                row = self.submitted_order.loc[self.submitted_order['orderID'] == orderID]
                #check if the size of executed order is the same as the size of submitted order
                if size < row['size'].values[0]:
                    self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'currStatus'] = 'PartiallyFilled'
                    #update the size of submitted order
                    self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'size'] -= size
                elif size > row['size'].values[0]:
                    print('Error: size of executed order is more than the size of submitted order')
                    return None
                else:
                    # update the currStatus and currStatusTime
                    self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'currStatus'] = 'Filled'
                self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'currStatusTime'] = timeStamp
                self.submitted_order.to_csv('./submitted_order.csv', index=False)

            # update current position with ticker
            if direction == 'Buy':
                if ticker in self.current_position:
                    self.current_position[ticker] += size
                else:
                    self.current_position[ticker] = size
            elif direction == 'Sell':
                if ticker in self.current_position:
                    self.current_position[ticker] -= size
                else:
                    self.current_position[ticker] = -size
            else:
                print('Error: direction is neither Buy nor Sell')
                return None

            # save executions order to a local csv file, path is hardcoded "./"
            self.executed_order = self.executed_order.append({'date':date, 'ticker':ticker, 'timeStamp':timeStamp, 'execID':execID, 'orderID':orderID, 'direction':direction, 'price':price, 'size':size, 'comm':comm}, ignore_index=True)
            self.executed_order.to_csv('./executed_order.csv', index=False)

            current_cash = self.cash.iloc[-1]['cash']

            # update cash and networth and save to a local csv file, path is hardcoded "./"
            if direction == 'Buy':
                current_cash -= (price * size + comm)
            elif direction == 'Sell':
                current_cash += price * size - comm

            self.cash = self.cash.append({'date':date, 'timestamp':timeStamp, 'cash':current_cash}, ignore_index=True)
            self.cash.to_csv('./cash.csv', index=False)

            current_networth = current_cash

            for ticker in self.current_position:
                current_networth += self.current_position[ticker] * price

            self.networth = self.networth.append({'date':date, 'timestamp':timeStamp, 'networth':current_networth}, ignore_index=True)
            self.networth.to_csv('./networth.csv', index=False)

            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            current_market_data = marketData.outputAsDataFrame()
            current_date = current_market_data.iloc[0]['date']
            current_time = current_market_data.iloc[0]['time']
            #handle new market data, then create a new order and send it via quantTradingPlatform if needed
            #If it is the first time to receive market data, initialize the networth, cash
            if self.networth.empty:
                self.networth = self.networth.append({'date':current_date, 'timestamp':current_time, 'networth':1000000}, ignore_index=True)
            if self.cash.empty:
                self.cash = self.cash.append({'date':current_date, 'timestamp':current_time, 'cash':1000000}, ignore_index=True)

            #update networth if there is an open position, and save all self dataframe to a local csv file (override), path is hardcoded "./"

            #update current cash with the cash from the last row of self.cash
            current_cash = self.cash.iloc[-1]['cash']
            #check if self.cash has the same date as current_date, if not, add a new row to self.cash
            if self.cash.iloc[-1]['date'] != current_date:
                self.cash = self.cash.append({'date': current_date, 'timestamp': current_time, 'cash': current_cash},ignore_index=True)

            current_networth = current_cash
            if len(self.current_position) > 0:
                #loop through all the positions (keys of self.current_position) to find the current price from current_market_data, the current price is the average price of askPrice1 and bidPrice1
                for ticker in self.current_position.keys():
                    #check if the ticker is in the current_market_data, if not, skip this ticker
                    if ticker not in current_market_data['ticker'].values:
                        continue
                    current_price = (current_market_data.loc[current_market_data['ticker'] == ticker]['askPrice1'].values[0] + current_market_data.loc[current_market_data['ticker'] == ticker]['bidPrice1'].values[0])/2
                    self.position_price = self.position_price.append({'date':current_date, 'timestamp':current_time, 'ticker':ticker, 'price':current_price}, ignore_index=True)

                for ticker in self.current_position.keys():
                    current_networth += self.current_position[ticker] * self.position_price.loc[self.position_price['ticker'] == ticker].iloc[-1]['price']

            # check if self.networth has the same date as current_date, if not, add a new row to self.networth, if so, update the networth
            if self.networth.iloc[-1]['date'] != current_date:
                self.networth = self.networth.append({'date': current_market_data.iloc[0]['date'], 'timestamp': marketData.iloc[0]['time'],'networth': current_networth}, ignore_index=True)
            else:
                self.networth.loc[self.networth['date'] == current_date, 'networth'] = current_networth

            #save all self dataframe to a local csv file (override), path is hardcoded "./"
            self.networth.to_csv('./networth.csv', index=False)
            self.cash.to_csv('./cash.csv', index=False)
            self.position_price.to_csv('./position_price.csv', index=False)

            tradeOrder = None

            if random.choice([True, False]):
                ticker = "testTicker"
                direction = random.choice(["Buy", "Sell"])

                #debug only, if there is no current position and the direction is Sell, change the direction to Buy
                if (len(self.current_position) == 0):
                    direction = 'Buy'
                else:
                    if ticker in self.current_position.keys():
                        if self.current_position[ticker] <= 0:
                            direction = 'Buy'


                current_price = (current_market_data.loc[current_market_data['ticker'] == ticker]['askPrice1'].values[0] +current_market_data.loc[current_market_data['ticker'] == ticker]['bidPrice1'].values[0]) / 2
                tradeOrder = SingleStockOrder('testTicker', '2019-07-05', time.asctime(time.localtime(time.time())),time.asctime(time.localtime(time.time())), 'New', direction, current_price, 100, 'MO')
                date, ticker, submissionTime, orderID, currStatus, currStatusTime, direction, price, size, type = tradeOrder.outputAsArray()
                self.submitted_order = self.submitted_order.append({'date':date, 'submissionTime':submissionTime, 'ticker':ticker, 'orderID':orderID, 'currStatus':currStatus, 'currStatusTime':currStatusTime, 'direction':direction, 'price':price, 'size':size, 'type':type}, ignore_index=True)
                self.submitted_order.to_csv('./submitted_order.csv', index=False)

            return tradeOrder
        else:
            return None
                
        