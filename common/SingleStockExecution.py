# -*- coding: utf-8 -*-
"""
Created on Sat Jul 3 07:11:28 2019

@author: hongs
"""
import random
class SingleStockExecution():
    
    def __init__(self, ticker, date, timeStamp, orderID):
        self.execID = random.randint(0,1000)
        self.orderID = orderID
        self.ticker = ticker
        self.date = date
        self.timeStamp = timeStamp
        self.direction = None
        self.price = None
        self.size = None
        self.comm = 5 #commission for this transaction

    def outputAsArray(self):
        output = []
        output.append(self.date)
        output.append(self.ticker)
        output.append(self.timeStamp)
        output.append(self.execID)
        output.append(self.orderID)
        output.append(self.direction)
        output.append(self.price)
        output.append(self.size)
        output.append(self.comm)

        return output