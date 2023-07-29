# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:47:43 2019

@author: hongsong chou
"""
import random
class SingleStockOrder():
    
    def __init__(self, ticker, date, submissionTime,currStatusTime,currStatus,direction,price,size,type):
        self.orderID = random.randint(0,100000)
        self.ticker = ticker
        self.date = date
        self.submissionTime = submissionTime
        self.currStatusTime = currStatusTime
        self.currStatus = currStatus #"New", "Filled", "PartiallyFilled", "Cancelled"
        self.direction = direction
        self.price = price
        self.size = size
        self.type = type #"MLO", "LO", "MO", "TWAP"

    def outputAsArray(self):
        output = []
        output.append(self.date)
        output.append(self.ticker)
        output.append(self.submissionTime)
        output.append(self.orderID)
        output.append(self.currStatus)
        output.append(self.currStatusTime)
        output.append(self.direction)
        output.append(self.price)
        output.append(self.size)
        output.append(self.type)
        
        return output
    
    def copyOrder(self):
        returnOrder = SingleStockOrder(self.ticker, self.date)
        returnOrder.orderID = self.orderID
        returnOrder.submissionTime = self.submissionTime
        returnOrder.currStatusTime = self.currStatusTime
        returnOrder.currStatus = self.currStatus
        returnOrder.direction = self.direction
        returnOrder.price = self.price
        returnOrder.size = self.size
        returnOrder.type = self.type
        
        return returnOrder

