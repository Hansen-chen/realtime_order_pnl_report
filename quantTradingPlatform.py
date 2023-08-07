# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:15:48 2020

@author: hongsong chou
"""

import threading
import os
from QuantStrategy import QuantStrategy
import time

class TradingPlatform:
    quantStrat = None
    
    def __init__(self, marketData_2_platform_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call Platform.init" % (os.getpid(),))
        
        #Instantiate individual strategies
        self.quantStrat = QuantStrategy("tf_1","quantStrategy","hongsongchou","JBF_3443","20230706")

        t_md = threading.Thread(name='platform.on_marketData', target=self.consume_marketData, args=(platform_2_exchSim_order_q, marketData_2_platform_q,))
        t_md.start()
        
        t_exec = threading.Thread(name='platform.on_exec', target=self.handle_execution, args=(exchSim_2_platform_execution_q, ))
        t_exec.start()

        t_cancel = threading.Thread(name='platform.on_cancel', target=self.handle_order_cancelation , args=(platform_2_exchSim_order_q,))
        t_cancel.start()

    def consume_marketData(self, platform_2_exchSim_order_q, marketData_2_platform_q):
        print('[%d]Platform.consume_marketData' % (os.getpid(),))
        while True:
            time.sleep(1)
            res = marketData_2_platform_q.get()
            print('[%d] Platform.on_md' % (os.getpid()))
            print(res.outputAsDataFrame())
            result = self.quantStrat.run(res, None)
            if result is None:
                pass
            else:
                #do something with the new order
                platform_2_exchSim_order_q.put(result)
    
    def handle_execution(self, exchSim_2_platform_execution_q):
        print('[%d]Platform.handle_execution' % (os.getpid(),))
        while True:
            time.sleep(1)
            execution = exchSim_2_platform_execution_q.get()
            print('[%d] Platform.handle_execution' % (os.getpid()))
            print(execution.outputAsArray())
            self.quantStrat.run(None, execution)

    def handle_order_balance(self, platform_2_exchSim_order_q):
        print('[%d]Platform.handle_balance_order' % (os.getpid(),))
        while True:
            time.sleep(2)
            res = self.quantStrat.handle_order_balance()
            for order in res:
                platform_2_exchSim_order_q.put(order)

