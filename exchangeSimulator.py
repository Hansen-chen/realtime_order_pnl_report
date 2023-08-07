import threading
import os
import time
import datetime
from common.SingleStockExecution import SingleStockExecution


class ExchangeSimulator:

    def __init__(self, marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call ExchSim.init" % (os.getpid(),))
        self.stockdata = None
        self.futuredata = None
        self.future2stock = {'JBF': '3443', 'QWF': '2388', 'HCF': '2498', 'DBF': '2610', 'EHF': '1319', 'IPF': '3035',
                             'IIF': '3006', 'QXF': '2615', 'PEF': '5425', 'NAF': '3105'}
        t_md = threading.Thread(name='exchsim.on_md', target=self.consume_md, args=(marketData_2_exchSim_q,))
        t_md.start()

        t_order = threading.Thread(name='exchsim.on_order', target=self.consume_order,
                                   args=(platform_2_exchSim_order_q, exchSim_2_platform_execution_q,))
        t_order.start()

    def consume_md(self, marketData_2_exchSim_q):
        while True:
            marketData = marketData_2_exchSim_q.get()
            if str(marketData.ticker) in list(self.future2stock.values()):
                self.stockdata = marketData
            elif str(marketData.ticker) in list(self.future2stock.keys()):
                self.futuredata = marketData
            # print('[%d]ExchSim.consume_md' % (os.getpid()))
            # print(marketData.outputAsDataFrame())

    def consume_order(self, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        while True:
            res = platform_2_exchSim_order_q.get()
            # print('[%d]ExchSim.on_order' % (os.getpid()))
            # print(res.outputAsArray())
            self.produce_execution(res, exchSim_2_platform_execution_q)

    def produce_execution(self, order, exchSim_2_platform_execution_q):
        execution = self.Trade(order, self.stockdata, self.futuredata)
        for n in range(len(execution)):
            exchSim_2_platform_execution_q.put(execution[n])
        # print('[%d]ExchSim.produce_execution' % (os.getpid()))
        # print(execution.outputAsArray())

    def Trade(self, order, stockdata, futuredata):
        t = time.time()
        exec_list = []
        market_data = None
        if str(order.ticker) in list(self.future2stock.values()):
            market_data = stockdata
        elif str(order.ticker) in list(self.future2stock.keys()):
            market_data = futuredata
        remain_size = order.size
        execution_size = []
        if order.type == 'LO':  # Limit Order
            if order.direction == 'buy':
                execute_price = [market_data.askPrice1, market_data.askPrice2, market_data.askPrice3,
                                 market_data.askPrice4, market_data.askPrice5]
                ask_size = [market_data.askSize1, market_data.askSize2, market_data.askSize3, market_data.askSize4,
                            market_data.askSize5]
                ask_size = [x * 250 for x in ask_size]
                for i in range(5):
                    if remain_size > 0 and order.price >= execute_price[i]:
                        Ask_execution_size = remain_size - max(0, remain_size - ask_size[i])
                        remain_size -= Ask_execution_size
                        execution_size.append(Ask_execution_size)
                    else:
                        break
            elif order.direction == 'sell':
                execute_price = [market_data.bidPrice1, market_data.bidPrice2, market_data.bidPrice3,
                                 market_data.bidPrice4, market_data.bidPrice5]
                bid_size = [market_data.bidSize1, market_data.bidSize2, market_data.bidSize3, market_data.bidSize4,
                            market_data.bidSize5]
                bid_size = [x * 250 for x in bid_size]
                for i in range(5):
                    if remain_size > 0 and order.price <= execute_price[i]:
                        Bid_execution_size = remain_size - max(0, remain_size - bid_size[i])
                        remain_size -= Bid_execution_size
                        execution_size.append(Bid_execution_size)
                    else:
                        break

        if order.type == 'MO':  # Market Order
            if order.direction == 'buy':
                execute_price = [market_data.askPrice1, market_data.askPrice2, market_data.askPrice3,
                                 market_data.askPrice4, market_data.askPrice5]
                ask_size = [market_data.askSize1, market_data.askSize2, market_data.askSize3, market_data.askSize4,
                            market_data.askSize5]
                ask_size = [x * 250 for x in ask_size]
                for i in range(5):
                    if remain_size > 0:
                        Ask_execution_size = remain_size - max(0, remain_size - ask_size[i])
                        remain_size -= Ask_execution_size
                        execution_size.append(Ask_execution_size)
                    else:
                        break
            elif order.direction == 'sell':
                execute_price = [market_data.bidPrice1, market_data.bidPrice2, market_data.bidPrice3,
                                 market_data.bidPrice4, market_data.bidPrice5]
                bid_size = [market_data.bidSize1, market_data.bidSize2, market_data.bidSize3, market_data.bidSize4,
                            market_data.bidSize5]
                bid_size = [x * 250 for x in bid_size]

                for i in range(5):
                    if remain_size > 0:
                        Bid_execution_size = remain_size - max(0, remain_size - bid_size[i])
                        remain_size -= Bid_execution_size
                        execution_size.append(Bid_execution_size)
                    else:
                        break

        # ticker, date, timeStamp, orderID, size, price, direction
        if remain_size == 0 or (remain_size != 0 and len(execution_size) != 0):
            for i in range(len(execution_size)):
                ex = SingleStockExecution(order.ticker, order.date, order.submissionTime, "", execution_size[i],
                                          execute_price[i], order.direction)
                ex.orderID = str(order.orderID)
                ex.execID = str(order.orderID) + '-' + str(i + 1)
                ex.timeStamp = order.submissionTime + datetime.timedelta(seconds=time.time() - t)
                exec_list.append(ex)
                print("Execution : " + ex.orderID)
                print(ex.outputAsArray())
        else:
            ex = SingleStockExecution(order.ticker, order.date, order.submissionTime, "", 0, 0, order.direction)
            ex.orderID = str(order.orderID)
            ex.execID = str(order.orderID) + '-' + str(1)
            ex.timeStamp = order.submissionTime + datetime.timedelta(seconds=time.time() - t)
            exec_list.append(ex)
            print("Execution : " + ex.orderID)
            print(ex.outputAsArray())
        return exec_list