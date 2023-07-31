#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou
"""

import os
import time, datetime
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels
from common.Strategy import Strategy
from common.SingleStockOrder import SingleStockOrder
from common.SingleStockExecution import SingleStockExecution
import pandas as pd
import random
import dash
from dash import dcc, dash_table,html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import multiprocessing
import numpy as np

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
        self.current_position_dataframe = pd.DataFrame(columns=['ticker','quantity','price'])
        # self metrics dataframe
        self.metrics = pd.DataFrame(columns=['cumulative_return','portfolio_volatility','max_drawdown'])
        #Save all the dataframes to local csv files, path is hardcoded "./"
        self.networth.to_csv('./networth.csv', index=False)
        self.cash.to_csv('./cash.csv', index=False)
        self.position_price.to_csv('./position_price.csv', index=False)
        self.submitted_order.to_csv('./submitted_order.csv', index=False)
        self.executed_order.to_csv('./executed_order.csv', index=False)
        self.current_position_dataframe.to_csv('./current_position.csv', index=False)
        self.metrics.to_csv('./metrics.csv', index=False)


        # initiate dash plotly app
        # Set up the app
        app = dash.Dash(__name__)

        # Define the layout
        app.layout = dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.H2(
                            "HFT Quantitative Strategy Dashboard",
                            className="text-center bg-primary text-white p-2",
                        ),
                    )
                ),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='my-graph', animate=False),
                        dcc.Interval(
                            id='interval-component-1',
                            interval=3000,  # Refresh every 3 second
                            n_intervals=0
                        )
                    ]),
                    dbc.Col([
                        dcc.Graph(id='position-graph', animate=False),
                        dcc.Interval(
                            id='interval-component-4',
                            interval=3000,  # Refresh every 3 second
                            n_intervals=0
                        )
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(data=self.current_position_dataframe.to_dict('records'), page_size=10,
                                             id='position-table',
                                             columns=[{"name": i, "id": i} for i in self.current_position_dataframe.columns]),
                        dcc.Interval(
                            id='interval-component-5',
                            interval=3000,  # Refresh every 3 second
                            n_intervals=0
                        )
                    ]),
                ]),
                dbc.Row(
                    dbc.Col(
                        html.H4(
                            "Portfolio Metrics (%)",
                            className="text-center bg-primary text-white p-2",
                        ),
                    )
                ),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(data=self.metrics.to_dict('records'), page_size=10,
                                             id='metrics-table',
                                             columns=[{"name": i, "id": i} for i in self.metrics.columns]),
                        dcc.Interval(
                            id='interval-component-3',
                            interval=3000,  # Refresh every 3 second
                            n_intervals=0
                        )
                    ]),
                ]),
                dbc.Row(
                    dbc.Col(
                        html.H4(
                            "Submitted Orders",
                            className="text-center bg-primary text-white p-2",
                        ),
                    )
                ),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(data=self.submitted_order.to_dict('records'), page_size=10,
                                             id='order-table',
                                             columns=[{"name": i, "id": i} for i in self.submitted_order.columns]),
                        dcc.Interval(
                            id='interval-component-2',
                            interval=3000,  # Refresh every 3 second
                            n_intervals=0
                        )
                    ])
                ])
            ],fluid=True)

        # Define the callback function
        @app.callback(Output('my-graph', 'figure'),Input('interval-component-1', 'n_intervals'))
        def update_graph(n):
            # Read the CSV file
            df = pd.read_csv('./networth.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.index = df['timestamp']

            #get the latest timestamp and convert it to string
            latest_timestamp = str(df.iloc[-1]['timestamp'])

            # Create the graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['networth'], mode='lines', name='Networth'))
            fig.update_layout(title='Portfolio Value at '+latest_timestamp, xaxis_title='Date', yaxis_title='Networth')

            return fig

        @app.callback(
            Output('order-table', 'data'),
            Input('interval-component-2', 'n_intervals')
        )
        def update_table1(n):
            # Load data into Pandas DataFrame
            df = pd.read_csv('./submitted_order.csv')

            #sort the dataframe by submissionTime in descending order, and then currStatus = 'New' first, other currStatus second
            df = df.sort_values(by=['submissionTime','currStatus'], ascending=[False, True])

            # Return the DataFrame as a list of dictionaries
            return df.to_dict('records')
        @app.callback(
            Output('metrics-table', 'data'),
            Input('interval-component-3', 'n_intervals')
        )
        def update_table2(n):
            # Load data into Pandas DataFrame
            df = pd.read_csv('./metrics.csv')

            # Return the DataFrame as a list of dictionaries
            return df.to_dict('records')
        @app.callback(
            Output('position-graph', 'figure'),
            Input('interval-component-4', 'n_intervals')
        )
        def update_graph2(n):
            # Load data into Pandas DataFrame
            df = pd.read_csv('./current_position.csv')

            # Create the graph as a pie chart
            fig = go.Figure(data=[go.Pie(labels=df['ticker'], values=abs(df['quantity']*df['price']),textinfo="label+percent",textposition="inside")])
            fig.update_layout(title='Current Position')

            return fig
        @app.callback(
            Output('position-table', 'data'),
            Input('interval-component-5', 'n_intervals')
        )
        def update_table2(n):
            # Load data into Pandas DataFrame
            df = pd.read_csv('./current_position.csv')

            # Return the DataFrame as a list of dictionaries
            return df.to_dict('records')

        # Define the function to run the server
        def run_server():
            app.run_server(debug=True)

        server_process = multiprocessing.Process(target=run_server)
        server_process.start()

    def getStratDay(self):
        return self.day
    
    def run(self, marketData, execution):
        print(self.current_position)
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
                submitted_size = self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'size'][0]
                #check if the size of executed order is the same as the size of submitted order
                if size < submitted_size:
                    self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'currStatus'] = 'PartiallyFilled'
                    #update the size of submitted order
                    self.submitted_order.loc[self.submitted_order['orderID'] == orderID, 'size'] -= size
                elif size > submitted_size:
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
            self.executed_order = pd.concat([self.executed_order, pd.DataFrame({'date':date, 'ticker':ticker, 'timeStamp':timeStamp, 'execID':execID, 'orderID':orderID, 'direction':direction, 'price':price, 'size':size, 'comm':comm}, index=[0])])
            self.executed_order.to_csv('./executed_order.csv', index=False)

            current_cash = self.cash.iloc[-1]['cash']

            # update cash and networth and save to a local csv file, path is hardcoded "./"
            if direction == 'Buy':
                current_cash -= (price * size + comm)
            elif direction == 'Sell':
                current_cash += price * size - comm

            self.cash = pd.concat([self.cash, pd.DataFrame({'date':date, 'timestamp':timeStamp, 'cash':current_cash}, index=[0])])
            self.cash.to_csv('./cash.csv', index=False)

            current_networth = current_cash

            for ticker in self.current_position:
                current_networth += self.current_position[ticker] * price

            self.networth = pd.concat([self.networth, pd.DataFrame({'date':date, 'timestamp':timeStamp, 'networth':current_networth}, index=[0])])
            self.networth.to_csv('./networth.csv', index=False)

            #update self.current_position_dataframe's row with ticker, price and self.current_position
            if ticker in self.current_position_dataframe['ticker'].values:
                self.current_position_dataframe.loc[self.current_position_dataframe['ticker'] == ticker, 'price'] = price
                self.current_position_dataframe.loc[self.current_position_dataframe['ticker'] == ticker, 'quantity'] = self.current_position[ticker]
            else:
                self.current_position_dataframe = pd.concat([self.current_position_dataframe, pd.DataFrame({'ticker':ticker, 'price':price, 'quantity':self.current_position[ticker]}, index=[0])])

            #update self.current_position_dataframe's ticker "cash" with price = 1 and current_cash
            if 'cash' in self.current_position_dataframe['ticker'].values:
                self.current_position_dataframe.loc[self.current_position_dataframe['ticker'] == 'cash', 'price'] = 1
                self.current_position_dataframe.loc[self.current_position_dataframe['ticker'] == 'cash', 'quantity'] = current_cash
            else:
                self.current_position_dataframe = pd.concat([self.current_position_dataframe, pd.DataFrame({'ticker':'cash', 'price':1, 'quantity':current_cash}, index=[0])])


            self.current_position_dataframe.to_csv('./current_position.csv', index=False)

            #update self.metrics with self.networth if self.networth has more than 1 row
            if self.networth.shape[0] > 1:
                cumulative_return = (self.networth.iloc[-1]['networth'] / self.networth.iloc[0]['networth'] - 1) * 100

                # calculate portfolio volatility
                portfolio_volatility = self.networth['networth'].pct_change().std() * 100

                # calculate max drawdown
                max_drawdown = 0
                for i in range(1, len(self.networth)):
                    if self.networth.iloc[i]['networth'] > self.networth.iloc[i - 1]['networth']:
                        continue
                    else:
                        drawdown = (self.networth.iloc[i]['networth'] / self.networth.iloc[i - 1]['networth'] - 1) * 100
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown

                self.metrics = pd.DataFrame({'cumulative_return': cumulative_return, 'portfolio_volatility': portfolio_volatility,'max_drawdown': max_drawdown}, index=[0])
                self.metrics.to_csv('./metrics.csv', index=False)

            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):
            #TODO: save market data to a local csv file, path is hardcoded "./"

            current_market_data = marketData.outputAsDataFrame()

            current_date = current_market_data.iloc[0]['date']
            current_time = current_market_data.iloc[0]['time']
            #handle new market data, then create a new order and send it via quantTradingPlatform if needed
            #If it is the first time to receive market data, initialize the networth, cash
            if self.networth.empty:
                self.networth = pd.DataFrame({'date':current_date, 'timestamp':current_time, 'networth':10000.0}, index=[0])
            if self.cash.empty:
                self.cash = pd.DataFrame({'date':current_date, 'timestamp':current_time, 'cash':10000.0}, index=[0])

            if self.current_position_dataframe.empty:
                self.current_position_dataframe = pd.DataFrame({'ticker':'cash','quantity':10000.0,'price':1}, index=[0])

            if self.metrics.empty:
                self.metrics = pd.DataFrame({'cumulative_return':0,'portfolio_volatility':0,'max_drawdown':0}, index=[0])

            #update networth and current_position_dataframe if there is an open position, and save all self dataframe to a local csv file (override), path is hardcoded "./"

            #update current cash with the cash from the last row of self.cash
            current_cash = self.cash.iloc[-1]['cash']
            #check if self.cash has the same time as current_time, if not, add a new row to self.cash
            if self.cash.iloc[-1]['timestamp'] != current_time:
                self.cash = pd.concat([self.cash, pd.DataFrame({'date': current_date, 'timestamp': current_time, 'cash': current_cash}, index=[0])])

            current_networth = current_cash
            if len(self.current_position) > 0:
                #loop through all the positions (keys of self.current_position) to find the current price from current_market_data, the current price is the average price of askPrice1 and bidPrice1
                for ticker in self.current_position.keys():
                    #check if the ticker is in the current_market_data, if not, skip this ticker
                    if ticker not in current_market_data['ticker'].values:
                        continue
                    current_price = (current_market_data.loc[current_market_data['ticker'] == ticker]['askPrice1'].values[0] + current_market_data.loc[current_market_data['ticker'] == ticker]['bidPrice1'].values[0])/2
                    self.position_price = pd.concat([self.position_price, pd.DataFrame({'date':current_date, 'timestamp':current_time, 'ticker':ticker, 'price':current_price}, index=[0])])

                for ticker in self.current_position.keys():
                    current_networth += self.current_position[ticker] * self.position_price.loc[self.position_price['ticker'] == ticker].iloc[-1]['price']

            # check if self.networth has the same date as current_date, if not, add a new row to self.networth, if so, update the networth
            if self.networth.iloc[-1]['timestamp'] != current_time:
                self.networth = pd.concat([self.networth, pd.DataFrame({'date': current_date, 'timestamp': current_time,'networth': current_networth}, index=[0])])
            else:
                self.networth.loc[self.networth['timestamp'] == current_time, 'networth'] = current_networth


            #construct self.current_position_dataframe from self.current_position, price and current cash
            self.current_position_dataframe = pd.DataFrame({'ticker':list(self.current_position.keys()), 'quantity':list(self.current_position.values()), 'price':list(self.position_price.loc[self.position_price['timestamp'] == current_time]['price'])})
            self.current_position_dataframe = pd.concat([self.current_position_dataframe, pd.DataFrame({'ticker':'cash','quantity':current_cash,'price':1}, index=[0])])

            #update self.metrics with self.networth if self.networth has more than 1 row
            if self.networth.shape[0] > 1:
                cumulative_return = (self.networth.iloc[-1]['networth'] / self.networth.iloc[0]['networth'] - 1)*100

                #calculate portfolio volatility
                portfolio_volatility = self.networth['networth'].pct_change().std() * 100

                #calculate max drawdown
                max_drawdown = 0
                for i in range(1, len(self.networth)):
                    if self.networth.iloc[i]['networth'] > self.networth.iloc[i-1]['networth']:
                        continue
                    else:
                        drawdown = (self.networth.iloc[i]['networth'] / self.networth.iloc[i-1]['networth'] - 1)*100
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown

                self.metrics = pd.DataFrame({'cumulative_return': cumulative_return,'portfolio_volatility': portfolio_volatility, 'max_drawdown': max_drawdown}, index=[0])


            #save all self dataframe to a local csv file (override), path is hardcoded "./"
            self.networth.to_csv('./networth.csv', index=False)
            self.cash.to_csv('./cash.csv', index=False)
            self.position_price.to_csv('./position_price.csv', index=False)
            self.current_position_dataframe.to_csv('./current_position.csv', index=False)
            self.metrics.to_csv('./metrics.csv', index=False)

            tradeOrder = None


            #TODO: decide the tradeOrder

            if random.choice([True, False]):
                ticker = "testTicker"
                direction = random.choice(["Buy", "Sell"])

                current_price = (current_market_data.loc[current_market_data['ticker'] == ticker]['askPrice1'].values[0] +current_market_data.loc[current_market_data['ticker'] == ticker]['bidPrice1'].values[0]) / 2
                quantity = 100
                #check if there is enough cash to buy
                if (current_cash < current_price*quantity) and (direction == 'Buy'):
                    print('Not enough cash to buy')
                    return None

                tradeOrder = SingleStockOrder('testTicker', datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(), datetime.datetime.now(), 'New', direction, current_price,quantity , 'MO')
                date, ticker, submissionTime, orderID, currStatus, currStatusTime, direction, price, size, type = tradeOrder.outputAsArray()
                self.submitted_order = pd.concat([self.submitted_order, pd.DataFrame({'date':date, 'submissionTime':submissionTime, 'ticker':ticker, 'orderID':orderID, 'currStatus':currStatus, 'currStatusTime':currStatusTime, 'direction':direction, 'price':price, 'size':size, 'type':type}, index=[0])])
                self.submitted_order.to_csv('./submitted_order.csv', index=False)

            return tradeOrder
        else:
            return None
                
        