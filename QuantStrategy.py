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
import lightgbm as lgb

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.automap import automap_base



class QuantStrategy(Strategy):

    def __init__(self, stratID, stratName, stratAuthor, ticker, day):
        super(QuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        self.initial_cash = 10000.0
        # Create the declarative base
        self.Base = declarative_base()

        class Networth(self.Base):
            __tablename__ = 'networth'
            date = Column(String())
            timestamp = Column(DateTime(), primary_key=True)
            networth = Column(Float())

        class Current_position(self.Base):
            __tablename__ = 'current_position'
            price = Column(Float())
            inception_timestamp = Column(DateTime())
            ticker = Column(String(), primary_key=True)
            quantity = Column(Float())

        class Submitted_order(self.Base):
            __tablename__ = 'submitted_order'
            date = Column(String())
            submissionTime = Column(DateTime())
            ticker = Column(String())
            orderID = Column(String(), primary_key=True)
            currStatus = Column(String())
            currStatusTime = Column(DateTime())
            direction = Column(String())
            price = Column(Float())
            size = Column(Integer())
            type = Column(String())

        class Metrics(self.Base):
            __tablename__ = 'portfolio_metrics'
            metricsID = Column(Integer(), primary_key=True)
            cumulative_return = Column(Float())
            portfolio_volatility = Column(Float())
            max_drawdown = Column(Float())



        self.networth = pd.DataFrame(columns=['date','timestamp','networth'])

        self.current_position = pd.DataFrame(columns=['ticker', 'quantity', 'price','dollar_amount'])


        self.submitted_order = pd.DataFrame(
            columns=['date', 'submissionTime', 'ticker', 'orderID', 'currStatus', 'currStatusTime', 'direction',
                     'price', 'size', 'type'])


        self.metrics = pd.DataFrame(columns=['cumulative_return', 'portfolio_volatility', 'max_drawdown'])
        # Load model
        self.all_market_data = {}
        self.last_position_time = {}
        self.future2stock = {'JBF': 3443, 'QWF': 2388, 'HCF': 2498, 'DBF': 2610, 'EHF': 1319,
                             'IPF': 3035, 'IIF': 3006, 'QXF': 2615, 'PEF': 5425, 'NAF': 3105}
        self.stock2future = {v: k for k, v in self.future2stock.items()}
        self.future_tickers = self.future2stock.keys()
        self.stock_tickers = list(self.future2stock.values())
        self.if_enough_data = False
        self.models = {}
        for future_ticker in self.future2stock.keys():
            self.models[future_ticker] = lgb.Booster(model_file='./modelParamsProd/model_{}_Y_M_1.txt'.format(future_ticker))

        # Create an engine and session
        self.engine = create_engine('sqlite:///database.db')  # Database Abstraction and Portability
        Session = sessionmaker(bind=self.engine)
        session = Session()
        session.begin()

        # Create the table in the database
        self.Base.metadata.create_all(self.engine)  # Database Schema Management

        # Commit the session to persist the changes to the database
        session.commit()  # Query Building and Execution


        # Close the session
        session.close()  # Query Building and Execution

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
                            interval=5000,  # Refresh every 1 second
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
                            interval=5000,  # Refresh every 1 second
                            n_intervals=0
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='position-graph', animate=False),
                        dcc.Interval(
                            id='interval-component-4',
                            interval=5000,  # Refresh every 1 second
                            n_intervals=0
                        )
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(data=self.current_position.to_dict('records'), page_size=10,
                                             id='position-table',
                                             columns=[{"name": i, "id": i} for i in self.current_position.columns]),
                        dcc.Interval(
                            id='interval-component-5',
                            interval=5000,  # Refresh every 1 second
                            n_intervals=0
                        )
                    ]),
                ]),

                dbc.Row(
                    dbc.Col(
                        html.H4(
                            "Submitted Orders (Latest 15)",
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
                            interval=5000,  # Refresh every 1 second
                            n_intervals=0
                        )
                    ])
                ])
            ],fluid=True)

        # Define the callback function
        @app.callback(Output('my-graph', 'figure'),
                      Output('order-table', 'data'),
                      Output('metrics-table', 'data'),
                      Output('position-graph', 'figure'),
                      Output('position-table', 'data'),
                      Input('interval-component-1', 'n_intervals'))
        def update_app(n):

            df = pd.read_sql_table('networth', con=self.engine.connect())
            #get the latest timestamp and convert it to string
            latest_timestamp = "inception"
            if len(df) > 1:
                latest_timestamp = str(df.iloc[-1]['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # sort the dataframe by timestamp, the latest timestamp will be at the end
            df = df.sort_values(by=['timestamp'])

            # Create the graph
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=df['timestamp'], y=df['networth'], mode='lines', name='Networth'))
            fig_pnl.update_layout(title='Portfolio Value at ' + latest_timestamp, xaxis_title='Date',
                              yaxis_title='Networth')

            df_order = pd.read_sql_table('submitted_order', con=self.engine.connect())
            # sort the dataframe by submissionTime in descending order, and then currStatus = 'New' first, other currStatus second
            df_order = df_order.sort_values(by=['submissionTime', 'currStatus'], ascending=[False, True])
            #take the first 15 rows
            df_order = df_order.head(15)

            df_position = pd.read_sql_table('current_position', con=self.engine.connect())

            #add a column to calculate the absolute total value of each position called dollar_amount
            df_position['dollar_amount'] = abs(df_position['quantity'] * df_position['price'])

            # Create the graph as a pie chart
            fig_position = go.Figure(data=[
                go.Pie(labels=df_position['ticker'], values=df_position['dollar_amount'], textinfo="label+percent",
                       textposition="inside")])
            fig_position.update_layout(title='Current Position')

            df_metrics = pd.read_sql_table('portfolio_metrics', con=self.engine.connect())

            # drop metricsID column
            df_metrics = df_metrics.drop(columns=['metricsID'])

            return fig_pnl, df_order.to_dict('records'), df_metrics.to_dict('records'), fig_position, df_position.to_dict('records')


        # Define the function to run the server
        def run_server():
            app.run_server(debug=True)

        server_process = multiprocessing.Process(target=run_server)
        server_process.start()

    def getStratDay(self):
        return self.day


    def run(self, marketData, execution):
        Base = automap_base()
        Base.prepare(autoload_with=self.engine)

        Submitted_order = Base.classes.submitted_order
        Current_position = Base.classes.current_position
        Metrics = Base.classes.portfolio_metrics
        Networth = Base.classes.networth

        #TODO: double check logic below? and add real time print screen?

        if (marketData is None) and (execution is None):
            return None
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            print('[%d] Strategy.handle_execution' % (os.getpid()))
            date, ticker, timeStamp, execID, orderID, direction, price, size, comm = execution.outputAsArray()
            execID = str(execID)
            orderID = str(orderID)
            #direction to lower case
            direction = direction.lower()

            #TODO: check ticker

            Session = sessionmaker(bind=self.engine)
            session = Session()
            session.begin()

            # locate the row in self.submitted_order with orderID, then update the currStatus and currStatusTime, check if the size of executed order is the same as the size of submitted order, if less than, the status is PartiallyFilled, if equal, the status is Filled, if more than, return Non and print error

            submitted_order = session.query(Submitted_order).filter_by(orderID=orderID).first()

            #check if the orderID is in self.submitted_order
            if submitted_order is None:
                print('Error: orderID not in submitted_order')
                return None
            else:
                #locate the row in self.submitted_order with orderID
                submitted_size = submitted_order.size
                #check if the size of executed order is the same as the size of submitted order
                if size < submitted_size:
                    submitted_order.currStatus = 'PartiallyFilled'
                    #update the size of submitted order
                    submitted_order.size = submitted_order.size - size
                elif size > submitted_size:
                    print('Error: size of executed order is more than the size of submitted order')
                    return None
                else:
                    # update the currStatus and currStatusTime
                    submitted_order.currStatus = 'Filled'
                submitted_order.currStatusTime = timeStamp
                session.commit()  # Query Building and Execution


            current_position = session.query(Current_position).filter_by(ticker=ticker).first()

            # update current position with ticker
            if direction == 'buy':
                if current_position is not None:
                    current_position.quantity = current_position.quantity + size
                    current_position.inception_timestamp=timeStamp
                    session.commit()
                else:
                    new_position = Current_position(price=price, inception_timestamp=timeStamp, ticker=ticker, quantity=size)
                    session.add(new_position)
                    session.commit()
            elif direction == 'sell':
                if current_position is not None:
                    current_position.quantity = current_position.quantity - size
                    current_position.inception_timestamp = timeStamp
                    session.commit()
                else:
                    new_position = Current_position(price=price, inception_timestamp=timeStamp, ticker=ticker, quantity=-size)
                    session.add(new_position)
                    session.commit()
            else:
                print('Error: direction is neither Buy nor Sell')
                return None

            cash_position = session.query(Current_position).filter_by(ticker="cash").first()

            # update cash and networth and save to a local csv file, path is hardcoded "./"
            if direction == 'buy':
                cash_position.quantity = cash_position.quantity - (price * size + comm)
            elif direction == 'sell':
                cash_position.quantity = cash_position.quantity + price * size - comm

            session.commit()

            positions = session.query(Current_position).all()

            current_networth = Networth(date=date, timestamp=timeStamp, networth=0)

            for position in positions:
                current_networth.networth = current_networth.networth + position.price * position.quantity

            session.add(current_networth)
            session.commit()

            networthes = pd.read_sql_table('networth', con=self.engine.connect())

            #update self.metrics with self.networth if self.networth has more than 1 row
            if networthes.shape[0] > 1:
                cumulative_return = (networthes.iloc[-1]['networth'] / networthes.iloc[0]['networth'] - 1) * 100

                # calculate portfolio volatility
                portfolio_volatility = networthes['networth'].pct_change().std() * 100

                # calculate max drawdown
                max_drawdown = 0
                for i in range(1, len(networthes)):
                    if networthes.iloc[i]['networth'] > networthes.iloc[i - 1]['networth']:
                        continue
                    else:
                        drawdown = (networthes.iloc[i]['networth'] / networthes.iloc[i - 1]['networth'] - 1) * 100
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown

                metrics = session.query(Metrics).filter_by(metricsID=1).first()
                if metrics is None:
                    metrics = Metrics(cumulative_return=cumulative_return, portfolio_volatility=portfolio_volatility,
                                      max_drawdown=max_drawdown)
                    session.add(metrics)
                else:
                    metrics.cumulative_return = cumulative_return
                    metrics.portfolio_volatility = portfolio_volatility
                    metrics.max_drawdown = max_drawdown
                session.commit()
            session.close()
            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):

            current_market_data = marketData.outputAsDataFrame()

            #check if askPrice1 or bidPrice1 is empty, if it is, print error and then return None
            if current_market_data.iloc[0]['askPrice1'] == 0 or current_market_data.iloc[0]['bidPrice1'] == 0:
                print('Error: askPrice1 or bidPrice1 is empty')
                return None

            current_date = current_market_data.iloc[0]['date']
            current_time = current_market_data.iloc[0]['time']
            Session = sessionmaker(bind=self.engine)
            session = Session()
            session.begin()

            networthes = session.query(Networth).filter_by(networth=10000.0).first()
            if networthes is None:
                networth = Networth(date=current_date, timestamp=current_time, networth=10000.0)
                session.add(networth)
                session.commit()
                position = Current_position(price=1, inception_timestamp=current_time, ticker='cash', quantity=10000.0)
                session.add(position)
                session.commit()
                metrics = Metrics(cumulative_return=0, portfolio_volatility=0, max_drawdown=0)
                session.add(metrics)
                session.commit()
                print("initialize networth, current_position and portfolio_metrics")

            #handle new market data, then create a new order and send it via quantTradingPlatform if needed

            #update networth and current_position if there is an open position related to this ticker
            ticker = current_market_data.iloc[0]['ticker']
            # if it is a futrue ticker with month, transfer it
            if ticker not in self.stock2future.keys():
                ticker = ticker[:3]
                if ticker not in self.stock2future.keys():
                    raise ValueError('Future ticker probelm')

            related_position = session.query(Current_position).filter_by(ticker=ticker).first()
            if related_position is not None:
                #get the current price of the ticker
                current_price = (current_market_data.iloc[0]['askPrice1'] + current_market_data.iloc[0]['bidPrice1']) / 2
                #update the price of the position
                related_position.price = current_price
                session.commit()
                #update the networth
                positions = session.query(Current_position).all()
                current_networth = Networth(date=current_date, timestamp=current_time, networth=0)
                for position in positions:
                    current_networth.networth = current_networth.networth + position.price * position.quantity
                session.add(current_networth)
                session.commit()

            networthes = pd.read_sql_table('networth', con=self.engine.connect())

            #update self.metrics with self.networth if self.networth has more than 1 row
            if networthes.shape[0] > 1:
                cumulative_return = (networthes.iloc[-1]['networth'] / networthes.iloc[0]['networth'] - 1) * 100

                # calculate portfolio volatility
                portfolio_volatility = networthes['networth'].pct_change().std() * 100

                # calculate max drawdown
                max_drawdown = 0
                for i in range(1, len(networthes)):
                    if networthes.iloc[i]['networth'] > networthes.iloc[i - 1]['networth']:
                        continue
                    else:
                        drawdown = (networthes.iloc[i]['networth'] / networthes.iloc[i - 1]['networth'] - 1) * 100
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown

                metrics = session.query(Metrics).filter_by(metricsID=1).first()
                if metrics is None:
                    metrics = Metrics(cumulative_return=cumulative_return, portfolio_volatility=portfolio_volatility, max_drawdown=max_drawdown)
                    session.add(metrics)
                else:
                    metrics.cumulative_return = cumulative_return
                    metrics.portfolio_volatility = portfolio_volatility
                    metrics.max_drawdown = max_drawdown
                session.commit()


            tradeOrder = None
            current_cash_query = session.query(Current_position).filter_by(ticker="cash").first()

            current_cash = 0

            if current_cash_query is not None:
                current_cash = current_cash_query.quantity

            # Save market data to self.all_market_data
            if ticker not in self.all_market_data.keys():
                self.all_market_data[ticker] = current_market_data
            self.all_market_data[ticker] = pd.concat([self.all_market_data[ticker], current_market_data], ignore_index=True)
            # Check future/stock
            # Check if future ticker consists of month
            if ticker in self.future_tickers.keys():
                return None
            # Check if we have enough data to make decision
            if self.if_enough_data == False:
                stk_time_delta = (self.all_market_data[ticker]['time'].iloc[-1] - self.all_market_data[ticker]['time'].iloc[0]).total_seconds()
                future_time_delta = (self.all_market_data[self.stock2future[ticker]]['time'].iloc[-1] - self.all_market_data[self.stock2future[ticker]]['time'].iloc[0]).total_seconds()
                if stk_time_delta < 110 or future_time_delta < 110:
                    return None
            self.if_enough_data = True
            # Check if stock in current position
            current_position = session.query(Current_position).filter_by(ticker=ticker).first()
            if current_position is not None and current_position.quantity != 0:
                # if holding time < 10s, we will return None
                last_position_time = current_position.inception_timestamp
                if (current_time - last_position_time).total_seconds() < 10:
                    # Keep this position
                    return None
                else:
                    # Balance the position
                    direction = 'buy' if current_position.quantity < 0 else 'sell'
                    quantity = abs(self.current_position[ticker])
                    tradeOrder = SingleStockOrder(ticker, datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(),
                                                  datetime.datetime.now(), 'New', direction, current_price,quantity , 'MO')
                    return tradeOrder
            # if we don't have this ticker's position, we will make a new order decision
            # get the latest 100 seconds market data and downsample by 10s and get the last data in each 10s
            delay_100s = current_time - datetime.timedelta(seconds=100)
            input_stock_data = self.all_market_data[ticker].loc[self.all_market_data[ticker]['time'] > delay_100s].resample('10s', on='time').last().reset_index()
            input_future_data = self.all_market_data[self.stock2future[ticker]].loc[self.all_market_data[self.stock2future[ticker]]['time'] > delay_100s].resample('10s', on='time').last().reset_index()
            # get feature
            features_df = self.generate_features(input_stock_data, input_future_data)
            # get prediction
            prediction = self.models[ticker].predict(features_df).iloc[-1]
            # make decision
            if prediction > 0:
                direction = 'buy'
            elif prediction < 0:
                direction = 'sell'
            else:
                return None
            quantity = self.initial_cash * 0.1 // current_price
            tradeOrder = SingleStockOrder(ticker, datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(),
                                            datetime.datetime.now(), 'New', direction, current_price, quantity, 'MO')
            date, ticker, submissionTime, orderID, currStatus, currStatusTime, direction, price, size, type = tradeOrder.outputAsArray()
            new_order = Submitted_order(date=date, submissionTime=submissionTime, ticker=ticker, orderID=orderID, currStatus=currStatus, currStatusTime=currStatusTime, direction=direction, price=price, size=size, type=type)
            session.add(new_order)
            session.commit()
            session.close()

            return tradeOrder
        else:
            return None

    def cal_slope(self, df):
        bidSizes = [f'bidSize{i}' for i in range(1, 6)]
        bidPrices = [f'bidPrice{i}' for i in range(1, 6)]
        askSizes = [f'askSize{i}' for i in range(1, 6)]
        askPrices = [f'askPrice{i}' for i in range(1, 6)]

        df_bid = df[bidSizes + bidPrices].copy()
        df_ask = df[askSizes + askPrices].copy()
        df_bid.loc[:, bidPrices] = df_bid[bidPrices] / df[bidPrices[0]].values.reshape(-1, 1)
        df_ask.loc[:, askPrices] = df_ask[askPrices] / df[askPrices[0]].values.reshape(-1, 1)

        bid_data = df_bid.values
        ask_data = df_ask.values

        cum_bid_sizes = np.cumsum(bid_data[:, :5], axis=1) / bid_data[:, :5].sum(axis=1, keepdims=True)
        cum_ask_sizes = np.cumsum(ask_data[:, :5], axis=1) / ask_data[:, :5].sum(axis=1, keepdims=True)

        bid_price = bid_data[:, 5:]
        ask_price = ask_data[:, 5:]

        X_bid = cum_bid_sizes - cum_bid_sizes.mean(axis=1, keepdims=True)
        Y_bid = bid_price - bid_price.mean(axis=1, keepdims=True)
        slope_b = (X_bid * Y_bid).sum(axis=1) / ((X_bid ** 2).sum(axis=1) + 1e-10)

        X_ask = cum_ask_sizes - cum_ask_sizes.mean(axis=1, keepdims=True)
        Y_ask = ask_price - ask_price.mean(axis=1, keepdims=True)
        slope_a = (X_ask * Y_ask).sum(axis=1) / ((X_ask ** 2).sum(axis=1) + 1e-10)

        return -slope_b, slope_a

    def generate_features(self, futureData_date, stockData_date):
        basicCols = ['date', 'time', 'sAskPrice1','sBidPrice1','sMidQ', 'fAskPrice1','fBidPrice1', 'fMidQ', 'spreadRatio']
        featureCols = []

        for i in range(1, 11):
            featureCols.extend(['fLaggingRtn_{}'.format(str(i))])
            featureCols.extend(['spreadRatio_{}'.format(str(i))])
            featureCols.extend(['volumeImbalanceRatio_{}'.format(str(i))])
            featureCols.extend(['slope_b_{}'.format(str(i))])
            featureCols.extend(['slope_a_{}'.format(str(i))])
            featureCols.extend(['slope_ab_{}'.format(str(i))])
            featureCols.extend(['sLaggingRtn_{}'.format(str(i))])
            featureCols.extend(['stockSpreadRatio_{}'.format(str(i))])
            featureCols.extend(['stockVolumeImbalanceRatio_{}'.format(str(i))])
            featureCols.extend(['stockSlope_b_{}'.format(str(i))])
            featureCols.extend(['stockSlope_a_{}'.format(str(i))])
            featureCols.extend(['stockSlope_ab_{}'.format(str(i))])

            for j in range(1, 6):
                featureCols.extend(['fAskSize{}_{}'.format(str(j), str(i))])
                featureCols.extend(['fBidSize{}_{}'.format(str(j), str(i))])
                featureCols.extend(['sAskSize{}_{}'.format(str(j), str(i))])
                featureCols.extend(['sBidSize{}_{}'.format(str(j), str(i))])

        df = pd.DataFrame(index=stockData_date.index, columns=basicCols+featureCols)
        df['date'] = stockData_date['date']
        df['time'] = stockData_date['time']

        # Normalize the size
        fAskSizeMax = futureData_date[['askSize1', 'askSize2', 'askSize3', 'askSize4', 'askSize5']].max(axis=1)
        fBidSizeMax = futureData_date[['bidSize1', 'bidSize2', 'bidSize3', 'bidSize4', 'bidSize5']].max(axis=1)
        sAskSizeMax = stockData_date[['askSize1', 'askSize2', 'askSize3', 'askSize4', 'askSize5']].max(axis=1)
        sBidSizeMax = stockData_date[['bidSize1', 'bidSize2', 'bidSize3', 'bidSize4', 'bidSize5']].max(axis=1)

        for i in range(1, 6):
            df['fAskPrice{}'.format(str(i))] = futureData_date['askPrice{}'.format(str(i))]
            df['fBidPrice{}'.format(str(i))] = futureData_date['bidPrice{}'.format(str(i))]
            df['fAskSize{}'.format(str(i))] = futureData_date['askSize{}'.format(str(i))] / fAskSizeMax
            df['fBidSize{}'.format(str(i))] = futureData_date['bidSize{}'.format(str(i))] / fBidSizeMax

            df['sAskPrice{}'.format(str(i))] = stockData_date['askPrice{}'.format(str(i))]
            df['sBidPrice{}'.format(str(i))] = stockData_date['bidPrice{}'.format(str(i))]
            df['sAskSize{}'.format(str(i))] = stockData_date['askSize{}'.format(str(i))] / sAskSizeMax
            df['sBidSize{}'.format(str(i))] = stockData_date['bidSize{}'.format(str(i))] / sBidSizeMax

        df['fMidQ'] = (df['fAskPrice1'] + df['fBidPrice1']) / 2
        df['slope_b'], df['slope_a'] = self.cal_slope(futureData_date)
        df['slope_ab'] = df['slope_a'] - df['slope_b']

        # Order Imbalance Ratio (OIR)
        ask = np.array([df['fAskPrice{}'.format(str(i))] * df['fAskSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        bid = np.array([df['fBidPrice{}'.format(str(i))] * df['fBidSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        df['spreadRatio'] = (ask - bid) / (ask + bid)

        # Order Flow Imbalance (Only 1 level)
        delta_size_bid = np.where(df['fBidPrice1'] < df['fBidPrice1'].shift(1), 0, np.where(df['fBidPrice1'] == df['fBidPrice1'].shift(1), df['fBidSize1'] - df['fBidSize1'].shift(1), df['fBidSize1']))
        delta_size_ask = np.where(df['fAskPrice1'] > df['fAskPrice1'].shift(1), 0, np.where(df['fAskPrice1'] == df['fAskPrice1'].shift(1), df['fAskSize1'] - df['fAskSize1'].shift(1), df['fAskSize1']))
        df['fOrderImbalance'] = (delta_size_bid - delta_size_ask) / (delta_size_bid + delta_size_ask)
        # df['fOrderImbalance'] = (df['fOrderImbalance'] - df['fOrderImbalance'].rolling(10).mean()) / df['fOrderImbalance'].rolling(10).std()

        for i in range(1, 11):
            df['fLaggingRtn_{}'.format(str(i))] = np.log(df['fMidQ']) - np.log(df['fMidQ'].shift(i))
            df['spreadRatio_{}'.format(str(i))] = df['spreadRatio'].shift(i)
            df['volumeImbalanceRatio_{}'.format(str(i))] = df['fOrderImbalance'].shift(i)
            df['slope_b_{}'.format(str(i))] = df['slope_b'].shift(i)
            df['slope_a_{}'.format(str(i))] = df['slope_a'].shift(i)
            df['slope_ab_{}'.format(str(i))] = df['slope_ab'].shift(i)

            for j in range(1, 6):
                df['fAskSize{}_{}'.format(str(j), str(i))] = df['fAskSize{}'.format(str(j))].shift(i)
                df['fBidSize{}_{}'.format(str(j), str(i))] = df['fBidSize{}'.format(str(j))].shift(i)
                df['sAskSize{}_{}'.format(str(j), str(i))] = df['sAskSize{}'.format(str(j))].shift(i)
                df['sBidSize{}_{}'.format(str(j), str(i))] = df['sBidSize{}'.format(str(j))].shift(i)

        # Add stock data
        df['sMidQ'] = (stockData_date['askPrice1'] + stockData_date['bidPrice1']) / 2
        df['sAskPrice1'] = stockData_date['askPrice1']
        df['sBidPrice1'] = stockData_date['bidPrice1']
        df['stockSlope_b'], df['stockSlope_a'] = self.cal_slope(stockData_date)
        df['stockSlope_ab'] = df['stockSlope_a'] - df['stockSlope_b']

        ask = np.array([df['sAskPrice{}'.format(str(i))] * df['sAskSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        bid = np.array([df['sBidPrice{}'.format(str(i))] * df['sBidSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
        df['stockSpreadRatio'] = (ask - bid) / (ask + bid)

        delta_size_bid = np.where(df['sBidPrice1'] < df['sBidPrice1'].shift(1), 0, np.where(df['sBidPrice1'] == df['sBidPrice1'].shift(1), df['sBidSize1'] - df['sBidSize1'].shift(1), df['sBidSize1']))
        delta_size_ask = np.where(df['sAskPrice1'] > df['sAskPrice1'].shift(1), 0, np.where(df['sAskPrice1'] == df['sAskPrice1'].shift(1), df['sAskSize1'] - df['sAskSize1'].shift(1), df['sAskSize1']))
        df['stockOrderImbalance'] = (delta_size_bid - delta_size_ask) / (delta_size_bid + delta_size_ask)
        # df['stockOrderImbalance'] = (df['stockOrderImbalance'] - df['stockOrderImbalance'].rolling(10).mean()) / df['stockOrderImbalance'].rolling(10).std()

        for i in range(1, 11):
            df['sLaggingRtn_{}'.format(str(i))] = np.log(df['sMidQ']) - np.log(df['sMidQ'].shift(i))
            df['stockSpreadRatio_{}'.format(str(i))] = df['stockSpreadRatio'].shift(i)
            df['stockVolumeImbalanceRatio_{}'.format(str(i))] = df['stockOrderImbalance'].shift(i)
            df['stockSlope_b_{}'.format(str(i))] = df['stockSlope_b'].shift(i)
            df['stockSlope_a_{}'.format(str(i))] = df['stockSlope_a'].shift(i)
            df['stockSlope_ab_{}'.format(str(i))] = df['stockSlope_ab'].shift(i)

            for j in range(1, 6):
                df['sAskSize{}_{}'.format(str(j), str(i))] = df['sAskSize{}'.format(str(j))].shift(i)
                df['sBidSize{}_{}'.format(str(j), str(i))] = df['sBidSize{}'.format(str(j))].shift(i)
        # Convert inf to nan to 0
        df = df[featureCols].iloc[[-1]]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # get the last row
        return df

        
