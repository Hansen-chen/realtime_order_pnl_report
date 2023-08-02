#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou
"""

import os
import time, datetime
from datetime import timedelta
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
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
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
            one_min_return = Column(Float())
            ten_min_return = Column(Float())
            portfolio_volatility = Column(Float())
            max_drawdown = Column(Float())



        self.networth = pd.DataFrame(columns=['date','timestamp','networth'])

        self.current_position = pd.DataFrame(columns=['ticker', 'quantity', 'price','dollar_amount'])


        self.submitted_order = pd.DataFrame(
            columns=['date', 'submissionTime', 'ticker', 'orderID', 'currStatus', 'currStatusTime', 'direction',
                     'price', 'size', 'type'])


        self.metrics = pd.DataFrame(columns=['cumulative_return', 'one_min_return', 'ten_min_return', 'portfolio_volatility', 'max_drawdown'])



        # Create an engine and session
        self.engine = create_engine('sqlite:///database.db')  # Database Abstraction and Portability
        self.session_factory = sessionmaker(bind=self.engine)
        Session = scoped_session(self.session_factory)
        session = Session()

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

    def cancel_not_filled_orders(self):
        Base = automap_base()
        Base.prepare(autoload_with=self.engine)

        Submitted_order = Base.classes.submitted_order
        timeStamp = datetime.datetime.now() - timedelta(seconds=5)
        Session = scoped_session(self.session_factory)
        orders_to_cancel = []

        #cancel order if the submissionTime compared to the current time is more than 5 seconds .
        session = Session()
        cancel_orders = session.query(Submitted_order).filter(Submitted_order.submissionTime < timeStamp, Submitted_order.currStatus == 'New').all()
        session.close()
        if cancel_orders is not None:
            for order in cancel_orders:
                _order = SingleStockOrder(order.ticker, order.date, order.submissionTime, order.currStatusTime, order.currStatus,'cancel' , order.price, order.size, order.type)
                _order.orderID = order.orderID
                orders_to_cancel.append(_order)

        return orders_to_cancel


    def run(self, marketData, execution):
        Base = automap_base()
        Base.prepare(autoload_with=self.engine)

        Submitted_order = Base.classes.submitted_order
        Current_position = Base.classes.current_position
        Metrics = Base.classes.portfolio_metrics
        Networth = Base.classes.networth
        Session = scoped_session(self.session_factory)

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

            # locate the row in self.submitted_order with orderID, then update the currStatus and currStatusTime, check if the size of executed order is the same as the size of submitted order, if less than, the status is PartiallyFilled, if equal, the status is Filled, if more than, return Non and print error

            submitted_order = session.query(Submitted_order).filter_by(orderID=orderID).first()

            #check if the orderID is in self.submitted_order
            if submitted_order is None:
                print('Error: orderID not in submitted_order')
                return None
            else:
                if direction == 'cancel':
                    submitted_order.currStatus = 'Cancelled'
                    session.commit()
                    session.close()
                    return None
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
                    # if we have balanced the position, we will update the inception_timestamp
                    # or keep the inception_timestamp
                    new_quantity = current_position.quantity + size
                    current_position.quantity = new_quantity
                    if new_quantity == 0:
                        current_position.inception_timestamp = timeStamp
                    session.commit()
                else:
                    new_position = Current_position(price=price, inception_timestamp=timeStamp, ticker=ticker, quantity=size)
                    session.add(new_position)
                    session.commit()
            elif direction == 'sell':
                if current_position is not None:
                    new_quantity = current_position.quantity - size
                    current_position.quantity = new_quantity
                    if new_quantity == 0:
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

                # filter the networthes df with timestamp within 1 minute before the current timestamp (networthes.iloc[-1]['timestamp'])
                one_minutes_networthes = networthes[networthes['timestamp'] >= networthes.iloc[-1]['timestamp'] - timedelta(minutes=1)]
                one_min_return = (one_minutes_networthes.iloc[-1]['networth'] / one_minutes_networthes.iloc[0][
                    'networth'] - 1) * 100

                # filter the networthes df with timestamp within 10 minute before the current timestamp (networthes.iloc[-1]['timestamp'])
                ten_minutes_networthes = networthes[networthes['timestamp'] >= networthes.iloc[-1]['timestamp'] - timedelta(minutes=10)]
                ten_min_return = (ten_minutes_networthes.iloc[-1]['networth'] / ten_minutes_networthes.iloc[0][
                    'networth'] - 1) * 100

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
                                      max_drawdown=max_drawdown, one_min_return=one_min_return, ten_min_return=ten_min_return)
                    session.add(metrics)
                else:
                    metrics.cumulative_return = cumulative_return
                    metrics.portfolio_volatility = portfolio_volatility
                    metrics.max_drawdown = max_drawdown
                    metrics.one_min_return = one_min_return
                    metrics.ten_min_return = ten_min_return
                session.commit()
            session.close()
            print(execution.outputAsArray())
            return None
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels))) and (execution is None):

            #TODO: save market data to sqlite?

            current_market_data = marketData.outputAsDataFrame()

            #check if askPrice1 or bidPrice1 is empty, if it is, print error and then return None
            if current_market_data.iloc[0]['askPrice1'] == 0 or current_market_data.iloc[0]['bidPrice1'] == 0:
                print('Error: askPrice1 or bidPrice1 is empty')
                return None

            current_date = current_market_data.iloc[0]['date']
            current_time = current_market_data.iloc[0]['time']
            session = Session()

            networthes = session.query(Networth).filter_by(networth=self.initial_cash).first()
            if networthes is None:
                networth = Networth(date=current_date, timestamp=current_time, networth=self.initial_cash)
                session.add(networth)
                session.commit()
                position = Current_position(price=1, inception_timestamp=current_time, ticker='cash', quantity=self.initial_cash)
                session.add(position)
                session.commit()
                metrics = Metrics(cumulative_return=0, portfolio_volatility=0, max_drawdown=0.0, one_min_return=0.0, ten_min_return=0.0)
                session.add(metrics)
                session.commit()
                print("initialize networth, current_position and portfolio_metrics")

            #handle new market data, then create a new order and send it via quantTradingPlatform if needed

            #update networth and current_position if there is an open position related to this ticker
            ticker = current_market_data.iloc[0]['ticker']

            #TODO: check ticker?


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

                #filter the networthes df with timestamp within 1 minute before the current timestamp (networthes.iloc[-1]['timestamp'])
                one_minutes_networthes = networthes[networthes['timestamp'] >= networthes.iloc[-1]['timestamp'] - timedelta(minutes=1)]
                one_min_return = (one_minutes_networthes.iloc[-1]['networth'] / one_minutes_networthes.iloc[0]['networth'] - 1) * 100

                # filter the networthes df with timestamp within 10 minute before the current timestamp (networthes.iloc[-1]['timestamp'])
                ten_minutes_networthes = networthes[networthes['timestamp'] >= networthes.iloc[-1]['timestamp'] - timedelta(minutes=10)]
                ten_min_return = (ten_minutes_networthes.iloc[-1]['networth'] / ten_minutes_networthes.iloc[0]['networth'] - 1) * 100


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
                    metrics = Metrics(cumulative_return=cumulative_return, portfolio_volatility=portfolio_volatility, max_drawdown=max_drawdown, one_min_return=one_min_return, ten_min_return=ten_min_return)
                    session.add(metrics)
                else:
                    metrics.cumulative_return = cumulative_return
                    metrics.portfolio_volatility = portfolio_volatility
                    metrics.max_drawdown = max_drawdown
                    metrics.one_min_return = one_min_return
                    metrics.ten_min_return = ten_min_return
                session.commit()


            tradeOrder = None
            current_cash_query = session.query(Current_position).filter_by(ticker="cash").first()

            current_cash = 0

            if current_cash_query is not None:
                current_cash = current_cash_query.quantity

            #TODO: decide the tradeOrder

            if random.choice([True, False]):
                ticker = current_market_data['ticker'].values[0]
                direction = random.choice(["buy", "sell"])

                current_price = (current_market_data.iloc[0]['askPrice1'] + current_market_data.iloc[0]['bidPrice1']) / 2
                quantity = 100
                #check if there is enough cash to buy
                if (current_cash < current_price*quantity) and (direction == 'buy'):
                    print('Error: Not enough cash to buy')
                    return None

                tradeOrder = SingleStockOrder(ticker, datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now(), datetime.datetime.now(), 'New', direction, current_price,quantity , 'MO')
                date, ticker, submissionTime, orderID, currStatus, currStatusTime, direction, price, size, type = tradeOrder.outputAsArray()
                new_order = Submitted_order(date=date, submissionTime=submissionTime, ticker=ticker, orderID=orderID, currStatus=currStatus, currStatusTime=currStatusTime, direction=direction, price=price, size=size, type=type)
                session.add(new_order)
                session.commit()
            session.close()

            return tradeOrder
        else:
            return None

        