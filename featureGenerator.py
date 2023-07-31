import pandas as pd


# Generate features and labels
# for future_ticker, futureData in future_data.items():
#     stockData = stock_data[future2stock[future_ticker]]
#     unique_dates = np.unique(stockData['date'])
#     dfs = []
#     for date in unique_dates:
#         stockData_date = stockData[stockData['date'] == date]
#         futureData_date = futureData[futureData['date'] == date]

#         # Continue to next iteration if stockData_date or futureData_date is empty
#         if stockData_date.empty or futureData_date.empty:
#             continue

#         df = pd.DataFrame(index=stockData_date.index, columns=basicCols+labelCols+featureCols)
#         df['date'] = stockData_date['date']
#         df['time'] = stockData_date['time']   
             
#         # Normalize the size
#         fAskSizeMax = futureData_date[['askSize1', 'askSize2', 'askSize3', 'askSize4', 'askSize5']].max(axis=1)
#         fBidSizeMax = futureData_date[['bidSize1', 'bidSize2', 'bidSize3', 'bidSize4', 'bidSize5']].max(axis=1)
#         sAskSizeMax = stockData_date[['SV1', 'SV2', 'SV3', 'SV4', 'SV5']].max(axis=1)
#         sBidSizeMax = stockData_date[['BV1', 'BV2', 'BV3', 'BV4', 'BV5']].max(axis=1)
        
#         for i in range(1, 6):
#             df['fAskPrice{}'.format(str(i))] = futureData_date['askPrice{}'.format(str(i))]
#             df['fBidPrice{}'.format(str(i))] = futureData_date['bidPrice{}'.format(str(i))]
#             df['fAskSize{}'.format(str(i))] = futureData_date['askSize{}'.format(str(i))] / fAskSizeMax
#             df['fBidSize{}'.format(str(i))] = futureData_date['bidSize{}'.format(str(i))] / fBidSizeMax
            
#             df['sAskPrice{}'.format(str(i))] = stockData_date['SP{}'.format(str(i))]
#             df['sBidPrice{}'.format(str(i))] = stockData_date['BP{}'.format(str(i))]
#             df['sAskSize{}'.format(str(i))] = stockData_date['SV{}'.format(str(i))] / sAskSizeMax
#             df['sBidSize{}'.format(str(i))] = stockData_date['BV{}'.format(str(i))] / sBidSizeMax
        
#         df['fMidQ'] = (df['fAskPrice1'] + df['fBidPrice1']) / 2
#         df['slope_b'], df['slope_a'] = cal_slope(futureData_date)
#         df['slope_ab'] = df['slope_a'] - df['slope_b']
        
#         # Order Imbalance Ratio (OIR)
#         ask = np.array([df['fAskPrice{}'.format(str(i))] * df['fAskSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
#         bid = np.array([df['fBidPrice{}'.format(str(i))] * df['fBidSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
#         df['spreadRatio'] = (ask - bid) / (ask + bid)

#         # Order Flow Imbalance (Only 1 level)
#         delta_size_bid = np.where(df['fBidPrice1'] < df['fBidPrice1'].shift(1), 0, np.where(df['fBidPrice1'] == df['fBidPrice1'].shift(1), df['fBidSize1'] - df['fBidSize1'].shift(1), df['fBidSize1']))
#         delta_size_ask = np.where(df['fAskPrice1'] > df['fAskPrice1'].shift(1), 0, np.where(df['fAskPrice1'] == df['fAskPrice1'].shift(1), df['fAskSize1'] - df['fAskSize1'].shift(1), df['fAskSize1']))
#         df['fOrderImbalance'] = (delta_size_bid - delta_size_ask) / (delta_size_bid + delta_size_ask)
#         # df['fOrderImbalance'] = (df['fOrderImbalance'] - df['fOrderImbalance'].rolling(10).mean()) / df['fOrderImbalance'].rolling(10).std()
        
#         for i in range(1, 11):
#             df['fLaggingRtn_{}'.format(str(i))] = np.log(df['fMidQ']) - np.log(df['fMidQ'].shift(i))
#             df['spreadRatio_{}'.format(str(i))] = df['spreadRatio'].shift(i)
#             df['volumeImbalanceRatio_{}'.format(str(i))] = df['fOrderImbalance'].shift(i)
#             df['slope_b_{}'.format(str(i))] = df['slope_b'].shift(i)
#             df['slope_a_{}'.format(str(i))] = df['slope_a'].shift(i)
#             df['slope_ab_{}'.format(str(i))] = df['slope_ab'].shift(i)
            
#             for j in range(1, 6):
#                 df['fAskSize{}_{}'.format(str(j), str(i))] = df['fAskSize{}'.format(str(j))].shift(i)
#                 df['fBidSize{}_{}'.format(str(j), str(i))] = df['fBidSize{}'.format(str(j))].shift(i)
#                 df['sAskSize{}_{}'.format(str(j), str(i))] = df['sAskSize{}'.format(str(j))].shift(i)
#                 df['sBidSize{}_{}'.format(str(j), str(i))] = df['sBidSize{}'.format(str(j))].shift(i)

#         # Add stock data
#         df['sMidQ'] = (stockData_date['SP1'] + stockData_date['BP1'])/2
#         df['sAskPrice1'] = stockData_date['SP1']
#         df['sBidPrice1'] = stockData_date['BP1']
#         df['sMidQ'] = (stockData_date['SP1'] + stockData_date['BP1'])/2
#         df['stockSlope_b'], df['stockSlope_a'] = cal_slope(stockData_date)
#         df['stockSlope_ab'] = df['stockSlope_a'] - df['stockSlope_b']
        
#         ask = np.array([df['sAskPrice{}'.format(str(i))] * df['sAskSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
#         bid = np.array([df['sBidPrice{}'.format(str(i))] * df['sBidSize{}'.format(str(i))] * (1 - (i - 1) / 5) for i in range(1, 6)]).sum(axis=0)
#         df['stockSpreadRatio'] = (ask - bid) / (ask + bid)

#         delta_size_bid = np.where(df['sBidPrice1'] < df['sBidPrice1'].shift(1), 0, np.where(df['sBidPrice1'] == df['sBidPrice1'].shift(1), df['sBidSize1'] - df['sBidSize1'].shift(1), df['sBidSize1']))
#         delta_size_ask = np.where(df['sAskPrice1'] > df['sAskPrice1'].shift(1), 0, np.where(df['sAskPrice1'] == df['sAskPrice1'].shift(1), df['sAskSize1'] - df['sAskSize1'].shift(1), df['sAskSize1']))
#         df['stockOrderImbalance'] = (delta_size_bid - delta_size_ask) / (delta_size_bid + delta_size_ask)
#         # df['stockOrderImbalance'] = (df['stockOrderImbalance'] - df['stockOrderImbalance'].rolling(10).mean()) / df['stockOrderImbalance'].rolling(10).std()
#         for i in range(1, 11):
#             df['Y_M_{}'.format(str(i))] = np.log(df['sMidQ'].shift(-i)) - np.log(df['sMidQ'])
            
#             df['sLaggingRtn_{}'.format(str(i))] = np.log(df['sMidQ']) - np.log(df['sMidQ'].shift(i))
#             df['stockSpreadRatio_{}'.format(str(i))] = df['stockSpreadRatio'].shift(i)
#             df['stockVolumeImbalanceRatio_{}'.format(str(i))] = df['stockOrderImbalance'].shift(i)
#             df['stockSlope_b_{}'.format(str(i))] = df['stockSlope_b'].shift(i)
#             df['stockSlope_a_{}'.format(str(i))] = df['stockSlope_a'].shift(i)
#             df['stockSlope_ab_{}'.format(str(i))] = df['stockSlope_ab'].shift(i)
            
#             for j in range(1, 6):
#                 df['sAskSize{}_{}'.format(str(j), str(i))] = df['sAskSize{}'.format(str(j))].shift(i)
#                 df['sBidSize{}_{}'.format(str(j), str(i))] = df['sBidSize{}'.format(str(j))].shift(i)
#         dfs.append(df)
#     # Convert inf to nan to 0
#     df_all[future_ticker] = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)

def generate_features(future_data, stock_data):
    # Columns in future data and stock data
    # outputCols = ['ticker','date','time', \
    #                 'askPrice5','askPrice4','askPrice3','askPrice2','askPrice1', \
    #                 'bidPrice1','bidPrice2','bidPrice3','bidPrice4','bidPrice5', \
    #                 'askSize5','askSize4','askSize3','askSize2','askSize1', \
    #                 'bidSize1','bidSize2','bidSize3','bidSize4','bidSize5']
    
    
        
            