from pymongo import MongoClient
from sqlalchemy.orm.collections import collection
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from sklearn.decomposition import PCA

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['bluecrest_data']
collection = db['stock_data_prod1']


def data_fetch(ticker_list, data_type='Close', start=None, end=None):
    if ((start is None) and (end is None)):
        query = {
            "Ticker": {"$in": ticker_list},
        }
    else:
        query = {
            "Ticker": {"$in": ticker_list},
            "Date": {"$gte": start, "$lte": end}
        }
    documents = collection.find(query,  {'Date': 1, 'Ticker':1, data_type: 1})
    df = pd.DataFrame(list(documents))
    df = df[['Date', 'Ticker', data_type]]
    df = df.pivot(index = 'Date', columns = 'Ticker', values = data_type)
    df.sort_index(inplace=True)
    return df


def data_fetch_backtest(ticker_list, start=None, end=None):
    if ((start is None) and (end is None)):
        query = {
            "Ticker": {"$in": ticker_list},
        }
    else:
        query = {
            "Ticker": {"$in": ticker_list},
            "Date": {"$gte": start, "$lte": end}
        }
    documents = collection.find(query,  {'Date': 1, 'Ticker':1, 'Adj Close': 1, 'Open': 1, 'Close': 1})
    df = pd.DataFrame(list(documents))
    df = df[['Date', 'Ticker', 'Adj Close', 'Open', 'Close']]
    df = df.set_index('Date')
    df.sort_index(inplace=True)
    return df


def kalman_filter_similarity(stock1, stock2):
    n = len(stock1)

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[0.]])  # initial state (initial guess for beta)
    kf.F = np.array([[1.]])  # state transition matrix
    kf.H = np.array([[0.]])  # measurement function (to be updated each step)
    kf.P = np.array([[1.]])  # covariance matrix (initial uncertainty)
    kf.R = 0.01  # measurement noise
    kf.Q = 0.01  # process noise

    beta_estimates = []
    for t in range(n):
        kf.H = np.array([[stock2[t]]])  # Update measurement matrix with stock2 at time t
        kf.predict()
        kf.update(stock1[t])
        beta_estimates.append(kf.x[0, 0])
    return np.array(beta_estimates)


def calculate_distance(df, tickers, method):
    the_matrix = pd.DataFrame(index=tickers, columns=tickers)

    if method == "Correlation":
        the_matrix = df.corr().abs()
    elif method == "OLS R2":  # in the uni-variate regression, R-square is the square of the correlation
        the_matrix = ((df.corr()) ** 2)
    elif method == "Kalman":
        for x, ticker_x in enumerate(tickers):
            for y, ticker_y in enumerate(tickers):
                if y >= x:
                    betas = kalman_filter_similarity(df[ticker_x], df[ticker_y])
                    print(ticker_x, ticker_y, betas.mean(), betas.std())
                    the_matrix.iloc[x, y] = -1 * betas.std()
    elif method == "Cointegration":
        for x, ticker_x in enumerate(tickers):
            for y, ticker_y in enumerate(tickers):
                if y >= x:
                    score, p_value, _ = coint(df[ticker_x], df[ticker_y])
                    the_matrix.iloc[x, y] = -1 * p_value
    elif method == "Normalized Euclidean Distance":
        df_norm = (df - df.mean()) / df.std()
        for x, ticker_x in enumerate(tickers):
            for y, ticker_y in enumerate(tickers):
                if y >= x:
                    distance = np.linalg.norm(df_norm[ticker_x] - df_norm[ticker_y])
                    the_matrix.iloc[x, y] = -1 * distance
    elif method == "PCA explained ratio":
        pca = PCA(n_components=2)
        for x, ticker_x in enumerate(tickers):
            for y, ticker_y in enumerate(tickers):
                if y > x:
                    pca.fit(df[[ticker_x, ticker_y]])
                    the_matrix.iloc[x, y] = pca.explained_variance_ratio_[0]
    return the_matrix


def get_trade(the_pair, N, K, df_backtest):

    df_close = df_backtest[['Ticker', 'Close']].reset_index().pivot(index='Date', columns='Ticker', values='Close')
    df_open = df_backtest[['Ticker', 'Open']].reset_index().pivot(index='Date', columns='Ticker', values='Open')
    sa, sb = the_pair[0], the_pair[1]

    df_close['spread'] = df_close[sa] - df_close[sb]
    df_open['spread'] = df_open[sa] - df_open[sb]
    df_close['spread_ma'] = df_close['spread'].rolling(N).mean()
    df_close['spread_std'] = df_close['spread'].rolling(N).std()
    df_close['upper_band'] = df_close['spread_ma'] + K * df_close['spread_std']
    df_close['lower_band'] = df_close['spread_ma'] - K * df_close['spread_std']
    df_close['long_spread'] = df_close['spread'] < df_close['lower_band']
    df_close['short_spread'] = df_close['spread'] > df_close['upper_band']
    df_close['exit'] = (df_close['spread'] < df_close['spread_ma']) & (df_close['spread'] > df_close['lower_band']) | \
                       (df_close['spread'] > df_close['spread_ma']) & (df_close['spread'] < df_close['upper_band'])

    df_input = df_close[['long_spread', 'short_spread', 'exit']]
    df_input['long_spread'] = df_input['long_spread'].apply(lambda x: 1 if x else 0)
    df_input['short_spread'] = df_input['short_spread'].apply(lambda x: -1 if x else 0)
    df_input['exit'] = df_input['exit'].apply(lambda x: 0)
    df_input['final'] = df_input['long_spread'] + df_input['short_spread'] + df_input['exit']
    df_input = df_input.shift().dropna().reset_index()

    df_input['trade'] = df_input['final']
    for i in range(1, len(df_input)):
        if df_input.loc[i, 'final'] == 0:
            if df_input.loc[i - 1, 'final'] == 0:
                df_input.loc[i, 'trade'] = 0
            else:
                j = i - 1
                consecutive_sum = 0
                while j >= 0 and df_input.loc[j, 'final'] != 0:
                    consecutive_sum += df_input.loc[j, 'final']
                    j -= 1
                df_input.loc[i, 'trade'] = -consecutive_sum
    df_input.set_index('Date', inplace=True)
    df_input['hold'] = df_input['trade'].cumsum()

    return df_input, df_close, df_open


def plot_pnl_spread(the_pair, df_input):
    curr_fig = make_subplots(specs=[[{"secondary_y": True}]])
    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['Spread'], name='Spread price', mode='lines+text'))
    for item in the_pair:
        curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input[item], name=item + ' price', mode='lines+text'))

    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['total_value'], name='Strategy PnL', mode='lines+text'), secondary_y=True)
    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['long_action'], name='Long Spread', mode='markers', marker=dict(color='green', symbol='triangle-up')))
    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['short_action'], name='Short Spread', mode='markers', marker=dict(color='red', symbol='triangle-down')))

    curr_fig.update_layout(title_text='Strategy Backtest', height=500)
    curr_fig.update_xaxes(title_text='Date')
    curr_fig.update_yaxes(title_text='Close Price', secondary_y=False)
    curr_fig.update_yaxes(title_text='PnL', secondary_y=True)
    st.plotly_chart(curr_fig, theme='streamlit', use_container_width=True)


def plot_pnl_ratio(df_input):

    curr_fig = make_subplots(specs=[[{"secondary_y": True}]])
    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['Ratio'], name='Spread price', mode='lines+text'))

    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['total_value'], name='Strategy PnL', mode='lines+text'), secondary_y=True)
    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['long_action'], name='Long Spread', mode='markers', marker=dict(color='green', symbol='triangle-up')))
    curr_fig.add_trace(go.Scatter(x=df_input.index, y=df_input['short_action'], name='Short Spread', mode='markers', marker=dict(color='red', symbol='triangle-down')))

    curr_fig.update_layout(title_text='Strategy Backtest', height=500)
    curr_fig.update_xaxes(title_text='Date')
    curr_fig.update_yaxes(title_text='Close Price', secondary_y=False)
    curr_fig.update_yaxes(title_text='PnL', secondary_y=True)
    st.plotly_chart(curr_fig, theme='streamlit', use_container_width=True)


def get_signal_diff(df_close, sa, sb, N, K):

    df_close['spread'] = df_close[sa] - df_close[sb]
    df_close['spread_ma'] = df_close['spread'].rolling(N).mean()
    df_close['spread_std'] = df_close['spread'].rolling(N).std()
    df_close['upper_band'] = df_close['spread_ma'] + K * df_close['spread_std']
    df_close['lower_band'] = df_close['spread_ma'] - K * df_close['spread_std']
    df_close['long_spread'] = df_close['spread'] < df_close['lower_band']
    df_close['short_spread'] = df_close['spread'] > df_close['upper_band']
    df_close['exit'] = (df_close['spread'] < df_close['spread_ma']) & (df_close['spread'] > df_close['lower_band']) | \
                       (df_close['spread'] > df_close['spread_ma']) & (df_close['spread'] < df_close['upper_band'])
    return df_close


def get_signal_ratio(df_close, sa, sb, N, K):

    df_close['ratio'] = df_close[sa] / df_close[sb]
    df_close['ratio_ma'] = df_close['ratio'].rolling(N).mean()
    df_close['ratio_std'] = df_close['ratio'].rolling(N).std()
    df_close['upper_band'] = df_close['ratio_ma'] + K * df_close['ratio_std']
    df_close['lower_band'] = df_close['ratio_ma'] - K * df_close['ratio_std']
    df_close['long_ratio'] = df_close['ratio'] < df_close['lower_band']
    df_close['short_ratio'] = df_close['ratio'] > df_close['upper_band']
    df_close['exit'] = (df_close['ratio'] < df_close['ratio_ma']) & (df_close['ratio'] > df_close['lower_band']) | \
                       (df_close['ratio'] > df_close['ratio_ma']) & (df_close['ratio'] < df_close['upper_band'])

    return df_close


def get_pnl_diff(df_close, df_open):
    df_input = df_close[['long_spread', 'short_spread', 'exit']]
    df_input['long_spread'] = df_input['long_spread'].apply(lambda x: 1 if x else 0)
    df_input['short_spread'] = df_input['short_spread'].apply(lambda x: -1 if x else 0)
    df_input['exit'] = df_input['exit'].apply(lambda x: 0)
    df_input['final'] = df_input['long_spread'] + df_input['short_spread'] + df_input['exit']
    df_input = df_input.shift().dropna().reset_index()

    df_input['trade'] = df_input['final']
    for i in range(1, len(df_input)):
        if df_input.loc[i, 'final'] == 0:
            if df_input.loc[i - 1, 'final'] == 0:
                df_input.loc[i, 'trade'] = 0
            else:
                j = i - 1
                consecutive_sum = 0
                while j >= 0 and df_input.loc[j, 'final'] != 0:
                    consecutive_sum += df_input.loc[j, 'final']
                    j -= 1
                df_input.loc[i, 'trade'] = -consecutive_sum
    df_input.set_index('Date', inplace=True)
    df_input['hold'] = df_input['trade'].cumsum()

    df_input['open_spread'] = df_open['spread']
    df_input['close_spread'] = df_close['spread']
    df_input['trade_pnl'] = df_input['trade'] * df_input['open_spread'] * (-1)
    df_input['hold_value'] = df_input['hold'] * df_input['close_spread']
    df_input['total_value'] = df_input['trade_pnl'].cumsum() + df_input['hold_value']
    df_input['daily_pnl'] = df_input['total_value'] - df_input['total_value'].shift()

    return df_input


def get_pnl_ratio(df_close, df_open, the_pair):
    df_input = df_close[['long_ratio', 'short_ratio', 'exit']]
    df_input['long_spread'] = df_input['long_ratio'].apply(lambda x: 1 if x else 0)
    df_input['short_spread'] = df_input['short_ratio'].apply(lambda x: -1 if x else 0)
    df_input['exit'] = df_input['exit'].apply(lambda x: 0)
    df_input['final'] = df_input['long_spread'] + df_input['short_spread'] + df_input['exit']
    df_input[the_pair[0] + 'open'] = df_open[the_pair[0]]
    df_input[the_pair[1] + 'open'] = df_open[the_pair[1]]
    df_input[the_pair[0] + 'close'] = df_close[the_pair[0]]
    df_input[the_pair[1] + 'close'] = df_close[the_pair[1]]
    df_input[the_pair[0]] = df_close[the_pair[0]]
    df_input[the_pair[1]] = df_close[the_pair[1]]
    df_input['Ratio'] = df_input[the_pair[0]] / df_input[the_pair[1]]
    df_input['open_ratio'] = df_open['ratio']
    df_input['close_ratio'] = df_close['ratio']
    df_input = df_input.shift().dropna().reset_index()

    trading_pnl, pos_value, count0, count1 = 0, 0, 0, 0
    trading_pnl_lst, pos_value_lst = [], []
    for i in range(df_input.shape[0]):
        ratio = df_input['open_ratio'][i]
        if df_input['final'][i] == 1:
            trading_pnl = 0
            count0 += 1
            count1 -= ratio
            pos_value = df_input[the_pair[0] + 'close'][i] * count0 + df_input[the_pair[1] + 'close'][i] * count1
        elif df_input['final'][i] == -1:
            trading_pnl = 0
            count0 -= 1
            count1 += ratio
            pos_value = df_input[the_pair[0] + 'close'][i] * count0 + df_input[the_pair[1] + 'close'][i] * count1
        else:
            trading_pnl = df_input[the_pair[0] + 'close'][i] * count0 + df_input[the_pair[1] + 'close'][i] * count1
            count0, count1, pos_value = 0, 0, 0
        trading_pnl_lst.append(trading_pnl)
        pos_value_lst.append(pos_value)
    df_input['trade_pnl'] = trading_pnl_lst
    df_input['hold_value'] = pos_value_lst

    return df_input


def backtester_diff(the_pair, N, K, start_date, end_date):

    df_backtest = data_fetch_backtest(list(the_pair), start_date, end_date)
    df_close = df_backtest[['Ticker', 'Close']].reset_index().pivot(index='Date', columns='Ticker', values='Close')
    df_open = df_backtest[['Ticker', 'Open']].reset_index().pivot(index='Date', columns='Ticker', values='Open')
    sa, sb = the_pair[0], the_pair[1]

    df_open['spread'] = df_open[sa] - df_open[sb]
    df_close = get_signal_diff(df_close, sa, sb, N, K)

    df_input = get_pnl_diff(df_close, df_open)

    df_input[the_pair[0]] = df_close[the_pair[0]]
    df_input[the_pair[1]] = df_close[the_pair[1]]
    df_input['Spread'] = df_input[the_pair[0]] - df_input[the_pair[1]]
    df_input['long_action'] = df_input.loc[:, 'Spread']
    df_input['long_action'][df_input['final'] < 1] = None
    df_input['short_action'] = df_input.loc[:, 'Spread']
    df_input['short_action'][df_input['final'] > -1] = None

    plot_pnl_spread(the_pair, df_input)

    return df_input


def backtester_ratio(the_pair, N, K, start_date, end_date):

    df_backtest = data_fetch_backtest(list(the_pair), start_date, end_date)
    df_close = df_backtest[['Ticker', 'Close']].reset_index().pivot(index='Date', columns='Ticker', values='Close')
    df_open = df_backtest[['Ticker', 'Open']].reset_index().pivot(index='Date', columns='Ticker', values='Open')
    sa, sb = the_pair[0], the_pair[1]

    df_open['ratio'] = df_open[sa] / df_open[sb]
    df_close = get_signal_ratio(df_close, sa, sb, N, K)

    df_input = get_pnl_ratio(df_close, df_open, the_pair)

    df_input['total_value'] = df_input['trade_pnl'].cumsum() + df_input['hold_value']
    df_input['daily_pnl'] = df_input['total_value'] - df_input['total_value'].shift()
    df_input['long_action'] = df_input.loc[:, 'Ratio']
    df_input['long_action'][df_input['final'] < 1] = None
    df_input['short_action'] = df_input.loc[:, 'Ratio']
    df_input['short_action'][df_input['final'] > -1] = None
    plot_pnl_ratio(df_input)

    return df_input