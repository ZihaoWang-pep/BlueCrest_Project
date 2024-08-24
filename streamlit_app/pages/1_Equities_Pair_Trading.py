import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


from data_load.ticker_loader import get_final_tickers
from data_load.helper_function import data_fetch
from data_load.helper_function import kalman_filter_similarity, calculate_distance, backtester_diff, backtester_ratio

client = MongoClient('localhost', 27017)
db = client['bluecrest_data']
collection = db['stock_data']
## Start of the page
st.set_page_config(layout="wide")
st.title("Page1: Equities Pair Trading Analysis")
col1, col2 =st.columns((4,1))
with col1:
    st.write("This page is designed to conduct equity pair-trading analysis & backtesting.")


st.write('---------------------------------')
col3, col4 = st.columns((2,1))
with col3:
    tickers = st.multiselect('Selected Tickers', get_final_tickers(), ['MSFT', 'NVDA', 'META', 'AMZN', 'GOOG', 'TSLA', 'AAPL'])
with col4:
    col5, col6 = st.columns((1,1))
    with col5:
        start_date = st.date_input('Start Date', datetime.date(2020,1,1))
    with col6:
        end_date = st.date_input('End Date', datetime.date(2023,12,31))
start_date = pd.to_datetime(start_date.strftime('%Y-%m-%d'))
end_date = pd.to_datetime(end_date.strftime('%Y-%m-%d'))
df = data_fetch(tickers, 'Close', start_date, end_date)
df = df.dropna()
with col3:
    fig = make_subplots()
    for item in df.columns:
        fig.add_trace(go.Scatter(x = df.index, y = df[item], mode = 'lines+text', name = item))
    fig.update_layout(title_text = 'Close Price Over Selected Window')
    fig.update_layout(height=500)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
with col4:
    df_corr = df.corr().round(3)
    mask = np.zeros_like(df_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna(how='all')
    fig2 = px.imshow(df_corr_viz, text_auto=True)
    fig2.update_layout(title_text = 'Correlation Matrix')
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, theme='streamlit', use_container_width=True)


st.write('---------------------------------')
col7, col8 = st.columns((1,1))
with col7:
    method = st.selectbox("How would you like to measure the distance of stocks?", ("Correlation", 'Cointegration', "OLS R2", "Kalman", 'Normalized Euclidean Distance', 'PCA explained ratio'))
    distance_matrix = calculate_distance(df, tickers, method)
    sorted_pairs = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False).index.tolist()
    try:
        display_pairs = sorted_pairs[:3]
    except:
        display_pairs = sorted_pairs[:]
st.markdown(f'''The top 3 pairs are: :blue[{display_pairs}]''')
with col8:
    the_selection = st.selectbox("Choose the pair of interest (for backtesting)", [(pair, 'Rank #' + str(i+1)) for i, pair in enumerate(sorted_pairs)])
    the_pair = list(the_selection[0])
df_the_pair = data_fetch(list(the_pair))
s1, s2 = df_the_pair[the_pair[0]], df_the_pair[the_pair[1]]
corr_10, corr_126 = s1.rolling(10).corr(s2), s1.rolling(126).corr(s2)
df_the_pair['corr_2W'] = corr_10
df_the_pair['corr_6M'] = corr_126
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
for item in the_pair:
    fig2.add_trace(go.Scatter(x=df_the_pair.index, y=df_the_pair[item], name=item+' Close', mode='lines+text'), secondary_y=False)
for item in ['corr_2W', 'corr_6M']:
    fig2.add_trace(go.Scatter(x=df_the_pair.index, y=df_the_pair[item], mode='lines+text', name=item), secondary_y=True)
fig2.update_layout(title_text='Historical Corr plot', height=500)
fig2.update_xaxes(title_text='Date')
fig2.update_yaxes(title_text='Close Price', secondary_y=False)
fig2.update_yaxes(title_text='Correlation', secondary_y=True)
st.plotly_chart(fig2, theme='streamlit', use_container_width=True)


st.write('---------------------------------')
st.markdown(f'''Monetize the price difference of selected Pair''')
col9, col10, col11, col12 = st.columns((1,1,1,1))
with col9:
    N = st.number_input("Look back window of bollinger bands", value = 63, placeholder="Type a number...")
with col10:
    K = st.number_input("Times of Standard deviation", value = 2.00, placeholder="Type a number...")
with col11:
    s_date = st.date_input('BackTest Starter', datetime.date(2020,1,1))
    s_date = pd.to_datetime(s_date.strftime('%Y-%m-%d'))
with col12:
    e_date = st.date_input('BackTest Ender', datetime.date(2023,8,22))
    e_date = pd.to_datetime(e_date.strftime('%Y-%m-%d'))

col13, col14= st.columns((1,5))
with col14:
    df_res1 = backtester_diff(the_pair, N, K, s_date, e_date)
with col13:
    st.write('Key Statistics of Backtesting')
    sh_r1 = df_res1['daily_pnl'].mean() / df_res1['daily_pnl'].std() * np.sqrt(252)
    so_r1 = (df_res1['daily_pnl'].mean() - 0) / np.sqrt(np.mean(np.minimum(0, df_res1['daily_pnl'] - 0) ** 2)) * np.sqrt(252)
    st.write('Sharpe Ratio:', sh_r1.round(3))
    st.write('Sortino Ratio:', so_r1.round(3))
    running_max1 = np.maximum.accumulate(df_res1['total_value'])
    drawdown1 = (running_max1 - df_res1['total_value']) / running_max1
    st.write('% MaxDrawdown ', (np.max(drawdown1[running_max1 > 0]) * 100).round(2), '%')
    drawdown1 = (running_max1 - df_res1['total_value'])
    st.write('$ MaxDrawdown ', (np.max(drawdown1[running_max1 > 0])).round(2))

st.write('---------------------------------')
st.markdown(f'''Monetize the price ratio of selected Pair''')
col15, col16, col17, col18 = st.columns((1,1,1,1))
with col15:
    N2 = st.number_input("Look back window of bollinger bands", value = 126, placeholder="Type a number...")
with col16:
    K2 = st.number_input("Times of Standard deviation", value = 1.25, placeholder="Type a number...")
with col17:
    s_date2 = st.date_input('BackTest  Starter', datetime.date(2020,1,1))
    s_date2 = pd.to_datetime(s_date2.strftime('%Y-%m-%d'))
with col18:
    e_date2 = st.date_input('BackTest  Ender', datetime.date(2023,8,22))
    e_date2 = pd.to_datetime(e_date2.strftime('%Y-%m-%d'))

col19, col20= st.columns((1,5))
with col20:
    df_res2 = backtester_ratio(the_pair, N2, K2, s_date2, e_date2)
with col19:
    st.write('Key Statistics of Backtesting')
    sh_r2 = df_res2['daily_pnl'].mean() / df_res2['daily_pnl'].std() * np.sqrt(252)
    so_r2 = (df_res2['daily_pnl'].mean() - 0) / np.sqrt(np.mean(np.minimum(0, df_res2['daily_pnl'] - 0) ** 2)) * np.sqrt(252)
    st.write('Sharpe Ratio:', sh_r2.round(3))
    st.write('Sortino Ratio:', so_r2.round(3))
    running_max2 = np.maximum.accumulate(df_res2['total_value'])
    drawdown2 = (running_max2 - df_res2['total_value']) / running_max2
    st.write('% MaxDrawdown ', (np.max(drawdown2[running_max2 > 0]) * 100).round(2), '%')
    drawdown2 = (running_max2 - df_res2['total_value'])
    st.write('$ MaxDrawdown ', (np.max(drawdown2[running_max2 > 0])).round(2))












