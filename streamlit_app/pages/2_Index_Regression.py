import streamlit as st
import datetime
import pandas as pd
import statsmodels.api as sm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import warnings
warnings.simplefilter("ignore")
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

from data_load.ticker_loader import get_final_tickers
from data_load.helper_function import data_fetch


def display_table(table, if_select = False):
    table = table.round(3)
    gb = GridOptionsBuilder.from_dataframe(table)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    if if_select:
        gb.configure_selection(selection_mode='multiple', use_checkbox=True)
    gb.configure_side_bar()
    for item in table.columns:
        gb.configure_column(item, maxWidth=100)
    gridoptions = gb.build()
    response = AgGrid(table, height=400, gridOptions=gridoptions, enable_enterprise_modules=True, update_mode=GridUpdateMode.MODEL_CHANGED, data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                      git_columns_on_grid_load=True, header_checkbox_selection_filtered_only=True, use_checkbox=True)
    return response

def display_plot(table, title):
    fig = make_subplots()
    for item in table.columns:
        fig.add_trace(go.Scatter(x = table.index, y = table[item], mode = 'lines+text', name = item))
    fig.update_layout(title_text = title)
    fig.update_layout(height=400)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

ticker_map = {"S&P 500" : '^GSPC', "Nasdaq 100" : '^NDX',  "Russell 2000" : '^RUT'}


## Start of the page
st.set_page_config(layout="wide")
st.title("Page2: Multi-Variate Index Regression Analysis")
col1, col2 =st.columns((4,1))
with col1:
    st.write("This page allows the user to see how much variance of the selected index can be explained by securities.")


st.write('---------------------------------')
col3, col4, col5 = st.columns((1,1,1))
with col3:
    target_ticker = st.selectbox(label = 'Target Index', options = ("S&P 500", "Nasdaq 100", "Russell 2000"), index=0, placeholder="Target Index")
with col4:
    start_date = st.date_input('Start Date', datetime.date(2023,1,1))
    start_date = pd.to_datetime(start_date.strftime('%Y-%m-%d'))
with col5:
    end_date = st.date_input('End Date', datetime.date(2023,12,31))
    end_date = pd.to_datetime(end_date.strftime('%Y-%m-%d'))
col6, col7 = st.columns((1000,1))
with col6:
    securities = st.multiselect('Selected Tickers', get_final_tickers(), ['MSFT', 'NVDA', 'META', 'AMZN', 'GOOG', 'TSLA', 'AAPL'], max_selections=10)

price_df = data_fetch([ticker_map[target_ticker]] + securities, 'Close', start_date, end_date)
price_df = price_df[[ticker_map[target_ticker]] + securities]
return_df = price_df.pct_change().iloc[1:, :].dropna(axis = 1)
lr_model = sm.OLS( return_df[ticker_map[target_ticker]], sm.add_constant(return_df[securities]) ).fit()

col8, col9, col10 = st.columns((1.5,3, 4.5))
with col8:
    st.write('General Info')
    data_table = pd.Series({'R-square': lr_model.rsquared, 'Adj R-square': lr_model.rsquared_adj, 'F-stat': lr_model.fvalue, 'AIC': lr_model.aic, 'BIC': lr_model.bic}).reset_index()
    data_table.columns = ['stat', 'value']
    data_table['value'] = data_table['value'].astype(float)
    display_table(data_table, if_select=False)
with col9:
    st.write('Coefficient Info')
    data_table = pd.DataFrame({'Coef': lr_model.params, 'TValue': lr_model.tvalues, 'PValue':lr_model.pvalues })
    res2 = display_table(data_table.reset_index(names = 'Ticker'), if_select=True)
with col10:
    st.write('Time Series Plot')
    try:
        selected_names = list(res2['selected_rows']['Ticker'])
    except:
        selected_names = []
    data_plot = return_df.loc[:, [ticker_map[target_ticker]] + selected_names]
    data_plot[ticker_map[target_ticker] + ' Pred'] = lr_model.predict()
    cumprod_df = (data_plot + 1).cumprod()
    display_plot(cumprod_df, title = 'Cumprod Return of Target vs Prediction, with selected names')

st.write('---------------------------------')
st.markdown(f'''The tool that identifies the set of K securities that best explains the index''')
col11, col12, col13 = st.columns((1, 1, 1))
prev_K, prev_method = None, None
with col11:
    K = st.selectbox(label = 'Select Top K stocks', options = (i+1 for i in range(10)), index=1, placeholder="Top K stocks")
with col12:
    method = st.selectbox(label = 'Fitting Criterion', options = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], index=0, placeholder="Fitting Criterion")
st.markdown(f'''Top {K} Securities which best explained the select index, according to selected {method}. :red[Could be very slow for large K]'''  )

with col13:
    run = st.button('Run Analysis')
all_stock = get_final_tickers()
all_stock = [item for item in all_stock if item not in ticker_map.values()]
price_df2 = data_fetch([ticker_map[target_ticker]] + all_stock, 'Close', start_date, end_date)
return_df2 = price_df2.pct_change().iloc[1:, :].dropna(axis=1)

try:
    _ = st.session_state.keep_data
except:
    st.session_state.keep_data = False

if run:
    st.markdown(f''':blue[Running analysis for Top {K} names according to {method}...]''')
    linear_reg = LinearRegression()
    forward = SequentialFeatureSelector(linear_reg, n_features_to_select=K, direction='forward', scoring = method)
    prev_K, prev_method = K, method
    X = return_df2.loc[: , ~return_df2.columns.isin([ticker_map[target_ticker]])]
    sf = forward.fit(X, return_df2[[ticker_map[target_ticker]]])
    selected_stocks = list(sf.get_feature_names_out())
    st.markdown(f'''According to {method}, top {K} names which best explained {ticker_map[target_ticker]} are :blue[{selected_stocks}]''')
    st.session_state.selected_stocks = selected_stocks
    st.session_state.keep_data = True
else:
    if not st.session_state.keep_data:
        st.markdown('''Please Click Button for analysis!''')
    else:
        pass

if run or st.session_state.keep_data:
    if not run:
        selected_stocks = st.session_state.selected_stocks
    col14, col15, col16 = st.columns((1.5, 3, 4.5))
    return_df2_small = return_df2[[ticker_map[target_ticker]] + selected_stocks]
    lr_model2 = sm.OLS( return_df2_small[ticker_map[target_ticker]], sm.add_constant(return_df2_small[selected_stocks])).fit()
    with col14:
        st.write('General Info')
        data_table2 = pd.Series({'R-square': lr_model2.rsquared, 'Adj R-square': lr_model2.rsquared_adj, 'F-stat': lr_model2.fvalue, 'AIC': lr_model2.aic, 'BIC': lr_model2.bic}).reset_index()
        data_table2.columns = ['stat', 'value']
        data_table2['value'] = data_table2['value'].astype(float)
        display_table(data_table2, if_select=False)
    with col15:
        st.write('Coefficient Info')
        data_table2 = pd.DataFrame({'Coef': lr_model2.params, 'TValue': lr_model2.tvalues, 'PValue': lr_model2.pvalues})
        res3 = display_table(data_table2.reset_index(names='Ticker'), if_select=True)
    with col16:
        try:
            selected_names2 = list(res3['selected_rows']['Ticker'])
        except:
            selected_names2 = []
        data_plot2 = return_df2_small.loc[:, [ticker_map[target_ticker]] + selected_names2]
        data_plot2[ticker_map[target_ticker] + ' Pred'] = lr_model2.predict()
        cumprod_df2 = (data_plot2 + 1).cumprod()
        display_plot(cumprod_df2, title='Cumprod Return of Target vs Prediction, with selected names')


