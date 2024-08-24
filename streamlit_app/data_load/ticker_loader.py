import requests
from bs4 import BeautifulSoup
import pandas as pd


sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
nas100_url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
russell2000_url = 'https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund'

rename_dic = {'CRDA' : 'CRDA.L',
              'METCV' : 'METC',
              'MOGA': 'MOG-A',
              'INH': 'INBX',
              'BF.B': 'BF-B',
              'LGFB': 'LGF-B',
              'LGFA': 'LGF-A',
              'BRK.B': 'BRK-B',
              'GEFB': 'GEF-B'}
remove_lst = ['GTXI', '\xa0', 'PDLI', 'CAD', 'P5N994', 'ADRO', 'RTYU4', 'XTSLA', 'MSFUT', ]

def get_sp500_tickers():
    response = requests.get(sp500_url)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    sp500_tickers = df['Symbol'].tolist()
    return sp500_tickers


def get_nasdaq100_tickers():
    response = requests.get(nas100_url)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    nasdaq100_tickers = df['Ticker'].tolist()
    return nasdaq100_tickers



def get_russell2000_tickers():
    df = pd.read_csv(russell2000_url, skiprows=9)
    russell2000_tickers = df['Ticker'].tolist()
    russell2000_tickers = [item for item in russell2000_tickers if item != '-']
    del russell2000_tickers[-1]
    return russell2000_tickers


def get_index_tickers():
    sp500_index = '^GSPC'  # S&P 500
    nasdaq100_index = '^NDX'  # Nasdaq 100
    russell2000_index = '^RUT'  # Russell 2000

    return {
        'S&P 500': sp500_index,
        'Nasdaq 100': nasdaq100_index,
        'Russell 2000': russell2000_index
    }


def get_all_tickers():
    sp500_tickers = get_sp500_tickers()
    nasdaq100_tickers = get_nasdaq100_tickers()
    russell2000_tickers = get_russell2000_tickers()
    index_tickers = get_index_tickers()

    return {
        'S&P 500': sp500_tickers,
        'Nasdaq 100': nasdaq100_tickers,
        'Russell 2000': russell2000_tickers,
        'Indices': list(index_tickers.values())
    }


def get_unique_tickers():
    all_tickers = get_all_tickers()
    res = []
    for key, item in all_tickers.items():
        res += item
    return list(set(res))


def get_final_tickers():
    unique_tickers = get_unique_tickers()
    unique_tickers = [item for item in unique_tickers if item not in remove_lst]
    unique_tickers = [rename_dic[item] if item in rename_dic else item for item in unique_tickers]
    return unique_tickers



