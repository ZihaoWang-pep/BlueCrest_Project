from pymongo import MongoClient
import yfinance as yf
import datetime
from streamlit_app.data_load.ticker_loader import get_final_tickers

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['bluecrest_data']
collection = db['stock_data']
test_collection = db['stock_data_test']
prod_collection = db['stock_data_prod1']


def backfill_stock_data(ticker, start_date, end_date, collection = test_collection):
    # Download data from Yahoo Finance
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.insert(loc=0, column='Ticker', value=ticker)

        # Prepare data for MongoDB insertion
        data.reset_index(inplace=True)
        records = data.to_dict('records')

        # Add ticker information to each record
        for record in records:
            record['Ticker'] = ticker

        # Insert data into MongoDB collection
        collection.insert_many(records)
        print(f"Data for {ticker} from {start_date} to {end_date} back-filled successfully.")
    except Exception:
        print(f"An error occurred while back-filling data for {ticker}.")
        return ticker




if __name__ == "__main__":
    tickers = get_final_tickers()
    print(f"Number of unique stocks are {len(tickers)}." )
    start_date = '2020-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    issue_lst = []
    for ticker in tickers:
        res = backfill_stock_data(ticker, start_date, end_date, prod_collection)
        if res:
            issue_lst.append(res)
