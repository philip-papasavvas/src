"""
Created 13 Oct 2019
How to use mongoDB, initialise library
Write to it (and with time-series data using Arctic too)
"""

import json
from typing import List, Union

import pandas as pd
import pymongo
import yfinance as yf
from arctic import Arctic

from get_paths import get_config_path


def download_yf_stock_data(stocks: Union[List[str], str],
                           start_date: str = "2019-01-01",
                           end_date: str = "2019-12-31") -> pd.DataFrame:
    """Wrapper on Yahoo Finance on which to download stock data (daily returns), will return
    a melted dataframe

    Args
        stocks: specify a list of stocks from the acceptable universe
        start_date
        end_date

    Returns
        pd.DataFrame: with columns ["Open", "High", "Low", "Close", "Adj Close", "Volume"], index
        is the dates in datetime

    """
    print(f"Downloading data from YahooFinance for stocks: {stocks}")
    data_download = yf.download(tickers=stocks,
                                start=start_date,
                                end=end_date,
                                interval="1d")

    melted_data = data_download.unstack().reset_index()
    melted_data.columns = ["measure", "stock", "date", "value"]

    return melted_data


if __name__ == "__main__":

    SAMPLE_STOCK_DATA = download_yf_stock_data(stocks=["TSLA", "AMZN"])

    with open(get_config_path("mongo_private.json")) as mongo_json:
        MONGO_CONFIG = json.load(mongo_json)

    with open(get_config_path("security_data.json")) as sec_json:
        SEC_MAP = json.load(sec_json)

    # Initialise the database
    USER = MONGO_CONFIG["mongo_user"]
    PASSWORD = MONGO_CONFIG["mongo_pwd"]
    MONGO_URL = MONGO_CONFIG["url_cluster"]

    HOST_URL = "".join(["mongodb+srv://", USER, ":", PASSWORD, "@", MONGO_URL])
    CLIENT = pymongo.MongoClient(HOST_URL)

    CLIENT.list_database_names()  # list databases that exist

    SECURITIES = CLIENT["arctic"]  # type pymongo.database.Database

    # use arctic to upload time-series financial data to mongoDB
    STORE = Arctic(CLIENT)
    STORE.list_libraries()  # see what libraries exist

    # initialise
    # store.initialize_library("security_data")

    LIBRARY = STORE["security_data"]
    LIBRARY.list_symbols()

    # DOWNLOAD SAMPLE DATA AND STORE IN MONGODB using Arctic
    SEC_DATA = {}
    for stock_name, stock_ticker in SEC_MAP.items():
        SEC_DATA[stock_name] = yf.download(tickers=stock_ticker,
                                           start="2017-01-01",
                                           end="2019-08-31",
                                           interval="1d")
        LIBRARY.write(stock_name, SEC_DATA[stock_name], metadata={"source": "yahoo finance"})

    LIBRARY.list_symbols()  # smithson, apple, amazon

    sample = LIBRARY.read("amazon")
    # sample.data # view the data
