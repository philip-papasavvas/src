# Created 13 Oct 2019. How to use mongoDB, initialise library,
# write to it (and with time-series data using Arctic too)

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
        pd.DataFrame: with columns ["Open", "High", "Low", "Close", "Adj Close", "Volume"], index is the dates in
        datatime

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

    sample_stock_data = download_yf_stock_data(stocks=["TSLA", "AMZN"])

    with open(get_config_path("mongo_private.json")) as mongo_json:
        mongo_config = json.load(mongo_json)

    with open(get_config_path("security_data.json")) as sec_json:
        sec_map = json.load(sec_json)

    # Initialise the database
    user = mongo_config["mongo_user"]
    password = mongo_config["mongo_pwd"]
    mongo_url = mongo_config["url_cluster"]

    host_url = "".join(["mongodb+srv://", user, ":", password, "@", mongo_url])
    client = pymongo.MongoClient(host_url)

    client.list_database_names()  # list databases that exist

    securities = client["arctic"]  # type pymongo.database.Database

    # use arctic to upload time-series financial data to mongoDB
    store = Arctic(client)
    store.list_libraries()  # see what libraries exist

    # initialise
    # store.initialize_library("security_data")

    library = store["security_data"]
    library.list_symbols()

    # DOWNLOAD SAMPLE DATA AND STORE IN MONGODB using Arctic
    sec_data = {}
    for k, v in sec_map.items():
        sec_data[k] = yf.download(tickers=v,
                                  start="2017-01-01",
                                  end="2019-08-31",
                                  interval="1d")
        library.write(k, sec_data[k], metadata={"source": "yahoo finance"})

    library.list_symbols()  # smithson, apple, amazon

    sample = library.read("amazon")
    sample.data  # view the data that has been written down
