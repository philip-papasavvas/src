"""
Created: 7 Aug 2020
Data retrieval for stock data (mainly using YahooFinance API)
"""
from typing import Union, List

import pandas as pd
import yfinance as yf


def return_stock_data(stocks: Union[List[str], str],
                      start_date: str = "2019-01-01",
                      end_date: str = "2019-12-31",
                      return_price: bool = True) -> pd.DataFrame:
    """Wrapper on Yahoo Finance on which to download stock data (daily returns), will return
    a melted dataframe

    Args
        stocks: list of stocks from the acceptable universe
        start_date
        end_date

    Returns
        pd.DataFrame: with columns ["Open", "High", "Low", "Close", "Adj Close", "Volume"], index
        is the dates in datetime

    """
    if not isinstance(stocks, List):
        stocks = [stocks]

    print(f"Downloading data from YahooFinance for stocks: {stocks}")
    data_download = yf.download(tickers=stocks,
                                start=start_date,
                                end=end_date,
                                interval="1d")

    melted_data = data_download.unstack().reset_index()

    rename_cols_dict = {
        'level_0': 'measure',
        'level_1': 'stock',
        'Date': 'date',
        0: 'value'
    }

    if len(stocks) == 1:
        melted_data['stock'] = stocks[0]
        rename_cols_dict.pop('level_1')
        rename_cols_dict['stock'] = 'stock'

    melted_data.rename(columns=rename_cols_dict,
                       inplace=True)

    if return_price:
        print(f"Returning daily price data for {stocks}")
        return melted_data.loc[melted_data['measure'] == 'Adj Close']
    else:
        print(f"Returning all data for {stocks}")
        return data_download.unstack().reset_index()


def is_valid_ticker(ticker: str) -> None:
    """Method to check if the ticker is valid and therefore can download data
    from yfinance API"""

    try:
        yf._download_one(ticker=ticker) # use the hidden method to see if the symbol exists
        print(f"Success! Ticker: {ticker} exists")
    except ValueError:
        print(f'Ticker {ticker} does not exist/cannot be downloaded from yfinance API.'
              f' Try another!')


if __name__ == '__main__':
    stock_data_multiple = return_stock_data(stocks=["TSLA", "AMZN"],
                                            return_price=True)

    stock_data_single = return_stock_data(stocks='AMZN',
                                          return_price=False)

    stock_data_single = return_stock_data(stocks=['AMZN'],
                                          return_price=False)

    is_valid_ticker(ticker='GZC')
