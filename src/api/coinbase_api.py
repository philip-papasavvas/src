"""
Created on: 29 Apr 2021

Examples of using the coinbase pro API
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from requests import get
import mplfinance as mpf

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:,.2f}'.format


def get_product_stats(product_id: str) -> dict:
    """
    Fetches the 24-hour statistics for a specific product from Coinbase API

    Parameters:
    - product_id (str): The product ID for which the stats are to be fetched

    Returns:
    - stats (dict): A dictionary containing the statistics for the product
    """
    client_public = cbp.PublicClient()
    stats = client_public.get_product_24hr_stats(product_id=product_id)
    return stats


def get_products_from_api(url_product: str) -> pd.DataFrame:
    """
    Fetches product data from Coinbase API and returns it as a DataFrame

    Parameters:
    - url_product (str): The API endpoint from which to fetch the product data

    Returns:
    - info_df (DataFrame): A DataFrame containing the product data
    """
    response_api = get(url=url_product)
    response_text = response_api.text
    info_df = pd.read_json(response_text)
    return info_df


def get_ccy_pair_stats(ccy_pair: str,
                       base_url: str = 'https://api.pro.coinbase.com/products') -> dict:
    """
    Fetches the statistics for a specific currency pair from Coinbase API

    Parameters:
    - ccy_pair (str): The currency pair for which the stats are to be fetched
    - base_url (str, optional): The base API endpoint (default is 'https://api.pro.coinbase.com/products')

    Returns:
    - ccy_content (dict): A dictionary containing the statistics for the currency pair
    """
    print(f'Getting CCYPAIR stats for {ccy_pair}')
    conn = get(url=f'{base_url}/{ccy_pair}/stats')
    ccy_content = json.loads(conn.content)
    ccy_content['pair'] = ccy_pair
    return ccy_content


def get_historic_data(pair: str,
                      lookback_days: int,
                      granularity: int = 86400) -> pd.DataFrame:
    """
    Fetches historic data for a specific currency pair from Coinbase API and returns it as a DataFrame

    Parameters:
    - pair (str): The currency pair for which the historic data is to be fetched
    - lookback_days (int): The number of days in the past to fetch data for
    - granularity (int, optional): The granularity of the data in seconds (default is 86400, i.e., daily data)

    Returns:
    - candles_df (DataFrame): A DataFrame containing the historic data
    """
    today_date = datetime.now()
    start_date = today_date - timedelta(days=lookback_days)
    candles_data = get(
        url=f'https://api.pro.coinbase.com/products/{pair}/candles',
        params={
            'start': start_date,
            'end': today_date,
            'granularity': granularity
        }
    )
    candles_df = pd.read_json(candles_data.text)
    candles_df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
    candles_df['date'] = pd.to_datetime(candles_df['time'], unit='s')
    candles_df.drop('time', axis=1, inplace=True)
    candles_df.sort_values('date', ascending=True, inplace=True)
    return candles_df


def plot_price_history(df: pd.DataFrame, pair: str) -> None:
    """
    Plots the price history for a specific currency pair using mplfinance

    Parameters:
    - df (DataFrame): The DataFrame containing the historic data
    - pair (str): The currency pair for which the data is to be plotted
    """
    mpf.plot(df[['date', 'open', 'close', 'high', 'low', 'volume']].set_index('date'),
             type='candle',
             mav=(3, 6, 9),
             volume=True,
             title=f'Price History for {pair}')


if __name__ == '__main__':
    # see an example of the stats
    stats_example = get_product_stats(product_id='BTC-USD')

    # get the BTC USD pair information from the api
    info_df = get_products_from_api(url_product='https://api.pro.coinbase.com/products/BTC-USD')

    # example for currency pair stats
    ccy_pair_stats = pd.DataFrame(list(map(get_ccy_pair_stats,
                                           ['BTC-USD', 'BTC-GBP', 'ETH-USD', 'ETH-GBP'])))
    candles_df = get_historic_data(pair='BTC-USD', lookback_days=300)

    # plot the price history for BTC USD
    plot_price_history(df=candles_df, pair='BTC-USD')
