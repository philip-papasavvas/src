# Created on 29 Apr 2021
# Calls from CoinbasePro API for specific pairs
import json
from datetime import datetime, timedelta
from pprint import pprint

# third party imports
import coinbasepro as cbp
import mplfinance as mpf
import pandas as pd
from requests import get

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:,.2f}'.format

# --------------
# public client API using third party software
client_public = cbp.PublicClient()
stats_example = client_public.get_product_24hr_stats(product_id='BTC-USD')
pprint(stats_example)  # pretty print

# ---------
# REST API
# parameters
url_api = 'https://api.pro.coinbase.com'
url_product = f"{url_api}/products"
ccy_pairs = ['BTC-USD', 'BTC-GBP', 'ETH-USD', 'ETH-GBP']

# response codes for URL:
# 200 - ok, 404 - not found, 400- bad request
response_api = get(url=url_product)
response_text = response_api.text
info_df = pd.read_json(response_text)

# look at the snippet of data just for desired currencies
info_df[info_df['id'].isin(ccy_pairs)]


def get_ccy_pair_stats(ccy_pair: str,
                       base_url: str = 'https://api.pro.coinbase.com/products') -> dict:
    """Method to get the statistics from Coinbase API for desired ccy pair"""
    print(f'Getting CCYPAIR stats for {ccy_pair}')
    conn = requests.get(url=f'{base_url}/{ccy_pair}/stats')
    ccy_content = json.loads(conn.content)
    ccy_content['pair'] = ccy_pair
    return ccy_content


# example for one pair
btc_pair_stats = get_ccy_pair_stats(ccy_pair='BTC-USD')

# all stats
ccy_pair_stats = pd.DataFrame(list(map(get_ccy_pair_stats, ccy_pairs)))

# get the historic data
today_date = datetime.now()
lookback_days = 300
start_date = today_date - timedelta(days=lookback_days)
pair = 'BTC-USD'
candles_data = get(
    url=f'https://api.pro.coinbase.com/products/{pair}/candles',
    params={
        'start': start_date,
        'end': today_date,
        'granularity': 86400}
)
candles_df = pd.read_json(candles_data.text)
# columns in line with Coinbase Pro API documentation
candles_df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
candles_df.head()

# need to format the time column to get a date
candles_df['date'] = pd.to_datetime(candles_df['time'], unit='s')
candles_df.drop('time', axis=1, inplace=True)
# sort in ascending time order
candles_df.sort_values('date', ascending=True, inplace=True)

# to plot the candle chart, download matplotlibfinance > mplfinance

mpf.plot(candles_df[['date', 'open', 'close', 'high', 'low', 'volume']].set_index('date'),
         type='candle',
         mav=(3, 6, 9),  # moving average
         volume=True,
         title=f'Price History for {pair}')
