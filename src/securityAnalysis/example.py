"""
Created on: 16 Mar 2019
Example of using functions to analyse stock data (using yfinance and investpy),
and utils_finance methods
"""
import numpy as np
import pandas as pd
import yfinance  # third party import

from securityAnalysis.utils_finance import (calculate_annualised_return_df, return_sortino_ratio,
                                            calculate_return_df, return_sharpe_ratio,
                                            calculate_annual_volatility_df)

# ----------------------------
# EXAMPLE STOCK DATA RETRIEVAL
# ----------------------------

# ticker = 'TSLA'
# price_data = yfinance.download(
#     tickers=ticker,
#     start="2019-01-01", end="2020-01-01")
# df = price_data[
#     ['Adj Close']].rename(columns={'Adj Close': 'price'}).reset_index().copy(True)
# df.columns = ['date', ticker]

ticker = ['AMZN', 'GOOGL']
price_data_two = yfinance.download(
    tickers=ticker,
    start="2017-01-01", end="2020-01-01")

df = price_data_two[
    ['Adj Close']].rename(columns={'Adj Close': 'price'}).reset_index().copy(True)
df.columns = ['date'] + ticker  # since it is a multi-index

# config = {
#     "startDate": "2018-01-01",
#     "endDate": "2019-01-01"
# }
# from utils_generic import convert_config_dates  # local import
# config = convert_config_dates(config)

# -------------------------
# Utilise the utils_finance
# -------------------------
daily_return = calculate_return_df(data=df.set_index('date'),
                                   is_relative_return=True)

# Annualised return
ann_rtn = calculate_annualised_return_df(data=df.set_index('date'))

# Annualised volatility
vol = calculate_annual_volatility_df(data=df.set_index('date'))

# Sharpe Ratio - annualised return divided by volatility
sharpe_df = return_sharpe_ratio(data=df.set_index('date')) # or ann_rtn/vol

# Sortino Ratio
return_sortino_ratio(data=df.set_index('date'))


# -------------
# DOWNLOAD DATA
# -------------
from utils_generic import df_columns_to_dict
dir = '/Users/philip_p/Downloads'

codes = pd.read_csv(dir+'/ic100_codes.csv')
codes_subset = codes[codes['To Get Data']]

codes_to_fund_map = df_columns_to_dict(
    df=codes,
    columns=['yfin_code', 'Fund'])

yfinance.download(tickers=list(codes_subset['yfin_code'][:5]))

import investpy
funds_isin = investpy.get_funds(country='United Kingdom')['isin'].unique()

funds_exist_investpy = codes_subset['yfin_code'][
    np.in1d(codes_subset['yfin_code'].values, funds_isin)]

