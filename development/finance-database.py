"""
Download data using yfinance and create a mongoDB to store the data

Created on: 16/07/2019
"""

import numpy as np
import pandas as pd
import os


import yfinance as yf

# data = yf.download(  # or pdr.get_data_yahoo(...
#         # tickers list or string as well
#         tickers = "SPY", \
#
#         # use "period" instead of start/end
#         # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#         # (optional, default is '1mo')
#         period = "ytd", \
#
#         # fetch data by interval (including intraday if period < 60 days)
#         # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#         # (optional, default is '1d')
#         interval = "1d", \
#
#         # group by ticker (to access via data['SPY'])
#         # (optional, default is 'column')
#         group_by = 'ticker', \
#
#         # adjust all OHLC automatically
#         # (optional, default is False)
#         auto_adjust = True, \
#
#         # download pre/post regular market hours data
#         # (optional, default is False)
#         prepost = True, \
#
#         # use threads for mass downloading? (True/False/Integer)
#         # (optional, default is True)
#         treads = True, \
#
#         # proxy URL scheme use use when downloading?
#         # (optional, default is None)
#         proxy = None \
#     )

# ---------------
# Financial data
# ---------------

a = yf.download(tickers=['AAPL MSFT'], period="1m", interval="1d")

# Smithson Equity
c = yf.download(tickers=['SSON.L'], period="ytd", interval="1d")

# Fundsmith Equity I Acc
d = yf.download(tickers=['0P0000RU81.L'], period="ytd", interval="1d")

# Gabelli Value Plus & Trust
e = yf.download(tickers=['GVP.L'], period="ytd", interval="1d")

ex_dict = {'pp_funds': {'Fundsmith Equity': '0P0000RU81.L',
                        'Gabelli Value': 'GVP.L'}}

for k,v in ex_dict['pp_funds'].items():
    print(k,v)

data = []
for k,v in ex_dict['pp_funds'].items():
    data.append(yf.download(tickers= v, period= "ytd", intervals = "1d"))


mapping_list = pd.read_csv(r"C:\Users\Philip\Documents\python\input\security_data\mapping_table.csv")

f_map = mapping_list[['GROUP', 'SECURITY NAME', 'YFINANCE']]
f_map = f_map[f_map['YFINANCE'].isin(['FALSE',np.nan])]

# --------------
# Database stuff
# --------------

# start up the mongoDB
# connect to Arctic
from pymongo import MongoClient
from arctic import Arctic

client = MongoClient('mongodb://localhost:27017/')

store = Arctic(client)
store.list_libraries() # see what libraries are present

store.initialize_library('security_data')

library = store['security_data']
library.write('Fundsmith Equity', data[0], metadata={'source':'yahoo finance'})
library.write('Gabelli Value', data[1], metadata={'source':'yahoo finance'})

library.list_symbols()

item = library.read('Gabelli Value')

#view/access the data
item.data


# ------------
# JSON reading
# ------------



