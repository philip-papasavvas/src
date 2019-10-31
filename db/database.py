"""
Created on 13 Oct 2019
author: Philip P

Module to give an example of how to go into the MongoDB, intialise library, write to it
(and with time-series data using Arctic too)
"""
import json
import pymongo
import dns
import numpy as np
import pandas as pd
import os
from utils import get_config_path

import yfinance as yf
from arctic import Arctic

if __name__ == "__main__":

    with open(get_config_path("mongo_private.json")) as mongo_json:
        mongo_config = json.load(mongo_json)

    with open(get_config_path("security_data.json")) as sec_json:
        sec_map = json.load(sec_json)

    # Initialise the database
    user = mongo_config['mongo_user']
    password = mongo_config['mongo_pwd']
    mongo_url = mongo_config['url_cluster']

    host_url = "".join(["mongodb+srv://", user, ":", password, "@", mongo_url])
    client = pymongo.MongoClient(host_url)

    client.list_database_names() # list databases that exist

    securities = client['arctic']  # type pymongo.database.Database

    # use arctic to upload time-series financial data to MongoDB
    store = Arctic(client)
    store.list_libraries() # see what libraries exist

    # initialise
    # store.initialize_library('security_data')

    library = store['security_data']
    library.list_symbols()

    # DOWNLOAD SAMPLE DATA AND STORE IN MONGODB using Arctic
    sec_data = {}
    for k,v in sec_map.items():
        sec_data[k] = yf.download(tickers=v,start="2017-01-01", end="2019-08-31", interval="1d")
        library.write(k, sec_data[k], metadata={'source': 'yahoo finance'})

    library.list_symbols() # smithson, apple, amazon

    sample = library.read('amazon')
    sample.data # to view the data that has been written down