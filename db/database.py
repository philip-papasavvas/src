"""
Created on 13 Oct 2019
author: Philip P

Module to house database, call it and add to it
"""

import pymongo
import dns
import numpy as np
import pandas as pd
import os

import yfinance as yf
from arctic import Arctic

if __name__ == "__main__":

    # SET UP THE DATABASE
    # sub out <password> for the actual password for master_user
    client = pymongo.MongoClient("mongodb+srv://master_user:<password>@cluster0-iooan.mongodb.net/test?retryWrites=true&w=majority")

    client.list_database_names() # list the databases that exist

    # DOWNLOAD SAMPLE DATA AND STORE IN MONGODB using Arctic
    sson_data = yf.download(tickers=['SSON.L'], period="ytd", interval="1d")

    store = Arctic(client)
    store.list_libraries() # see what libraries exist

    # initialise and write to library
    store.initialize_library('security_data')

    library = store['security_data']
    library.write('smithson', sson_data['Adj Close'], metadata={'source': 'yahoo finance'})
    library.list_symbols() # smithson

    item = library.read('smithson')
    item.data # to view the data that has been written down


    # Write things to database which is not time series
    # Initialise a database for CF WOD email
    db = client.email
    quotes = db.quotes #new database called email has been created

    inspir_quotes = {'Muhammad Ali': 'Service to others is the rent you pay for your room here on earth.',
                     'Josh Bridges': 'Hard work pays off',
                     'Walt Disney': 'The Way Get Started Is To Quit Talking And Begin Doing.',
                     'Winston Churchill': 'The Pessimist Sees Difficulty In Every Opportunity. The Optimist Sees Opportunity In Every Difficulty.'}
    anon_quotes = {'anon': "If you're tired of starting over, stop giving up."}

     # creates a new collection for the database
    quotes.insert_one(inspir_quotes) # <pymongo.results.InsertOneResult at <where memory is stored> if it's done with success
    quotes.insert_one(anon_quotes)

    # see all documents within one collection
    db.inventory.find( {} )