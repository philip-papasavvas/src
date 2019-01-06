"""
Script to scrape price of a product from Amazon
"""

import bs4 #for beautiful soup 4
from development import requests


def getAmazonPrice(productURL):

    # res = requests.get("https://www.amazon.co.uk/Tattooist-Auschwitz-heart-breaking-unforgettable-international/dp/1785763644")
    res = requests.get(productURL)
    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    elems = soup.select('#buyNewSection > div:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(2) > a:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(2) > span:nth-of-type(1) > span:nth-of-type(1)')
    return elems[0].text.strip()

price = getAmazonPrice("https://www.amazon.co.uk/Tattooist-Auschwitz-heart-breaking-unforgettable-international/dp/1785763644")
print('The price is ' + price)

#change formatting below since it doesn't work for some webpages
#buyNewSection > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > a:nth-child(1) > div:nth-child(1) > div:nth-child(2) > span:nth-child(1) > span:nth-child(1)
#buyNewSection > div:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(2) > a:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(2) > span:nth-of-type(1) > span:nth-of-type(1)

#FTSE100 price from yahoo finance
page = requests.get("https://finance.yahoo.com/quote/%5EFTSE/")

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

data = pdr.get_data_yahoo("SPY", start="2018-01-02", end="2018-01-3")
