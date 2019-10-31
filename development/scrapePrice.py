"""
Script to scrape price of a product from Amazon
"""

import bs4 #for beautiful soup 4
from archive import requests


def getAmazonPrice(productURL):
    # res = requests.get("https://www.amazon.co.uk/Tattooist-Auschwitz-heart-breaking-unforgettable-international/dp/1785763644")
    res = requests.get(productURL)
    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    elems = soup.select('#buyNewSection > div:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(2) > a:nth-of-type(1) > div:nth-of-type(1) > div:nth-of-type(2) > span:nth-of-type(1) > span:nth-of-type(1)')
    return elems[0].text.strip()

price = getAmazonPrice("https://www.amazon.co.uk/Tattooist-Auschwitz-heart-breaking-unforgettable-international/dp/1785763644")
print('The price is ' + price)