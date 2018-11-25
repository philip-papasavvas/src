import bs4 #for beautiful soup 4
import requests

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
