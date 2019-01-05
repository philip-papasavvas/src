import requests
res  = requests.get('http://automatetheboringstuff.com/files/rj.txt')

playFile = open('RomeoandJuliet.txt', 'wb')

for chunk in res.iter_content(100000):
    playFile.write(chunk)
    #
playFile.close()
