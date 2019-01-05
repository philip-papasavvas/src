import random
import os

basePath = 'C:\\Users\\ppapasav\\Documents\\python'
with open(basePath + "\sowpods.txt") as f:
    words = list(f)
print(random.choice(words).strip())

