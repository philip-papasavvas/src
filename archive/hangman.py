
from utils import find
import random
import os

basePath = 'C:\\Users\\ppapasav\\Documents\\python'

a = find(folderPath = "C:\\Users\Philip\PycharmProjects\PythonSkills\data", pattern='sowpods', fullPath=True)

with open(basePath + "\sowpods.txt") as f:
    words = list(f)
print(random.choice(words).strip())

os.listdir()