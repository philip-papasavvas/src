"""
Password generator in Python:
specify length and it will create one

https://www.practicepython.org/exercise/2014/05/28/16-password-generator.html
"""
# Step 1: generate a list of alphanumeric characters
# Step 2: specify what mix of characters to have
# Development - to remove some weird characters not wanted

import string
from string import punctuation
import random
# import pyperclip
# import sys

# sys.path.append('C:\\Program Files\\Python\\Python36\\lib\\site-packages')

def passwordGen(length):
    lowers = list(string.ascii_lowercase)
    uppers = list(string.ascii_uppercase)
    # letters = lowers + uppers
    nums = [str(i) for i in range(0,10)]
    #notAllowed = '\"<",.>\'][)('
    punct = list(punctuation)
    charList = lowers + uppers + nums #+ punct
    result = "".join([random.choice(charList) for i in range(0, length)])
    return result

pw = passwordGen(length = 8)
print(pw)

#copy to clipboard
# pyperclip.copy(pw)

# List of characters of punctuation
# punct = list(punctuation)
# [punct[i] for i in range(0, len(punct)) if punct[i] in list(punctuation)]

