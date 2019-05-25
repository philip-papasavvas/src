"""
Created: November 2018

Password generator in Python: specify length and if you want special characters

https://www.practicepython.org/exercise/2014/05/28/16-password-generator.html
"""
# Step 1: generate a list of alphanumeric characters
# Step 2: specify what mix of characters to have

import string
import random
from string import punctuation
# import sys
# import pyperclip

# sys.path.append('C:\\Program Files\\Python\\Python36\\lib\\site-packages')

def passwordGen(length, special=False):
    """Generates a password of a user given length, and can specify if want special characters"""
    lowers = list(string.ascii_lowercase)
    uppers = list(string.ascii_uppercase)
    # letters = lowers + uppers
    nums = [str(i) for i in range(0,10)]
    # notAllowed = '\"<",.>\'][)('
    # punct = list(punctuation)
    specialChar = list("!\"£$%^&*()#@?<>")

    if special:
        charList = lowers + uppers + nums + specialChar #punct
    else:
        charList = lowers + uppers + nums

    result = "".join([random.choice(charList) for i in range(0, length)])
    return result

pw = passwordGen(length = 8, special=True)
print(pw)

