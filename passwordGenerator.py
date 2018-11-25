"""
Password generator in Python:
specify length and it will create one

https://www.practicepython.org/exercise/2014/05/28/16-password-generator.html

"""
# Step 1: generate a list of alphanumeric characters
# Step 2: specify what mix of characters to have
# Improvement - to remove some weird characters that
# are not wanted

import string
from string import punctuation
import random
# import pyperclip
# import sys

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

#copy it to the clipboard
#print(sys.path)
print(passwordGen(length = 8))

#[punct[i] for i in range(0, len(punct)) if punct[i] in list(punctuation)]
