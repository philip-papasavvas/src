"""
Created on 25 Nov 2018

Password generator in Python: specify length and if you want special characters
https://www.practicepython.org/exercise/2014/05/28/16-password-generator.html
"""
import random
import string


def generate_password(length: int, special_char: bool = False) -> str:
    """Generates a password of a user given length, specify if special characters"""
    lowers = list(string.ascii_lowercase)
    uppers = list(string.ascii_uppercase)
    nums = [str(i) for i in range(0, 10)]
    special_characters = list("!\"£$%^&*()#@?<>")

    if special_char:
        allowed_char_list = lowers + uppers + nums + special_characters
    else:
        allowed_char_list = lowers + uppers + nums

    result = "".join([random.choice(allowed_char_list) for i in range(0, length)])
    return result


if __name__ == '__main__':
    pw = generate_password(length=8, special_char=True)
    print(pw)
