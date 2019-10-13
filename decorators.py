import numpy as np
from time import time

def timer(method):
    def timed(*args, **kwargs):
        ts = time()
        result = method(*args, **kwargs)
        te = time()

        print('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed

# @timer
# def func(num):
#     return np.sqrt(num)

func(200)