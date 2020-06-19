# Created on 30 Jul 2019. Python decorators

import functools
from time import time

def timer(method):
    """Return the calculation time of methods/functions"""
    @functools.wraps(method)
    def timed(*args, **kwargs):
        ts = time()
        result = method(*args, **kwargs)
        te = time()

        print('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed