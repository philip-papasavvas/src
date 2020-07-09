"""
Created on 30 Jul 2019
Python decorators
"""

import functools
from time import time


def timer(method):
    """Return the calculation time of methods/functions"""
    @functools.wraps(method)
    def timed(*args, **kwargs):
        start_time = time()
        result = method(*args, **kwargs)
        end_time = time()

        print(f'{method.__name__} took {end_time - start_time: 2.2f} sec')
        return result

    return timed
