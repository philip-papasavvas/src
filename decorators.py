"""
Created on 30 Jul 2019
Python decorators
"""
import functools
import warnings
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


def deprecated(_func=None, *, print_msg=None):
    """
    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.

    Args:
        print_msg: Information to point user to newest version.
    """

    def decorator_deprecated(func):

        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):
            with warnings.catch_warnings():
                if print_msg is None:
                    warnings.warn(f"\nFunction deprecated: {func.__name__}",
                                  category=DeprecationWarning, stacklevel=2)
                else:
                    warnings.warn(f"\nFunction deprecated: {func.__name__}"
                                  f"\n{print_msg}",
                                  category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper_decorator

    if _func is None:
        return decorator_deprecated
    else:
        return decorator_deprecated(_func)


if __name__ == '__main__':
    pass
