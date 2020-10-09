"""
Created 17 Jun 2019
For examples of the methods, see the unit tests in test_utils_lists
"""
from math import ceil
from typing import Union


def list_as_comma_sep(lst: list) -> str:
    """Gets a list and returns one single string with elements separated by a comma"""
    return ",".join(lst)


def has_duplicates(lst: list) -> bool:
    """Checks if a list has duplicate values"""
    return len(lst) != len(set(lst))


def all_unique(lst: list) -> bool:
    """Check if a given list has duplicate elements"""
    return len(lst) == len(set(lst))


def chunk(lst: list, chunk_size: int) -> list:
    """Split a list into a list of smaller lists defined by chunk_size"""
    return list(
        map(lambda x: lst[x * chunk_size: x * chunk_size + chunk_size],
            list(range(0, ceil(len(lst) / chunk_size))))
    )


def count_occurrences(lst: list, value: Union[bool, str, int, float]) -> int:
    """Function to count occurrences of value in a list"""
    return len([x for x in lst if x == value and type(x) == type(value)])


def flatten(lst: list) -> list:
    """Flatten a list using recursion"""
    res = []
    res.extend(flatten_list(list(map(lambda x: flatten(x) if type(x) == list else x, lst))))
    return res


def flatten_list(arg) -> list:
    """Function used for flattening lists recursively"""
    ls = []
    for i in arg:
        if isinstance(i, list):
            ls.extend(i)
        else:
            ls.append(i)
    return ls


if __name__ == '__main__':
    pass
