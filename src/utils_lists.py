# Created 17 June 2019. Translplanted from utils_generic.py
from math import ceil


def list_as_comma_sep(lst):
    """Gets a list and returns one single string with elements separated by a comma
    >>> list_as_comma_sep(["hello","this","is","an", "example"]) # 'hello,this,is,an,example'
    """
    return ",".join(lst)


def has_duplicates(lst):
    """
    Checks if a list has duplicate values
    >>> has_duplicates([1,2,4,5]) # False
    >>> has_duplicates([1,2,2,5]) # True
    """
    return len(lst) != len(set(lst))


def all_unique(lst):
    """
    Check if a given list has duplicate elements
    >>> all_unique([1,2,3,4]) # True
    >>> all_unique([1,2,2,4]) # False
    """
    return len(lst) == len(set(lst))


def chunk(lst, chunk_size):
    """
    Split a list into a list of smaller lists defined by chunk_size
    >>> chunk([1,2,4,5,6,6],5)
    >>> chunk([1,2,4,5,6,6],2)
    """
    return list(
        map(lambda x: lst[x * chunk_size: x * chunk_size + chunk_size],
            list(range(0, ceil(len(lst) / chunk_size))))
    )


def count_occurences(lst, value):
    """
    Function to count occurrences of value in a list
    >>> count_occurences(lst=[1,2,3,4,4,4,4], value=4) # 4
    """
    return len([x for x in lst if x == value and type(x) == type(value)])


def flatten(lst):
    """
    Flatten a list using recursion
    >>> flatten(lst=[1,2,[3,4,5,[6,7]]]) # [1, 2, 3, 4, 5, 6, 7]
    """
    res = []
    res.extend(flatten_list(list(map(lambda x: flatten(x) if type(x) == list else x, lst))))
    return res


def flatten_list(arg):
    """Flatten lists
    Function used for flattening lists
    >>> flatten_list([2,3,5,[7,8]])
    """
    ls = []
    for i in arg:
        if isinstance(i, list):
            ls.extend(i)
        else:
            ls.append(i)
    return ls

if __name__ == '__main__':
    pass