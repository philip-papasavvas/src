"""
Created on: 6 May 2019
Utils module for generic useful functions, divided into classes
"""

import datetime as dt
import os
import re
from typing import Union

import numpy as np
import pandas as pd


def convert_config_dates(config: dict) -> dict:
    for key, val in config.items():
        if "date" in key.lower():
            config[key] = np.datetime64(val)
    return config


def to_array(*args: Union[np.ndarray, list, tuple, pd.Series, np.datetime64, dt.datetime]):
    """Turning x into np.ndarray

    Yields:
        :class:'np.ndarray'

    Raises:
        ValueError if x is not in the listed type

    Example
    >>> import numpy as np
    >>> x, y, z = to_array(2, ["a","b"], None)
    >>> date_array, =  to_array(np.datetime64("2019-01-01"))
    """

    for x in args:
        if isinstance(x, dt.date):
            yield np.array([x.strftime('%Y-%m-%d')], dtype='datetime64[D]')
        elif isinstance(x, (list, tuple, np.ndarray)):
            yield np.array(x)
        elif isinstance(x, (pd.Series, pd.core.indexes.base.Index)):
            yield x.values
        elif isinstance(x, (int, np.int32, np.int64, float, str)):
            yield np.array([x], dtype=type(x))
        elif isinstance(x, np.datetime64):
            yield np.array([x], 'datetime64[D]')
        elif x is None:
            yield np.array([])
        else:
            raise ValueError('unable to convert to array')


def average(*args):
    """
    Finds arithmetic mean of an array input
    >>> average(*[1,2,3]) #  2.0
    """
    return sum(args, 0.0) / len(args)


def difference(a, b):
    """Give difference between two iterables by keeping values in first
    >>> difference([3,10,9],[3,4,10]) #{9}
    """
    set_a, set_b = set(a), set(b)
    return set_a.difference(set_b)


def match(x: Union[list, np.ndarray, pd.Series],
          y: Union[list, np.ndarray, pd.Series],
          strict: bool = True):
    """Finds the index of x's elements in y. This is the same function as R implements.

    Args:
        x
        y  (list or np.ndarray or pd.Series)
        strict (bool): Whether to raise error if some elements in x are not found in y

    Returns:
        list or np.ndarray of int

    Raises
        AssertionError: If any element of x is not in y
    """
    # to handle for pd.Series, but this should be used for arrays
    x, y = to_array(x, y)
    mask = x[:, None] == y

    rowMask = mask.any(axis=1)

    if strict:
        # this is 40x faster pd.core.algorithms.match(x,y)
        assert rowMask.all(), "%s not found, uniquely : %s " % (
            (~rowMask).sum(), np.array(x)[~rowMask])
        out = np.argmax(mask, axis=1)  # returns the index of the first match
    else:
        # this is 26x faster than pd.core.algorithms.match(x,y,np.nan)
        # return floats where not found elements are returned as np.nan
        out = np.full(np.array(x).shape, np.nan)
        out[rowMask] = np.argmax(mask[rowMask], axis=1)

    return out


def find(folder_path, pattern='.*', full_path=False, expect_one=True):
    """
    To find path(s) of file(s), especially useful for searching the same file pattern in multiple folders

    Args:
        folder_path (str, list, np.ndarray): Folder path(s). If multiple, it will be searched in its order
        pattern (str, list/tuple): regex pattern. Use list/tuple if you need multiple conditions
        full_path (bool, default False): if the full path of the files are needed
        expect_one (bool, default True): True will raise AssertionError if more than one file is found

    Returns:
        str: If one file is found
        list of str: If multiple files are found

    Note:
        This function can only handle same search pattern for every folderPath, and will return the first one it finds \
        if there are multiple folderPath. If it cannot find any, it will raise exceptions about the first folderPath

    Raises:
        FileNotFoundError: If folderPath(s) or pattern(s) does not match any findings
        FileExistsError: If more than one file is found when expectOne = True
    """
    if isinstance(folder_path, str):
        folder_path, = to_array(folder_path)

    for i, path in enumerate(folder_path):

        try:
            listOfFiles = os.listdir(path=path)
        except (FileNotFoundError, OSError) as err:
            if i < len(folder_path) - 1:
                print(err.args[-1] + ' for "%s",... trying next' % path)
                continue  # go for next folderPath
            # out of luck
            err.args = (err.args[0], err.args[1] + ': %s' % path)  # raise with first folderPath
            raise

        if isinstance(pattern, (list, tuple)):
            n = len(pattern)
            # multi condition pattern matching
            ipattern = '|'.join(pattern)
            # strict matching of all required patters
            files = [f for f in listOfFiles if
                     np.unique(re.findall(ipattern, f, re.IGNORECASE)).size == n]
        else:
            files = [f for f in listOfFiles if re.findall(pattern, f, re.IGNORECASE)]

        try:
            if len(files) == 0:
                # remove some special characters before raising error
                if isinstance(pattern, str):
                    pattern = re.sub('[^A-Za-z0-9_.-]+', '', pattern)
                else:
                    pattern = [re.sub('[^A-Za-z0-9_.-]+', '', pat) for pat in pattern]
                raise FileNotFoundError(
                    '%s exists but no file names with pattern: %s' % (path, pattern))

            elif len(files) > 1 and expect_one:
                raise FileExistsError('%s exists but %s files found' % (path, len(files)))

        except (FileNotFoundError, FileExistsError) as err:
            if i < len(folder_path) - 1:
                print(err.args[0] + ',... trying next')
                continue  # go for next folderPath
            # out of luck, raise with first folderPath
            err.args = (
                re.sub(path.replace('\\', '/'), folder_path[0],
                       err.args[0]),)  # re.sub doesn't like double backslashes
            raise

        break  # stop loop when we got the files we wanted

    if full_path:
        if path[-1] == '/':
            path = path[:-1]
        files = ['/'.join((path, f)) for f in files]

    if len(files) == 1 and expect_one:
        files = files[0]

    return files


def concat_columns(sep='', *args):
    """Concatenate multiple columns of pd.DataFrame with sep"""
    df = pd.DataFrame()
    for arg in args:
        df = pd.concat([df, arg], axis=1, ignore_index=True)
    try:
        out = df.astype(str).add(sep).sum(axis=1).str.replace('%s+$' % re.escape(sep),
                                                              '')  # removes trailing sep
        # need to make any columns with nan to output NaN, which is the result when 'A' + '_' + 'NaN'
        mask = df.isnull().any(axis=1)
        out[mask] = np.nan
    except AttributeError:
        # incase of empty data frame
        out = pd.Series()
    return out


def format_csv_commas(path: str) -> list:
    """ 
    Feed in filepath of CSV to be edited and returns List of cleaned data, replacing 
    "," with ""
    """

    data = []
    with open(path, newline='') as f:
        for lines in f:
            new_line = lines.replace(", ", "")
            data.append(new_line)

    # compile lines and remove special characters
    data = pd.Series(data).str.split(',', expand=True).replace({'\\n': '', '\\r': ''}, regex=True)
    data.columns = data.iloc[0, :]
    data = data.iloc[1:].reset_index(drop=True)
    return data


# DICT METHODS
def flatten_dict(d):
    """Flatten dictionary d

    Example
        >>> flatten_dict(d={"a":{1}, "b":{"yes":{"more detail"}, "no": "level below" }})
        returns {'a': {1}, 'b.yes': {'more detail'}, 'b.no': 'level below'}
    """

    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value

    return dict(items())


def return_dict_keys(dct):
    """Returns keys of a dict in a list
    >>> return_dict_keys({'a':1, 'b':2, 'c':3})
    """
    return list(dct.keys())


def return_dict_values(dct):
    """
    Returns keys of a dict in a list
    >>> return_dict_values({'a':1, 'b':2, 'c':3})
    [1, 2, 3]
    """
    return list(dct.values())


def change_dict_keys(in_dict, text):
    """Change the keys of an input dictionary as with the text specified"""
    return {text + "_" + str(key): (change_dict_keys(value) if
                                    isinstance(value, dict) else
                                    value) for key, value in in_dict.items()}


def dict_from_df_cols(df: pd.DataFrame, columns: list):
    """Convenience function to create a dict from the dataframe columns"""
    assert len(columns) == 2, "Cannot produce a dict if two columns not specified"
    return dict(zip(df[columns[0]], df[columns[1]]))


def drop_null_columns_df(data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns from the dataframe with null values"""
    original_columns = list(data.columns)
    cleaned_data = data.dropna(axis=1)
    new_columns = list(cleaned_data.columns)
    cut_columns = [x for x in original_columns if x not in new_columns]

    print(f"Columns: {cut_columns}  \n have been dropped from the dataframe as they contain NaNs")
    return cleaned_data


def linear_bucketing(x: np.array, y: np.array) -> np.ndarray:
    """Returns a matrix of weighting from linear bucketing from x to y

    Args:
        x: Source buckets
        y: Destination buckets

    Returns:
        np.ndarray: Weights to apply for linear bucketing, for x as axis 0,
        y as axis 1
    """
    # weights on input and output buckets
    index = np.interp(x, y, np.arange(y.size, dtype=float))

    # indices which buckets are involved
    t1, t2 = np.floor(index).astype(int), np.ceil(index).astype(int)

    # weights
    weight_far = index % 1
    weight_near = 1 - weight_far

    # construct a vector to be multiplied with weights for mapping on buckets
    t1_vec, t2_vec = np.zeros([y.size, t1.size]), np.zeros([y.size, t2.size])
    t1_vec[t1, np.arange(t1.size)] = 1
    t2_vec[t2, np.arange(t2.size)] = 1

    return (t1_vec * weight_near[None, :] + t2_vec * weight_far[None, :]).T


if __name__ == '__main__':

    # example of using match - find index of elements from one list in another
    match(x=[46, 15, 5], y=[5, 4, 46, 6, 15, 1, 70])

    # dict_from_df_cols - get a dict from dataframe columns
    sample_df = pd.DataFrame(
        {"a": ["liquid", "arrogant", "imagine", "knock", "share"],
         "b": range(5)})

    dict_from_df_cols(df=sample_df, columns=['a', 'b'])
