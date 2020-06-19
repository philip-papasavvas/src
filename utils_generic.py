"""
Created on: 6 May 2019
Created by: Philip.P_adm

Utils module for generic useful functions, divided into classes
"""

# usual (built-in) imports
import pandas as pd
import numpy as np
import os
from re import escape
import re
import datetime as dt

# third party imports
import pyperclip
import xlrd


# Date utilities functions
from utils_date import char_to_date


# Generic Utilities functions
def to_array(*args):
    """Turning x into np.ndarray

    Args:
        x (list, tuple, np.ndarray, pd.Series, np.datetime64, datetime.datetime)

    Yields:
        :class:'np.ndarray'

    Raises:
        ValueError if x is not in the listed type

    Example
    >>> import numpy as np
    >>> x,y,z = to_array(2,["a","b"],None)
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


def match(cls, x, y, strict=True):
    """Finds the index of x's elements in y. This is the same function as R implements.

    Args:
        x,y  (list or np.ndarray or pd.Series)
        strict (bool): Whether to raise error if some elements in x are not found in y

    Returns:
        list or np.ndarray of int

    Raises
        AssertionError: If any element of x is not in y
    """

    # just be sure it handles pd.Series as well, but in reality one should not use this function for pd.Series
    x, y = cls.to_array(x, y)
    mask = x[:, None] == y

    rowMask = mask.any(axis=1)

    if strict:
        # this is 40x faster pd.core.algorithms.match(x,y)
        assert rowMask.all(), "%s not found, uniquely : %s " % ((~rowMask).sum(), np.array(x)[~rowMask])
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
            files = [f for f in listOfFiles if np.unique(re.findall(ipattern, f, re.IGNORECASE)).size == n]
        else:
            files = [f for f in listOfFiles if re.findall(pattern, f, re.IGNORECASE)]

        try:
            if len(files) == 0:
                # remove some special characters before raising error
                if isinstance(pattern, str):
                    pattern = re.sub('[^A-Za-z0-9_.-]+', '', pattern)
                else:
                    pattern = [re.sub('[^A-Za-z0-9_.-]+', '', pat) for pat in pattern]
                raise FileNotFoundError('%s exists but no file names with pattern: %s' % (path, pattern))

            elif len(files) > 1 and expect_one:
                raise FileExistsError('%s exists but %s files found' % (path, len(files)))

        except (FileNotFoundError, FileExistsError) as err:
            if i < len(folder_path) - 1:
                print(err.args[0] + ',... trying next')
                continue  # go for next folderPath
            # out of luck, raise with first folderPath
            err.args = (
                re.sub(path.replace('\\', '/'), folder_path[0], err.args[0]),)  # re.sub doesn't like double backslashes
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
        out = df.astype(str).add(sep).sum(axis=1).str.replace('%s+$' % escape(sep), '')  # removes trailing sep
        # need to make any columns with nan to output NaN, which is the result when 'A' + '_' + 'NaN'
        mask = df.isnull().any(axis=1)
        out[mask] = np.nan
    except AttributeError:
        # incase of empty data frame
        out = pd.Series()
    return out


def read_excel(file):
    """
    Read excel files

    Args:
        file (str):  Excel file path

    Returns
        dict with keys as sheet names and items as sheet contents as pd.dataframe
    """

    xl_workbook = xlrd.open_workbook(file)
    sheet_names = xl_workbook.sheet_names()
    out = {}
    for sheet in sheet_names:
        fl = pd.read_excel(file, sheet_names[0])
        out.update({sheet: fl})

    return out


def rolling_window(a, size):
    """
    Returns a rolling window of a n-dimensional array

    Parameters
    ----------
        a : np.ndarray
        size : int
            size of rolling window
    Returns
    -------
        n + 1 dimensional array
            the extra dimension added at the end

    Example
    -------
    >>> import numpy as np
    >>> rolling_window(np.random.randn(20),5).mean(axis=-1) # find the 5 element rolling mean of an array

    """
    a_ext = np.concatenate((np.full(a.shape[:-1] + (size - 1,), np.nan), a), axis=-1)
    strides = a_ext.strides + (a_ext.strides[-1],)
    return np.lib.stride_tricks.as_strided(a_ext, shape=(a.shape + (size,)), strides=strides)


def array_to_clipboard(array):
    """Copies an array into a string format acceptable by Excel.
    Note: Columns separated by "\\t", rows separated by "\\n"
    """
    # Create string from array
    line_strings = []

    array = array.astype(str)

    if array.ndim > 2:
        raise ValueError('could not operate on array with ndim >2')
    elif array.ndim == 1:
        for line in array:
            line_strings.append(line.replace("\n", ""))
    else:
        for line in array:
            line_strings.append('\t'.join([l for l in line]).replace("\n", ""))

    array_string = "\r\n".join(line_strings)

    # Put string into clipboard
    pyperclip.copy(array_string)


def format_csv_commas(path):
    """ Reads in data and replaces ", " with "" before returning List of cleaned data"""

    data = []
    with open(path, newline='') as f:
        for lines in f:
            new_line = lines.replace(", ", "")
            data.append(new_line)

    # compile lines and remove special charaters
    data = pd.Series(data).str.split(',', expand=True).replace({'\\n': '', '\\r': ''}, regex=True)
    data.columns = data.iloc[0, :]
    data = data.iloc[1:].reset_index(drop=True)
    return data


def prep_fund_data(df_path, date_col="Date"):
    """Prep fund data (csv) using char to date and setting 'Date' as the index

    Args:
        df_path (str): Path to dataframe
        date_col (str): Date column labelled in bloomberg dataframe

    Returns:
        df (dataframe): Dataframe with date columns converted to np.datetime64
    """
    df = pd.read_csv(df_path)
    df = char_to_date(df)

    assert date_col in df.columns, f"The date column: {date_col} is not specified in the function"

    df.set_index('Date', inplace=True)
    return df


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


def return_dict_keys(dict):
    """Returns keys of a dict in a list
    >>> return_dict_keys({'a':1, 'b':2, 'c':3})
    """
    return list(dict.keys())


def return_dict_values(dict):
    """
    Returns keys of a dict in a list
    >>> return_values({'a':1, 'b':2, 'c':3})
    """
    return list(dict.values())


def change_dict_keys(in_dict, text):
    """Change the keys of an input dictionary as with the text specified"""
    return {text + "_" + str(key): (change_dict_keys(value) if
                                    isinstance(value, dict) else
                                    value) for key, value in in_dict.items()}



# merge dictionaries: {**a,**b}
# merge two lists: dict(zip(list_one,list_two))

# if a function returns multiple arguments, label as follows for the variable unpacking: a,*_, b = var1, ...., var2
# example
# def func(dict):
#     return list(dict.keys())[0], list(dict.keys())[1], list(dict.keys())[2],list(dict.keys())[3]
# (a, *_, c) = func({'a':1, 'b':2, 'c':3, 'd':4}) --> a= 'a', c='d'
