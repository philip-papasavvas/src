"""
Created on: 14 Apr 2021

Boilerplate Parser class accepting one of multiple data frames and a list of parser
keys are the *args. Outputs a single pd.DataFrame

This file contains the since parser functions where the first parameter is pd.DataFrame
and returns pd.DataFrame
"""
from typing import List, Union, Dict

import pandas as pd


# generic parsing methods
def get_columns(dataframe: pd.DataFrame,
                columns: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
    """Get the column names, and can rename according to list"""
    return dataframe[list(columns)].copy(True)


def rename_columns(dataframe: pd.DataFrame,
                   columns: Dict) -> pd.DataFrame:
    """Rename columns"""
    return dataframe.rename(columns=columns)


def name_columns(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Names columns if column names are not of type 'object'. Useful if reading a file
    without a header"""
    if dataframe.columns.dtype != 'O':
        # reading flat file
        dataframe.columns = columns

    return dataframe


def sort_columns(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Sort the dataframe by column values for columns defined"""
    return dataframe.sort_values(columns)


