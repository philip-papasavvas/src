"""
Created on: 14 Apr 2021

Boilerplate Parser class accepting one of multiple data frames and a list of parser
keys are the *args. Outputs a single pd.DataFrame

This file contains the since parser functions where the first parameter is pd.DataFrame
and returns pd.DataFrame
"""
from typing import List, Union

import pandas as pd


# generic parsing
def get_columns(dataframe: pd.DataFrame,
                columns: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
    """Get the column names, and can rename according to list"""
    return dataframe[list(columns)].copy(True)
