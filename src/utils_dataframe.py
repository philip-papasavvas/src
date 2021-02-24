"""
Created on: 26 Feb 2021
Utils module for comparing pd DataFrames
"""
import re
from typing import Union, List

import numpy as np
import pandas as pd


def get_selected_column_names(df: pd.DataFrame,
                              cols_to_exclude: Union[List[str], str]) -> List[str]:
    """Return a list with all the column names in the data frame except those specified"""
    return [x for x in list(df.columns) if x not in cols_to_exclude]


def compare_dataframe_col(df_one: pd.DataFrame,
                          df_two: pd.DataFrame,
                          index_col: str,
                          merge_col: str,
                          suffixes: tuple = ('_x', '_y')) -> pd.DataFrame:
    """
    Compare two dataframes, specifically for a column choose the common index.
    Percentage difference will be relative to the first suffix dataframe

    Args:
        df_one: first dataframe to compare
        df_two: second dataframe to compare
        index_col: common column (in both data frames) on which to use as the 'index'
        merge_col: common column on which to carry out the merge, and compute the differences
        suffixes: specifed to give detail to different data frames compared

    Returns:
        pd.DataFrame: A dataframe with columns:
        [index, merge_col_first_suffix, merge_col_second_suffix, absolute_diff, pc_diff]
        Note: the percentage diff is given in decimals. 2% = 0.02
    """
    print(f"Performing data frame compare with index: \t {index_col}. \n"
          f"Merge column: \t {merge_col} ")

    merged_df = pd.merge(
        left=df_one.set_index(index_col)[[merge_col]],
        right=df_two.set_index(index_col)[[merge_col]],
        left_on=index_col,
        right_on=index_col,
        suffixes=suffixes,
        how='outer'
    ).fillna(0)

    merged_df['absolute_diff'] = np.abs(merged_df[merge_col + suffixes[0]].values -
                                        merged_df[merge_col + suffixes[1]].values)
    merged_df['pc_diff'] = \
        np.abs(np.abs(merged_df['absolute_diff']) / np.abs(merged_df[merge_col + suffixes[0]]))

    return merged_df


def reconcile_dataframes_numeric(df_one: pd.DataFrame,
                                 df_two: pd.DataFrame,
                                 tolerance: float = 1E-12) -> pd.DataFrame:
    """Method to reconcile two dataframes. This is different to
    pd.testing.assert_frame_equal since it allows the user to set a tolerance
    the difference between the array values.
    It will check that the index and columns are the same. If the columns are the
    same but ordered differently, it will sort them before the comparison takes place

    Args:
        df_one: pd.DataFrame
        df_two: pd.DataFrame
        tolerance: specify the tolerance between the values in the array

    Returns:
        pd.DataFrame: returns the difference between the two dataframes, with
        labelled columns
    """
    assert isinstance(df_one, pd.DataFrame), 'df_one is not a dataframe'
    assert isinstance(df_two, pd.DataFrame), 'df_two is not a dataframe'
    assert all(df_one.index == df_two.index), 'indices do not match'
    assert df_one.shape == df_two.shape, 'shapes of the dataframes do not match'

    assert all(np.in1d(df_one.columns, df_two.columns)), 'column values do not match'

    compare_mat = df_two.loc[:, df_one.columns]

    if np.max(np.absolute(compare_mat.values - df_one.values)) < tolerance:
        print("Data frames reconcile")
    else:
        print("Data frames did not reconcile")

    return pd.DataFrame(np.absolute(compare_mat.values - df_one.values),
                        columns=df_one.columns)


def drop_null_columns_df(data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns from the dataframe with null values"""
    original_columns = list(data.columns)
    cleaned_data = data.dropna(axis=1)
    new_columns = list(cleaned_data.columns)
    cut_columns = [x for x in original_columns if x not in new_columns]

    print(f"Columns: {cut_columns}  \n have been dropped from the dataframe as they contain NaNs")
    return cleaned_data


def replace_underscores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace underscores in a pd.DataFrame"""
    return df.replace({"_": ""}, regex=True)


def concat_columns(sep: str = '', *args) -> pd.DataFrame:
    """Concatenate multiple columns of pd.DataFrame with sep"""
    df = pd.DataFrame()
    for arg in args:
        df = pd.concat([df, arg], axis=1, ignore_index=True)
    try:
        out = df.astype(str).add(sep).sum(axis=1).str.replace(
            '%s+$' % re.escape(sep), '', regex=True)  # removes trailing sep
        # need to make any columns with nan to output NaN, which is the result when 'A' + '_' +
        # 'NaN'
        mask = df.isnull().any(axis=1)
        out[mask] = np.nan
    except AttributeError:
        # incase of empty data frame
        out = pd.Series()
    return out


if __name__ == '__main__':
    sample_df = pd.DataFrame(
        {"a": ["liquid", "arrogant", "imagine", "knock", "share"],
         "b": range(5)})

    concat_columns('_', sample_df[['a', 'b']])
