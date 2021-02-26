# Created 26 Feb 2021
import unittest
import pandas as pd
import numpy as np

from utils_dataframe import (replace_underscores_df, drop_null_columns_df,
                             compare_dataframe_col)

np.random.seed(10)


class TestUtilsDataframe(unittest.TestCase):
    def assert_dataframe_equal(self, a, b, msg):
        """utilise pandas.testing module to check if dataframes are the same"""
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_replace_underscores_df(self):
        sample_df = pd.DataFrame({'a': ['here_there', 'are_underscores', 'underscores__']})
        pd.testing.assert_frame_equal(
            replace_underscores_df(df=sample_df),
            pd.DataFrame({'a': {0: 'herethere', 1: 'areunderscores', 2: 'underscores'}})
        )

    def test_drop_null_columns_df(self):
        # used random numbers using np.random.seed(10)
        pd.testing.assert_frame_equal(
            drop_null_columns_df(data=pd.DataFrame(
                {'a': np.repeat(np.nan, 10),
                 'b': np.random.random_sample(10),
                 'c': np.repeat(1, 10)})),
            pd.DataFrame(
                {'b': {0: 0.771320643266746,
                       1: 0.0207519493594015,
                       2: 0.6336482349262754,
                       3: 0.7488038825386119,
                       4: 0.4985070123025904,
                       5: 0.22479664553084766,
                       6: 0.19806286475962398,
                       7: 0.7605307121989587,
                       8: 0.16911083656253545,
                       9: 0.08833981417401027},
                 'c': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}}
            )
        )

    def test_compare_dataframe_col(self):
        df_one = pd.DataFrame(
            {'a': np.arange(1, 6),
             'b': np.arange(11, 16),
             'c': np.linspace(start=10, stop=12, num=5)}
        )
        df_two = pd.DataFrame(
            {'a': np.arange(1, 6),
             'b': [round(x * 1.1, 2) for x in np.arange(11, 16)],
             'c': [x + 0.5 for x in np.linspace(start=10, stop=12, num=5)]}
        )

        pd.testing.assert_frame_equal(
            compare_dataframe_col(df_one=df_one,
                                  df_two=df_two,
                                  suffixes=('_one', '_two'),
                                  index_col='a',
                                  merge_col='b').reset_index(drop=True),
            pd.DataFrame(
                {'b_one': {0: 11, 1: 12, 2: 13, 3: 14, 4: 15},
                 'b_two': {0: 12.1, 1: 13.2, 2: 14.3, 3: 15.4, 4: 16.5},
                 'absolute_diff': {0: 1.0999999999999996,
                                   1: 1.1999999999999993,
                                   2: 1.3000000000000007,
                                   3: 1.4000000000000004,
                                   4: 1.5},
                 'pc_diff': {0: 0.09999999999999996,
                             1: 0.09999999999999994,
                             2: 0.10000000000000006,
                             3: 0.10000000000000002,
                             4: 0.1}}
            )
        )


if __name__ == '__main__':
    unittest.main()
