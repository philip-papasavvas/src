# Created 26 Feb 2021
import string
import unittest

import numpy as np
import pandas as pd

from utils.utils_dataframe import (replace_underscores_df, drop_null_columns_df,
                                   compare_dataframe_col, reconcile_dataframes_numeric,
                                   return_reconciliation_summary_table)

np.random.seed(10)


class TestUtilsDataframe(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_data = {'a': np.repeat(np.nan, 10),
                       'b': np.random.random_sample(10),
                       'c': np.repeat(1, 10)}
        cls.sample_df = pd.DataFrame(sample_data)

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
        self.assertListEqual(
            drop_null_columns_df(data=self.sample_df).columns.to_list(),
            ['b', 'c']
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

    def test_reconcile_dataframes_numeric(self):
        df_example = pd.DataFrame(data=np.random.rand(25).reshape(5, 5),
                                  columns='A,B,C,D,E'.split(','))
        df_two = df_example + np.exp(0.5)

        pd.testing.assert_frame_equal(
            reconcile_dataframes_numeric(df_one=df_example,
                                         df_two=df_two),
            pd.DataFrame(data=np.array(
                [[1.64872127, 1.64872127, 1.64872127, 1.64872127, 1.64872127],
                 [1.64872127, 1.64872127, 1.64872127, 1.64872127, 1.64872127],
                 [1.64872127, 1.64872127, 1.64872127, 1.64872127, 1.64872127],
                 [1.64872127, 1.64872127, 1.64872127, 1.64872127, 1.64872127],
                 [1.64872127, 1.64872127, 1.64872127, 1.64872127, 1.64872127]]),
                columns='A,B,C,D,E'.split(',')
            )
        )

    def test_return_reconciliation_summary_table(self):
        df_one = pd.DataFrame({
            'case': np.concatenate(
                [np.tile('lowers', len(string.ascii_lowercase)),
                 np.tile('uppers', len(string.ascii_uppercase))]),
            'letters': list(map(str, string.ascii_letters)),
            'nums': np.random.random(len(string.ascii_letters))
        })

        df_two = df_one[['case', 'letters']]
        df_two['nums'] = np.random.randn(len(string.ascii_letters))

        dataframe_differences = compare_dataframe_col(
            df_one=df_one,
            df_two=df_two,
            index_col=['case', 'letters'],
            merge_col='nums',
            suffixes=('_one', '_two')
        )

        pd.testing.assert_frame_equal(
            return_reconciliation_summary_table(differences_df=dataframe_differences,
                                                groupby_key='case').reset_index(),
            pd.DataFrame(
                {'case': {0: 'lowers', 1: 'uppers'},
                 'absolute_diff_mean': {0: 0.6466922873690832, 1: 1.053783120181656},
                 'absolute_diff_max': {0: 1.709343846516738, 1: 2.8512440721987904},
                 'pc_diff_mean': {0: 3.3140623932672395, 1: 3.2560714420173884},
                 'pc_diff_max': {0: 29.435891064487656, 1: 20.944493382113617}}
            )
        )


if __name__ == '__main__':
    unittest.main()
