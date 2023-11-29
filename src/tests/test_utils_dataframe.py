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
            'nums': [0.68535982, 0.95339335, 0.00394827, 0.51219226, 0.81262096,
                     0.61252607, 0.72175532, 0.29187607, 0.91777412, 0.71457578,
                     0.54254437, 0.14217005, 0.37334076, 0.67413362, 0.44183317,
                     0.43401399, 0.61776698, 0.51313824, 0.65039718, 0.60103895,
                     0.8052232, 0.52164715, 0.90864888, 0.31923609, 0.09045935,
                     0.30070006, 0.11398436, 0.82868133, 0.04689632, 0.62628715,
                     0.54758616, 0.819287, 0.19894754, 0.8568503, 0.35165264,
                     0.75464769, 0.29596171, 0.88393648, 0.32551164, 0.1650159,
                     0.39252924, 0.09346037, 0.82110566, 0.15115202, 0.38411445,
                     0.94426071, 0.98762547, 0.45630455, 0.82612284, 0.25137413,
                     0.59737165, 0.90283176]
        })

        df_two = df_one[['case', 'letters']]
        df_two['nums'] = [2.39470366, 0.91745894, -0.11227247, -0.36218045, -0.23218226,
                          -0.5017289, 1.12878515, -0.69781003, -0.08112218, -0.52929608,
                          1.04618286, -1.41855603, -0.36249918, -0.12190569, 0.31935642,
                          0.4609029, -0.21578989, 0.98907246, 0.31475378, 2.46765106,
                          -1.50832149, 0.62060066, -1.04513254, -0.79800882, 1.98508459,
                          1.74481415, -1.85618548, -0.2227737, -0.06584785, -2.13171211,
                          -0.04883051, 0.39334122, 0.21726515, -1.99439377, 1.10770823,
                          0.24454398, -0.06191203, -0.75389296, 0.71195902, 0.91826915,
                          -0.48209314, 0.08958761, 0.82699862, -1.95451212, 0.11747566,
                          -1.90745689, -0.92290926, 0.46975143, -0.14436676, -0.40013835,
                          -0.29598385, 0.84820861]

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
                 'absolute_diff_mean': {0: 0.9466936123076923, 1: 0.9533933273076923},
                 'absolute_diff_max': {0: 2.31354469, 1: 2.8517175999999997},
                 'pc_diff_mean': {0: 3.8199560197379125, 1: 2.6630068865247094},
                 'pc_diff_max': {0: 29.43586431525706, 1: 17.284562899682026}}
            )
        )


if __name__ == '__main__':
    unittest.main()
