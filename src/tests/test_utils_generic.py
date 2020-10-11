# Created 12 Jun 2020
import unittest

import numpy as np
import pandas as pd

from utils_generic import (average, difference, flatten_dict, return_dict_keys,
                           return_dict_values, change_dict_keys, df_columns_to_dict,
                           convert_config_dates, drop_null_columns_df)

np.random.seed(10)


class TestUtilsGeneric(unittest.TestCase):
    def assert_dataframe_equal(self, a, b, msg):
        """utilise pandas.testing module to check if dataframes are the same"""
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_average__tuple(self):
        self.assertEqual(average(2, 2, 5),
                         3,
                         "Average has not calculated correctly")

    def test_average__list(self):
        self.assertEqual(average(*[1, 2, 3]),
                         2,
                         "Average has not calculated correctly")

    def test_difference(self):
        self.assertEqual(difference([3, 10, 9], [3, 4, 10]),
                         {9},
                         "Difference function not working as expected")

    def test_utils_flatten_dict(self):
        self.assertEqual(
            flatten_dict(d={'a': 'first level',
                            'b': {'more detail': {'third level'},
                                  'second level': [0]}}
                         ),
            {'a': 'first level', 'b.more detail': {'third level'},
             'b.second level': [0]},
            "Dict should've been flatten to have "
            "two sub keys on b level: more detail, second level")

    def test_utils_return_dict_keys(self):
        self.assertEqual(return_dict_keys(dct={'a': 1, 'b': 2, 'c': 3}),
                         ['a', 'b', 'c'],
                         "Should've returned ['a', 'b', 'c']")

    def test_utils_return_dict_values(self):
        self.assertEqual(return_dict_values(dct={'a': 1, 'b': 2, 'c': 3}),
                         [1, 2, 3],
                         "Should've returned [1,2,3]")

    def test_utils_change_dict_keys(self):
        self.assertEqual(change_dict_keys(in_dict={'a': [1], 'b': [2]}, text='test'),
                         {'test_a': [1], 'test_b': [2]},
                         "Should've returned keys 'test_a', 'test_b' ")

    def test_df_columns_to_dict(self):
        self.assertEqual(
            df_columns_to_dict(df=pd.DataFrame(
                {'A': [1, 2, 3, 4],
                 'B': ['one', 'two', 'three', 'four']}),
                columns=['A', 'B']
            ),
            {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        )

    def test_convert_config_dates(self):
        self.assertEqual(
            convert_config_dates(config={'date_one': "2020-01-01",
                                         "date_two": "2019-01-01",
                                         "DATE": "2010-12-25"}),
            {'date_one': np.datetime64('2020-01-01'),
             'date_two': np.datetime64('2019-01-01'),
             'DATE': np.datetime64('2010-12-25')}
        )

    def test_drop_null_columns_df(self):
        # used random numbers using np.random.seed(1)
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


if __name__ == '__main__':
    unittest.main()
