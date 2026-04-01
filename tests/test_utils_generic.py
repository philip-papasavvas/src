# Created 12 Jun 2020
import unittest

import numpy as np
import pandas as pd

from aialpha.utils.generic import (average, difference, flatten_dict,
                                   change_dict_keys, dict_from_df_cols,
                                   convert_config_dates, to_array, match,
                                   linear_bucketing)


class TestUtilsGeneric(unittest.TestCase):

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

    def test_utils_change_dict_keys(self):
        self.assertEqual(change_dict_keys(in_dict={'a': [1], 'b': [2]}, text='test'),
                         {'test_a': [1], 'test_b': [2]},
                         "Should've returned keys 'test_a', 'test_b' ")

    def test_df_columns_to_dict(self):
        self.assertEqual(
            dict_from_df_cols(df=pd.DataFrame(
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

    def test_to_array__list(self):
        result, = to_array([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_to_array__int(self):
        result, = to_array(5)
        np.testing.assert_array_equal(result, np.array([5]))

    def test_to_array__none(self):
        result, = to_array(None)
        np.testing.assert_array_equal(result, np.array([]))

    def test_to_array__datetime64(self):
        result, = to_array(np.datetime64("2020-01-01"))
        self.assertEqual(result[0], np.datetime64("2020-01-01"))

    def test_to_array__series(self):
        s = pd.Series([10, 20, 30])
        result, = to_array(s)
        np.testing.assert_array_equal(result, np.array([10, 20, 30]))

    def test_match__strict(self):
        result = match(x=[46, 15, 5], y=[5, 4, 46, 6, 15, 1, 70])
        np.testing.assert_array_equal(result, np.array([2, 4, 0]))

    def test_match__non_strict(self):
        result = match(x=[46, 99, 5], y=[5, 4, 46, 6, 15, 1, 70], strict=False)
        self.assertEqual(result[0], 2)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 0)

    def test_linear_bucketing(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = linear_bucketing(x, y)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_change_dict_keys_nested(self):
        result = change_dict_keys(in_dict={'a': {'inner': 1}, 'b': 2}, text='prefix')
        self.assertEqual(result, {'prefix_a': {'prefix_inner': 1}, 'prefix_b': 2})


if __name__ == '__main__':
    unittest.main()
