# Created 20 Dec 2019

import json
import os
import unittest

import pandas as pd
import pandas.testing as pd_testing

import utils_generic
from securityAnalysis import utils_finance


class Testutils(unittest.TestCase):
    """Unit tests for utils module"""

    def assert_dataframe_equal(self, a, b, msg):
        """utilise pandas.testing module to check if dataframes are the same"""
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assert_dataframe_equal)

    def test_utils_funcs(self):
        self.assertEqual(utils_generic.average(2, 2, 5), 3, "Average has not calculated correctly")

        self.assertEqual(utils_generic.average(*[1, 2, 3]), 2, "Average has not calculated correctly")

        self.assertEqual(utils_generic.difference([3, 10, 9], [3, 4, 10]),
                         {9},
                         "Difference function not working as expected")

    def test_utils_dict_funcs(self):
        self.assertEqual(utils_generic.flatten_dict(d={'a': 'first level',
                                               'b': {'more detail': {'third level'},
                                                     'second level': [0]}
                                                       }
                                                    ),
                         {'a': 'first level', 'b.more detail': {'third level'},
                          'b.second level': [0]},
                         "Dict should've been flatten to have "
                         "two sub keys on b level: more detail, second level")

        self.assertEqual(utils_generic.return_dict_keys(dict={'a': 1, 'b': 2, 'c': 3}),
                         ['a', 'b', 'c'],
                         "Should've returned ['a', 'b', 'c']")

        self.assertEqual(utils_generic.return_dict_values(dict={'a': 1, 'b': 2, 'c': 3}),
                         [1, 2, 3],
                         "Should've returned [1,2,3]")

        self.assertEqual(utils_generic.change_dict_keys(in_dict={'a': [1], 'b': [2]}, text='test'),
                         {'test_a': [1], 'test_b': [2]},
                         "Should've returned keys 'test_a', 'test_b' ")

    def test_utils_securities_funcs(self):
        with open(os.path.join("/Users/philip_p/python/src", "securityAnalysis",
                               "data", "market_data.json"), "r") \
                as mkt_data:
            k = json.load(mkt_data)

        sample_data = pd.DataFrame(data=k)

        self.assertEqual(first=utils_finance.log_daily_returns(data=sample_data.iloc[:3, :]),
                         second=pd.DataFrame(data=
                                             {'INDU Index': {'01/06/2011': -0.007019973807377511,
                                                             '02/05/2011': 0.04122269078824914},
                                              'MXWO Index': {'01/06/2011': -0.004377976690467911,
                                                             '02/05/2011': 0.041267840353913954}
                                              }
                                             ),
                         msg="Data-frames not equal")


if __name__ == "__main__":
    unittest.main()
