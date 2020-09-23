# Created 20 Dec 2019

import json
import os
import unittest

import pandas as pd
import pandas.testing as pd_testing

from src.securityAnalysis import utils_finance


class TestUtils(unittest.TestCase):
    """Unit tests for utils module"""

    def assert_dataframe_equal(self, a, b, msg):
        """utilise pandas.testing module to check if dataframes are the same"""
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assert_dataframe_equal)

    def test_utils_securities_funcs(self):
        with open(os.path.join("/Users/philip_p/python/src", "securityAnalysis",
                               "data", "market_data.json"), "r") \
                as mkt_data:
            k = json.load(mkt_data)

        sample_data = pd.DataFrame(data=k)

        self.assertEqual(first=utils_finance.calculate_log_returns(data=sample_data.iloc[:3, :]),
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
