# Created 20 Dec 2019

import json
import os
import unittest

import pandas as pd
import pandas.testing as pd_testing

from src.securityAnalysis.utils_finance import calculate_return_df

market_data_dir = f"{os.path.dirname(os.path.abspath(__file__))}/data"


class TestSecurities(unittest.TestCase):
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
        market_data = json.load(open(f"{market_data_dir}/test_market_data.json", "r"))

        sample_data = pd.DataFrame(data=market_data)

        pd.testing.assert_series_equal(
            pd.Series(calculate_return_df(data=sample_data,
                                          is_log_return=True).sum()),
            pd.Series(
                {'INDU Index': -0.020600636770248835,
                 'MXWO Index': -0.025305383694803556},
            )
        )


if __name__ == "__main__":
    unittest.main()
