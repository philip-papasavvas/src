# Created on 20 Sep 2020
import unittest

import numpy as np
import pandas as pd

from securityAnalysis.utils_finance import (
    calculate_relative_return_from_array, calculate_return_df,
    calculate_annualised_return_df
)

np.random.seed(1)  # set the random seed so the unit tests use synthetic data


class TestFinanceUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {'stock_a': {'03/06/2019': 325.7,
                         '04/06/2019': 323.3,
                         '05/06/2019': 325.5,
                         '06/06/2019': 324.1,
                         '07/06/2019': 323.9},
             'stock_b': {'03/06/2019': 857.0,
                         '04/06/2019': 861.0,
                         '05/06/2019': 862.0,
                         '06/06/2019': 867.0,
                         '07/06/2019': 883.0}
             })

    def test_calculate_relative_return_arr(self):
        np.testing.assert_array_almost_equal(
            calculate_relative_return_from_array(np.random.random_sample(5)),
            np.array([1.72730572e+00, 1.58782352e-04, 2.64334912e+03, 4.85412106e-01])
        )

    def test_calculate_return_from_df__absolute(self):
        # calculate the absolute return
        pd.testing.assert_frame_equal(
            calculate_return_df(data=self.data,
                                is_log_return=False,
                                is_relative_return=False,
                                is_absolute_return=True),
            pd.DataFrame(
                {'stock_a': {'04/06/2019': -2.3999999999999773,
                             '05/06/2019': 2.1999999999999886,
                             '06/06/2019': -1.3999999999999773,
                             '07/06/2019': -0.20000000000004547},
                 'stock_b': {'04/06/2019': 4.0,
                             '05/06/2019': 1.0,
                             '06/06/2019': 5.0,
                             '07/06/2019': 16.0}}
            )
        )

    def test_calculate_return_from_df__log(self):
        # calculate the log return
        pd.testing.assert_frame_equal(
            calculate_return_df(data=self.data,
                                is_log_return=True,
                                is_relative_return=False,
                                is_absolute_return=False),
            pd.DataFrame(
                {'stock_a': {'04/06/2019': -0.007396027550799822,
                             '05/06/2019': 0.006781776917236471,
                             '06/06/2019': -0.004310351501122689,
                             '07/06/2019': -0.0006172839702180966},
                 'stock_b': {'04/06/2019': 0.004656585829950544,
                             '05/06/2019': 0.001160766235962285,
                             '06/06/2019': 0.0057837061168486414,
                             '07/06/2019': 0.01828622382341827}}
            )
        )

    def test_calculate_return_from_df__relative(self):
        # calculate the absolute return
        pd.testing.assert_frame_equal(
            calculate_return_df(data=self.data,
                                is_log_return=False,
                                is_relative_return=True,
                                is_absolute_return=False),
            pd.DataFrame(
                {'stock_a': {'04/06/2019': -0.007368744243168468,
                             '05/06/2019': 0.006804825239715484,
                             '06/06/2019': -0.004301075268817178,
                             '07/06/2019': -0.000617093489663878},
                 'stock_b': {'04/06/2019': 0.004667444574095736,
                             '05/06/2019': 0.0011614401858304202,
                             '06/06/2019': 0.0058004640371229765,
                             '07/06/2019': 0.018454440599769306}}
            )
        )

    def test_calculate_annualised_return_df(self):
        pd.testing.assert_series_equal(
            calculate_annualised_return_df(data=self.data),
            pd.Series({'stock_a': -0.34537152900184453, 'stock_b': 1.8952787319995616})
        )


if __name__ == '__main__':
    unittest.main()
