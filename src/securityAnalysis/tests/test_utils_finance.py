# Created on 20 Sep 2020
import unittest

import numpy as np
import pandas as pd

from securityAnalysis.utils_finance import (calculate_relative_return, calculate_log_return_from_df,
                                            calculate_daily_return_from_df,
                                            calculate_annualised_return_from_df)

np.random.seed(1)  # set the random seed so the unit tests use synthetic data


class TestFinanceUtils(unittest.TestCase):
    def test_calculate_relative_returns(self):
        np.testing.assert_array_almost_equal(
            calculate_relative_return(np.random.random_sample(5)),
            np.array([1.72730572e+00, 1.58782352e-04, 2.64334912e+03, 4.85412106e-01])
        )

    def test_calculate_log_returns(self):
        pd.testing.assert_frame_equal(
            calculate_log_return_from_df(data=pd.DataFrame(
                {"A": [1, np.exp(1), np.exp(2), np.exp(3)]}
            )),
            pd.DataFrame({'A': {1: 1.0, 2: 1.0, 3: 1.0}})
        )

    def test_calculate_daily_return_from_df(self):
        pd.testing.assert_frame_equal(
            calculate_daily_return_from_df(data=pd.DataFrame(
                {"A": [2**x for x in range(1, 5)]})
            ),
            pd.DataFrame({'A': {1: 1.0, 2: 1.0, 3: 1.0}})
        )

    def test_calculate_annualised_return_from_df(self):
        np.testing.assert_array_almost_equal(
            calculate_annualised_return_from_df(pd.DataFrame(
                {'JPMorgan Fund ICVC - Emerging': {'03/06/2019': 325.7,
                                               '04/06/2019': 323.3,
                                               '05/06/2019': 325.5,
                                               '06/06/2019': 324.1,
                                               '07/06/2019': 323.9,
                                               '10/06/2019': 330.0,
                                               '11/06/2019': 333.6,
                                               '12/06/2019': 332.1,
                                               '13/06/2019': 332.2,
                                               '14/06/2019': 331.9},
             'Smithson Investment Trust PLC': {'03/06/2019': 1160.0,
                                               '04/06/2019': 1168.0,
                                               '05/06/2019': 1170.0,
                                               '06/06/2019': 1188.0,
                                               '07/06/2019': 1198.0,
                                               '10/06/2019': 1214.0,
                                               '11/06/2019': 1208.0,
                                               '12/06/2019': 1208.0,
                                               '13/06/2019': 1214.0,
                                               '14/06/2019': 1214.0},
             'Monks Investment Trust PLC': {'03/06/2019': 857.0,
                                            '04/06/2019': 861.0,
                                            '05/06/2019': 862.0,
                                            '06/06/2019': 867.0,
                                            '07/06/2019': 883.0,
                                            '10/06/2019': 891.0,
                                            '11/06/2019': 889.0,
                                            '12/06/2019': 883.0,
                                            '13/06/2019': 888.0,
                                            '14/06/2019': 884.0}})),
            pd.Series({'JPMorgan Fund ICVC - Emerging': 0.5365252785780994,
                       'Smithson Investment Trust PLC': 1.2821520597496012,
                       'Monks Investment Trust PLC': 0.8766239000922136})
        )


if __name__ == '__main__':
    unittest.main()
