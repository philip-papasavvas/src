# Created on 20 Sep 2020
import unittest

import numpy as np
import pandas as pd

from securityAnalysis.utils_finance import (
    calculate_relative_return_from_array, calculate_security_returns,
    calculate_annual_return, calculate_annual_volatility,
    return_sharpe_ratio, return_sortino_ratio, return_information_ratio
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
        return_arr = [0.5, 0.6, 0.7, 0.8, 0.9]
        np.testing.assert_array_almost_equal(
            calculate_relative_return_from_array(np.array(return_arr)),
            np.array([0.2, 0.16666667, 0.14285714, 0.125])
        )

    def test_calculate_return_from_df__absolute(self):
        # calculate the absolute return
        pd.testing.assert_frame_equal(
            calculate_security_returns(data=self.data,
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
            calculate_security_returns(data=self.data,
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
            calculate_security_returns(data=self.data,
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
            calculate_annual_return(data=self.data),
            pd.Series({'stock_a': -0.34537152900184453, 'stock_b': 1.8952787319995616})
        )

    def test_calculate_annual_volatility(self):
        result = calculate_annual_volatility(data=self.data)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result > 0))

    def test_return_sharpe_ratio(self):
        result = return_sharpe_ratio(security_prices=self.data,
                                     risk_free_rate_float=0.0)
        self.assertEqual(len(result), 2)

    def test_return_sharpe_ratio_with_rfr(self):
        result_zero = return_sharpe_ratio(security_prices=self.data,
                                          risk_free_rate_float=0.0)
        result_nonzero = return_sharpe_ratio(security_prices=self.data,
                                             risk_free_rate_float=0.05)
        # sharpe should be lower with a positive risk-free rate
        self.assertTrue(all(result_nonzero <= result_zero))

    def test_return_sortino_ratio(self):
        result = return_sortino_ratio(security_prices=self.data,
                                      target_return=0.0,
                                      risk_free=0.0)
        self.assertEqual(len(result), 2)

    def test_return_information_ratio(self):
        result = return_information_ratio(data=self.data)
        self.assertEqual(len(result), 2)

    def test_information_ratio_raises_on_nan(self):
        data_with_nan = self.data.copy()
        data_with_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            return_information_ratio(data=data_with_nan)

    def test_calculate_security_returns_raises_no_type(self):
        with self.assertRaises(ValueError):
            calculate_security_returns(data=self.data)

    def test_calculate_security_returns_raises_single_col(self):
        with self.assertRaises(ValueError):
            calculate_security_returns(data=self.data[['stock_a']],
                                       is_relative_return=True)


if __name__ == '__main__':
    unittest.main()
