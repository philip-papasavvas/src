# Created on 24 Dec 2019

import datetime
import unittest

import numpy as np
import pandas as pd

from utils.utils_date import np_dt_to_str, excel_date_to_np, datetime_to_str, time_delta_to_days


class TestDateFuncs(unittest.TestCase):
    def setUp(self) -> None:
        dates = np.arange("2010-01-01", "2010-01-05", dtype='datetime64[D]')
        self.dates = dates
        self.dataframe_dates = pd.DataFrame(
            {'date': dates,
             'date_as_str': dates.astype(str)}
        )

    def test_date_to_str(self):
        np.testing.assert_equal(np_dt_to_str(np.datetime64("2020-01-01")),
                                "20200101")

    def test_x2pdate(self):
        self.assertEqual(list(excel_date_to_np(xl_date=43100)),
                         [np.datetime64('2017-12-31')],
                         "Should be 2017-12-31 in a list")

    def test_datetime_to_str(self):
        self.assertEqual(
            datetime_to_str(input_date=datetime.datetime(2020, 1, 1)),
            "20200101"
        )

    def test_time_delta_to_days(self):
        td = pd.Series(pd.to_timedelta(['1 days', '3 days', '7 days']))
        np.testing.assert_array_equal(
            time_delta_to_days(td),
            np.array([1, 3, 7])
        )

    def test_excel_date_to_np_single(self):
        result = excel_date_to_np(xl_date=1)
        self.assertEqual(list(result), [np.datetime64('1899-12-31')])

    def test_np_dt_to_str_with_hyphens(self):
        result = np_dt_to_str(np.datetime64("2023-12-25"))
        self.assertEqual(result, "20231225")


if __name__ == '__main__':
    unittest.main()
