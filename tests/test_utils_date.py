# Created on 24 Dec 2019

import unittest

import numpy as np
import pandas as pd

from utils_date import date_to_str, char_to_date, excel_date_to_np


class TestDateFuncs(unittest.TestCase):
    def setUp(self) -> None:
        dates = np.arange("2010-01-01", "2010-01-05", dtype='datetime64[D]')
        self.dates = dates
        self.dataframe_dates = pd.DataFrame(
            {'date': dates,
             'date_as_str': dates.astype(str)}
        )

    def test_date_to_str(self):
        np.testing.assert_equal(date_to_str(np.datetime64("2020-01-01")),
                                "20200101")

    def test_char_to_date__series_as_dt(self):
        # test for dataframe
        np.testing.assert_array_equal(
            char_to_date(self.dataframe_dates['date']),
            self.dates
        )

    def test_char_to_date__series_as_str(self):
        # test for dataframe
        np.testing.assert_array_equal(
            char_to_date(self.dataframe_dates['date_as_str']),
            self.dates
        )

    def test_x2pdate(self):
        self.assertEqual(list(excel_date_to_np(xl_date=43100)),
                         [np.datetime64('2017-12-31')],
                         "Should be 2017-12-31 in a list")


if __name__ == '__main__':
    unittest.main()
