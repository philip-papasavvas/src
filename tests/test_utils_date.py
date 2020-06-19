# Created on 24 Dec 2019

import unittest

import numpy as np
import pandas as pd

from utils_date import date_to_str, char_to_date, excel_date_to_np


class TestDateFuncs(unittest.TestCase):
    def test_date_to_str(self):
        np.testing.assert_equal(date_to_str(np.datetime64("2020-01-01")),
                                "20200101")

        np.testing.assert_equal(date_to_str(np.datetime64("2100-01-01")),
                                "21000101")

    def test_char_to_date(self):
        d1 = np.arange("2010-01-01", "2011-01-01", dtype='datetime64[D]')
        df = pd.DataFrame({'date1': d1, 'date2': d1.astype(str)})

        # test for dataframe
        df1 = char_to_date(df)

        for date_col in ['date1', 'date2']:
            np.testing.assert_array_equal(df1[date_col].values,
                                          d1)

        # series
        for date_col in ['date1', 'date2']:
            np.testing.assert_array_equal(char_to_date(df[date_col]),
                                          d1)

    def test_x2pdate(self):
        self.assertEqual(list(excel_date_to_np(xl_date=43100)),
                         [np.datetime64('2017-12-31')],
                         "Should be 2017-12-31 in a list")


if __name__ == '__main__':
    unittest.main()
